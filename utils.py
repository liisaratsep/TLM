# coding=utf-8

import dynet as dy
from collections import Counter
import random
import math
import logging
import pickle


class TLM:
    def __init__(self, options, train, dev):
        resume = options.resume
        score = options.score
        last_epoch = options.last_epoch
        self._filename = options.model
        self._dir = options.output

        self._train = train
        self._dev = dev

        if (not options.conll_train and not options.conll_dev) or resume:  # loading options from pickle
            with open(self._dir.rstrip('/') + '/' + self._filename + '.pickle', 'rb') as pkl:
                options, self._int2word, self._word2int, self._vocab_size = pickle.load(pkl)

        else:  # processing training data to create vocabulary data structures
            self._int2word = [w for w, _ in Counter([word[1] for line in self._train
                                                     for word in line]).most_common(options.vocab)]
            self._int2word.append("<UNK>")
            self._word2int = {c: i for i, c in enumerate(self._int2word)}  # dict {word:id}

            self._vocab_size = len(self._int2word)  # number of uniq words

            with open(self._dir.rstrip('/') + '/' + self._filename + '.pickle', 'wb') as pkl:
                pickle.dump((options, self._int2word, self._word2int, self._vocab_size), pkl)

        self._use_syntax = options.syntax
        self._stack_window = options.stack_window
        self._lstm = options.lstm

        self._batch = options.batch

        self._model = dy.Model()

        self._wemb = options.wemb
        self._hidden_generate = options.hidden_generate
        self._hidden_parse = options.hidden_parse

        if self._use_syntax:
            self._layers1 = self._layers2 = options.layers
        else:
            self._layers1 = math.floor(options.layers / 2)
            self._layers2 = math.ceil(options.layers / 2)

        self._dropout = options.dropout

        # Input size and multiplier
        if self._use_syntax:
            input_count = self._stack_window
        else:
            input_count = 1

        self._epochs = range(options.epochs)

        # Word lookup embedding martix:
        self._word_lookup = self._model.add_lookup_parameters((self._vocab_size + 1, self._wemb))  # + empty
        # LSTM used for generating word representations
        self._word_lstm = dy.VanillaLSTMBuilder(self._layers1, self._wemb, self._wemb, self._model)

        if self._lstm:
            # LSTM used for word generation
            self._gen_lstm = \
                dy.VanillaLSTMBuilder(self._layers2, self._wemb * input_count, self._hidden_generate, self._model)
            self._generate_W = self._model.add_parameters((self._vocab_size, self._hidden_generate))
            self._generate_bias = self._model.add_parameters((self._vocab_size,))
        else:  # if new word is generated with MLP
            self._gen_W2 = self._model.add_parameters((self._hidden_generate, self._wemb * input_count))
            self._gen_bias2 = self._model.add_parameters((self._hidden_generate,))
            self._gen_W1 = self._model.add_parameters((self._hidden_generate, self._hidden_generate))
            self._gen_bias1 = self._model.add_parameters((self._hidden_generate,))
            self._gen_W_out = self._model.add_parameters((self._vocab_size, self._hidden_generate))
            self._gen_bias_out = self._model.add_parameters((self._vocab_size,))

        if self._use_syntax:
            # MLP for parsing actions
            self._action_W2 = self._model.add_parameters((self._hidden_parse, self._wemb * 4))
            self._action_bias2 = self._model.add_parameters((self._hidden_parse,))
            self._action_W1 = self._model.add_parameters((self._hidden_parse, self._hidden_parse))
            self._action_bias1 = self._model.add_parameters((self._hidden_parse,))
            self._action_W_out = self._model.add_parameters((3, self._hidden_parse))  # output: LA, RA, SHIFT
            self._action_bias_out = self._model.add_parameters((3,))

        self._trainer = dy.AdamTrainer(self._model)

        self._generate_lstm_state = None
        self._word_lstm_state = None

        self._empty = None
        self._stack = []
        self._buffer = []
        self._generated = []
        self._word_counter = 0

        # Logger configuration
        self._logger = logging.getLogger(self._filename)
        self._logger.setLevel(logging.INFO)
        handler = logging.FileHandler(self._dir.rstrip('/') + '/' + self._filename + '.log')
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)
        self._logger.addHandler(logging.StreamHandler())

        self._score = None  # Top perplexity score
        self._best_epoch = None

        if resume:
            self._score = score
            self._epochs = range(last_epoch + 1, options.epochs)

    def _get_word_vectors(self, sentence):
        wembs = [self._word_lookup[self._vocab_size]]
        for i in range(len(sentence)):
            word = sentence[i]
            if not word[1] in self._word2int:
                sentence[i][1] = '<UNK>'
            wembs.append(self._word_lookup[self._word2int[word[1]]])

        wembs = self._word_lstm.initial_state().transduce(wembs)

        self._empty = wembs[0]

        for i in range(len(sentence)):
            if len(sentence[i]) > 3:
                sentence[i][3] = wembs[i + 1]
            else:
                sentence[i].append(wembs[i + 1])

        return sentence

    def _init_sentence(self):
        if self._lstm:
            self._generate_lstm_state = self._gen_lstm.initial_state()

        self._stack = []
        self._buffer = []
        self._generated = []
        self._word_counter = 0

    def _get_word_output(self):
        if self._use_syntax:
            input_vec = []
            for i in range(self._stack_window):
                input_vec.append(self._stack[-i - 1][3] if len(self._stack) > i else self._empty)
        else:
            input_vec = [self._generated[-1][3] if len(self._generated) > 0 else self._empty]

        input_vec = dy.concatenate(input_vec)

        if self._lstm:
            output = dy.softmax(
                self._generate_bias + self._generate_W * self._generate_lstm_state.transduce([input_vec])[-1])
        else:
            output = dy.softmax(self._gen_bias_out + self._gen_W_out *
                                (dy.tanh(self._gen_bias1 + self._gen_W1 *
                                         dy.tanh(self._gen_bias2 + self._gen_W2 * input_vec))))

        return output

    def _get_action_output(self):
        input_vec = [self._buffer[0][3] if len(self._buffer) > 0 else self._empty]
        for i in range(3):
            input_vec.append(self._stack[-i - 1][3] if len(self._stack) > i else self._empty)

        input_vec = dy.concatenate(input_vec)

        output = dy.softmax(self._action_bias_out + self._action_W_out *
                            (dy.tanh(self._action_bias1 + self._action_W1 *
                                     dy.tanh(self._action_bias2 + self._action_W2 * input_vec))))

        return output

    def _save(self, epoch):
        if epoch is None:
            self._model.save(self._dir.rstrip('/') + '/' + self._filename + '.model')
        else:
            self._model.save(self._dir.rstrip('/') + '/' + str(epoch) + '-' + self._filename + '.model')

    def load(self, last_epoch=None):
        if last_epoch:
            self._model.populate(self._dir.rstrip('/') + '/' + str(last_epoch) + '-' + self._filename + '.model')
        else:
            self._model.populate(self._dir.rstrip('/') + '/' + self._filename + '.model')

    def evaluate(self, corpus):
        def test_sentence_tree():
            if len(sent) > 1:
                self._init_sentence()
                sentence = self._get_word_vectors(sent)

                prob = 0

                while True:
                    if len(self._buffer) == 0 and (len(self._generated) == 0 or self._generated[-1][2] != -1):  # gen
                        output = self._get_word_output()
                        word = sentence[self._word_counter]
                        prob = prob + math.log(output[self._word2int[word[1]]].value(), 2)

                        if word[1] == "<EOS>":
                            self._generated.append([0, word[1], -1, word[3]])
                            self._buffer.append([0, word[1], -1, word[3]])
                        else:
                            self._generated.append([self._word_counter + 1, word[1], None, word[3]])
                            self._buffer.append([self._word_counter + 1, word[1], None, word[3]])
                        self._word_counter += 1

                    else:  # pick action
                        output = self._get_action_output()

                        left_arc_conditions = len(self._stack) > 0 and len(self._buffer) > 0
                        right_arc_conditions = len(self._stack) > 1 and self._stack[-1][0] != 0
                        shift_conditions = len(self._buffer) > 0 and self._buffer[0][0] != 0

                        best = None
                        rnd = random.random()
                        for i, p in enumerate(output):
                            if (i == 0 and left_arc_conditions) or (i == 1 and right_arc_conditions) or (
                                    i == 2 and shift_conditions):
                                best = i
                                rnd -= p.value()
                                if rnd <= 0 and best < 3:
                                    break

                        if best == 0:  # left-arc
                            self._generated[self._stack[-1][0] - 1][2] = self._buffer[0][0]
                            del self._stack[-1]
                        elif best == 1:  # right-arc
                            self._generated[self._stack[-1][0] - 1][2] = self._stack[-2][0]
                            del self._stack[-1]
                        elif best == 2:  # shift
                            self._stack.append(self._buffer[0])
                            del self._buffer[0]

                    if len(self._stack) == 0 and len(self._buffer) == 1 and self._buffer[0][0] == 0:
                        uas_corr = 0
                        uas_tot = 0
                        if sentence[0][2] != '-':
                            for i in range(len(sentence)):
                                if sentence[i][2] == self._generated[i][2]:
                                    uas_corr += 1
                                uas_tot += 1
                        return self._word_counter, prob, uas_tot, uas_corr
            else:
                return 0, 0, 0, 0

        def test_sentence():
            if len(sent) > 1:
                self._init_sentence()
                sentence = self._get_word_vectors(sent)

                end = False
                prob = 0

                while True:
                    if len(self._generated) == 0 or self._generated[-1][2] != -1:  # generate
                        output = self._get_word_output()
                        word = sentence[self._word_counter]
                        prob = prob + math.log(output[self._word2int[word[1]]].value(), 2)

                        if word[1] == "<EOS>":
                            end = True
                            self._generated.append([0, word[1], -1, word[3]])
                        else:
                            self._generated.append([self._word_counter + 1, word[1], None, word[3]])

                        self._generated.append(word)
                        self._word_counter += 1

                    if end:
                        return self._word_counter, prob
            else:
                return 0, 0

        n = 0
        perplexity = 0
        uas_t = 0
        uas_c = 0
        try:
            self._logger.info("Starting evaluation")
            s = 1
            for sent in corpus:
                dy.renew_cg()
                if sent[0][2] is None or not self._use_syntax:
                    c, perp = test_sentence()
                    perplexity = perplexity + perp
                    n = n + c
                else:
                    c, perp, uast, uasc = test_sentence_tree()
                    perplexity = perplexity + perp
                    n = n + c
                    uas_t += uast
                    uas_c += uasc
                s += 1
                if s % 100 == 0 or s == len(corpus):
                    self._logger.info("Processed sentence " + str(s) + " of " + str(len(corpus)))

            perplexity = 2 ** ((-1 / n) * perplexity)
            self._logger.info("Perplexity: " + str(perplexity))
            if self._use_syntax:
                if uas_c > 0:
                    uas = 100 * uas_c / uas_t
                else:
                    uas = 0
                self._logger.info("UAS: " + str(round(uas, 4)) + ' %')

            return perplexity
        except ValueError:
            self._logger.info("Infinite perplexity reached. Stopping evaluation")
            return None

    def train(self):
        def learn_sentence_tree():
            self._init_sentence()

            sentence = self._get_word_vectors(sent)

            while not (len(self._stack) == 0 and len(self._buffer) == 1 and self._buffer[0][0] == 0):
                if len(self._buffer) == 0 and (len(self._generated) == 0 or self._generated[-1][2] != -1):  # generate
                    output = self._get_word_output()

                    self._buffer.append(sentence[self._word_counter])

                    word = sentence[self._word_counter][1]
                    if word not in self._word2int:
                        word = "<UNK>"
                    errs.append(-dy.log(dy.pick(output, self._word2int[word])))

                    self._word_counter += 1
                else:  # pick action
                    output = self._get_action_output()

                    if len(self._stack) > 0 and self._stack[-1][2] == self._buffer[0][0]:
                        valid = 0
                        del self._stack[-1]
                    elif len(self._stack) > 1 and self._stack[-1][2] == self._stack[-2][0] \
                            and self._stack[-1][0] not in [sentence[dep][2] for dep in
                                                           range(self._word_counter, len(sentence))] \
                            and not self._stack[-1][0] == self._buffer[0][2]:
                        valid = 1
                        del self._stack[-1]
                    elif self._buffer[0][0] != 0:
                        valid = 2
                        self._stack.append(self._buffer[0])
                        del self._buffer[0]
                    else:  # conll format error
                        break

                    errs.append(-dy.log(dy.pick(output, valid)))

        def learn_sentence():
            self._init_sentence()

            sentence = self._get_word_vectors(sent)

            while len(self._generated) == 0 or self._generated[-1][2] != -1:
                output = self._get_word_output()

                self._generated.append(sentence[self._word_counter])

                word = sentence[self._word_counter][1]
                if word not in self._word2int:
                    word = "<UNK>"
                errs.append(-dy.log(dy.pick(output, self._word2int[word])))

                self._word_counter += 1

        self._logger.info("Starting training, vocab size: " + str(self._vocab_size))
        for epoch in self._epochs:
            self._logger.info("Starting epoch " + str(epoch))

            if self._dropout:
                self._word_lstm.set_dropout(self._dropout)

            s = 1
            s_batch = 1
            random.shuffle(self._train)

            dy.renew_cg()
            errs = []
            for sent in self._train:
                if len(sent) > 1:
                    if sent[0][2] is None or not self._use_syntax:
                        learn_sentence()
                    else:
                        learn_sentence_tree()
                if s % self._batch == 0 or s == len(self._train):
                    loss = dy.esum(errs) / len(errs)
                    loss_value = loss.value()
                    if s % (self._batch * 10) == 0 or s == len(self._train):
                        self._logger.info("Processed sentence " + str(s) + " of " + str(len(self._train)))
                        self._logger.info('Loss: ' + str(loss_value))
                    loss.backward()
                    self._trainer.update()
                    dy.renew_cg()
                    errs = []
                    s_batch = 0
                s += 1
                s_batch += 1

            if self._dropout:
                self._word_lstm.disable_dropout()

            perplexity = self.evaluate(self._dev)
            self._save(epoch)
            # Overwrites best score if there is none or it is the best one yet, and saves the model as the default one.
            if perplexity is not None and (self._score is None or self._score > perplexity):
                self._save(None)
                self._score = perplexity
                self._best_epoch = epoch
            elif self._best_epoch is not None and self._best_epoch < epoch - 1:  # stop training if no improvement
                break
        self.load()  # Resets model to the best epoch

    def generate(self, n):
        self._logger.info('Starting sentence genration')
        for _ in range(n):
            dy.renew_cg()
            self._init_sentence()

            self._word_lstm_state = self._word_lstm.initial_state()
            self._empty = self._word_lstm_state.transduce([self._word_lookup[self._vocab_size]])[-1]

            while self._word_counter < 100 and not (len(self._generated) > 0 and self._generated[-1][2] == -1):
                if (len(self._buffer) == 0 or not self._use_syntax) and \
                        (len(self._generated) == 0 or self._generated[-1][2] != -1):  # generate
                    output = self._get_word_output()

                    rnd = random.random()
                    i = 0
                    for i, p in enumerate(output):
                        rnd -= p.value()
                        if rnd <= 0:
                            break

                    word = self._int2word[i]
                    word_vec = self._word_lstm_state.add_input(self._word_lookup[i]).output()

                    if word == "<EOS>":
                        self._generated.append([0, word, -1, word_vec])
                        self._buffer.append([0, word, -1, word_vec])
                    else:
                        self._generated.append([self._word_counter + 1, word, None, word_vec])
                        self._buffer.append([self._word_counter + 1, word, None, word_vec])

                    self._word_counter += 1

                elif self._use_syntax:  # pick action
                    output = self._get_action_output()

                    left_arc_conditions = len(self._stack) > 0 and len(self._buffer) > 0
                    right_arc_conditions = len(self._stack) > 1 and self._stack[-1][0] != 0
                    shift_conditions = len(self._buffer) > 0 and self._buffer[0][0] != 0

                    best = None
                    rnd = random.random()
                    for i, p in enumerate(output):
                        if (i == 0 and left_arc_conditions) or (i == 1 and right_arc_conditions) or (
                                i == 2 and shift_conditions):
                            best = i
                            rnd -= p.value()
                            if rnd <= 0 and best < 3:
                                break

                    if best == 0:
                        self._generated[self._stack[-1][0] - 1] = (
                            self._stack[-1][0], self._stack[-1][1], self._buffer[0][0], self._stack[-1][3])
                        del self._stack[-1]
                    elif best == 1:
                        self._generated[self._stack[-1][0] - 1] = (
                            self._stack[-1][0], self._stack[-1][1], self._stack[-2][0], self._stack[-1][3])
                        del self._stack[-1]
                    elif best == 2:
                        self._stack.append(self._buffer[0])
                        del self._buffer[0]

            self._logger.info(str([word[1] for word in self._generated]))


def conll_transform(conll):
    with open(conll, 'r') as f:
        lines = f.readlines()

    sentences = [[]]
    for line in lines:
        line = line.strip()
        if len(line) == 0:
            if sentences[-1]:
                sentences[-1].append([0, "<EOS>", -1])
                sentences.append([])
        elif line[0] != "#":
            cols = line.split('\t')
            if '-' not in cols[0]:
                if cols[6] != '_':
                    sentences[-1].append([int(cols[0]), cols[1], int(cols[6])])
                else:
                    sentences[-1].append([int(cols[0]), cols[1], None])
    if not sentences[-1]:
        sentences.pop()
    elif sentences[-1][-1] != [0, "<EOS>", -1]:
        sentences[-1].append([0, "<EOS>", -1])

    return sentences
