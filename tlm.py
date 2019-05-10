# coding=utf-8
from optparse import OptionParser
from utils import *

if __name__ == '__main__':
    parser = OptionParser()
    # Datasets
    parser.add_option("--train", dest="conll_train", help='Training corpus in CoNLL-U format')
    parser.add_option("--dev", dest="conll_dev", help="Development corpus in CoNLL-U format")
    parser.add_option("--test", dest="conll_test", help="Test corpus in CoNLL-U format")

    # Model naming and location
    parser.add_option("--model", dest="model", help="Model name for loading or saving", default="lm-syn")
    parser.add_option("--outdir", dest="output", help="The destination of all models and generated files",
                      default=".")
    # Hyperparameters
    parser.add_option("--wemb", dest="wemb", type="int", help="Word embedding dimensions", default=512)
    parser.add_option("--layers", dest="layers", type="int", help="Number of LSTM layers", default=2)
    parser.add_option("--hidden-gen", dest="hidden_generate", type="int",
                      help="Number of hidden units for word generation", default=512)
    parser.add_option("--hidden-parse", dest="hidden_parse", type="int", help="Number of hidden units for parsing",
                      default=256)
    parser.add_option("--dropout", dest="dropout", type="float", default=0.0,
                      help="LSTM dropout rate")

    parser.add_option("--epochs", dest="epochs", type="int", help="Number of epochs", default=10)
    parser.add_option("--stack", dest="stack_window", type="int",
                      help="Number of top items in stack used as generation input", default=2)
    parser.add_option("--vocab", dest="vocab", type="int", help="Vocabulary size", default=33247)
    parser.add_option("--batch", dest="batch", type="int", help="Batch size", default=16)

    # Running mode options
    parser.add_option("--mlp", action="store_false", dest="lstm", default=True,
                      help="For generating words with MLP (not LSTM)")
    parser.add_option("--no-syntax", action="store_false", dest="syntax", default=True,
                      help="For training without syntactic info.")
    parser.add_option("--generate", dest="generate", type="int", default=0,
                      help="The number of sentences to generate")
    parser.add_option("--dynet-mem")
    parser.add_option("--dynet-autobatch")
    parser.add_option("--dynet-seed")
    parser.add_option("--resume", dest="resume", action="store_true", default=False,
                      help="For resuming training")
    parser.add_option("--last-epoch", dest="last_epoch", type="int", help="Number of the last completed epoch",
                      default=-1)
    parser.add_option("--score", dest="score", type="int", help="Highest perplexity score from completed epochs",
                      default=0)

    options, _ = parser.parse_args()

    if options.conll_train and options.conll_dev:
        train = conll_transform(options.conll_train)
        dev = conll_transform(options.conll_dev)
        model = TLM(options, train, dev)
        if options.resume:
            model.load(options.last_epoch)
        model.train()
        if options.conll_test:
            test = conll_transform(options.conll_test)
            model.evaluate(test)
        model.generate(options.generate)
    elif options.conll_test or options.generate > 0:
        model = TLM(options, None, None)
        model.load()
        if options.conll_test:
            test = conll_transform(options.conll_test)
            model.evaluate(test)
        model.generate(options.generate)
