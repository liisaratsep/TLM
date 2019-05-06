## A Generative Transition-based Dependency Language Model

A language model that uses the operations of an arc-hybrid transition-based parser while generating new words to the buffer based on the current items in the stack.

#### Requirements
- Python3
- DyNet library for Python3 (version 2.1)

To train the model, a corpora in the CoNLL-U format is required (the model uses ID, FORM and HEAD columns).

#### Usage

To train a baseline model:

    python3 tml.py --train train.conll --dev dev.conll --dynet-autobatch 1 --no-syntax --model modelname
    
To train a transition-based model that uses an MLP to generate new words:

    python3 tml.py --train train.conll --dev dev.conll --dynet-autobatch 1 --model modelname

To train a transition-based model that uses LSTM units to generate new words:

    python3 tml.py --train train.conll --dev dev.conll --dynet-autobatch 1 --lstm --model modelname
    
To evaluate a trained model using test data:

    python3 tml.py --test test.conll  --model modelname
    
To generate sentences using the trained model:

    python3 tml.py --generate 20 --model modelname

By using all relevant flags at once, the testing and/or generation can be done right after training automatically. Details on all available paramaters can be checked with `python3 tml.py --help`.
