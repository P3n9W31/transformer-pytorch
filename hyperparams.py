class Hyperparams:
    """Hyperparameters"""

    # data
    source_train = "corpora/cn.txt"
    target_train = "corpora/en.txt"
    source_test = "corpora/cn.test.txt"
    target_test = "corpora/en.test.txt"

    # training
    batch_size = 48  # alias = N
    batch_size_valid = 32
    lr = (
        0.0001  # learning rate. In paper, learning rate is adjusted to the global step.
    )
    logdir = "logdir"  # log directory

    model_dir = "./models/"  # saving directory

    # model
    maxlen = 50  # Maximum number of words in a sentence. alias = T.
    min_cnt = 0  # words whose occurred less than min_cnt are encoded as <UNK>.
    hidden_units = 512  # alias = C
    num_blocks = 12  # number of encoder/decoder blocks
    num_epochs = 50
    num_heads = 8
    dropout_rate = 0.4
    sinusoid = False  # If True, use sinusoid. If false, positional embedding.
    eval_epoch = 1  # epoch of model for eval
    eval_script = 'scripts/validate.sh'
    check_frequency = 10  # checkpoint frequency
