from hyperparams import Hyperparams as hp

import numpy as np
import codecs
import regex
import random


def load_vocab(language):
    assert language in ["cn", "en"]
    vocab = [
        line.split()[0]
        for line in codecs.open(
            "preprocessed/{}.txt.vocab.tsv".format(language), "r", "utf-8"
        )
        .read()
        .splitlines()
        if int(line.split()[1]) >= hp.min_cnt
    ]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word


def load_cn_vocab():
    word2idx, idx2word = load_vocab("cn")
    return word2idx, idx2word


def load_en_vocab():
    word2idx, idx2word = load_vocab("en")
    return word2idx, idx2word


def create_data(source_sents, target_sents):
    cn2idx, idx2cn = load_cn_vocab()
    en2idx, idx2en = load_en_vocab()

    # Index
    x_list, y_list, Sources, Targets = [], [], [], []
    for source_sent, target_sent in zip(source_sents, target_sents):
        x = [
            cn2idx.get(word, 1) for word in (source_sent + " </S>").split()
        ]  # 1: OOV, </S>: End of Text
        y = [en2idx.get(word, 1) for word in (target_sent + " </S>").split()]
        if max(len(x), len(y)) <= hp.maxlen:
            x_list.append(np.array(x))
            y_list.append(np.array(y))
            Sources.append(source_sent)
            Targets.append(target_sent)

    # Pad
    X = np.zeros([len(x_list), hp.maxlen], np.int32)
    Y = np.zeros([len(y_list), hp.maxlen], np.int32)
    for i, (x, y) in enumerate(zip(x_list, y_list)):
        X[i] = np.lib.pad(
            x, [0, hp.maxlen - len(x)], "constant", constant_values=(0, 0)
        )
        Y[i] = np.lib.pad(
            y, [0, hp.maxlen - len(y)], "constant", constant_values=(0, 0)
        )

    return X, Y, Sources, Targets


def load_data(data_type):
    if data_type == "train":
        source, target = hp.source_train, hp.target_train
    elif data_type == "test":
        source, target = hp.source_test, hp.target_test
    assert data_type in ["train", "test"]
    cn_sents = [
        regex.sub("[^\s\p{L}']", "", line)
        for line in codecs.open(source, "r", "utf-8").read().split("\n")
        if line and line[0] != "<"
    ]
    en_sents = [
        regex.sub("[^\s\p{L}']", "", line)
        for line in codecs.open(target, "r", "utf-8").read().split("\n")
        if line and line[0] != "<"
    ]

    X, Y, Sources, Targets = create_data(cn_sents, en_sents)
    return X, Y, Sources, Targets


def load_train_data():
    X, Y, _, _ = load_data("train")
    return X, Y


def load_test_data():
    X, _, Sources, Targets = load_data("test")
    return X, Sources, Targets


def get_batch_indices(total_length, batch_size):
    assert (
        batch_size <= total_length
    ), "Batch size is large than total data length. Check your data or change batch size."
    current_index = 0
    indexs = [i for i in range(total_length)]
    random.shuffle(indexs)
    while 1:
        if current_index + batch_size >= total_length:
            break
        current_index += batch_size
        yield indexs[current_index : current_index + batch_size], current_index
