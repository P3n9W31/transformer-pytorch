from hyperparams import Hyperparams as hp
import codecs
import os
import regex
from collections import Counter


def make_vocab(fpath, fname):
    """Constructs vocabulary.
    
    Args:
      fpath: A string. Input file path.
      fname: A string. Output file name.
    
    Writes vocabulary line by line to `preprocessed/fname`
    """
    text = codecs.open(fpath, "r", "utf-8").read()
    text = regex.sub("[^\s\p{L}']", "", text)
    words = text.split()
    word2cnt = Counter(words)
    if not os.path.exists("preprocessed"):
        os.mkdir("preprocessed")
    with codecs.open("preprocessed/{}".format(fname), "w", "utf-8") as fout:
        fout.write(
            "{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n".format(
                "<PAD>", "<UNK>", "<S>", "</S>"
            )
        )
        for word, cnt in word2cnt.most_common(len(word2cnt)):
            fout.write(u"{}\t{}\n".format(word, cnt))


if __name__ == "__main__":
    make_vocab(hp.source_train, "cn.txt.vocab.tsv")
    make_vocab(hp.target_train, "en.txt.vocab.tsv")
    print("Done")
