# -*- coding: utf-8 -*-
# /usr/bin/python3
"""
Feb. 2019 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer.

Preprocess the iwslt 2016 datasets.
"""

import os
import errno
import sentencepiece as spm
import re
import logging

logging.basicConfig(level=logging.INFO)

# modify this dir
data_dir = "/home/cnndm_data/truncate_500"


def preprocess():
    logging.info("check raw files exist")
    train1 = os.path.join(data_dir, "train.art")
    train2 = os.path.join(data_dir, "train.abs")
    valid1 = os.path.join(data_dir, "valid.art")
    valid2 = os.path.join(data_dir, "valid.abs")
    test1 = os.path.join(data_dir, "test.art")
    test2 = os.path.join(data_dir, "test.abs")
    vocab_size = 32000
    for f in (train1, train2, valid1, valid2, test1, test2):
        if not os.path.isfile(f):
          raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), f)
  
    def _precess_abs_line(line):
        sen_list = re.split("<t>|</t>", line)
        sen_list = [sen.strip() for sen in sen_list if len(sen.strip()) > 0]
        line = ' '.join(sen_list)
        return line
    logging.info("# Preprocessing")
    _prepro_article = lambda x: [line.strip() for line in open(x, 'r').read().splitlines()]
    _prepro_abstract = lambda x: [_precess_abs_line(line) for line in open(x, 'r').read().splitlines()]
    prepro_train1, prepro_train2 = _prepro_article(train1), _prepro_abstract(train2)
    assert len(prepro_train1) == len(prepro_train2)
    prepro_valid1, prepro_valid2 = _prepro_article(valid1), _prepro_abstract(valid2)
    assert len(prepro_valid1) == len(prepro_valid2)
    prepro_test1, prepro_test2 = _prepro_article(test1), _prepro_abstract(test2)
    assert len(prepro_test1) == len(prepro_test2)

    logging.info("# write preprocessed files")
    os.makedirs(os.path.join(data_dir, "prepro"), exist_ok=True)

    def _write(sents, fname):
        with open(fname, 'w') as writer:
            writer.write("\n".join(sents))

    _write(prepro_train1, os.path.join(data_dir, "prepro", "train.art"))
    _write(prepro_train2, os.path.join(data_dir, "prepro", "train.abs"))
    _write(prepro_train1+prepro_train2, data_dir + "/prepro/train")
    _write(prepro_valid1, os.path.join(data_dir, "prepro", "valid.art"))
    _write(prepro_valid2, os.path.join(data_dir, "prepro", "valid.abs"))
    _write(prepro_test1, os.path.join(data_dir, "prepro", "test.art"))
    _write(prepro_test2, os.path.join(data_dir, "prepro", "test.abs"))

    logging.info("# Train a joint BPE model with sentencepiece")
    os.makedirs(os.path.join(data_dir, "segmented"), exist_ok=True)
    train = '--input={} --pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 --model_prefix={} --vocab_size={} --model_type=bpe' \
        .format(data_dir+"/prepro/train", data_dir+"/segmented/bpe", vocab_size)
    spm.SentencePieceTrainer.Train(train)

    logging.info("# Load trained bpe model")
    sp = spm.SentencePieceProcessor()
    sp.Load(data_dir + "/segmented/bpe.model")

    logging.info("# Segment")

    def _segment_and_write(sents, fname):
        with open(fname, 'w') as writer:
            for sent in sents:
                pieces = sp.EncodeAsPieces(sent)
                writer.write(" ".join(pieces) + "\n")

    _segment_and_write(prepro_train1, os.path.join(data_dir, "segmented", "train.art"))
    _segment_and_write(prepro_train2, os.path.join(data_dir, "segmented", "train.abs"))
    _segment_and_write(prepro_valid1, os.path.join(data_dir, "segmented", "valid.art"))
    _segment_and_write(prepro_valid2, os.path.join(data_dir, "segmented", "valid.abs"))
    _segment_and_write(prepro_test1, os.path.join(data_dir, "segmented", "test.art"))
    _segment_and_write(prepro_test2, os.path.join(data_dir, "segmented", "test.abs"))


if __name__ == '__main__':
    preprocess()
    logging.info("Done")
