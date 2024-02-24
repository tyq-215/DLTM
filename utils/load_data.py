import time

import pandas as pd
import torchtext
from torchtext.legacy import data
from utils import tensor2text
import regex as re
import selfies
import copy
import numpy as np


def replace_halogen(string):
    """Regex to replace Br and Cl with single letters"""
    br = re.compile('Br')
    cl = re.compile('Cl')
    string = br.sub('R', string)
    string = cl.sub('L', string)

    return string


def tokenize(smiles):
    """Takes a SMILES string and returns a list of tokens.
    This will swap 'Cl' and 'Br' to 'L' and 'R' and treat
    '[xx]' as one token."""
    regex = '(\[[^\[\]]{1,6}\])'
    smiles = replace_halogen(smiles)
    char_list = re.split(regex, smiles)
    tokenized = []
    for char in char_list:
        if char.startswith('['):
            tokenized.append(char)
        else:
            chars = [unit for unit in char]
            [tokenized.append(unit) for unit in chars]
    return tokenized


class DatasetIterator(object):
    def __init__(self, high_iter, low_iter):
        self.high_iter = high_iter
        self.low_iter = low_iter

    def __iter__(self):
        for batch_high, batch_low in zip(iter(self.high_iter), iter(self.low_iter)):
            if batch_high.text.size(0) == batch_low.text.size(0):
                yield batch_high.text, batch_low.text


class DatasetIterator2(object):
    def __init__(self, all_iter):
        self.all_iter = all_iter

    def __iter__(self):
        for i, batch in enumerate(iter(self.all_iter)):
                yield batch.text


def load_dataset(config, train_high='', train_low='',
                 test_high='', test_low='', all_data=""):

    root = config.data_path
    TEXT = data.Field(batch_first=True, tokenize=tokenize, eos_token='<eos>')
    
    dataset_fn = lambda name: data.TabularDataset(
        path=root + name,
        format='tsv',
        fields=[('text', TEXT)]
    )

    train_high_set, train_low_set = map(dataset_fn, [train_high, train_low])
    test_high_set, test_low_set = map(dataset_fn, [test_high, test_low])
    all_data_set, = map(dataset_fn, [all_data])
    TEXT.build_vocab(all_data_set,train_high_set, train_low_set, min_freq=config.min_freq)

    if config.load_pretrained_embed:
        start = time.time()
        
        vectors = torchtext.vocab.GloVe('6B', dim=config.embed_size, cache=config.pretrained_embed_path)
        TEXT.vocab.set_vectors(vectors.stoi, vectors.vectors, vectors.dim)
        print('vectors', TEXT.vocab.vectors.size())
        
        print('load embedding took {:.2f} s.'.format(time.time() - start))

    vocab = TEXT.vocab
    dataiter_fn = lambda dataset, train: data.BucketIterator(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=train,
        repeat=train,
        sort_key=lambda x: len(x.text),
        sort_within_batch=False,
        device=config.device
    )

    train_high_iter, train_low_iter = map(lambda x: dataiter_fn(x, True), [train_high_set, train_low_set])
    test_high_iter, test_low_iter = map(lambda x: dataiter_fn(x, False), [test_high_set, test_low_set])
    all_iter, = map(lambda x: dataiter_fn(x, False), [all_data_set])

    train_iters = DatasetIterator(train_high_iter, train_low_iter)
    test_iters = DatasetIterator(test_high_iter, test_low_iter)
    all_iters = DatasetIterator2(all_iter)
    return train_iters, test_iters, vocab, all_iters


if __name__ == '__main__':
    train_iter, _, _, vocab = load_dataset('')
    print(len(vocab))
    for batch in train_iter:
        text = tensor2text(vocab, batch.text)
        print('\n'.join(text))
        print(batch.label)
        break
