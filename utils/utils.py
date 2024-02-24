import torch
import random
import numpy as np


def decode(self, matrix):
    """Takes an array of indices and returns the corresponding SMILES"""
    chars = []
    for i in matrix:
        if i == self.vocab['EOS']: break
        chars.append(self.reversed_vocab[i])
    smiles = "".join(chars)
    smiles = smiles.replace("L", "Cl").replace("R", "Br")
    return smiles


# 根据字典找到单词
def tensor2text(vocab, tensor):
    text = []
    index2word = vocab.itos
    eos_idx = vocab.stoi['<eos>']
    unk_idx = vocab.stoi['<unk>']
    # stop_idxs = [vocab.stoi['!'], vocab.stoi['.'], vocab.stoi['?']]
    for sample in tensor:
        sample_filtered = []
        # prev_token = None
        for idx in list(sample):
            if idx == eos_idx:
                break
            sample_filtered.append(index2word[idx])
        sample = ''.join(sample_filtered)
        sample = sample.replace("L", "Cl").replace("R", "Br")
        text.append(sample)

    return text


def calc_ppl(log_probs, tokens_mask):
    return (log_probs.sum() / tokens_mask.sum()).exp()


def idx2onehot(x, num_classes):
    y = x.unsqueeze(-1)
    x_onehot = torch.zeros_like(y.expand(x.size() + torch.Size((num_classes, ))))
    x_onehot.scatter_(-1, y, 1)
    return x_onehot.float()


def word_shuffle(x, l, shuffle_len):
    if not shuffle_len:
        return x

    noise = torch.rand(x.size(), dtype=torch.float).to(x.device)
    high_idx = torch.arange(x.size(1)).unsqueeze(0).expand_as(x).to(x.device)
    pad_mask = (high_idx >= l.unsqueeze(1)).float()

    scores = high_idx.float() + ((1 - pad_mask) * noise + pad_mask) * shuffle_len
    x2 = x.clone()
    x2 = x2.gather(1, scores.argsort(1))

    return x2


def word_dropout_raw(x, l, unk_drop_prob, rand_drop_prob, vocab):
    if not unk_drop_prob and not rand_drop_prob:
        return x

    assert unk_drop_prob + rand_drop_prob <= 1

    noise = torch.rand(x.size(), dtype=torch.float).to(x.device)
    high_idx = torch.arange(x.size(1)).unsqueeze(0).expand_as(x).to(x.device)
    token_mask = high_idx < l.unsqueeze(1)

    x2 = x.clone()
    
    # drop to <unk> token
    if unk_drop_prob:
        unk_idx = vocab.stoi['<unk>']
        unk_drop_mask = (noise < unk_drop_prob) & token_mask
        x2.masked_fill_(unk_drop_mask, unk_idx)

    # drop to random_mask
    if rand_drop_prob:
        rand_drop_mask = (noise > 1 - rand_drop_prob) & token_mask
        rand_tokens = torch.randint_like(x, len(vocab))
        rand_tokens.masked_fill_(1 - rand_drop_mask, 0)
        x2.masked_fill_(rand_drop_mask, 0)
        x2 = x2 + rand_tokens
    
    return x2


def unk_dropout_(x, l, drop_prob, unk_idx):
    noise = torch.rand(x.size(), dtype=torch.float).to(x.device)
    high_idx = torch.arange(x.size(1)).unsqueeze(0).expand_as(x).to(x.device)
    token_mask = high_idx < l.unsqueeze(1)
    unk_drop_mask = (noise < drop_prob) & token_mask
    x.masked_fill_(unk_drop_mask, unk_idx)


def rand_dropout_(x, l, drop_prob, vocab_size):
    noise = torch.rand(x.size(), dtype=torch.float).to(x.device)
    high_idx = torch.arange(x.size(1)).unsqueeze(0).expand_as(x).to(x.device)
    token_mask = high_idx < l.unsqueeze(1)
    rand_drop_mask = (noise < drop_prob) & token_mask
    rand_tokens = torch.randint_like(x, vocab_size)
    rand_tokens.masked_fill_(1 - rand_drop_mask, 0)
    x.masked_fill_(rand_drop_mask, 0)
    x += rand_tokens


def word_dropout_new(x, l, unk_drop_fac, rand_drop_fac, drop_prob, vocab):
    if not unk_drop_fac and not rand_drop_fac:
        return x

    assert unk_drop_fac + rand_drop_fac <= 1

    batch_size = x.size(0)
    unk_idx = vocab.stoi['<unk>']
    unk_drop_idx = int(batch_size * unk_drop_fac)
    rand_drop_idx = int(batch_size * rand_drop_fac)

    shuffle_idx = torch.argsort(torch.rand(batch_size))
    orignal_idx = torch.argsort(shuffle_idx)

    x2 = x.clone()
    x2 = x2[shuffle_idx]

    if unk_drop_idx:
        unk_dropout_(x2[:unk_drop_idx], l[:unk_drop_idx], drop_prob, unk_idx)

    if rand_drop_idx:
        rand_dropout_(x2[-rand_drop_idx:], l[-rand_drop_idx:], drop_prob, len(vocab))

    x2 = x2[orignal_idx]

    return x2


def word_dropout(x, l, drop_prob, unk_idx):
    if not drop_prob:
        return x

    noise = torch.rand(x.size(), dtype=torch.float).to(x.device)
    high_idx = torch.arange(x.size(1)).unsqueeze(0).expand_as(x).to(x.device)
    token_mask = high_idx < l.unsqueeze(1)

    drop_mask = (noise < drop_prob) & token_mask
    x2 = x.clone()
    x2.masked_fill_(drop_mask, unk_idx)
    
    return x2


def word_drop(x, l, drop_prob, pad_idx):
    if not drop_prob:
        return x

    noise = torch.rand(x.size(), dtype=torch.float).to(x.device)
    high_idx = torch.arange(x.size(1)).unsqueeze(0).expand_as(x).to(x.device)
    token_mask = high_idx < (l.unsqueeze(1) - 1)

    drop_mask = (noise < drop_prob) & token_mask
    x2 = x.clone()
    high_idx.masked_fill_(drop_mask, x.size(1) - 1)
    high_idx = torch.sort(high_idx, 1)[0]
    x2 = x2.gather(1, high_idx)

    return x2


def add_noise(words, lengths, shuffle_len, drop_prob, unk_idx):
    words = word_shuffle(words, lengths, shuffle_len)
    words = word_dropout(words, lengths, drop_prob, unk_idx)
    return words


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    random.SystemRandom(seed)
    np.random.seed(seed)
    np.random.RandomState(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def choose_best_smiles(output, input):
    pass


def get_lengths(tokens, eos_idx):
    lengths = torch.cumsum(tokens == eos_idx, 1)
    lengths = (lengths == 0).long().sum(-1)
    lengths = lengths + 1  # +1 for <eos> token
    return lengths


def batch_preprocess(batch, pad_idx, eos_idx, reverse=False):
    # batch里面包含两个矩阵，一个high，一个low，一个high里面又有两个矩阵，即两个句子
    if isinstance(batch, tuple):
        batch_high, batch_low = batch
        # 两个句子的长度差异
        diff = batch_high.size(1) - batch_low.size(1)
        if diff < 0:
            pad = torch.full_like(batch_low[:, :-diff], pad_idx)
            batch_high = torch.cat((batch_high, pad), 1)
        elif diff > 0:
            pad = torch.full_like(batch_high[:, :diff], pad_idx)
            batch_low = torch.cat((batch_low, pad), 1)

        high_styles = torch.ones_like(batch_high[:, 0])
        low_styles = torch.zeros_like(batch_low[:, 0])

        if reverse:
            batch_high, batch_low = batch_low, batch_high
            high_styles, low_styles = low_styles, high_styles
        tokens = torch.cat((batch_high, batch_low), 0)
        lengths = get_lengths(tokens, eos_idx)
        domain_labels = torch.cat((high_styles, low_styles), 0)
    else:
        domain_labels = torch.zeros_like(batch[:, 0])
        tokens = batch
        lengths = get_lengths(tokens, eos_idx)
    return tokens, lengths, domain_labels