import math
import torch
from torch import nn
import torch.nn.functional as F
from utils import idx2onehot
import numpy as np
class Classfier(nn.Module):
    def __init__(self, config, vocab):
        super(Classfier, self).__init__()
        num_domain_labels, num_layers = config.num_domain_labels, config.num_layers
        d_model, max_length = config.d_model, config.max_length
        h, dropout = config.h, config.dropout
        learned_pos_embed = config.learned_pos_embed
        load_pretrained_embed = config.load_pretrained_embed
        num_classes = config.num_classes

        self.pad_idx = vocab.stoi['<pad>']
        self.domain_label_embed = Embedding(num_domain_labels, d_model)
        self.embed = EmbeddingLayer(
            vocab, d_model, max_length,
            self.pad_idx,
            learned_pos_embed,
            load_pretrained_embed
        )
        self.cls_token = nn.Parameter(torch.randn(d_model))
        self.encoder = Encoder(num_layers, d_model, len(vocab), h, dropout)
        self.classifier = Linear(d_model, num_classes)

    def forward(self, inp_tokens, inp_lengths, domain_label=None):
        batch_size = inp_tokens.size(0)
        num_extra_token = 1 if domain_label is None else 2
        max_seq_len = inp_tokens.size(1)

        pos_idx = torch.arange(max_seq_len).unsqueeze(0).expand((batch_size, -1)).to(inp_lengths.device)
        mask = pos_idx >= inp_lengths.unsqueeze(-1)
        for _ in range(num_extra_token):
            mask = torch.cat((torch.zeros_like(mask[:, :1]), mask), 1)
        mask = mask.view(batch_size, 1, 1, max_seq_len + num_extra_token)

        cls_token = self.cls_token.view(1, 1, -1).expand(batch_size, -1, -1)

        enc_input = cls_token
        if domain_label is not None:
            domain_label_emb = self.domain_label_embed(domain_label).unsqueeze(1)
            enc_input = torch.cat((enc_input, domain_label_emb), 1)

        enc_input = torch.cat((enc_input, self.embed(inp_tokens, pos_idx)), 1)

        encoded_features = self.encoder(enc_input, mask)
        x = encoded_features[:, 0]
        logits = self.classifier(encoded_features[:, 0])

        return F.log_softmax(logits, -1)

class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, vocab_size, h, dropout):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, h, dropout) for _ in range(num_layers)])
        self.norm = LayerNorm(d_model)

    def forward(self, x, mask):
        y = x

        assert y.size(1) == mask.size(-1)

        for layer in self.layers:
            y = layer(y, mask)

        return self.norm(y)

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab, d_model, max_length, pad_idx, learned_pos_embed, load_pretrained_embed):
        super(EmbeddingLayer, self).__init__()
        self.token_embed = Embedding(len(vocab), d_model)
        self.pos_embed = Embedding(max_length, d_model)
        self.vocab_size = len(vocab)
        if load_pretrained_embed:
            self.token_embed = nn.Embedding.from_pretrained(vocab.vectors)
            print('embed loaded.')

    def forward(self, x, pos):
        if len(x.size()) == 2:
            y = self.token_embed(x) + self.pos_embed(pos)
        else:
            y = torch.matmul(x, self.token_embed.weight) + self.pos_embed(pos)

        return y


class EncoderLayer(nn.Module):
    def __init__(self, d_model, h, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, h, dropout)
        self.pw_ffn = PositionwiseFeedForward(d_model, dropout)
        self.sublayer = nn.ModuleList([SublayerConnection(d_model, dropout) for _ in range(2)])

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.pw_ffn)


def scaled_attention(query, key, value, mask):
    d_k = query.size(-1)
    scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(d_k)
    scores.masked_fill_(mask, float('-inf'))
    attn_weight = F.softmax(scores, -1)
    attn_feature = attn_weight.matmul(value)

    return attn_feature, attn_weight

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h, dropout):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.head_projs = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask):
        batch_size = query.size(0)

        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for x, l in zip((query, key, value), self.head_projs)]

        attn_feature, _ = scaled_attention(query, key, value, mask)

        attn_concated = attn_feature.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.fc(attn_concated)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.mlp = nn.Sequential(
            Linear(d_model, 4 * d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        return self.mlp(x)


class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout):
        super(SublayerConnection, self).__init__()
        self.layer_norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        y = sublayer(self.layer_norm(x))
        return x + self.dropout(y)

    def incremental_forward(self, x, sublayer):
        y = sublayer(self.layer_norm(x))
        return x[:, -1:] + self.dropout(y)


def Linear(in_features, out_features, bias=True, uniform=True):
    m = nn.Linear(in_features, out_features, bias)
    if uniform:
        nn.init.xavier_uniform_(m.weight)
    else:
        nn.init.xavier_normal_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.xavier_uniform_(m.weight)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def LayerNorm(embedding_dim, eps=1e-6):
    m = nn.LayerNorm(embedding_dim, eps)
    return m
