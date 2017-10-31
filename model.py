# -*- coding: utf-8 -*-

import random
import torch
import torch.nn as nn

from torch import LongTensor
from torch.autograd import Variable


class Bundler(nn.Module):

    def forward(self, data):
        raise NotImplementedError

    def forward_i(self, data):
        raise NotImplementedError

    def forward_o(self, data):
        raise NotImplementedError


class Word2Vec(Bundler):
    '''Modified Word2Vec with combined product/firm embeddings.'''

    def __init__(self,
            vocab_size,
            embedding_dim=30,
            use_gpu=False):
        super(Word2Vec, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.use_gpu = use_gpu
        self.vectors = nn.Embedding(self.vocab_size, self.embedding_dim)
        if self.gpu:
            self.vectors = self.vectors.cuda()

    def forward(self, data):
        '''Turns a batch of IDs into vectors.
        
        data = a list of ints'''
        v = Variable(LongTensor(data), requires_grad=False)
        if self.gpu:
            v = v.cuda()
        return self.vectors(v)


class SGNS(nn.Module):
    '''Industry Classification SGNS (Skip-Gram Negative Sampling)'''

    def __init__(
            self, embedding,
            debugging=False,
            use_gpu=False):
        '''Params:
        embedding - Word2Vec instance
        '''
        super(SGNS, self).__init__()
        self.embedding = embedding
        self.debugging = debugging
        self.use_gpu = use_gpu

        assert isinstance(embedding, Word2Vec), 'embedding must be Word2Vec'
        assert self.use_gpu == embedding.use_gpu, 'use_gpu must be consistent'

    def forward(self, iword, owords):
        # black magic from https://github.com/kefirski/pytorch_NEG_loss
        nwords = self.sample(iword, owords)
        ivectors = self.embedding.forward_i(LongTensor(iword).repeat(1, self.window_size).contiguous().view(-1))
        ovectors = self.embedding.forward_o(LongTensor(owords).contiguous().view(-1))
        nvectors = self.embedding.forward_o(LongTensor(nwords).contiguous().view(self.batch_size * self.window_size, -1)).neg()
        oloss = (ivectors * ovectors).sum(1).squeeze().sigmoid().log()
        nloss = torch.bmm(nvectors, ivectors.unsqueeze(2)).sigmoid().log().sum(1).squeeze()
        return -(oloss + nloss).sum() / self.batch_size
