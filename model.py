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
        if self.use_gpu:
            self.vectors = self.vectors.cuda()

    def forward(self, data):
        '''Turns a batch of IDs into vectors.
        
        data = a list of ints'''
        v = Variable(LongTensor(data), requires_grad=False)
        if self.use_gpu:
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

    def forward(self, _unused, data):
        '''Returns:
            loss - number to optimize
            details - loss broken down'''
        sgns_loss = self.sgns_loss(
                data['product'],
                data['winner'],
                data['negatives'])
        return sgns_loss, {
            'sgns_loss': sgns_loss
            # TODO: dollar loss
        }

    def sgns_loss(self, iword, oword, nwords):
        '''iword: the given "target" word (product)
           oword: the "context" to predict (firm)
           nwords: the negatives (losing firms)'''
        if self.debugging:
            batch_size = iword.size(0)
            assert iword.size() == (batch_size,)
            assert oword.size() == (batch_size,)
            assert nwords.dim() == 2
            assert nwords.size(0) == batch_size

        # black magic from https://github.com/kefirski/pytorch_NEG_loss
        ivectors = self.embedding(iword)
        ovectors = self.embedding(oword)
        nvectors = self.embedding(nwords.view(self.batch_size, -1)).neg()

        if self.debugging:
            batch_size = iword.size(0)
            n_neg = nwords.size(1)
            assert nvectors.size() == (batch_size, embedding.embedding_dim, n_neg)

        oloss = (ivectors * ovectors).sum(1).squeeze().sigmoid().log()
        nloss = torch.bmm(nvectors, ivectors.unsqueeze(2)).sigmoid().log().sum(1).squeeze()

        if self.debugging:
            batch_size = iword.size(0)
            n_neg = nwords.size(1)
            assert oloss.size == (batch_size,)
            assert nloss.size == (batch_size, n_neg)

        # oloss + nloss will broadcast oloss.
        return -(oloss + nloss).sum() / self.batch_size
