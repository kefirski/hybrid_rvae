import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.init import xavier_normal
from .encoder import Encoder
from .decoder import Decoder


class VAE(nn.Module):
    def __init__(self, vocab_size, embed_size, latent_size, decoder_size, decoder_num_layers):
        super(VAE, self).__init__()

        self.latent_size = latent_size
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        self.embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.embed.weight = xavier_normal(self.embed.weight)

        self.encoder = Encoder(self.embed_size, self.latent_size)

        self.context_to_mu = nn.Linear(self.latent_size, self.latent_size)
        self.context_to_logvar = nn.Linear(self.latent_size, self.latent_size)

        self.decoder = Decoder(self.vocab_size, self.latent_size, decoder_size, decoder_num_layers)

    def forward(self, drop_prob, input=None, z=None, batch_size=None, use_cuda=None):
        """
        :param drop_prob: Probability of units to be dropped out
        :param input: An long tensor with shape of [batch_size, seq_len]
        :param z: An float tensor with shape of [batch_size, latent_variable_size] in case if sampling is performed
        :param batch_size: Exactly batch size
        :param use_cuda: whether to use cuda for sampling z
        :return: logits for main model and auxiliary logits
                     of probabilities distribution over various tokens in sequence,
                 estimated latent loss
        """

        if input is not None:
            [batch_size, _] = input.size()
            input = self.embed(input)
            context = self.encoder(input)

            mu = self.context_to_mu(context)
            logvar = self.context_to_logvar(context)
            std = t.exp(0.5 * logvar)

            z = Variable(t.randn([batch_size, self.latent_size]))
            if use_cuda:
                z = z.cuda()
            z = z * std + mu
            z = F.dropout(z, drop_prob, training=True)

            kld = (-0.5 * t.sum(logvar - t.pow(mu, 2) - t.exp(logvar) + 1, 1)).mean()
        else:
            z = Variable(t.randn([batch_size, self.latent_size])) if z is None else z
            if use_cuda:
                z = z.cuda()

            kld = None

        logits, aux_logits = self.decoder(z)

        return logits, aux_logits, kld

    def inference(self, input):

        input = self.embed(input)
        context = self.encoder(input)
        mu = self.context_to_mu(context)
        logvar = self.context_to_logvar(context)

        return mu, logvar
