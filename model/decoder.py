import torch as t
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, vocab_size, latent_variable_size, rnn_size, rnn_num_layers, embed_size):
        super(Decoder, self).__init__()

        self.vocab_size = vocab_size
        self.latent_variable_size = latent_variable_size
        self.rnn_size = rnn_size
        self.embed_size = embed_size
        self.rnn_num_layers = rnn_num_layers

        self.cnn = nn.Sequential(
            nn.ConvTranspose1d(self.latent_variable_size, 512, 4, 2, 0),
            nn.BatchNorm1d(512),
            nn.ELU(),

            nn.ConvTranspose1d(512, 512, 4, 2, 0, output_padding=1),
            nn.BatchNorm1d(512),
            nn.ELU(),

            nn.ConvTranspose1d(512, 256, 4, 2, 0),
            nn.BatchNorm1d(256),
            nn.ELU(),

            nn.ConvTranspose1d(256, 256, 4, 2, 0, output_padding=1),
            nn.BatchNorm1d(256),
            nn.ELU(),

            nn.ConvTranspose1d(256, 128, 4, 2, 0),
            nn.BatchNorm1d(128),
            nn.ELU(),

            nn.ConvTranspose1d(128, self.vocab_size, 4, 2, 0)
        )

        self.rnn = nn.GRU(input_size=self.vocab_size + self.embed_size,
                          hidden_size=self.rnn_size,
                          num_layers=self.rnn_num_layers,
                          batch_first=True)

        self.hidden_to_vocab = nn.Linear(self.rnn_size, self.vocab_size)

    def forward(self, latent_variable, decoder_input):
        """
        :param latent_variable: An float tensor with shape of [batch_size, latent_variable_size]
        :param decoder_input: An float tensot with shape of [batch_size, max_seq_len, embed_size]
        :return: two tensors with shape of [batch_size, max_seq_len, vocab_size]
                    for estimating likelihood for whole model and for auxiliary target respectively
        """

        aux_logits = self.conv_decoder(latent_variable)

        logits, _ = self.rnn_decoder(aux_logits, decoder_input, initial_state=None)

        return logits, aux_logits

    def conv_decoder(self, latent_variable):
        latent_variable = latent_variable.unsqueeze(2)

        out = self.cnn(latent_variable)
        return t.transpose(out, 1, 2).contiguous()

    def rnn_decoder(self, cnn_out, decoder_input, initial_state=None):
        logits, final_state = self.rnn(t.cat([cnn_out, decoder_input], 2), initial_state)

        [batch_size, seq_len, _] = logits.size()
        logits = logits.contiguous().view(-1, self.rnn_size)

        logits = self.hidden_to_vocab(logits)

        logits = logits.view(batch_size, seq_len, self.vocab_size)

        return logits, final_state
