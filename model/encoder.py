import torch as t
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, embed_size):
        super(Encoder, self).__init__()

        self.embed_size = embed_size

        self.cnn = nn.Sequential(
            nn.Conv1d(self.embed_size, 128, 4, 2),
            nn.BatchNorm1d(128),
            nn.ELU(),

            nn.Conv1d(128, 256, 4, 2),
            nn.BatchNorm1d(256),
            nn.ELU(),

            nn.Conv1d(256, 256, 4, 2),
            nn.BatchNorm1d(256),
            nn.ELU(),

            nn.Conv1d(256, 512, 4, 2),
            nn.BatchNorm1d(512),
            nn.ELU(),

            nn.Conv1d(512, 512, 4, 2),
            nn.BatchNorm1d(512),
            nn.ELU(),

            nn.Conv1d(512, 512, 4, 2),
            nn.BatchNorm1d(512),
            nn.ELU()
        )

    def forward(self, input):
        """
        :param input: An float tensor with shape of [batch_size, seq_len, embed_size]
        :return: An float tensor with shape of [batch_size, latent_variable_size]
        """

        '''
        Transpose input to the shape of [batch_size, embed_size, seq_len]
        '''
        input = t.transpose(input, 1, 2)

        result = self.cnn(input)
        return result.squeeze(2)
