from math import floor


class Parameters:
    def __init__(self, vocab_size):

        self.vocab_size = vocab_size
        self.embed_size = 80

        '''
        These sizes were actualy choosen
        in order to process sequence with max len equal to 210
        '''
        self.encoder_sizes = [[self.embed_size, 128, 4, 2],
                              [128, 256, 4, 2],
                              [256, 256, 4, 2],
                              [256, 512, 4, 2],
                              [512, 512, 4, 2],
                              [512, 512, 4, 2]]

        self.latent_size = self.encoder_sizes[-1][1]

        self.decoder_sizes = [[self.latent_size, 512, 4, 2, 0],
                              [512, 512, 4, 2, 1],
                              [512, 256, 4, 2, 0],
                              [256, 256, 4, 2, 1],
                              [256, 128, 4, 2, 0],
                              [128, self.vocab_size, 4, 2, 0]]
        self.decoder_rnn_size = 1000
        self.decoder_rnn_num_layers = 1

    @staticmethod
    def conv_out_len(in_len, kernel_size, padding, stride: float):
        return floor((in_len + 2 * padding - kernel_size)/stride + 1)

    @staticmethod
    def deconv_out_len(in_len, kernel_size, padding, stride, out_padding):
        return (in_len - 1) * stride - 2 * padding + kernel_size + out_padding
