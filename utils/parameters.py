from math import floor


class Parameters:
    def __init__(self, vocab_size, max_seq_len):

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
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

        self.latent_variable_size = self.encoder_sizes[-1][1]

        self.decoder_sizes = [[512, 512, 4, 2, 0],
                              [512, 512, 4, 2, 1],
                              [512, 256, 4, 2, 0],
                              [256, 256, 4, 2, 1],
                              [256, 128, 4, 2, 0],
                              [128, self.embed_size, 4, 2, 0]]

    @staticmethod
    def conv_out_len(in_len, kernel_size, padding, stride: float):
        return floor((in_len + 2 * padding - kernel_size)/stride + 1)

    @staticmethod
    def deconv_out_len(in_len, kernel_size, padding, stride, out_padding):
        return (in_len - 1) * stride - 2 * padding + kernel_size + out_padding
