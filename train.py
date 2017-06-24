import argparse
import numpy as np
import torch as t
from torch.optim import Adam
import torch.nn.functional as F
from utils.batchloader import BatchLoader
from utils.parameters import Parameters
from model.vae import VAE


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='VAE')
    parser.add_argument('--num-iterations', type=int, default=200000, metavar='NI',
                        help='num iterations (default: 200000)')
    parser.add_argument('--batch-size', type=int, default=30, metavar='BS',
                        help='batch size (default: 30)')
    parser.add_argument('--use-cuda', type=bool, default=False, metavar='CUDA',
                        help='use cuda (default: False)')
    parser.add_argument('--learning-rate', type=float, default=0.0005, metavar='LR',
                        help='learning rate (default: 0.0005)')
    parser.add_argument('--dropout', type=float, default=0.12, metavar='DR',
                        help='dropout (default: 0.12)')
    parser.add_argument('--aux', type=float, default=0.4, metavar='DR',
                        help='aux loss coef (default: 0.4)')
    parser.add_argument('--use-trained', type=bool, default=False, metavar='UT',
                        help='load pretrained model (default: False)')

    args = parser.parse_args()

    batch_loader = BatchLoader()
    parameters = Parameters(batch_loader.vocab_size)

    vae = VAE(parameters.vocab_size, parameters.embed_size, parameters.latent_size,
              parameters.decoder_rnn_size, parameters.decoder_rnn_num_layers)
    if args.use_trained:
        vae.load_state_dict(t.load('trained_VAE'))
    if args.use_cuda:
        vae = vae.cuda()

    optimizer = Adam(vae.parameters(), args.learning_rate)

    for iteration in range(args.num_iterations):

        '''Train step'''
        input, decoder_input, target = batch_loader.next_batch(args.batch_size, 'train', args.use_cuda)
        target = target.view(-1)

        logits, aux_logits, kld = vae(args.dropout, input, decoder_input)

        logits = logits.view(-1, batch_loader.vocab_size)
        cross_entropy = F.cross_entropy(logits, target, size_average=False)

        aux_logits = aux_logits.view(-1, batch_loader.vocab_size)
        aux_cross_entropy = F.cross_entropy(aux_logits, target, size_average=False)

        loss = cross_entropy + args.aux * aux_cross_entropy + kld

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        '''Validation'''
        input, decoder_input, target = batch_loader.next_batch(args.batch_size, 'valid', args.use_cuda)
        target = target.view(-1)

        logits, aux_logits, valid_kld = vae(args.dropout, input, decoder_input)

        logits = logits.view(-1, batch_loader.vocab_size)
        valid_cross_entropy = F.cross_entropy(logits, target, size_average=False)

        aux_logits = aux_logits.view(-1, batch_loader.vocab_size)
        valid_aux_cross_entropy = F.cross_entropy(aux_logits, target, size_average=False)

        loss = valid_cross_entropy + args.aux * valid_aux_cross_entropy + kld

        if iteration % 50 == 0:
            print('\n')
            print('|--------------------------------------|')
            print(iteration)
            print('|--------ce------aux-ce-----kld--------|')
            print('|----------------train-----------------|')
            print(cross_entropy.data.cpu().numpy()[0]/(210 * args.batch_size),
                  aux_cross_entropy.data.cpu().numpy()[0]/(210 * args.batch_size),
                  kld.data.cpu().numpy()[0])
            print('|----------------valid-----------------|')
            print(valid_cross_entropy.data.cpu().numpy()[0]/(210 * args.batch_size),
                  valid_aux_cross_entropy.data.cpu().numpy()[0]/(210 * args.batch_size),
                  valid_kld.data.cpu().numpy()[0])
            print('|--------------------------------------|')
            print(vae.sample(batch_loader, args.use_cuda))
            print('|--------------------------------------|')
