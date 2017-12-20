import argparse
import os
import torch

from torch import nn, optim
from tensorboardX import SummaryWriter
from time import gmtime, strftime

from model.BIMPM import BIMPM
from model.utils import SNLI


def train(args, data):
    """
    model = BIMPM(args, data)
    if args.gpu > -1:
        model.cuda(args.gpu)
    """

    #parameters = filter(lambda p: p.requires_grad, model.parameters())
    #optimizer = optim.Adadelta(lr=args.learning_rate)
    #criterion = nn.CrossEntropyLoss()

    for batch in iter(data.train_iter):
        print(batch.premise, batch.hypothesis)


def test():
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--char-dim', default=20, type=int)
    parser.add_argument('--char-hidden-size', default=50, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--hidden-size', default=100, type=int)
    parser.add_argument('--learning-rate', default=0.001, type=float)
    parser.add_argument('--num-perspective', default=20, type=int)
    parser.add_argument('--word-dim', default=300, type=int)
    args = parser.parse_args()

    print('loading SNLI data...')
    data = SNLI(args)

    setattr(args, 'char_vocab_size', len(data.char_vocab))
    setattr(args, 'vocab_size', len(data.TEXT.vocab))
    setattr(args, 'class_size', len(data.LABEL.vocab))

    print('training start!')
    best_model = train(args, data)

    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
    torch.save(best_model.state_dict(),
               'saved_models/' + strftime('%H:%M:%S', gmtime()) + '.pt')

    print('training finished!')


if __name__ == '__main__':
    main()
