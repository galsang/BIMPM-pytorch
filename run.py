import argparse
import os
import torch

from torch import nn, optim
from tensorboardX import SummaryWriter
from time import gmtime, strftime

from model.BIMPM import BIMPM
from model.utils import SNLI


def train(args, data):
    model = BIMPM(args, data)
    if args.gpu > -1:
        model.cuda(args.gpu)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    writer = SummaryWriter(log_dir='runs/' + strftime('%H:%M:%S', gmtime()))

    model.train()
    for e in range(1):
        loss = 0
        for i, batch in enumerate(iter(data.train_iter)):
            pred = model(batch.premise, batch.hypothesis)

            optimizer.zero_grad()
            batch_loss = criterion(pred, batch.label)
            batch_loss.backward()
            loss += batch_loss
            optimizer.step()

            print("batch loss:", batch_loss)

        writer.add_scalar('loss', loss, e)
        print("epoch:", e, "loss:", loss)

    writer.close()


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

    #setattr(args, 'char_vocab_size', len(data.char_vocab))
    setattr(args, 'word_vocab_size', len(data.TEXT.vocab))
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
