from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe

from nltk import word_tokenize


class SNLI():
    def __init__(self, args):
        self.TEXT = data.Field(batch_first=True, tokenize=word_tokenize, lower=True)
        self.LABEL = data.Field(sequential=False, unk_token=None)

        self.train, self.dev, self.test = datasets.SNLI.splits(self.TEXT, self.LABEL)

        self.TEXT.build_vocab(self.train, vectors=GloVe(name='840B', dim=300))
        self.LABEL.build_vocab(self.train)

        self.train_iter, self.dev_iter, self.test_iter = \
            data.BucketIterator.splits((self.train, self.dev, self.test),
                                       batch_size=args.batch_size,
                                       device=args.gpu)

        self.char_vocab = {}
        self.characterized_words = []
        if args.use_char_emb:
            self.build_char_vocab()

    def build_char_vocab(self):
        for word in self.TEXT.vocab.itos:
            chars = []
            for c in list(word):
                if c not in self.char_vocab:
                    self.char_vocab[c] = len(self.char_vocab)

                chars.append(self.char_vocab[c])

            self.characterized_words.append(chars)





