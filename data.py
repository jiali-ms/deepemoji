import pickle
import os
from tqdm import tqdm
from util import is_emoji

CORPUS_PATH = os.path.join('data', 'corpus.txt')
LEXICON_PATH = os.path.join('data', 'lexicon.pkl')
EMOJI_LEXICON_PATH = os.path.join('data', 'lexicon_emoji.pkl')

class EmojiVocab(object):
    """ Emoji vocab is used as y in this task. Also add a blank for cases where no emoji is predicted.
    """
    def __init__(self, size):
        self.lexicon = pickle.load(open(EMOJI_LEXICON_PATH, 'rb'))[:size]
        self.lexicon = [('<blank>', 0)] + self.lexicon
        self.w2i = {x[0]: i for i, x in enumerate(self.lexicon)}
        self.i2w = {v: k for k, v in self.w2i.items()}
        print('emoji vocab with size {} loaded'.format(size))

    def encode(self, emoji):
        return self.w2i[emoji]

    def decode(self, i):
        return self.i2w[i]

    def __len__(self):
        return len(self.w2i)

class Vocab(object):
    def __init__(self, size):
        self.lexicon = pickle.load(open(LEXICON_PATH, 'rb'))[:size]
        self.lexicon = [('<unk>', 0)] + [('<eos>', 1)] + self.lexicon
        self.w2i = {x[0]:i for i, x in enumerate(self.lexicon)}
        self.i2w = {v:k for k,v in self.w2i.items()}
        print('vocab with size {} loaded'.format(size))

    def encode(self, token):
        if token in self.w2i:
            return self.w2i[token]

        return self.w2i['<unk>']

    def decode(self, i):
        assert i in self.i2w

        return self.i2w[i]

    def __len__(self):
        return len(self.w2i)

class Corpus(object):
    def __init__(self, vocab, emoji_vocab, debug=False):
        self.vocab = vocab
        self.emoji_vocab = emoji_vocab

        self.encoded_train = self._encode_corpus(CORPUS_PATH, debug)

        #self.encoded_dev = self._encode_corpus(DEV_CORPUS_PATH, debug)
        #self.encoded_test = self._encode_corpus(TEST_CORPUS_PATH, debug)

    def _encode_corpus(self, path, debug=False):
        if os.path.exists(path + '.pkl'):
            print('load encoded corpus from dump: %s' % path + '.pkl')
            data = pickle.load(open(path + '.pkl', 'rb'))
            if debug:
                return (data[0][:1024*1000], data[1][:1024*1000])
            else:
                return data

        encoded_x = [self.vocab.encode('<eos>')]
        encoded_y = [self.emoji_vocab.encode('<blank>')]
        print('encode corpus: {}'.format(path))
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if debug:
                lines = lines[:1024*1000]
            for line in tqdm(lines):
                tokens = line.strip().split(' ')
                for token in tokens:
                    if token in self.emoji_vocab.w2i :  # and encoded_y[-1] not in self.emoji_vocab.i2w
                        # do not support continuous emoji case
                        encoded_y[-1] = self.emoji_vocab.encode(token)
                    else:
                        encoded_x.append(self.vocab.encode(token))
                        encoded_y.append(self.emoji_vocab.encode('<blank>'))

                encoded_x.append(self.vocab.encode('<eos>'))
                encoded_y.append(self.emoji_vocab.encode('<blank>'))

        assert len(encoded_y) == len(encoded_x)

        pickle.dump((encoded_x, encoded_y), open(path + '.pkl', 'wb'))

        return encoded_x, encoded_y

if __name__ == "__main__":
    vocab = Vocab(20000)
    emoji_vocab = EmojiVocab(100)
    corpus = Corpus(vocab, emoji_vocab, debug=False)
    decoded = []
    train_x, train_y = corpus.encoded_train
    for i in range(100):
        decoded.append(vocab.decode(train_x[i]))
        decoded.append(' ' if train_y[i] == 0 else ' %s ' % emoji_vocab.decode(train_y[i]))
    print(''.join(decoded))