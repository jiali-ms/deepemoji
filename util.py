import re
import os
import pickle
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from glob import glob

from keras.utils import to_categorical

CORPUS_PATH = os.path.join('data', 'corpus.txt')
LEXICON_PATH = os.path.join('data', 'lexicon.pkl')
EMOJI_LEXICON_PATH = os.path.join('data', 'lexicon_emoji.pkl')

# The regular expression for Emoji Unicode range, it cannot perfectly cover all the code point yet.
# This list is also from other sites
emoji_re = re.compile("["
                      u"\U00002712\U00002714\U00002716\U0000271d\U00002721\U00002728\U00002733\U00002734\U00002744\U00002747\U0000274c\U0000274e\U00002753-\U00002755\U00002757\U00002763\U00002764\U00002795-\U00002797\U000027a1\U000027b0\U000027bf\U00002934\U00002935\U00002b05-\U00002b07\U00002b1b\U00002b1c\U00002b50\U00002b55\U00003030\U0000303d\U0001f004\U0001f0cf\U0001f170\U0001f171\U0001f17e\U0001f17f\U0001f18e\U0001f191-\U0001f19a\U0001f201\U0001f202\U0001f21a\U0001f22f\U0001f232-\U0001f23a\U0001f250\U0001f251\U0001f300-\U0001f321\U0001f324-\U0001f393\U0001f396\U0001f397\U0001f399-\U0001f39b\U0001f39e-\U0001f3f0\U0001f3f3-\U0001f3f5\U0001f3f7-\U0001f4fd\U0001f4ff-\U0001f53d\U0001f549-\U0001f54e\U0001f550-\U0001f567\U0001f56f\U0001f570\U0001f573-\U0001f579\U0001f587\U0001f58a-\U0001f58d\U0001f590\U0001f595\U0001f596\U0001f5a5\U0001f5a8\U0001f5b1\U0001f5b2\U0001f5bc\U0001f5c2-\U0001f5c4\U0001f5d1-\U0001f5d3\U0001f5dc-\U0001f5de\U0001f5e1\U0001f5e3\U0001f5ef\U0001f5f3\U0001f5fa-\U0001f64f\U0001f680-\U0001f6c5\U0001f6cb-\U0001f6d0\U0001f6e0-\U0001f6e5\U0001f6e9\U0001f6eb\U0001f6ec\U0001f6f0\U0001f6f3\U0001f910-\U0001f918\U0001f980-\U0001f984\U0001f9c0\U00003297\U00003299\U000000a9\U000000ae\U0000203c\U00002049\U00002122\U00002139\U00002194-\U00002199\U000021a9\U000021aa\U0000231a\U0000231b\U00002328\U00002388\U000023cf\U000023e9-\U000023f3\U000023f8-\U000023fa\U000024c2\U000025aa\U000025ab\U000025b6\U000025c0\U000025fb-\U000025fe\U00002600-\U00002604\U0000260e\U00002611\U00002614\U00002615\U00002618\U0000261d\U00002620\U00002622\U00002623\U00002626\U0000262a\U0000262e\U0000262f\U00002638-\U0000263a\U00002648-\U00002653\U00002660\U00002663\U00002665\U00002666\U00002668\U0000267b\U0000267f\U00002692-\U00002694\U00002696\U00002697\U00002699\U0000269b\U0000269c\U000026a0\U000026a1\U000026aa\U000026ab\U000026b0\U000026b1\U000026bd\U000026be\U000026c4\U000026c5\U000026c8\U000026ce\U000026cf\U000026d1\U000026d3\U000026d4\U000026e9\U000026ea\U000026f0-\U000026f5\U000026f7-\U000026fa\U000026fd\U00002702\U00002705\U00002708-\U0000270d\U0000270f]|[#]\U000020e3|[*]\U000020e3|[0]\U000020e3|[1]\U000020e3|[2]\U000020e3|[3]\U000020e3|[4]\U000020e3|[5]\U000020e3|[6]\U000020e3|[7]\U000020e3|[8]\U000020e3|[9]\U000020e3|\U0001f1e6[\U0001f1e8-\U0001f1ec\U0001f1ee\U0001f1f1\U0001f1f2\U0001f1f4\U0001f1f6-\U0001f1fa\U0001f1fc\U0001f1fd\U0001f1ff]|\U0001f1e7[\U0001f1e6\U0001f1e7\U0001f1e9-\U0001f1ef\U0001f1f1-\U0001f1f4\U0001f1f6-\U0001f1f9\U0001f1fb\U0001f1fc\U0001f1fe\U0001f1ff]|\U0001f1e8[\U0001f1e6\U0001f1e8\U0001f1e9\U0001f1eb-\U0001f1ee\U0001f1f0-\U0001f1f5\U0001f1f7\U0001f1fa-\U0001f1ff]|\U0001f1e9[\U0001f1ea\U0001f1ec\U0001f1ef\U0001f1f0\U0001f1f2\U0001f1f4\U0001f1ff]|\U0001f1ea[\U0001f1e6\U0001f1e8\U0001f1ea\U0001f1ec\U0001f1ed\U0001f1f7-\U0001f1fa]|\U0001f1eb[\U0001f1ee-\U0001f1f0\U0001f1f2\U0001f1f4\U0001f1f7]|\U0001f1ec[\U0001f1e6\U0001f1e7\U0001f1e9-\U0001f1ee\U0001f1f1-\U0001f1f3\U0001f1f5-\U0001f1fa\U0001f1fc\U0001f1fe]|\U0001f1ed[\U0001f1f0\U0001f1f2\U0001f1f3\U0001f1f7\U0001f1f9\U0001f1fa]|\U0001f1ee[\U0001f1e8-\U0001f1ea\U0001f1f1-\U0001f1f4\U0001f1f6-\U0001f1f9]|\U0001f1ef[\U0001f1ea\U0001f1f2\U0001f1f4\U0001f1f5]|\U0001f1f0[\U0001f1ea\U0001f1ec-\U0001f1ee\U0001f1f2\U0001f1f3\U0001f1f5\U0001f1f7\U0001f1fc\U0001f1fe\U0001f1ff]|\U0001f1f1[\U0001f1e6-\U0001f1e8\U0001f1ee\U0001f1f0\U0001f1f7-\U0001f1fb\U0001f1fe]|\U0001f1f2[\U0001f1e6\U0001f1e8-\U0001f1ed\U0001f1f0-\U0001f1ff]|\U0001f1f3[\U0001f1e6\U0001f1e8\U0001f1ea-\U0001f1ec\U0001f1ee\U0001f1f1\U0001f1f4\U0001f1f5\U0001f1f7\U0001f1fa\U0001f1ff]|\U0001f1f4\U0001f1f2|\U0001f1f5[\U0001f1e6\U0001f1ea-\U0001f1ed\U0001f1f0-\U0001f1f3\U0001f1f7-\U0001f1f9\U0001f1fc\U0001f1fe]|\U0001f1f6\U0001f1e6|\U0001f1f7[\U0001f1ea\U0001f1f4\U0001f1f8\U0001f1fa\U0001f1fc]|\U0001f1f8[\U0001f1e6-\U0001f1ea\U0001f1ec-\U0001f1f4\U0001f1f7-\U0001f1f9\U0001f1fb\U0001f1fd-\U0001f1ff]|\U0001f1f9[\U0001f1e6\U0001f1e8\U0001f1e9\U0001f1eb-\U0001f1ed\U0001f1ef-\U0001f1f4\U0001f1f7\U0001f1f9\U0001f1fb\U0001f1fc\U0001f1ff]|\U0001f1fa[\U0001f1e6\U0001f1ec\U0001f1f2\U0001f1f8\U0001f1fe\U0001f1ff]|\U0001f1fb[\U0001f1e6\U0001f1e8\U0001f1ea\U0001f1ec\U0001f1ee\U0001f1f3\U0001f1fa]|\U0001f1fc[\U0001f1eb\U0001f1f8]|\U0001f1fd\U0001f1f0|\U0001f1fe[\U0001f1ea\U0001f1f9]|\U0001f1ff[\U0001f1e6\U0001f1f2\U0001f1fc"
                      "]+", flags=re.UNICODE)


def is_emoji(word):
    if emoji_re.match(word[0]):
        return True
    else:
        return False


def build_corpus_from_pkl():
    lines = []
    files = glob('data/*_en.pkl')
    print('%d files laoded' % len(files))
    for file in files:
        temp = pickle.load(open(file, 'rb'))
        print('%s %d' % (file, len(temp)))
        lines += temp

    print('%d lines total' % len(lines))
    with open(CORPUS_PATH, 'w', encoding='utf-8') as f:
        for line in tqdm(lines):
            f.write(' '.join(line) + '\n')


def build_lexicon():
    """ Build lexicon from the corpus. Assume the corpus is a text file with each sentence per line and
    all sentences segmented by space.
    """
    lexicon = defaultdict(int)
    emoji_lexicon = defaultdict(int)

    with open(CORPUS_PATH, encoding='utf-8') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            tokens = line.strip().split(' ')
            for token in tokens:
                if is_emoji(token):
                    emoji_lexicon[token] += 1
                else:
                    lexicon[token] += 1

    sorted_lexicon = sorted(lexicon.items(), key=lambda x: (-x[1], x[0]))
    sorted_emoji_lexicon = sorted(emoji_lexicon.items(), key=lambda x: (-x[1], x[0]))

    print('lexicon with size %d', len(sorted_lexicon))
    print(sorted_lexicon[:100])
    print('emoji with size %d', len(sorted_emoji_lexicon))
    print(sorted_emoji_lexicon[:100])

    pickle.dump(sorted_lexicon, open(LEXICON_PATH, 'wb'))
    pickle.dump(sorted_emoji_lexicon, open(EMOJI_LEXICON_PATH, 'wb'))


def split_corpus(shuffle=True):
    with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        if shuffle:
            np.random.shuffle(lines)
        N = len(lines)
        train_split = 0.8
        dev_split = 0.9
        train = lines[:int(N * train_split)]
        dev = lines[int(N * train_split):int(N * dev_split)]
        test = lines[int(N * dev_split):]

    def dump_corpus(lines, path):
        with open(path, 'w', encoding='utf-8') as f:
            # for line in lines:
            f.writelines(lines)

    dump_corpus(train, os.path.join('data', 'train.txt'))
    dump_corpus(dev, os.path.join('data', 'dev.txt'))
    dump_corpus(test, os.path.join('data', 'test.txt'))


def data_generator(raw_data, batch_size, num_steps, n_classes):
    X, Y = raw_data
    data_len = len(X)
    batch_len = data_len // batch_size
    data_x = np.zeros([batch_size, batch_len], dtype=np.int32)
    data_y = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        data_x[i] = X[batch_len * i:batch_len * (i + 1)]
        data_y[i] = Y[batch_len * i:batch_len * (i + 1)]
    epoch_size = batch_len // num_steps
    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    while True:
        for i in range(epoch_size):
            x = data_x[:, i * num_steps:(i + 1) * num_steps]
            y = data_y[:, i * num_steps:(i + 1) * num_steps]
            yield (x, to_categorical(y, num_classes=n_classes))


def generator_y_true(raw_data, batch_size, num_steps, n_classes):
    X, Y = raw_data
    data_len = len(X)
    batch_len = data_len // batch_size
    # data_x = np.zeros([batch_size, batch_len], dtype=np.int32)
    data_y = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        # data_x[i] = X[batch_len * i:batch_len * (i + 1)]
        data_y[i, :] = Y[batch_len * i:batch_len * (i + 1)]

    epoch_size = batch_len // num_steps
    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    y_true = []

    for i in range(epoch_size):
        # x = data_x[:, i * num_steps:(i + 1) * num_steps]
        y = data_y[:, i * num_steps:(i + 1) * num_steps]
        y_true.append(y)

    return y_true


if __name__ == "__main__":
    print('util')
    x = [10] * 1000
    y = [0] * 1000
    a = data_generator((x, y), 32, 10, 3)
    print(next(a))
    # print(np.array(generator_y_true((x, y), 32, 10, 3)).reshape(-1))

    # build_corpus_from_pkl()
    # build_lexicon()
    # split_corpus()
