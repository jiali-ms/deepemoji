import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", "-m", type=str, default='model_clstm_01_0.585939.h5', help="the model to use in evaluation")
args = parser.parse_args()

import os
from keras.models import load_model
from util import data_generator
from data import Vocab, EmojiVocab, Corpus
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score
import numpy as np

model = load_model(os.path.join('weight', args.model))

# load corpus and vocab
vocab = Vocab(20000) # 100k
emoji_vocab = EmojiVocab(500)
# corpus = Corpus(vocab, debug=True)

# punc_dict = set(['、', '。', '「', '」', '・', '）', '（', '，', '？', '！', '…', '〜', '．', '‐', '『', '』', '―', '：', '“', '”'])

original = "i will always love you <eos> don't want miss a thing <eos> this the best gift <eos> happy birthday"
input = original.split()
encoded_input = [vocab.encode(x) for x in input]
print(encoded_input)

pred = model.predict(np.array(encoded_input).reshape((1,-1)))

y = []
for i in range(pred.shape[1]):
    y.append(pred[0][i].argsort()[-5:][::-1])

decoded = []
for i in range(len(encoded_input)):
    decoded.append(vocab.decode(encoded_input[i]))
    decoded += [emoji_vocab.decode(x) for x in y[i]]

print(original)
print(''.join(decoded))