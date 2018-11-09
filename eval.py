import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", "-m", type=str, default='model_cbilstm_02_0.353676.h5', help="the model to use in evaluation")
parser.add_argument("--batch_size", "-bs", type=int, default=512, help="batch size")
parser.add_argument("--step_size", "-ts", type=int, default=40, help="step size")
parser.add_argument("--merge_punc", "-mp", action='store_true', help="merge all punctuation in evaluation")
args = parser.parse_args()

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from keras.models import load_model
from util import data_generator, generator_y_true
from data import Vocab, EmojiVocab, Corpus
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score
import numpy as np

model = load_model(os.path.join('weight', args.model))

# load corpus and vocab
vocab = Vocab(20000) # 20k
emoji_vocab = EmojiVocab(40)
corpus = Corpus(vocab, emoji_vocab, debug=False, eval=True)

encoded_test = corpus.encoded_test

# evaluation
y_pred = model.predict_generator(data_generator(encoded_test, args.batch_size, args.step_size, len(emoji_vocab)),
                                 len(encoded_test[0])//(args.batch_size * args.step_size),
                                 verbose=1)

target_names = [emoji_vocab.decode(x) for x in range(len(emoji_vocab))]
y_true = list(np.array(generator_y_true(encoded_test, args.batch_size, args.step_size, len(emoji_vocab))).reshape(-1))

y_pred = list(y_pred.reshape(-1, len(emoji_vocab)).argmax(axis=1))

assert len(y_true)== len(y_pred)

# print('Confusion Matrix')
# print(confusion_matrix(y_true, y_pred))
print("classification report")
print(classification_report(y_true, y_pred, target_names=target_names))
