

import numpy as np
import pandas as pd
import jieba
import re
import h5py
import pickle

import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Embedding
from keras.layers import LSTM
from keras.callbacks import EarlyStopping

import os
import tensorflow as tf
import random

#dropout 0.2 loss: 0.0950 - acc: 0.9751 - val_loss: 0.2876 - val_acc: 0.9115
#dropout 0.3 loss: 0.1097 - acc: 0.9697 - val_loss: 0.2404 - val_acc: 0.9165
#dropout 0.4 loss: 0.0929 - acc: 0.9768 - val_loss: 0.2883 - val_acc: 0.9109
#dropout 0.5 loss: 0.1323 - acc: 0.9631 - val_loss: 0.2643 - val_acc: 0.9107

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(77)
random.seed(77)
tf.set_random_seed(77)

pos = pd.read_excel('pos.xls', header=None)
pos['label'] = 1
neg = pd.read_excel('neg.xls', header=None)
neg['label'] = 0
all_ = pos.append(neg, ignore_index=True)


def clean_text(text):
    text = re.sub('\s+', "", text)
    r1 = u'[A-Za-z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    text = re.sub(r1, ' ', text)
    return text.strip()

all_[0] = all_[0].apply(lambda s: clean_text(s))
all_['words'] = all_[0].apply(lambda s: list(jieba.cut(s))) #调用结巴分词



MAX_SEQUENCE_LENGTH = 100 
WORD_COUNT_VALVE = 5 
WORD_COUNT_LIST_DUMP_FILE = 'word_count_list.dump'
MODEL_WEIGHT_SAVED_FILE = 'lstm_binary.h5'
TRAINING_COUNT = 15000

all_words = []
for i in all_['words']:
	all_words.extend(i)

word_count_list = pd.Series(all_words).value_counts()
word_count_list = word_count_list[word_count_list >= WORD_COUNT_VALVE]

word_count_list[:] = list(range(1, len(word_count_list)+1))
word_count_list[''] = 0
word_set = set(word_count_list.index)

#dump file
with open(WORD_COUNT_LIST_DUMP_FILE, "wb+") as f: 
	pickle.dump(word_count_list, f)

print('Word count list saved with count = ', len(word_count_list))	

def doc2num(s, MAX_SEQUENCE_LENGTH): 
    s = [i for i in s if i in word_set]
    s = s[:MAX_SEQUENCE_LENGTH] + ['']*max(0, MAX_SEQUENCE_LENGTH-len(s))
    return list(word_count_list[s])

all_['doc2num'] = all_['words'].apply(lambda s: doc2num(s, MAX_SEQUENCE_LENGTH))

#shuffle
idx = list(range(len(all_)))
np.random.shuffle(idx)
all_ = all_.loc[idx]

#reshape
x = np.array(list(all_['doc2num']))
y = np.array(list(all_['label']))
y = y.reshape((-1,1))


def create_model():
	model = Sequential()
	model.add(Embedding(len(word_count_list), 16, input_length=MAX_SEQUENCE_LENGTH, name="embedding_1"))
	model.add(LSTM(128, name="lstm_1")) 
	model.add(Dropout(0.3, name="dropout_1"))
	model.add(Dense(1, name="dense_1"))
	model.add(Activation('sigmoid', name="activation_1"))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.summary()
	return model

model = create_model()

#training
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
history = model.fit(x[:TRAINING_COUNT], y[:TRAINING_COUNT], batch_size = 128, epochs= 30 , 
	validation_data = (x[TRAINING_COUNT:], y[TRAINING_COUNT:]), callbacks = [early_stopping])


#save model and weight
model.save(MODEL_WEIGHT_SAVED_FILE)
print('Training model and weight saved as ', MODEL_WEIGHT_SAVED_FILE)


def plot_loss(history, key='binary_crossentropy'):
	plt.figure(figsize=(12,7))
	plt.plot(history.epoch, history.history['loss'])
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.xlim([0,max(history.epoch)])
	plt.show()


plot_loss(history)

def predict_one(s):
	s = clean_text(s)
	s = np.array(doc2num(jieba.lcut(s), MAX_SEQUENCE_LENGTH))
	s = s.reshape((1, s.shape[0]))
	return model.predict_classes(s, verbose=0)[0][0]