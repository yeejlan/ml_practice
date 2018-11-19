

import numpy as np
import pandas as pd
import jieba
import re
import h5py
import pickle

import matplotlib.pyplot as plt

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout, Embedding
from keras.layers import LSTM
from keras.callbacks import EarlyStopping

import os
import tensorflow as tf
import random

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(77)
random.seed(77)
tf.set_random_seed(77)

def clean_text(text):
    text = re.sub('\s+', "", text)
    r1 = u'[A-Za-z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    text = re.sub(r1, ' ', text)
    return text.strip()


MAX_SEQUENCE_LENGTH = 100 
WORD_COUNT_VALVE = 5 
WORD_COUNT_LIST_DUMP_FILE = 'word_count_list.dump'
MODEL_WEIGHT_SAVED_FILE = 'lstm_binary.h5'
#MODEL_WEIGHT_SAVED_FILE = 'lstm_pre_trained_emb.h5'
TRAINING_COUNT = 15000

word_count_list = []

#load word count list file
with open(WORD_COUNT_LIST_DUMP_FILE, "rb") as f: 
	word_count_list = pickle.load(f)

print('Word list loaded with count = ', len(word_count_list))

word_set = set(word_count_list.index)


def doc2num(s, MAX_SEQUENCE_LENGTH): 
    s = [i for i in s if i in word_set]
    s = s[:MAX_SEQUENCE_LENGTH] + ['']*max(0, MAX_SEQUENCE_LENGTH-len(s))
    return list(word_count_list[s])


#load model
model = load_model(MODEL_WEIGHT_SAVED_FILE)
print('Model loaded from', MODEL_WEIGHT_SAVED_FILE)
model.summary()


def predict_one(s):
	s = clean_text(s)
	s = np.array(doc2num(jieba.lcut(s), MAX_SEQUENCE_LENGTH))
	s = s.reshape((1, s.shape[0]))
	return model.predict(s, verbose=0)[0][0]

data_arr = [
	'做工不错，也够大，可以放两张A4纸还有剩余空间。是用来放箫谱的，原来谱子用板夹夹着立在桌上，高度和角度都不合适，有了这个方便多了。工欲善其事，必先利其器，再说价格也亲民，很好。',
	'非常实用。三角腿支撑稳定，高度调节灵活，架面可以放平。除了当谱架，想站站坐坐看书iPad都能摆上。',
	'有点重，太高了有点歪',
	'比想象中要大，质量应该不错',
	'东西好，物有所值，物流快。',
	'之前送的琴谱架坏了 就赶紧在京东买个 当天就送到家 质量非常不错 性价比很高',
	'物流慢，货物收到竟然会有瑕疵，希望大家们注意一下！',
	'谁买了这垃圾谁傻，穿了不到10天，客服不讲理，说什么也不给换，真垃圾，谁买谁傻！',
	'这次是最失败的一次购物，不方便去实体店买鞋子就在这家买了，买的时候说的好好的是秋冬季穿的，拿到货打开后就失望了，这可能夏天最热的时候才穿的吧，冬天穿这鞋太开玩笑了',
	'不好大家不要买了，看我图片，麻烦又不想换，下次不在这里买了',
]	

for comment in data_arr:
	print(predict_one(comment))

#0.9916773     
#0.9919246     
#0.007935761   
#0.99084353    
#0.9912998     
#0.9916924     
#0.0076073133  
#0.007534861   
#0.00883507    
#0.0076051066  	