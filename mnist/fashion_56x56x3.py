import numpy as np

import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Embedding, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping
from keras import backend as K

import os
import tensorflow as tf
import random
import gzip

import datetime
import cv2


os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(77)
random.seed(77)
tf.set_random_seed(77)

#data source [https://github.com/zalandoresearch/fashion-mnist]

#Epoch 12/30 loss: 0.1219 - acc: 0.9521 - val_loss: 0.2591 - val_acc: 0.9255


BASE_PATH = 'D:/work/source/fashion-mnist/data/fashion/'
#BASE_PATH = '/export/work/fashion-mnist/data/fashion/'
MODEL_WEIGHT_SAVED_FILE = 'fashion_56x65x3.h5'

img_rows, img_cols = 28, 28


def load_data(base_path):
    """Loads the Fashion-MNIST dataset.

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    dirname = os.path.join('datasets', 'fashion-mnist')
    base = base_path
    files = ['train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
             't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz']

    paths = []
    for fname in files:
        paths.append(os.path.join(base_path, fname))

    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(imgpath.read(), np.uint8,
                                offset=16).reshape(len(y_train), 28, 28)

    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(imgpath.read(), np.uint8,
                               offset=16).reshape(len(y_test), 28, 28)

    return (x_train, y_train), (x_test, y_test)



(x_train, y_train), (x_test, y_test) = load_data(BASE_PATH)

class_names = ['T-shirt/T恤', 'Trouser/长裤', 'Pullover/套衫', 'Dress/连衣裙', 'Coat/大衣', 
               'Sandal/凉鞋', 'Shirt/衬衫', 'Sneaker/运动鞋', 'Bag/包', 'Ankle boot/短靴']

num_classes = len(class_names)

#channel last
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

print('Apply image resize...')
start = datetime.datetime.now()
#image resize
x_train = np.array([cv2.resize(x, (img_rows*2, img_cols*2)) for x in x_train])
x_test = np.array([cv2.resize(x, (img_rows*2, img_cols*2)) for x in x_test])
end = datetime.datetime.now()
print('Image resize DONE, cost =', end-start)

#(img_rows, img_cols, 1) -> (img_rows, img_cols, 3)
x_train = np.squeeze(np.stack((x_train,)*3, axis=-1))
x_test = np.squeeze(np.stack((x_test,)*3, axis=-1))

input_shape = (img_rows*2, img_cols*2, 3)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255.0
x_test = x_test / 255.0            

#img = x_train[0]
#cv2.imshow("input", img)
#cv2.waitKey(10*1000)
#quit()

def show_samples():
	plt.figure(figsize=(7,7))
	for i in range(25):
		plt.subplot(5,5,i+1)
		plt.xticks([])
		plt.yticks([])
		plt.grid(False)
		plt.imshow(x_train[i], cmap=plt.cm.binary)
		plt.xlabel(class_names[y_train[i]], fontproperties='SimHei')
	plt.show()

#show_samples()	
#quit()

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)   

def create_model():
	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))
	model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
	model.summary()
	return model

model = create_model()

#training
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
history = model.fit(x_train, y_train, batch_size = 128, epochs= 30 , 
	validation_data = (x_test, y_test), callbacks = [early_stopping])


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

