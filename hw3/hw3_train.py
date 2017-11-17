import numpy as np
import pandas as pd
import sys
import random
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pydot
import itertools

import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
from keras.utils.vis_utils import plot_model
from sklearn.metrics import confusion_matrix


def get_session():
	gpu_options = tf.GPUOptions(allow_growth=True)
	return 	tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))

def get_training_validation_data():
	raw = np.array(pd.read_csv(sys.argv[1]))
	y = raw[:,0]
	
	category_index_list = []
	for i in range(7):
		category_index_list.append(list())

	for i in range(raw.shape[0]):
		category_index_list[raw[i][0]].append(i)

	validation_count = [1340, 144, 1340, 2393, 1627, 1053, 1627]
		
	validation_data = []
	validation_index = []
	for i in range(7):
		validation_index.extend(random.sample(category_index_list[i], validation_count[i]))
	for index in validation_index:
		validation_data.append(raw[index])
	validation_data = np.array(validation_data)

	training_data = []
	for i in range(raw.shape[0]):
		if not i in validation_index:
			training_data.append(raw[i])
	training_data = np.array(training_data)

	training_y = training_data[:,0]
	training_x = []

	for m, i in enumerate(training_data[:,1]):
		feature = np.array(i.split(' '))
		feature = feature.astype(float)
		feature = feature / 255
		feature = feature.reshape(48,48)
		training_x.append(feature)
	
	training_x = np.array(training_x).reshape(len(training_x), 48, 48, 1)
	training_y = np_utils.to_categorical(training_y)

	validation_y = validation_data[:,0]
	validation_x = []

	for m, i in enumerate(validation_data[:,1]):
		feature = np.array(i.split(' '))
		feature = feature.astype(float)
		feature = feature / 255
		feature = feature.reshape(48,48)
		validation_x.append(feature)

	validation_x = np.array(validation_x).reshape(len(validation_x), 48, 48, 1)
	validation_y = np_utils.to_categorical(validation_y)

	return training_x, training_y, validation_x, validation_y

def build_model():
	model = Sequential()

	model.add(Conv2D(filters=64, kernel_size=(5,5), input_shape=(48,48,1)))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(ZeroPadding2D((2,2)))
	model.add(Conv2D(filters=64, kernel_size=(3,3)))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(MaxPooling2D(pool_size=(5,5),strides=(2,2)))
	model.add(Dropout(0.25))

	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(filters=64, kernel_size=(3,3)))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(filters=64, kernel_size=(3,3)))
	model.add(BatchNormalization())
	model.add(Activation('relu'))	
	model.add(ZeroPadding2D((1,1)))
	model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
	model.add(Dropout(0.25))

	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(filters=64, kernel_size=(3,3)))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(filters=64, kernel_size=(3,3)))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
	model.add(Dropout(0.25))

	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(filters=64, kernel_size=(3,3)))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(filters=64, kernel_size=(3,3)))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(1024, activation='relu'))
	model.add(Dropout(0.25))
	model.add(Dense(1024, activation='relu'))
	model.add(Dropout(0.25))
	model.add(Dense(7, activation='softmax'))

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	return model

def build_data_generator():
	datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=40,  # randomly rotate images in the range (degrees, 0 to 180)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images
	
	return datagen

def main():
	
	ktf.set_session(get_session())

	model = build_model()
	data_generator = build_data_generator()

	train_x, train_y, validation_x, validation_y = get_training_validation_data()
	history = model.fit_generator(data_generator.flow(train_x, train_y,batch_size=128), validation_data=(validation_x, validation_y), samples_per_epoch=train_x.shape[0],nb_epoch=250)
		
	model.save("model.h5")

if __name__ == '__main__':
	main()