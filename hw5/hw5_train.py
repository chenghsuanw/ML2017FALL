import numpy as np
import pandas as pd

from keras.models import Model, load_model
from keras.layers import Embedding, Input
from keras.layers.merge import Dot, Add, Concatenate
from keras.layers.core import Flatten, Dense, Dropout
from keras.callbacks import EarlyStopping,ModelCheckpoint, ReduceLROnPlateau

import tensorflow as tf
import keras.backend.tensorflow_backend as ktf


train_file = 'train.csv'

train_proportion = 0.9
epochs = 150
batch_size = 128

latent_dim = 300

def get_session():
	gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.7)
	return 	tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))

def build_model_MF(num_user, num_movie):
	user_input = Input(shape=[1])
	movie_input = Input(shape=[1])
	user_vector = Embedding(num_user, latent_dim)(user_input)
	user_vector = Flatten()(user_vector)
	user_vector = Dropout(0.5)(user_vector)
	movie_vector = Embedding(num_movie, latent_dim)(movie_input)
	movie_vector = Flatten()(movie_vector)
	movie_vector = Dropout(0.5)(movie_vector)

	user_bias = Embedding(num_user, 1)(user_input)
	user_bias = Flatten()(user_bias)
	movie_bias = Embedding(num_movie, 1)(movie_input)
	movie_bias = Flatten()(movie_bias)

	out = Dot(axes=1)([user_vector, movie_vector])
	out = Add()([out, user_bias, movie_bias])

	model = Model([user_input, movie_input], out)
	model.compile(loss='mse', optimizer='adamax')

	return model

def main():

	ktf.set_session(get_session())

	train = pd.read_csv(train_file, sep=",", engine='python')	
	train = pd.DataFrame.as_matrix(train)
	
	num_movie = max(train[:,2])
	num_user = max(train[:,1])

	# shuffle the training data
	np.random.shuffle(train)

	users = train[:, 1]
	users -= 1
	movies = train[:, 2]
	movies -= 1
	ratings = train[:, 3]

	# split the training set and validation set
	train_size = int(train.shape[0] * train_proportion)
	users_train = users[:train_size]
	users_val = users[train_size:]
	movies_train = movies[:train_size]
	movies_val = movies[train_size:]
	ratings_train = ratings[:train_size]
	ratings_val = ratings[train_size:]
	
	model = build_model_MF(num_user, num_movie)
	model.summary()
	
	modelcheckpoint = ModelCheckpoint('hw5_model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
	earlystopping = EarlyStopping(monitor='val_loss', patience = 3, verbose=1, mode='min')
	reducelr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=0, min_lr=0)

	model.fit([users_train, movies_train], ratings_train, validation_data=([users_val, movies_val], 
		ratings_val), epochs=epochs, batch_size=batch_size, callbacks=[modelcheckpoint, earlystopping, reducelr])

if __name__ == '__main__':
	main()