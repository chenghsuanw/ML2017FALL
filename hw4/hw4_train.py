import numpy as np
import sys
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout, Bidirectional, GRU
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint

import tensorflow as tf
import keras.backend.tensorflow_backend as ktf

training_label_file = sys.argv[1]
training_nolabel_file = sys.argv[2]

words_num = 20000
sequence_length = 30
embedding_dim = 128
training_label_size = 190000

def get_session():
	gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.7)
	return 	tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))

def get_training_validation_data():
	f = open(training_label_file)
	labels = []
	sentences = []
	for line in f.readlines():
		l = line.strip().split(' +++$+++ ')
		labels.append(l[0])
		sentences.append(l[1])

	f.close()

	labels = np.array(labels)
	sentences = np.array(sentences)

	return sentences[:training_label_size], labels[:training_label_size], sentences[training_label_size:], labels[training_label_size:]

def build_token():
	text = []
	f = open(training_label_file)
	for line in f.readlines():
		l = line.strip().split(' +++$+++ ')
		text.append(l[1])
	f.close()
	f = open(training_nolabel_file)
	for line in f.readlines():
		line = line.strip()
		text.append(line)
	f.close()

	token = Tokenizer(num_words=words_num, filters='\t\n')
	token.fit_on_texts(text)

	with open("token",'wb') as f:
		pickle.dump(token, f)

	return token

def build_model():
	model = Sequential()

	model.add(Embedding(words_num, embedding_dim, input_length=sequence_length))
	model.add(Dropout(0.5))

	# model.add(Bidirectional(LSTM(128, dropout=0.5, recurrent_dropout=0.5, return_sequences=True)))
	model.add((LSTM(256, dropout=0.5, recurrent_dropout=0.5)))
	
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.5))
	# model.add(BatchNormalization())
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.5))
	
	model.add(Dense(1, activation='sigmoid'))

	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

	return model

def get_nolabel_data():
	text = []
	f = open(training_nolabel_file)
	for line in f.readlines():
		text.append(line.strip())

	text = np.array(text)

	return text

def get_semi_training_data(x_nolabel_all, y_nolabel_all_prob):
	y_nolabel_all_prob = np.squeeze(y_nolabel_all_prob)
	index = (y_nolabel_all_prob>0.9) + (y_nolabel_all_prob<0.1)

	x_nolabel_train = x_nolabel_all[index]
	y_nolabel_train = np.around(y_nolabel_all_prob).astype(int)
	y_nolabel_train = y_nolabel_train[index]

	return x_nolabel_train, y_nolabel_train

def to_sequences(token, text):
	text_seq = token.texts_to_sequences(text)
	text_seq = sequence.pad_sequences(text_seq, maxlen=sequence_length)
	return text_seq

def plot_history(history, path):
	plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['val_acc'], loc='upper left')
	plt.savefig(path)
	plt.clf()

def main():

	ktf.set_session(get_session())
	
	x_label, y_label, x_validation, y_validation = get_training_validation_data()

	token = build_token()

	x_label_seq = to_sequences(token, x_label)
	x_validation_seq = to_sequences(token, x_validation)

	model = build_model()
	
	# earlystopping = EarlyStopping(monitor='val_acc', patience = 1, verbose=1, mode='max')
	# checkpoint = ModelCheckpoint('model.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')

	# model.fit(x_label_seq, y_label, validation_data=(x_validation_seq, y_validation), epochs=15, batch_size=1024, callbacks=[earlystopping, checkpoint])
	
	# # semi_training
	# x_nolabel_all = get_nolabel_data()
	
	# for i in range(2):
	# 	model = load_model('model.h5')
	# 	x_nolabel_all_seq = to_sequences(token, x_nolabel_all)
	# 	y_nolabel_all_prob = model.predict(x_nolabel_all_seq, batch_size=1024, verbose=1)
	# 	x_nolabel_train, y_nolabel_train = get_semi_training_data(x_nolabel_all, y_nolabel_all_prob)
	# 	x_nolabel_train_seq = to_sequences(token, x_nolabel_train)
	# 	x_nolabel_train_seq = np.concatenate((x_nolabel_train_seq, x_label_seq), axis=0)
	# 	y_nolabel_train = np.concatenate((y_nolabel_train, y_label), axis=0)
	# 	model.fit(x_nolabel_train_seq, y_nolabel_train, validation_data=(x_validation_seq, y_validation), epochs=6, batch_size=128, callbacks=[earlystopping, checkpoint])


if __name__ == '__main__':
	main()