import numpy as np
import pandas as pd
import sys

from keras.models import Model, load_model
from keras.layers import Embedding, Input
from keras.layers.merge import Dot, Add
from keras.layers.core import Flatten
from keras.callbacks import EarlyStopping,ModelCheckpoint

import tensorflow as tf
import keras.backend.tensorflow_backend as ktf

test_file = sys.argv[1]
result_file = sys.argv[2]

def get_session():
	gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.7)
	return 	tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))

def main():

	ktf.set_session(get_session())

	test = pd.read_csv(test_file, sep = ",", engine='python')
	test = pd.DataFrame.as_matrix(test)

	users = test[:, 1]
	movies = test[:, 2]

	users -= 1
	movies -= 1

	model = load_model('model.h5')

	predict = model.predict([users, movies])

	f = open(result_file, 'w')
	f.write('TestDataID,Rating')
	for i in range(predict.shape[0]):
		f.write('\n' + str(i+1) + ',' + str(predict[i][0]))
	f.close()


if __name__ == '__main__':
	main()