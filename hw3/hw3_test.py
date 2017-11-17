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

def get_testing_data():
	raw_test = np.array(pd.read_csv(sys.argv[1]))
	
	x_test = []
	for i in range(raw_test.shape[0]):
		x_test.append((np.array(raw_test[i][1].split(' ')).astype(float) / 255).reshape(48,48))
	x_test = np.array(x_test).reshape(raw_test.shape[0], 48, 48, 1)
	
	return x_test

def main():

	model = load_model('model.h5')
	test_x = get_testing_data()
	result = model.predict(test_x)

	f = open(sys.argv[2],"w")

	f.write("id,label\n")
	
	for i in range(result.shape[0]):
		f.write(str(i)+","+str(np.argmax(result[i]))+"\n")
	f.close()

if __name__ == '__main__':
	main()