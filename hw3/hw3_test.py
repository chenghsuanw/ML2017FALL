import numpy as np
import pandas as pd
import sys

import keras
from keras.models import load_model

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

	with open(sys.argv[2], 'w') as f:
		f.write("id,label\n")
		for i in range(result.shape[0]):
			f.write(str(i)+","+str(np.argmax(result[i]))+"\n")

if __name__ == '__main__':
	main()