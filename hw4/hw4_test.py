import numpy as np
import sys
import pickle

from keras.models import load_model
from keras.preprocessing import sequence
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf

testing_file = sys.argv[1]
result_file = sys.argv[2]

sequence_length = 30

def get_session():
	gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.7)
	return 	tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))

def get_testing_data():
	f = open(testing_file)
	sentences = []
	f.readline()
	i = 0
	for line in f.readlines():
		sentences.append(line[len(str(i))+1:])
		i += 1
	f.close()

	return sentences

def to_sequences(token, text):
	text_seq = token.texts_to_sequences(text)
	text_seq = sequence.pad_sequences(text_seq, maxlen=sequence_length)
	return text_seq

def main():

	ktf.set_session(get_session())
	
	with open("token",'rb') as f:
		token = pickle.load(f)

	x_test = get_testing_data()
	x_test_seq = to_sequences(token, x_test)
	
	model = load_model('model.h5')

	result = np.around(model.predict(x_test_seq,  batch_size=1024, verbose=1)).astype(int)
	
	f = open(result_file,"w")
	f.write("id,label")
	for i in range(result.shape[0]):
		f.write("\n" + str(i) + "," + str(result[i][0]))
	f.close()


if __name__ == '__main__':
	main()