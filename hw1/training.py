import pandas as pd
import numpy as np
import model

iteration = 10000
train_hour = 9

#return training sets and validation sets
def preprocess():
	train_csv = pd.read_csv("train.csv", encoding = "big5")
	train_csv.replace("NR",0,inplace=True)
	train_raw = pd.DataFrame.as_matrix(train_csv)
	train_raw = np.delete(train_raw,range(3),1)
	train_raw = train_raw.astype(float)
	train_raw = train_raw.reshape([12,20,18,24])

	hour_list = []

	for month in range(12):
		for day in range(20):
			for hour in train_raw[month][day].T:
				hour_list.append(hour.T)

	train = np.array(hour_list) #5760(hours)*18(features)
	y_set = train[:,9]

	training_sets = []
	validation_sets = []

	return train, y_set

def train(model, y_set):
	error_set = []
	lr_w = 0
	lr_b = 0

	for i in range(iteration):
		#initial the gradient
		weight_grad = np.zeros([train_hour,model.training_matrix.shape[1]])
		bias_grad = 0

		for training_hour_start in range(model.training_matrix.shape[0] - (train_hour+1)):
			current_training_data = model.training_matrix[training_hour_start:training_hour_start+train_hour]
			y = y_set[training_hour_start+train_hour]

			weight_grad += (-2) * (y - (np.sum(model.weight * current_training_data)) - model.bias) * current_training_data
			bias_grad += (-2) * (y - (np.sum(model.weight * current_training_data)) - model.bias)

			err = (y - (np.sum(model.weight * current_training_data)) - model.bias)**2
			error_set.append(err)

		#update weight and bias
		weight_grad += (2 * model.regular_factor * model.weight)
		lr_w += weight_grad**2
		lr_b += bias_grad**2
		model.weight -= model.learning_rate/np.sqrt(lr_w) * weight_grad
		model.bias -= model.learning_rate/np.sqrt(lr_b) * bias_grad

		if i % 20 == 0:
			print(i,np.sqrt(np.mean(error_set[-(model.training_matrix.shape[0] - (train_hour+1)):])),sep='\t')

	model.error = np.sqrt(np.mean(error_set[-(model.training_matrix.shape[0] - (train_hour+1)):]))

def main():
	raw_training_matrix, y_set = preprocess()
	#PM2.5 square
	model_training_matrix = raw_training_matrix
	for i in range(18):
		model_training_matrix = np.column_stack((model_training_matrix, model_training_matrix[:,i]*model_training_matrix[:,9]))

	model_init_weight = np.array([[0.01] * (model_training_matrix.shape[1])] * train_hour)
	model_init_bias = 0
	model_init_learning_rate = 1.75e-2
	model_regular_factor = 0

	model = model.Model(model_training_matrix, model_init_weight, model_init_bias, model_init_learning_rate, model_regular_factor)
	train(model, y_set)
	print("model error", model.error)

	np.savetxt("weight.txt", model.weight)
	np.savetxt("bias.txt", [model.bias])

if __name__ == '__main__':
	main()
