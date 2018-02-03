import numpy as np
import pandas as pd
import sys

def main():
	x = pd.read_csv(sys.argv[1])
	y = pd.read_csv(sys.argv[2])

	x = np.array(x)
	y = np.squeeze(y)

	#add bias
	x = np.concatenate((np.ones((x.shape[0],1)),x),axis=1)

	#normalize
	x[:,1] = (x[:,1]-np.mean(x[:,1]))/np.std(x[:,1])
	x[:,2] = (x[:,2]-np.mean(x[:,2]))/np.std(x[:,2])
	x[:,4] = (x[:,4]-np.mean(x[:,4]))/np.std(x[:,4])
	x[:,6] = (x[:,6]-np.mean(x[:,6]))/np.std(x[:,6])

	w = np.zeros(len(x[0]))
	l_rate = 1.75e-2
	reg = 1
	repeat = 10000

	ada_sum = np.zeros(len(x[0]))
	#training
	for i in range(repeat):
		pred = (1/(1+np.exp(-np.dot(x,w))))
		loss = pred - y
		gra = np.dot(loss, x) + 2*reg*w
		ada_sum += gra**2
		ada = np.sqrt(ada_sum)
		w -= l_rate * gra / ada

		result = np.rint(pred)
		err = np.sum(np.absolute(result - y))
		print("iteration",i,"\t",1 - err/len(y),sep='')

	#testing
	x_test = pd.read_csv(sys.argv[3])
	x_test = np.array(x_test)
	x_test = np.concatenate((np.ones((x_test.shape[0],1)),x_test),axis=1)

	x_test[:,1] = (x_test[:,1]-np.mean(x_test[:,1]))/np.std(x_test[:,1])
	x_test[:,2] = (x_test[:,2]-np.mean(x_test[:,2]))/np.std(x_test[:,2])
	x_test[:,4] = (x_test[:,4]-np.mean(x_test[:,4]))/np.std(x_test[:,4])
	x_test[:,6] = (x_test[:,6]-np.mean(x_test[:,6]))/np.std(x_test[:,6])

	pred_test = (1/(1+np.exp(-np.dot(x_test,w))))
	result_test = np.rint(pred_test)
	result_test = result_test.astype(int)

	with open(sys.argv[4], 'w') as f:
		f.write("id,label\n")
		for i in range(len(result_test)):
			f.write(str(i+1)+","+str(result_test[i])+"\n")

if __name__ == '__main__':
	main()
