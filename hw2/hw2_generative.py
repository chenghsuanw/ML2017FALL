import numpy as np
import pandas as pd
import sys

def sigmoid(z):
	return 1/(1 + np.exp(-z))

def main():
	x = pd.read_csv(sys.argv[1])
	y = pd.read_csv(sys.argv[2])
	x_test = pd.read_csv(sys.argv[3])

	x = np.array(x)
	y = np.squeeze(y)
	x_test = np.array(x_test)

	c0 = []
	c1 = []

	for i in range(y.shape[0]):
		if y[i] == 0:
			c0.append(x[i])
		else:
			c1.append(x[i])

	mu0 = np.mean(c0, axis=0)
	mu1 = np.mean(c1, axis=0)

	cov0 = np.cov(c0, rowvar=False)
	cov1 = np.cov(c1, rowvar=False)

	cov = len(c0)/y.shape[0] * cov0 + len(c1)/y.shape[0] * cov1
	cov_inv = np.linalg.pinv(cov)

	with open(sys.argv[4], 'w') as f:
		f.write("id,label\n")
		for i in range(x_test.shape[0]):
			z = np.dot(np.dot((mu0 - mu1).T, cov_inv), x_test[i]) - 1/2 * np.dot(np.dot(mu0.T, cov_inv), mu0) + 1/2 * np.dot(np.dot(mu1.T, cov_inv), mu1) + np.log(len(c0)/len(c1))
			p = sigmoid(z)
			if p > 0.5:
				f.write(str(i+1)+",0\n")
			else:
				f.write(str(i+1)+",1\n")

if __name__ == '__main__':
	main()

