import numpy as np
import pandas as pd
import csv
import sys

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.externals import joblib


train_file = sys.argv[1]
test_file = sys.argv[2]
predict_file = sys.argv[3]

def cluster(kmeans):
	test = pd.read_csv(test_file).as_matrix()

	with open(predict_file, 'w', newline='') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow(['ID','Ans'])
		for i in range(test.shape[0]):
			image1 = test[i][1]
			image2 = test[i][2]
			l1 = kmeans.labels_[image1]
			l2 = kmeans.labels_[image2]
			if l1 == l2:
				writer.writerow([i, 1])
			else:
				writer.writerow([i, 0])

def main():

	x = np.load(train_file)
	x = x / 255

	# PCA
	pca = joblib.load('pca_model')
	x = pca.transform(x)
	
	kmeans = joblib.load('kmeans_model')

	cluster(kmeans)


if __name__ == '__main__':
	main()

