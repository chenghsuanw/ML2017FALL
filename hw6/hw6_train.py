import numpy as np
import pandas as pd
import sys

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.externals import joblib

train_file = sys.argv[1]
reduced_dim = 300

def main():
	
	x = np.load(train_file)
	x = x / 255

	pca = PCA(n_components=reduced_dim, whiten=True).fit(x)
	joblib.dump(pca, 'pca_model')
	x = pca.transform(x)
	
	kmeans = KMeans(n_clusters=2).fit(x)
	joblib.dump(kmeans, 'kmeans_model')


if __name__ == '__main__':
	main()

