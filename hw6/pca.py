import numpy as np
import os
from skimage import io
import sys

img_dir = sys.argv[1]
img_file = os.path.join(sys.argv[1], sys.argv[2])

def get_image():
	imgs = []
	files = os.listdir(img_dir)
	for f in files:
		img = io.imread('{}/{}'.format(img_dir, f))
		img = img.flatten()
		imgs.append(img)
	imgs = np.array(imgs)

	return imgs

def transform(M):
	M -= np.min(M)
	M /= np.max(M)
	M = (M*255).astype(np.uint8)

	return M

def reconstruct(img, e1, e2, e3, e4, X_mean):
	img = img - X_mean
	w1 = np.dot(img, e1)
	w2 = np.dot(img, e2)
	w3 = np.dot(img, e3)
	w4 = np.dot(img, e4)
	combine = w1*e1 + w2*e2 + w3*e3 + w4*e4 + X_mean

	return combine

def main():
	X = get_image()
	X_mean = np.mean(X, axis=0)

	U, s, V = np.linalg.svd((X - X_mean).transpose(), full_matrices=False)

	e1 = U[:, 0]
	e2 = U[:, 1]
	e3 = U[:, 2]
	e4 = U[:, 3]

	img = io.imread(img_file).flatten()
	re = reconstruct(img, e1, e2, e3, e4, X_mean)
	io.imsave('reconstruction.jpg', transform(re).reshape(600,600,3))
	
	
if __name__ == '__main__':
	main()