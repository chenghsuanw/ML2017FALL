# Principal Components Analysis (PCA) of colored faces
Implement Principal Components Analysis for dimension reduction on colored faces, and then using top 4 eigenfaces to reconstruct the faces.

| original | reconstruction | original | reconstruction |
| -------- | -------------- | -------- | -------------- |
| ![image](https://github.com/chenghsuanw/ML2017FALL/blob/master/hw6/asset/5.jpg) | ![image](https://github.com/chenghsuanw/ML2017FALL/blob/master/hw6/asset/5_re.jpg) | ![image](https://github.com/chenghsuanw/ML2017FALL/blob/master/hw6/asset/10.jpg) | ![image](https://github.com/chenghsuanw/ML2017FALL/blob/master/hw6/asset/10_re.jpg) |

## Environment
python3 <br>
numpy 1.14.0 <br>
scikit-image 0.13.1

## Dataset
Dataset is from Prof. Ian Craw of Aberdeen University.

#### [Faces](https://drive.google.com/open?id=1wplq6b13QA56YG4VObitXtLn-SfNmWUy)
There are 415 600*600 colored faces.

## Usage
```$ python pca.py [Image_Directory] [Image_File]```

# Image Clustering
Given images which are from 2 datasets, determine whether 2 images are from the same dataset with unsupervised learning.

## Environment
python3 <br>
numpy 1.14.0 <br>
pandas 0.20.3 <br>
scikit-learn 0.18.1

## Data
#### [Images](https://drive.google.com/open?id=1hGVzLiOGX7YvTFeOzhThLOf0EVnScA4x)
The data is .npy file. Using ```np.load()``` to get an (140000, 784) ndarray. Each row represents a 28*28 image.
#### [Testing](https://drive.google.com/open?id=1Ji43jPI3GWo58EWoCTaBYHIrgI0C5wo8)
Each row is [ID] [Image1_ID] [Image2_ID]. 

## Usage
#### Training
```$ python hw6_train.py [Image_File]```
#### Testing
```$ python hw6_test.py [Image_File] [Testing_File] [Output_File]```
