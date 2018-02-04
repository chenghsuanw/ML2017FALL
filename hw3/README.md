# Image Sentiment Classification
Given a facial image, classify the sentiment. There are total 7 sentiments.

| 0 (Angry) | 1 (Hate) | 2 (Fear) | 3 (Joy) | 4 (Sad) | 5 (Surprise) | 6 (Neutral) |
|---|---|---|---|---|---|---|
| ![image](https://github.com/chenghsuanw/ML2017FALL/blob/master/hw3/asset/0.jpg) | ![image](https://github.com/chenghsuanw/ML2017FALL/blob/master/hw3/asset/1.jpg) | ![image](https://github.com/chenghsuanw/ML2017FALL/blob/master/hw3/asset/2.jpg) | ![image](https://github.com/chenghsuanw/ML2017FALL/blob/master/hw3/asset/3.jpg) | ![image](https://github.com/chenghsuanw/ML2017FALL/blob/master/hw3/asset/4.jpg) | ![image](https://github.com/chenghsuanw/ML2017FALL/blob/master/hw3/asset/5.jpg) | ![image](https://github.com/chenghsuanw/ML2017FALL/blob/master/hw3/asset/6.jpg) |

## Environment
python3 <br>
numpy 1.14.0 <br>
pandas 0.20.3 <br>
keras 2.0.8

## Data
#### [Training](https://drive.google.com/open?id=1-pvd9QmMb4B5faaDgDtFRIBVn3ezUP9n)
280000 48*48-pixel images. Each image has a sentiment label.
#### [Testing](https://drive.google.com/open?id=1346FDFvOWFG68izBiwFIAGIvlZgOiGYh)
7000 48*48-pixel images.

## Usage
#### Training
```$ python hw3_train.py [Training_File]```

#### Testing (You can directly use the pretrained model)
```$ python hw3_test.py [Testing_File] [Output_File]```
