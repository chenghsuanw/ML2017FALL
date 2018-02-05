# Text Sentiment Classification
Given a sentence, using Recurrent Neural Network to determine whether the sentiment is positvive or negative.

## Environment
python3 <br>
numpy 1.14.0 <br>
pandas 0.20.3 <br>
keras 2.0.8

## Data
The data is tweets from twitter.

#### [Training_With_Label](https://drive.google.com/open?id=1WD1C6D8Z6VJ_ZUgG9E2OGKiCE4lSAdXd)
There are 200000 data with label. 1 for positive, 0 for negative.

#### [Training_Without_Label](https://drive.google.com/open?id=11qlu6yYqiohDO5jiKTHbVT-tUBF3x-1C)
There are 1200000 data without label for semi-supervised learning.

#### [Testing](https://drive.google.com/open?id=1gd8DZVGlYI2mdj2ob1M8yQ7NUgI17aVn)
There are 200000 testing data.

## Usage
#### Training
```$ python hw4_train.py [Training_With_Label] [Training_Without_Label]```

#### Testing (You can directly use the pretrained model)
```$ python hw4_test.py [Testing_File] [Output_File]```
