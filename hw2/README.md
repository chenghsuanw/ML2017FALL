# Income Prediction
Predict whether a person makes over 50K a year.

## Environment
python3 <br>
numpy 1.14.0 <br>
pandas 0.20.3 <br>
keras 2.0.8

## Dataset
### ADULT Dataset
Extraction was done by Barry Becker from the 1994 Census database. A set of reasonably clean records was extracted using the following conditions: ((AGE>16) && (AGI>100) && (AFNLWGT>1) && (HRSWK>0)).
### Training
#### [Feature](https://drive.google.com/open?id=1AxMDzpzwJi0QcWvwzmugHkomw7z3ujj4)
Each row contains one 106-dim feature represents a sample.
#### [Label](https://drive.google.com/open?id=1mg6pYekVeGMXlLUikTQ_ETXzrVmE5gO1)
0 means <= 50K, 1 means > 50K
### [Testing](https://drive.google.com/open?id=1l8K-IsxSEos4q6hEf7xJu3TvL7M9r1Vo)
Each row contains one 106-dim feature represents a sample.

## Usage
### Logistic Regression
```$ python hw2_logistic.py [Training_Feature_File] [Training_Label_File] [Testing_File] [Output_File]```

### Probabilstic Generative Model
```$ python hw2_generative.py [Training_Feature_File] [Training_Label_File] [Testing_File] [Output_File]```

### Deep Neural Network (DNN)
```$ python hw2_DNN.py [Training_Feature_File] [Training_Label_File] [Testing_File] [Output_File]```
