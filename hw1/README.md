# PM2.5 Prediction
Using 9 hours data, such as PM2.5, PM10, NO, CO and so on, to predict PM2.5 of the 10th hour.

## Environment
python3 <br>
numpy 1.14.0 <br>
pandas 0.20.3

## Data
#### [Training](https://drive.google.com/open?id=1SW-Xvr2M-sT1eSGjEEREwnBnAuBXkpGO)
Data of the first 20 days of each month from 豐原 weather station. <br>
#### [Testing](https://drive.google.com/open?id=19u06yx_WbvvwkgfJT4NZ6BbgVRhf6Bas)
Given 9 hours data, predict PM2.5 of the next hour.

## Usage
#### Training
```$ python training.py [Training_File_Path]```

#### Testing
```$ python hw1.py [Testing_File_Path] [Output_File_Path]```
