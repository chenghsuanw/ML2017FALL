# Movie Recommendation
Given the user’s rating history on movies, predict the rating of unseen (user, movie) pairs.

## Environment
python3 <br>
numpy 1.14.0 <br>
pandas 0.20.3 <br>
keras 2.0.8

## Data
Dataset is from GroupLens Research.

#### [Training](https://drive.google.com/open?id=1jl4E8s1LrqzMkZ-KR5ZCq9v1t81R1ucW)
Each row is [TrainDataID] [UserID] [MovieID] [Rating]
#### [Testing](https://drive.google.com/open?id=175_paq1mzT0sTUsbHbw5tWWjEof3qbik)
Each row is [TestDataID] [UserID] [MovieID]

## Usage
#### Training
```$ python hw5_train.py [Training_File]```
#### Testing (You can directly use the pretrained model)
```$ python hw5_test.py [Testing_File] [Output_File]```
