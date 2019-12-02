from pandas import read_csv
import random
import pathlib

# Get data from CSV
locationUsersFile = pathlib.Path(r'data/purchase_play.csv')
dataUsers = read_csv(locationUsersFile)

# get 20% of random elements (combination user-game) for test dataset
testUsers = dataUsers.sample(frac=0.2, replace=False)

# get the remaining elements for training dataset
trainUsers = dataUsers[~dataUsers.isin(testUsers)].dropna()

# output csv
testUsers.to_csv(pathlib.Path(r'data/steam_user_test.csv'), index=False)
trainUsers.to_csv(pathlib.Path(r'data/steam_user_train.csv'), index=False)
