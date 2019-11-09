from pandas import read_csv
import random
import pathlib

# Get data from CSV
locationUsersFile = pathlib.Path(r'data/steam-200k.csv')
dataUsers = read_csv(locationUsersFile, header=None, usecols=[0, 1, 2, 3],
                     names=["user_id", "game_name", "behavior", "hours"])

# get list of all users
userArray = dataUsers["user_id"].unique()

# get number of users and find 20% of random users for test dataset
userCount = len(userArray)
testUserCount = round(0.2*userCount)
testUsers = random.sample(list(userArray), testUserCount)

# split user in 2 sets and output csv
criteriaTest = dataUsers['user_id'].isin(testUsers)
trainDataset = dataUsers[~criteriaTest]
testDataset = dataUsers[criteriaTest]
testDataset.to_csv(pathlib.Path(r'data/steam_user_test.csv'), columns=["user_id", "game_name", "behavior", "hours"], index=False)
trainDataset.to_csv(pathlib.Path(r'data/steam_user_train.csv'), columns=["user_id", "game_name", "behavior", "hours"], index=False)

