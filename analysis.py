from pandas import read_csv
import sys
import random

# Get data from CSV
locationUsersFile = str(sys.argv[1])
locationGamesFile = str(sys.argv[2])
dataUsers = read_csv(locationUsersFile, header=None, usecols=[0, 1, 2, 3],
                     names=["user_id", "game_name", "behavior", "hours"])
dataGames = read_csv(locationGamesFile)

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
testDataset.to_csv(sys.argv[3], columns=["user_id", "game_name", "behavior", "hours"], index=False)
trainDataset.to_csv(sys.argv[4], columns=["user_id", "game_name", "behavior", "hours"], index=False)

