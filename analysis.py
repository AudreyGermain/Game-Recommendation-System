from pandas import read_csv
import sys

# Get data from CSV
locationUsersFile = str(sys.argv[1])
locationGamesFile = str(sys.argv[2])
dataUsers = read_csv(locationUsersFile, header=None, usecols=[0, 1, 2, 3],
                     names=["user_id", "game_name", "behavior", "hours", "nothing"])
dataGames = read_csv(locationGamesFile)


