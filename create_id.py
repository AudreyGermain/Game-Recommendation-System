import re
from pandas import read_csv
import pathlib

# Get games data from CSV
locationGamesFile = pathlib.Path(r'data/steam_games.csv')
dataGames = read_csv(locationGamesFile,
                     usecols=["name"])

locationUsersFile = pathlib.Path(r'data/steam-200k.csv')
dataUsers = read_csv(locationUsersFile, header=None, usecols=[0, 1, 2, 3],
                     names=["user_id", "game_name", "behavior", "hours"])

dataGames['name'] = dataGames['name'].fillna('')

dataGames["ID"] = ""
dataUsers["ID"] = ""

for i, row in dataGames.iterrows():
    clean = re.sub('[^A-Za-z0-9]+', '', row["name"])
    clean = clean.lower()
    dataGames.at[i, 'ID'] = clean

for i, row in dataUsers.iterrows():
    clean = re.sub('[^A-Za-z0-9]+', '', row["game_name"])
    clean = clean.lower()
    dataUsers.at[i, 'ID'] = clean

print(dataGames['ID'])
print(dataUsers['ID'])

dataUsers.to_csv(pathlib.Path(r'data/usersWithID.csv'), index=False)
dataGames.to_csv(pathlib.Path(r'data/gamesWithID.csv'), index=False)
