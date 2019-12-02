from pandas import read_csv
import pathlib

# Get games data from CSV
locationGamesFile = pathlib.Path(r'data/gamesWithID.csv')
dataGamesID = read_csv(locationGamesFile,
                     usecols=["name", "ID"])

locationUsersFile = pathlib.Path(r'data/usersWithID.csv')
dataUsers = read_csv(locationUsersFile, usecols=[4])

# Get games data from CSV
locationGamesFile = pathlib.Path(r'data/steam_games.csv')
dataGames = read_csv(locationGamesFile,
                     usecols=["name", "genre", "game_details", "popular_tags", "publisher", "developer"])

dataGamesID['ID'] = dataGamesID['ID'].fillna('')
gameArray = dataUsers["ID"].unique()
print(gameArray)
print(len(gameArray))
criteriaTest = dataGamesID['ID'].isin(gameArray)
usedGames = dataGamesID[criteriaTest]
print(len(usedGames))

usedGames.to_csv(pathlib.Path(r'data/games_ID_name_corresponding_table.csv'), index=False)


gameArray = usedGames["name"].unique()
print(gameArray)
print(len(gameArray))
criteriaTest = dataGames['name'].isin(gameArray)
usedGamesAll = dataGames[criteriaTest]
print(len(usedGamesAll))

usedGamesAll.to_csv(pathlib.Path(r'data/games_used.csv'), index=False)