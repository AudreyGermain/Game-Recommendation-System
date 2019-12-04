import re
from pandas import read_csv
import pathlib

# Get games data from CSV
locationGamesFile = pathlib.Path(r'../../data/raw_data/steam_games.csv')
dataGames = read_csv(locationGamesFile,
                     usecols=["name", "genre", "game_details", "popular_tags", "publisher", "developer"])

locationUsersFile = pathlib.Path(r'../../data/model_data/steam_user_train.csv')
dataUsers = read_csv(locationUsersFile, header=None, usecols=[0, 1, 2, 3],
                     names=["user_id", "game_name", "behavior", "hours"])

dataGames['name'] = dataGames['name'].fillna('')

# create column ID for game and user dataset
dataGames["ID"] = ""
dataUsers["ID"] = ""

# remove spaces and special character from game name in both dataset
for i, row in dataGames.iterrows():
    clean = re.sub('[^A-Za-z0-9]+', '', row["name"])
    clean = clean.lower()
    dataGames.at[i, 'ID'] = clean

for i, row in dataUsers.iterrows():
    clean = re.sub('[^A-Za-z0-9]+', '', row["game_name"])
    clean = clean.lower()
    dataUsers.at[i, 'ID'] = clean

# find all the games in the game dataset that match the games in user dataset
gameArrayUsers = dataUsers["ID"].unique()
print(len(gameArrayUsers))
criteriaTest = dataGames['ID'].isin(gameArrayUsers)
usedGames = dataGames[criteriaTest]
print(len(usedGames))

# relevant info for recommendation: genre game_details popular_tags publisher developer
usedGames.loc[:, 'genre'] = usedGames['genre'].fillna('')
usedGames.loc[:, 'game_details'] = usedGames['game_details'].fillna('')
usedGames.loc[:, 'popular_tags'] = usedGames['popular_tags'].fillna('')
usedGames.loc[:, 'publisher'] = usedGames['publisher'].fillna('')
usedGames.loc[:, 'developer'] = usedGames['developer'].fillna('')


def clean_data(x):
    if isinstance(x, str):
        return x.replace(" ", "")
    else:
        print(x)
        return x


# remove spaces between the word. This way
usedGames.loc[:, 'genre'] = usedGames['genre'].apply(clean_data)
usedGames.loc[:, 'game_details'] = usedGames['game_details'].apply(clean_data)
usedGames.loc[:, 'popular_tags'] = usedGames['popular_tags'].apply(clean_data)
usedGames.loc[:, 'publisher'] = usedGames['publisher'].apply(clean_data)
usedGames.loc[:, 'developer'] = usedGames['developer'].apply(clean_data)

# create some column containing a mix of different information
usedGames["genre_publisher_developer"] = usedGames['genre'] + usedGames['publisher'] + usedGames['developer']
usedGames["genre_popular_tags_developer"] = usedGames['genre'] + usedGames['popular_tags'] + usedGames['developer']
usedGames["genre_popular_tags_game_details"] = usedGames['genre'] + usedGames['popular_tags'] + usedGames['game_details']
usedGames["genre_publisher_developer_game_details"] = usedGames['genre'] + usedGames['publisher'] + usedGames['developer'] + usedGames['game_details']

usedGames.drop_duplicates("name")
usedGames.to_csv(pathlib.Path(r'../../data/intermediate_data/processed_games_for_content-based.csv'), index=False)

