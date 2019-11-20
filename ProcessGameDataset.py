from pandas import read_csv
import pathlib

# Get games data from CSV
locationGamesFile = pathlib.Path(r'data/games_used.csv')
dataGames = read_csv(locationGamesFile,
                     usecols=["name", "genre", "game_details", "popular_tags", "publisher", "developer"])

locationIDFile = pathlib.Path(r'data/games_ID_name_corresponding_table.csv')
dataUsersGames = read_csv(locationIDFile)
usedGames = dataGames

# relevant info for recommendation: genre game_details popular_tags publisher developer

usedGames['genre'] = usedGames['genre'].fillna('')
usedGames['game_details'] = usedGames['game_details'].fillna('')
usedGames['popular_tags'] = usedGames['popular_tags'].fillna('')
usedGames['publisher'] = usedGames['publisher'].fillna('')
usedGames['developer'] = usedGames['developer'].fillna('')


def clean_data(x):
    if isinstance(x, str):
        return x.replace(" ", "")
    else:
        print(x)
        return x


usedGames['genre'] = usedGames['genre'].apply(clean_data)
usedGames['game_details'] = usedGames['game_details'].apply(clean_data)
usedGames['popular_tags'] = usedGames['popular_tags'].apply(clean_data)
usedGames['publisher'] = usedGames['publisher'].apply(clean_data)
usedGames['developer'] = usedGames['developer'].apply(clean_data)

usedGames.to_csv(pathlib.Path(r'data/processed_games_for_content-based.csv'), index=False)
