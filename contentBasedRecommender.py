from pandas import read_csv, Series
import pathlib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Get games data from CSV
locationGamesFile = pathlib.Path(r'data/steam_games.csv')
dataGames = read_csv(locationGamesFile, nrows=1000)     # for test purpose just use the first 1000 games

# need to do some modification on data to make sure there is no NaN in column
dataGames['popular_tags'] = dataGames['popular_tags'].fillna('')
# Compute the Cosine Similarity matrix using the popular tags column
count = CountVectorizer(stop_words='english')
count_matrix_popular_tags = count.fit_transform(dataGames['popular_tags'])
cosine_sim_matrix_popular_tags = cosine_similarity(count_matrix_popular_tags, count_matrix_popular_tags)

# need to do some modification on data to make sure there is no NaN
dataGames['genre'] = dataGames['genre'].fillna('')
# Compute the Cosine Similarity matrix using the genre column
count = CountVectorizer(stop_words='english')
count_matrix_genre = count.fit_transform(dataGames['genre'])
cosine_sim_matrix_genre = cosine_similarity(count_matrix_genre, count_matrix_genre)

# Construct a reverse map of indices and game names
indices = Series(dataGames.index, index=dataGames['name']).drop_duplicates()


# Function that takes in game name and Cosine Similarity matrix as input and outputs most similar games
def get_recommendations(title, cosine_sim):
    # Get the index of the game that matches the name
    idx = indices[title]

    # Get the pairwise similarity scores of all games with that game
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the games based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar games
    # (not the first one because this games as a score of 1 (perfect score) similarity with itself)
    sim_scores = sim_scores[1:11]

    # Get the games indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar games
    return dataGames['name'].iloc[movie_indices]


print("example with game DOOM:")
print("recommendation by popular tags:")
print(get_recommendations('DOOM', cosine_sim_matrix_popular_tags))
print("recommendation by genre:")
print(get_recommendations('DOOM', cosine_sim_matrix_genre))

