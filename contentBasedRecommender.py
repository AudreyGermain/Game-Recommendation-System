from pandas import read_csv, Series, DataFrame, concat

import pathlib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

n_recommendation = 20

# Get games data from CSV
locationGamesFile = pathlib.Path(r'data/processed_games_for_content-based.csv')
dataGames = read_csv(locationGamesFile)

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

# get list of games we have info about
listGames = dataGames['name'].unique()


# Function that takes in game name and Cosine Similarity matrix as input and outputs most similar games
def get_recommendations(title, cosine_sim):

    if title not in listGames:
        return []

    # Get the index of the game that matches the name
    idx = indices[title]

    # if there's 2 games or more with same name (game RUSH)
    if type(idx) is Series:
        return []

    # Get the pairwise similarity scores of all games with that game
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the games based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the most similar games
    # (not the first one because this games as a score of 1 (perfect score) similarity with itself)
    sim_scores = sim_scores[1:n_recommendation + 1]

    # Get the games indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top most similar games
    return dataGames['name'].iloc[movie_indices].tolist()


# create dataframe for recommendations
col_names = list(map(str, range(1, n_recommendation + 1)))
col_names = ["user_id"] + col_names
recommendationByUserData = DataFrame(columns=col_names)

# get review info from csv
locationReviewFile = pathlib.Path(r'data/steam_games_reviews.csv')
dataReviews = read_csv(locationReviewFile, usecols=["name", "percentage_positive_review"],)


def make_recommendation_for_user(user_id, game_list, game_user_have):
    if type(game_list) is not list or len(game_list) == 0:
        # return empty one
        return DataFrame(data=[[user_id] + [""] * n_recommendation], columns=col_names)

    # get reviews of game recommendation, remove the games the user already has and order them by reviews
    recommendation_reviews = dataReviews.loc[dataReviews['name'].isin(game_list)]
    recommendation_reviews = recommendation_reviews.loc[~recommendation_reviews['name'].isin(game_user_have)]
    recommendation_reviews = recommendation_reviews.sort_values(by="percentage_positive_review", ascending=False)

    if len(recommendation_reviews.index) < n_recommendation:
        return DataFrame(data=[[user_id] + recommendation_reviews["name"].tolist() +
                               [""] * (n_recommendation - len(recommendation_reviews.index))],
                         columns=col_names)
    else:
        return DataFrame(data=[[user_id] + recommendation_reviews["name"].tolist()[0:n_recommendation]],
                         columns=col_names)


# exemple of recommendation
# print("example with game TERA:")
# print("recommendation by popular tags:")
# print(get_recommendations('TERA', cosine_sim_matrix_popular_tags))
# print("recommendation by genre:")
# print(get_recommendations('TERA', cosine_sim_matrix_genre))
#
# print("example with game DOOM:")
# print("recommendation by popular tags:")
# print(get_recommendations('DOOM', cosine_sim_matrix_popular_tags))
# print("recommendation by genre:")
# print(get_recommendations('DOOM', cosine_sim_matrix_genre))
#
# print("example with game Dota 2:")
# print("recommendation by popular tags:")
# print(get_recommendations('Dota 2', cosine_sim_matrix_popular_tags))
# print("recommendation by genre:")
# print(get_recommendations('Dota 2', cosine_sim_matrix_genre))

# Get users data from CSV
locationUsersFile = pathlib.Path(r'data/purchase_play.csv')   # data/purchase_play
dataUsers = read_csv(locationUsersFile)

previousId = ""
listSuggestion = list()
listGamesUserHas = list()

# loop on all row and get recommendations for user
for j, row in dataUsers.iterrows():
    if previousId != row["user_id"]:
        recommendationByUserData = concat([recommendationByUserData,
                                           make_recommendation_for_user(previousId, listSuggestion, listGamesUserHas)],
                                          ignore_index=True)
        previousId = row["user_id"]
        listSuggestion = list()
        listGamesUserHas = list()
    listGamesUserHas.extend([row["game_name"]])
    listSuggestion.extend(get_recommendations(row["game_name"], cosine_sim_matrix_popular_tags))

# add the last element for the last user
recommendationByUserData = concat([recommendationByUserData,
                                   make_recommendation_for_user(previousId, listSuggestion, listGamesUserHas)],
                                  ignore_index=True)

locationOutputFile = pathlib.Path(r'data/Content-based-recommender-output.csv')
recommendationByUserData.to_csv(locationOutputFile, index=False)



