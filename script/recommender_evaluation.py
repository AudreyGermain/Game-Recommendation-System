from pandas import read_csv
import pathlib

n_recommendation = 20

locationTestFile = pathlib.Path(r'../data/model_data/steam_user_test.csv')
dataTest = read_csv(locationTestFile)


def evaluate(name_algo, location_algo_output_file, eval_output_file):
    dataOutputs = read_csv(location_algo_output_file)

    dataOutputs["numberGamesUserHasInTest"] = 0
    dataOutputs["numberRecommendationUserHas"] = 0
    dataOutputs["ratio"] = 0

    for i, row in dataOutputs.iterrows():
        userGames = dataTest[dataTest["user_id"] == row["user_id"]]["game_name"].tolist()
        dataOutputs.at[i, 'numberGamesUserHasInTest'] = len(userGames)
        count = 0
        for j in range(1, n_recommendation+1):
            if row[j] in userGames:
                count += 1
        dataOutputs.at[i, "numberRecommendationUserHas"] = count
        if len(userGames) != 0:
            dataOutputs.at[i, "ratio"] = float(count/len(userGames))
    print(name_algo)
    print(dataOutputs["ratio"].describe(include=[float]))
    print(dataOutputs["numberRecommendationUserHas"].describe(include=[float]))
    print(dataOutputs["numberGamesUserHasInTest"].describe(include=[float]))
    dataOutputs.to_csv(eval_output_file,
                       columns=["user_id", "ratio", "numberRecommendationUserHas", "numberGamesUserHasInTest"], index=False)


evaluate("Collaborative with EM",
         pathlib.Path(r'../data/output_data/Collaborative_EM_output.csv'),
         pathlib.Path(r'../data/evaluation_data/Collaborative_EM_evaluation.csv'))
evaluate("Collaborative with ALS",
         pathlib.Path(r'../data/output_data/Collaborative_recommender_als_output.csv'),
         pathlib.Path(r'../data/evaluation_data/Collaborative_als_evaluation.csv'))
evaluate("Content based with genre",
         pathlib.Path(r'../data/output_data/content_based_recommender_output_genre.csv'),
         pathlib.Path(r'../data/evaluation_data/Content_based_evaluation_genre.csv'))
evaluate("Content based with genre, popular tags & developer",
         pathlib.Path(r'../data/output_data/content_based_recommender_output_genre_popular_tags_developer.csv'),
         pathlib.Path(r'../data/evaluation_data/Content_based_evaluation_genre_popular_tags_developer.csv'))
evaluate("Content based with genre, popular tags & game details",
         pathlib.Path(r'../data/output_data/content_based_recommender_output_genre_popular_tags_game_details.csv'),
         pathlib.Path(r'../data/evaluation_data/Content_based_evaluation_genre_popular_tags_game_details.csv'))
evaluate("Content based with genre, publisher & developer",
         pathlib.Path(r'../data/output_data/content_based_recommender_output_genre_publisher_developer.csv'),
         pathlib.Path(r'../data/evaluation_data/Content_based_evaluation_genre_publisher_developer.csv'))
evaluate("Content based with genre, publisher, developer & game details",
         pathlib.Path(r'../data/output_data/content_based_recommender_output_genre_publisher_developer_game_details.csv'),
         pathlib.Path(r'../data/evaluation_data/Content_based_evaluation_genre_publisher_developer_game_details.csv'))
evaluate("Content based with popular tags",
         pathlib.Path(r'../data/output_data/content_based_recommender_output_popular_tags.csv'),
         pathlib.Path(r'../data/evaluation_data/Content_based_evaluation_popular_tags.csv'))