from pandas import read_csv
import pathlib

n_recommendation = 20

locationAlgoOutputFile = pathlib.Path(r'../data/output_data/content_based_recommender_output_genre_publisher_developer.csv')
dataOutputs = read_csv(locationAlgoOutputFile)

locationTestFile = pathlib.Path(r'../data/model_data/steam_user_test.csv')
dataTest = read_csv(locationTestFile)

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

print(dataOutputs["ratio"].describe(include=[float]))
print(dataOutputs["numberRecommendationUserHas"].describe(include=[float]))
print(dataOutputs["numberGamesUserHasInTest"].describe(include=[float]))
dataOutputs.to_csv(pathlib.Path(r'../data/evaluation_data/Content_based_evaluation_genre_publisher_developer.csv'),
                   columns=["user_id", "ratio", "numberRecommendationUserHas", "numberGamesUserHasInTest"], index=False)
