from pandas import read_csv
import pathlib

n_recommendation = 20

locationAlgoOutputFile = pathlib.Path(r'data/Content-based-recommender-output.csv')
dataOutputs = read_csv(locationAlgoOutputFile)

locationTestFile = pathlib.Path(r'data/steam_user_test.csv')
dataTest = read_csv(locationTestFile)

dataOutputs["numberGames"] = 0
dataOutputs["numberGamesUserHas"] = 0
dataOutputs["ratio"] = 0

for i, row in dataOutputs.iterrows():
    userGames = dataTest[dataTest["user_id"] == row["user_id"]]["game_name"].tolist()
    dataOutputs.at[i, 'numberGames'] = len(userGames)
    count = 0
    for j in range(1, n_recommendation+1):
        if row[j] in userGames:
            count += 1
    dataOutputs.at[i, "numberGamesUserHas"] = count
    if len(userGames) != 0:
        dataOutputs.at[i, "ratio"] = float(count/len(userGames))

print(dataOutputs["ratio"].describe(include=[float]))
print(dataOutputs["numberGamesUserHas"].describe(include=[float]))
print(dataOutputs["numberGames "].describe(include=[float]))
dataOutputs.to_csv(pathlib.Path(r'data/Content-based-evaluation.csv'),
                   columns=["user_id", "ratio", "numberGamesUserHas", "numberGames"], index=False)
