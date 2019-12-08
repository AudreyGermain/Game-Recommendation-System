from pandas import read_csv, DataFrame, concat
import pathlib
import matplotlib.pyplot as plt

# get review info from csv
locationReviewFile = pathlib.Path(r'../../data/raw_data/steam_games.csv')
dataGames = read_csv(locationReviewFile, usecols=["game_details"])

x = []

for i, row in dataGames.iterrows():
    if type(row["game_details"]) is str:
        x = x + row["game_details"].split(',')

uniqueGenre = list(set(x))

df = DataFrame(columns=["game_details", "count"])
for genre in uniqueGenre:
    df2 = DataFrame(data=[[genre, x.count(genre)]], columns=["game_details", "count"])
    df = concat([df, df2], ignore_index=True)

df = df.sort_values(by="count", ascending=False)
df = df.head(20)

ax = df.plot.barh(x='game_details', y='count')

# Add title and axis names
plt.title('Recurrence of Game Details')
plt.xlabel('Number of Games')
plt.ylabel('Game Details')
plt.yticks(fontsize=8)

plt.show()
