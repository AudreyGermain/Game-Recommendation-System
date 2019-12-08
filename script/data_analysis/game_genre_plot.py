from pandas import read_csv, DataFrame, concat
import pathlib
import matplotlib.pyplot as plt

# get review info from csv
locationReviewFile = pathlib.Path(r'../../data/raw_data/steam_games.csv')
dataGames = read_csv(locationReviewFile, usecols=["genre"])

x = []

for i, row in dataGames.iterrows():
    if type(row["genre"]) is str:
        x = x + row["genre"].split(',')

uniqueGenre = list(set(x))

df = DataFrame(columns=["genre", "count"])
for genre in uniqueGenre:
    df2 = DataFrame(data=[[genre, x.count(genre)]], columns=["genre", "count"])
    df = concat([df, df2], ignore_index=True)

df = df.sort_values(by="count", ascending=False)

ax = df.plot.barh(x='genre', y='count')

# Add title and axis names
plt.title('Recurrence of Genre')
plt.xlabel('Number of Games')
plt.ylabel('Genre')
plt.yticks(fontsize=8)

plt.show()
