from pandas import read_csv, DataFrame, concat
import pathlib
import matplotlib.pyplot as plt

# get review info from csv
locationReviewFile = pathlib.Path(r'../../data/raw_data/steam_games.csv')
dataGames = read_csv(locationReviewFile, usecols=["popular_tags"])

x = []

for i, row in dataGames.iterrows():
    if type(row["popular_tags"]) is str:
        x = x + row["popular_tags"].split(',')

uniqueGenre = list(set(x))

df = DataFrame(columns=["popular_tags", "count"])
for genre in uniqueGenre:
    df2 = DataFrame(data=[[genre, x.count(genre)]], columns=["popular_tags", "count"])
    df = concat([df, df2], ignore_index=True)

df = df.sort_values(by="count", ascending=False)
print(df)
df = df.head(20)

ax = df.plot.barh(x='popular_tags', y='count')

# Add title and axis names
plt.title('Recurrence of Popular Tags')
plt.xlabel('Number of Games')
plt.ylabel('Popular Tags')
plt.yticks(fontsize=8)

plt.show()
