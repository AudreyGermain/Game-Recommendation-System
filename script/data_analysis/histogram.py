import pandas as pd
import numpy as np
import pathlib
from plotnine import *
from plotnine.data import *
import seaborn as sns
import matplotlib.pyplot as plt

locationUsersFile=pathlib.Path(r'D:/Game-Recommendation-System/data/raw_data/steam_users_purchase_play.csv')
steam_clean = pd.read_csv(locationUsersFile, header=1, names=['user', 'game', 'hrs', 'purchase', 'play'])
print(steam_clean)
game_total_hrs = steam_clean.groupby(by='game')['hrs'].sum()
most_played_games = game_total_hrs.sort_values(ascending=False)[:20]


# game with the highest number of users
game_freq = steam_clean.groupby(by='game').agg({'user': 'count', 'hrs': 'sum'}).reset_index()
top20 = game_freq.sort_values(by='user',ascending=False)[:20].reset_index()
print(top20)

# show histogram
plt.figure(figsize=(20, 10))
sns.set(font_scale = 2)
ax = sns.barplot(x='user', y='game', alpha=0.9,data=top20, palette='Blues_r',dodge=False)
ax.set(xlabel='Number of Users', ylabel='Game', title='Top 20 games with the most users(All)')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.show()

#consider the situation that only included play
index=steam_clean[steam_clean['play']==0].index
steam_clean.drop(index,inplace=True)
game_total_hrs = steam_clean.groupby(by='game')['hrs'].sum()
most_played_games = game_total_hrs.sort_values(ascending=False)[:20]
# game with the highest number of users
game_freq = steam_clean.groupby(by='game').agg({'user': 'count', 'hrs': 'sum'}).reset_index()
top20 = game_freq.sort_values(by='user',ascending=False)[:20].reset_index()

plt.figure(figsize=(20, 10))
sns.set(font_scale = 2)
ax = sns.barplot(x='user', y='game', alpha=0.9,data=top20, palette='Blues_r',dodge=False)
ax.set(xlabel='Number of Users', ylabel='Game', title='Top 20 games with the most users(Play)')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.show()






