import random
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Get data from CSV
locationUsersFile = pathlib.Path(r'data/steam-200k.csv')
locationGamesFile = pathlib.Path(r'data/steam_games.csv')
df_dataUsers = pd.read_csv(locationUsersFile, header=None, usecols=[0, 1, 2, 3],
                           names=['user_id', 'game_name', 'behavior', 'hours'])
df_dataGames = pd.read_csv(locationGamesFile)

# Split purchase and played time information
df_purchase = df_dataUsers.loc[df_dataUsers['behavior'] == 'purchase'].copy(deep=True)
df_purchase.rename(columns={'hours': 'purchase'}, inplace=True)
df_purchase['purchase'] = df_purchase['purchase'].astype(int)
df_purchase.drop(columns='behavior', inplace=True)
df_play = df_dataUsers.loc[df_dataUsers['behavior'] == 'play'].copy(deep=True)
df_play['play'] = df_play['hours'].apply(lambda x: 1 if x > 0 else 0)
df_play.drop(columns='behavior', inplace=True)
# Clean dataset
df_dataUsers = df_dataUsers[['user_id', 'game_name']].drop_duplicates(subset=['user_id', 'game_name'],
                                                                      keep='last').copy(deep=True)
df_dataUsers = df_dataUsers.merge(df_play, on=['user_id', 'game_name'], how='left')
df_dataUsers = df_dataUsers.merge(df_purchase, on=['user_id', 'game_name'], how='left')
df_dataUsers[['hours', 'play']] = df_dataUsers[['hours', 'play']].fillna(value=0)
df_dataUsers['play'] = df_dataUsers['play'].astype(int)

# Get list of all users
userArray = df_dataUsers['user_id'].unique()

# Get number of users and find 20% of random users for test dataset
userCount = len(userArray)
testUserCount = round(0.2 * userCount)
testUsers = np.array(random.sample(list(userArray), testUserCount))

# Split user in 2 sets and output csv
criteriaTest = df_dataUsers['user_id'].isin(testUsers)
df_trainDataset = df_dataUsers[~criteriaTest].copy(deep=True)
df_testDataset = df_dataUsers[criteriaTest].copy(deep=True)
df_testDataset.to_csv(locationUsersFile.with_name(locationUsersFile.stem + '_test' + locationUsersFile.suffix),
                      index=False)
df_trainDataset.to_csv(locationUsersFile.with_name(locationUsersFile.stem + '_train' + locationUsersFile.suffix),
                       index=False)

# Heat-map (not sure heat-map is appropriate)
nb_games = 20
hm_user_count = df_play.groupby('game_name')['user_id'].agg('count').sort_values(ascending=False)
hm_hours_played = df_play.groupby('game_name')['hours'].agg(np.sum).sort_values(ascending=False)
df_heat_map = pd.DataFrame(
    {'game_name': hm_user_count.index, 'user_count': hm_user_count.values,
     'hours_played': hm_hours_played.values})[0:nb_games]
heat_map_data = df_heat_map.pivot('user_count', 'game_name', 'hours_played')
# sns.heatmap(heat_map_data,annot=True,cmap='RdYlGn',linewidths=0.4)
# plt.show()
