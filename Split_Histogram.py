import pandas as pd
import numpy as np
import pathlib
from plotnine import *
from plotnine.data import *
from pandas.api.types import CategoricalDtype
import seaborn as sns
import matplotlib.pyplot as plt


locationUsersFile = pathlib.Path(r'data/steam-200k.csv')
steam = pd.read_csv(locationUsersFile, header=None, usecols=[0, 1, 2, 3],
                    names=["user_id", "game_name", "behavior", "hours"])
#print(steam)  for check

#Split purchase and calculate time
steam['purchase']=1

#Split play
steam['play']=np.where(steam['behavior']=='play',1,0)
steam['hours']=steam['hours']+steam['play']-1

#Clean dataset
clean_steam=steam.drop_duplicates(subset=['user_id','game_name'],keep='last')
clean_steam=clean_steam.drop(columns=['behavior'])

#Export to csv
clean_steam.to_csv(r'data/purchase_play.csv',index=None)

#Most played game - hrs
index=clean_steam[clean_steam['play']==0].index
clean_steam.drop(index,inplace=True)  #drop the columns that the players only purchased

played=clean_steam.groupby(['game_name'])
phrs=played.agg({'hours':np.sum})
phrs=phrs.round(1)


#Most played game - users

played=played['game_name'].count().reset_index(name="alusers")


#Merge user - hrs
mp=pd.merge(phrs,played,on='game_name')
most=(mp.sort_values(by='alusers',ascending=False)).head(20)
Eb=pd.Categorical(most,ordered=True)

#Histogram
#print(most) for check
sns.set(style="darkgrid")
sns.barplot(x='alusers',y='game_name',hue='hours',alpha=0.9,data=most,palette='BuGn',dodge=False)
plt.title('Top 20 games with the most users')
plt.ylabel('Game', fontsize=12)
plt.xlabel('Top 20 games with the most users', fontsize=12)
plt.show()

#print(ggplot(Eb, aes(x = 'game_name', y = 'alusers', fill = 'hours')) +
#      geom_bar(stat = "identity") +
#     theme(axis_text_x=element_text(angle=90,hjust=0.5,vjust=1)) +
#     labs(title = "Top 20 games with the most users", x = "Game", y = "Number of users"))





