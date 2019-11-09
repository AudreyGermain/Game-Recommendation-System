import pandas as pd
import numpy as np
import pathlib
from plotnine import *
from plotnine.data import *

locationUsersFile = pathlib.Path(r'data/steam_user_test.csv')
steam = pd.read_csv(locationUsersFile,header=0)
print(steam)
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
played=clean_steam.groupby(['game_name'])
phrs=played.agg({'hours':np.sum})

#Most played game - users
played=played['game_name'].count().reset_index(name="alusers")

#Merge user - hrs
mp=pd.merge(phrs,played,on='game_name')
most=(mp.sort_values(by='alusers',ascending=False)).head(20)

#Histogram
print(most)
print(ggplot(most, aes(x = 'game_name', y = 'alusers', fill = 'hours')) +
      geom_bar(stat = "identity") +
      theme(axis_text_x=element_text(angle=90,hjust=0.5,vjust=1)) +
      labs(title = "Top 20 games with the most users", x = "Game", y = "Number of users"))




