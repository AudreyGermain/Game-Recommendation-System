import pandas as pd
import numpy as np
import csv
from pandas.api.types import CategoricalDtype
from plotnine import *
from plotnine.data import *

steam = pd.read_csv('C:\Python\steam-200k.csv',header=None,names=['user', 'game', 'purchase_play', 'hrs'])

#Split purchase and calculate time
steam['purchase']=1


#Split play
steam['play']=np.where(steam['purchase_play']=='play',1,0)
steam['hrs']=steam['hrs']+steam['play']-1
#Clean dataset
clean_steam=steam.drop_duplicates(subset=['user','game'],keep='last')
clean_steam=clean_steam.drop(columns=['purchase_play'])

#Export to csv
clean_steam.to_csv(r'C:\Python\purchase_play.csv',index=None)

#Most played game - hrs
played=clean_steam.groupby(['game'])
phrs=played.agg({'hrs':np.sum})

#Most played game - users
played=played['game'].count().reset_index(name="alusers")

#Merge user - hrs
mp=pd.merge(phrs,played,on='game')
most=(mp.sort_values(by='alusers',ascending=False)).head(20)

#Histogram
print(most)
print(ggplot(most, aes(x = 'game', y = 'alusers', fill = 'hrs')) + 
    geom_bar(stat = "identity") + 
    theme(axis_text_x=element_text(angle=90,hjust=0.5,vjust=1)) + 
    labs(title = "Top 20 games with the most users", x = "Game", y = "Number of users"))




