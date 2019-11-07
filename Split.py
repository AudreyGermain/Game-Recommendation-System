import pandas as pd
import numpy as np
import csv
steam = pd.read_csv('C:\Python\steam-200k.csv',header=None,names=['user', 'game', 'purchase_play', 'hrs'])

#Split purchase and calculate time
steam['purchase']=1
steam['hrs']=steam['hrs']-steam['purchase']

#Split play
steam['play']=np.where(steam['purchase_play']=='play',1,0)

#Clean dataset
clean_steam=steam.drop_duplicates(subset=['user','game'],keep='last')
clean_steam=clean_steam.drop(columns=['purchase_play'])

#Export to csv
clean_steam.to_csv(r'C:\Python\purchase_play.csv',index=None)


