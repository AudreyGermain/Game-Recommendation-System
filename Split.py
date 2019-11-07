import pandas as pd
import numpy as np
import csv
steam = pd.read_csv('C:\Python\steam.csv',header=None,names=['user', 'game', 'purchase_play', 'hrs'])

#split purchase and calculate time
steam['purchase']=1
steam['hrs']=steam['hrs']-steam['purchase']

#split play
steam['play']=np.where(steam['purchase_play']=='play',1,0)

#clean own/play
clean_steam=steam.drop_duplicates(subset=['user','game'],keep='last')

print(clean_steam)

