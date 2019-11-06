import pandas as pd
import numpy as np
import csv
steam = pd.read_csv('D:\steamk.csv', header = None,names=['user', 'game', 'purchase_play', 'hrs'])

#clean own/play

#split purchase
steam['purchase']=1

#split play
steam['play']=np.where(steam['purchase_play']=='play',1,0)



print(steam)
