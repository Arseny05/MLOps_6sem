import pandas as pd
import numpy as np
import sqlite3

'''df = pd.read_csv('london_merged.csv')


weather_dict = {
    1: 'clear',
    2: 'scattered clouds',
    3: 'broken clouds',
    4: 'cloudy',
    7: 'rain',
    10: 'rain with thunderstorm',
    26: 'snowfall',
    94: 'fog'
}

season_dict = {
    0: 'spring',
    1: 'summer',
    2: 'fall',
    3: 'winter'
}

df['weather_code'] = df['weather_code'].map(weather_dict)
df['season'] = df['season'].map(season_dict)

df.to_csv('table1.csv')'''

"""df = pd.read_csv('table1.csv')
df_tr = df.iloc[:15000]
df_vl = df.iloc[15000:]
df_tr.to_csv('train_data.csv')
df_vl.to_csv('test_data.csv')"""

df = pd.read_csv('train_data.csv')
df_rest = df.iloc[:,1:]
np.random.seed(42)
mask = np.random.random(df_rest.shape)<= 0.05
df_rest = df_rest.mask(mask)
df_raw = pd.concat([df.iloc[:,:1],df_rest],axis=1)
df_raw.to_csv('train_raw.csv')