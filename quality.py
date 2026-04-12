import numpy as np 
import pandas as pd
import sqlite3
import json

def quality():
    conn = sqlite3.connect('database.db')
    df = pd.read_sql_query('SELECT * FROM metaparameters', conn)
    quality = {}
    for idx, row in df.iterrows():
        quality['batch ' + str(row['batch_id'])] = {}
        quality['batch ' + str(row['batch_id'])]['quality'] = str(row['complete_records'] / row['total_record'] * 100)
        quality['batch ' + str(row['batch_id'])]['complete_records'] = str(row['complete_records'])
        quality['batch ' + str(row['batch_id'])]['total_records'] = str(row['total_record'])
        minimum = np.inf
        col_min = None
        col_max = None
        maximum = -np.inf
        for col_name, value in row.items():
            if col_name == 'batch_id' or col_name == 'complete_records' or col_name=='total_record':
                continue
            else:
                if value < minimum:
                    col_min = col_name
                    minimum = value
                if value > maximum:
                    col_max = col_name
                    maximum = value
        quality['batch ' + str(row['batch_id'])]['min_missings'] = col_min
        quality['batch ' + str(row['batch_id'])]['max_missings'] = col_max

    quality['total_quality'] = str(df['complete_records'].sum() / df['total_record'].sum() * 100)
    sums = df[['cnt_miss', 't_real_miss', 't_feel_miss', 'hum_miss', 'wind_speed_miss', 'weather_miss', 'is_holiday_miss', 'is_weekend_miss', 'season_miss']].sum()
    quality['min_missings'] = sums.idxmin()
    quality['max_missings'] = sums.idxmax()


    with open('data_quality.json','w') as f:
        json.dump(quality, f, ensure_ascii=False, indent=4)

    conn.close()
