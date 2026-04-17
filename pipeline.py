import missing
import drift
import encoder
import quality
import stream
import pandas as pd
import json
from missing import MissForest, Misser
from stream import DataLoader, get_param 
from drift import Drift
from encoder import Encoder
from quality import quality


format = input("Enter way to gather params...")
if format == '':
    default = True
else:
    default = False
flag = 'yes'
if not default:
    dict = get_param(format)
    db_path = dict['db_path']
    loader = DataLoader(db_path=dict['db_path'], csv_path=dict['csv_path'], batch_size=dict['batch_size'], flag=flag)
    loader.emulate_stream(delay_sec=dict['delay_sec'])
    loader.close()
else:
    db_path = 'database.db'
    loader = DataLoader(flag=flag)
    loader.emulate_stream()
    loader.close()
quality()
print('Look data quality at data_quality.json')
print('EDA is available at eda.ipynb')
detector = Drift(db_path)
detector.estimate()
cl = Misser(db_path=db_path)
res = cl.impute()
cl.write_data(res)
var_1 = Encoder(db_path=db_path)
var_1.ohe(['weather'], prefix='weather')
var_1.map('season', dct=Encoder.season_code)
var_1.trigonometry('season', 4)
var_1.trigonometry('hour', 24)
var_1.Standart(cols=['t_real', 't_feel', 'hum', 'wind_speed'])
var_1.write('var_1')
var_2 = Encoder(db_path=db_path)
var_2.label('weather')
var_2.map('season', dct=Encoder.season_code)
var_2.trigonometry('season', 4)
var_2.trigonometry('hour', 24)
var_2.MinMax(cols=['t_real', 't_feel', 'hum', 'wind_speed'])
var_2.write('var_2')
print('Database successfully updated!')

