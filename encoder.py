import pandas as pd
import numpy as np
import sqlite3
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

class Encoder:
    season_code = {'winter':0, 'spring':1, 'summer':2, 'fall':3}
    day_week_code = {'Sunday':6, 'Monday':0, 'Tuesday':1, 'Wednesday':2, 'Thursday':3, 'Friday':4, 'Saturday':5}

    def __init__(self, db_path='database.db'):
        self.conn = sqlite3.connect(db_path)
        self.df = pd.read_sql_query('SELECT * FROM clean', self.conn)

    def ohe(self, col, prefix):
        self.df = pd.get_dummies(self.df, columns=col, prefix=prefix, dtype=int)
    
    def label(self, col):
        le = LabelEncoder()
        self.df[col] = le.fit_transform(self.df[col])

    def trigonometry(self, col, n):
        cos = str(col) + '_cos'
        sin = str(col) + '_sin'
        self.df[cos] = np.cos(self.df[col]*np.pi*2/n)
        self.df[sin] = np.sin(self.df[col]*2*np.pi/n)
        self.df = self.df.drop([col], axis=1)
    
    def map(self, col, dct):
        self.df[col] = self.df[col].map(dct)

    def __str__(self):
        return str(self.df.head())
    def write(self, name):
        if self.df.isnull().any().any():
            print("Обнаружены NULL значения в колонках:")
            print(self.df.columns[self.df.isnull().any()].tolist())
            print("Количество NULL по колонкам:")
            print(self.df.isnull().sum())
        self.df.to_sql(name, self.conn, if_exists='replace', index=False)
        print(f'{name} table is ready!')
        #self.conn.close()

    def __del__(self):
        self.conn.close()
        print('Кодирование и обработка  переменных завершена!')

    def Standart(self, cols):
        sc = StandardScaler()
        self.df[cols] = sc.fit_transform(self.df[cols])

    def MinMax(self, cols):
        mm = MinMaxScaler()
        self.df[cols] = mm.fit_transform(self.df[cols])

if __name__ == '__main__':
    var_1 = Encoder()
    var_1.ohe(['weather'], prefix='weather')
    var_1.map('season', dct=Encoder.season_code)
    var_1.trigonometry('season', 4)
    var_1.trigonometry('hour', 24)
    var_1.Standart(cols=['t_real', 't_feel', 'hum', 'wind_speed'])
    var_1.write('var_1')
    
    var_2 = Encoder()
    var_2.label('weather')
    var_2.map('season', dct=Encoder.season_code)
    var_2.trigonometry('season', 4)
    var_2.trigonometry('hour', 24)
    var_2.MinMax(cols=['t_real', 't_feel', 'hum', 'wind_speed'])
    var_2.write('var_2')


