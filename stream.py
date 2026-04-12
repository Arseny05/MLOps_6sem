import pandas as pd
import numpy as np
import sqlite3
import time
import json
from datetime import datetime
from pathlib import Path
import math

class DataLoader:

    def __init__(self, db_path='database.db', csv_path='train_raw.csv', batch_size=1024, flag=False):
        self.db_path = db_path
        self.csv_path = csv_path
        self.batch_size = batch_size
        print('Begin loading csv-file...')
        self.df = pd.read_csv(csv_path)
        self.len = len(self.df)
        self.num_batches = math.ceil(self.len / self.batch_size)
        print(f'Получено представление потока из {self.num_batches} батчей')
        if flag:
            self.create_database()

    def create_database(self):
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute('PRAGMA foreign_keys = ON')

        self.conn.execute('''
                    CREATE TABLE IF NOT EXISTS batches (
                          batch_id INTEGER PRIMARY KEY AUTOINCREMENT,
                          received_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                          record_count INTEGER,
                          first_timestamp TIMESTAMP,
                          last_timestamp TIMESTAMP
                          )
                          ''')
        
        self.conn.execute('''
                    CREATE TABLE IF NOT EXISTS rents (
                          id INTEGER PRIMARY KEY AUTOINCREMENT,
                          batch_id INTEGER,
                          timestamp TIMESTAMP NOT NULL,
                          cnt INTEGER,
                          t_real REAL,
                          t_feel REAL,
                          hum REAL,
                          wind_speed REAL,
                          weather TEXT,
                          is_holiday INTEGER,
                          is_weekend INTEGER,
                          season TEXT,
                          FOREIGN KEY (batch_id) REFERENCES batches(batch_id)) 
                          ''')
        
        self.conn.execute('''
                    CREATE TABLE IF NOT EXISTS metaparameters (
                          batch_id INTEGER PRIMARY KEY,
                          total_record INTEGER,
                          complete_records INTEGER,
                          cnt_miss INTEGER,
                          t_real_miss INTEGER,
                          t_feel_miss INTEGER,
                          hum_miss INTEGER,
                          wind_speed_miss INTEGER,
                          weather_miss INTEGER,
                          is_holiday_miss INTEGER,
                          is_weekend_miss INTEGER,
                          season_miss INTEGER,
                          FOREIGN KEY (batch_id) REFERENCES batches(batch_id) )
                          ''')
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_batch ON rents(batch_id)')
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON rents(timestamp)')
        self.conn.commit()
        print("Databases created!")
    
    def emulate_stream(self, delay_sec=0.5):
        print('Beginning stream emulating...')
        total_processed = 0
        for butch_num in range(self.num_batches):
            start_idx = butch_num * self.batch_size
            end_idx = min(self.len, start_idx + self.batch_size)
            batch_data = self.df.iloc[start_idx:end_idx].copy()
            received_at = datetime.now()
            batch_id = self.save_batch(batch_data, received_at)
            self.analyze_batch_quality(batch_id, batch_data)
            total_processed += len(batch_data)
            print(f'Обработано {total_processed} данных')
            time.sleep(delay_sec)
    
    def save_batch(self, batch_data, received_at):
        first_ts = batch_data['timestamp'].min()
        last_ts = batch_data['timestamp'].max()
        cursor = self.conn.execute('''
                            INSERT INTO batches (received_at, record_count, first_timestamp, last_timestamp)
                                   VALUES (?,?,?,?)
                                   ''', (received_at, len(batch_data), first_ts, last_ts))
        batch_id = cursor.lastrowid
        batch_data_with_id = batch_data.copy()
        batch_data_with_id['batch_id'] = batch_id
        batch_data_with_id.to_sql('rents', self.conn, if_exists='append', index=False)
        self.conn.commit()
        return batch_id
    
    def analyze_batch_quality(self, batch_id, batch_data):
        total = len(batch_data)
        complete = len(batch_data.dropna())
        cnt_miss = int(batch_data['cnt'].isna().sum())
        t_real_miss = int(batch_data['t_real'].isna().sum())
        t_feel_miss = int(batch_data['t_feel'].isna().sum())
        hum_miss = int(batch_data['hum'].isna().sum())
        wind_speed_miss = int(batch_data['wind_speed'].isna().sum())
        weather_miss = int(batch_data['weather'].isna().sum())
        is_holiday_miss = int(batch_data['is_holiday'].isna().sum())
        is_weekend_miss = int(batch_data['is_weekend'].isna().sum())
        season_miss = int(batch_data['season'].isna().sum())
        self.conn.execute('''
                        INSERT INTO metaparameters(
                          batch_id, total_record, complete_records, cnt_miss, t_real_miss, t_feel_miss, hum_miss,
                          wind_speed_miss,weather_miss, is_holiday_miss, is_weekend_miss, season_miss)
                          VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                          ''', (batch_id, total, complete, cnt_miss, t_real_miss, t_feel_miss, hum_miss, wind_speed_miss, weather_miss, is_holiday_miss, is_weekend_miss, season_miss))
        self.conn.commit()

    def close(self):
        self.conn.close()

def get_param(path):
    with open(path, 'r') as f:
        dict = json.load(f)
        return dict

if __name__ == '__main__':
    default = True
    if not default:
        dict = get_param('gather.json')
        loader = DataLoader(db_path=dict['db_path'], csv_path=dict['csv_path'], batch_size=dict['batch_size'])
        loader.emulate_stream(delay_sec=dict['delay_sec'])
        loader.close()
    else:
        loader = DataLoader()
        loader.emulate_stream()
        loader.close()


        

    
        