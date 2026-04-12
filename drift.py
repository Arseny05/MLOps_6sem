import pandas as pd
import numpy as np
import sqlite3
from scipy.stats import ks_2samp, chi2_contingency
import warnings
warnings.filterwarnings('ignore')


class Drift:

    def __init__(self, db_path='database.db'):
        conn = sqlite3.connect(db_path)
        self.df = pd.read_sql_query('SELECT * FROM rents', conn)
        conn.close()

    def estimate(self, group_size=4):
        self.batches = {}
        for batch_id, batch_data in self.df.groupby('batch_id'):
            batch_data_clean = batch_data.drop(['timestamp', 'id', 'batch_id'], axis=1)
            self.batches[batch_id] = batch_data_clean
            print(f'Батч {batch_id}: {len(batch_data_clean)} строк')
        batch_ids = list(self.batches.keys())
        groups = []
        group_labels = []
        for i in range(0, len(batch_ids), group_size):
            group_batches = batch_ids[i:i+group_size]
            group_data = pd.concat([self.batches[b] for b in group_batches], ignore_index=True)
            groups.append(group_data)
            group_labels.append(group_batches)
    
        for i in range(0,len(groups)-1):
            ref_data = groups[i]
            batch_data = groups[i+1]
            drifted = []
            for col in ref_data.columns:
                try:
                    if pd.api.types.is_numeric_dtype(ref_data[col]):
                        stat, p_val = ks_2samp(ref_data[col].dropna(), batch_data[col].dropna())
                        if p_val < 0.01:
                            drifted.append(col)
                    else:
                        ref_counts = ref_data[col].value_counts()
                        curr_counts = batch_data[col].value_counts()
                        all_cats = ref_counts.index.union(curr_counts.index)
                        ref_arr = [ref_counts.get(cat, 0) for cat in all_cats]
                        curr_arr = [curr_counts.get(cat, 0) for cat in all_cats]
                        stat, p_val, dof, expected = chi2_contingency([ref_arr, curr_arr])
                        if p_val < 0.01:
                            drifted.append(col)
                except Exception as e:
                     print(f"  Ошибка при проверке колонки {col}: {e}")
                
            
            print(f"\nГруппа {group_labels[i]} -> {group_labels[i+1]}:")
            if drifted:
                print(f"  ДРИФТ в: {drifted}")
            else:
                print(f"  Дрифта нет")


if __name__ == '__main__':
    detector = Drift('database.db')
    detector.estimate()

            

