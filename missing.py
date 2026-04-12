import pandas as pd
import numpy as np
import sqlite3
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import json

class MissForest:
   def __init__(self, n_iterations = 10, n_estimators = 100, random_state = 42, categories = []):
      self.iterations = n_iterations
      self.estimators = n_estimators
      self.random_state = random_state
      self.categories = categories
      self.encoders = {}
      self.original_missing = None
   
   def fit_transform(self, X):
      if isinstance(X,pd.DataFrame):
         self.feature_names = X.columns.tolist()
         X = X.copy()
      else:
         X = pd.DataFrame(X)
      X_encoded = X.copy()
      self.original_missing = X.isna().copy()
      for col in self.categories:
         le  = LabelEncoder()
         all_vals = set()
         for val in X.iloc[:, col].dropna():
            all_vals.add(val)
         le.fit(list(all_vals))
         X_encoded.iloc[:,col] = X.iloc[:,col].map(lambda x: le.transform([x])[0] if pd.notna(x) else np.nan)
         self.encoders[col] = le

      X_imp = X_encoded.copy()
      n_features = X_imp.shape[1]
      for col in range(n_features):
         mask = X_imp.iloc[:,col].isna()
         if X_imp.iloc[:,col].isna().any():
            if col in self.categories:
               mode_val = X_imp.iloc[:,col].mode()
               if not mode_val.empty:
                  fill_value = int(mode_val.iloc[0])
                  #X_imp.iloc[:,col].fillna(fill_value, inplace = True)
                  X_imp.iloc[mask,col] = fill_value
            else:
               median_val = X_imp.iloc[:,col].median()
               #X_imp.iloc[:, col].fillna(median_val, inplace=True)
               mask = X_imp.iloc[:,col].isna()
               X_imp.iloc[mask,col] = median_val
      
      for iteration in range(self.iterations):
         X_old = X_imp.copy()

         for col in range(n_features):
            missing_mask = self.original_missing.iloc[:, col]
            if not missing_mask.any():
               continue
            
            other_cols = [c for c in range(n_features) if c!=col]

            train_X = X_imp.iloc[~missing_mask.values, other_cols]
            train_y = X_imp.iloc[~missing_mask.values, col]

            test_X = X_imp.iloc[missing_mask.values, other_cols]
            
            if col in self.categories:
               train_y = train_y.astype(int)
               model = RandomForestClassifier(n_estimators = self.estimators, random_state = self.random_state, n_jobs = -1)
            else:
               model = RandomForestRegressor(n_estimators = self.estimators, random_state = self.random_state, n_jobs = -1)
            
            model.fit(train_X,train_y)
            predictions = model.predict(test_X)
            X_imp.iloc[missing_mask.values, col] = predictions
         #if np.allclose(X_imp.values, X_old.values,equal_nan = True):
           # break
         
      result = X_imp.copy()
      for col, le in self.encoders.items():
         col_data = result.iloc[:, col]
         col_data = pd.to_numeric(col_data, errors = 'coerce').fillna(0).astype(int)
         result.iloc[:,col] = le.inverse_transform(col_data)

      if self.feature_names is not None:
         result = pd.DataFrame(result,columns=self.feature_names)
      return result
    
class Misser:
    def __init__(self, db_path='database.db'):
        self.conn = sqlite3.connect(db_path)
        self.df = pd.read_sql_query('SELECT * FROM rents', self.conn)
    
    @staticmethod
    def diagnosis(json_path='data_quality.json'):
        with open(json_path, 'r') as f:
            data = json.load(f)
        print('Quality data:')
        print(data)

    def get_data(self):
        return self.df.copy()
    
    def impute(self, categorical=[5,8]):
       data = self.get_data()
       data = data.drop(['id', 'batch_id'], axis=1)
       data['hour'] = pd.to_datetime(data['timestamp']).dt.hour
       data['month'] = pd.to_datetime(data['timestamp']).dt.month
       data = data.drop(['timestamp'], axis=1)
       data_correct = MissForest(categories=categorical).fit_transform(data)
       return data_correct
    
    def write_data(self, data):
       data.to_sql('clean', self.conn, if_exists='replace', index=False)
       print('Clean data are ready!')
       return
    
    def __del__(self):
       if hasattr(self, 'conn') and self.conn:
            self.conn.close()
            print(f"Соединение с БД закрыто")

if __name__ == '__main__':
   cl = Misser()
   res = cl.impute()
   cl.write_data(res)