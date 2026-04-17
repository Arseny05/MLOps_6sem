import sklearn
import torch
from torch.utils.data import DataLoader
import joblib
import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import sqlite3
import pandas as pd
import os
from sklearn.model_selection import KFold
import numpy as np
import json
from itertools import product
import threading
import matplotlib.pyplot as plt
from tabulate import tabulate
import copy
from monitor import Monitor
from nn_model import MyDataSet, MyModel



class ModelOrchestrator:
    abs_counter = 0
    database_path = ""
    ckp_folder = ""
    DEVICE = "cpu"
    MODELS = {}
    DEF_MODEL = "BEST"
    data_dir = ""
    DATASETS = {}
    CUR_DS = 0
    ds_counter = 0
    builders = {
        "nn": MyModel,
        "sgd": sklearn.linear_model.SGDRegressor,
        "tree": sklearn.tree.DecisionTreeRegressor,
        "for": sklearn.ensemble.RandomForestRegressor
    }
    param_grids = {
        "nn": { "lr": [1e-4, 3e-4, 1e-3, 3e-3],
                "epoch": [20, 40, 80],
                "weight_decay": [0.0, 1e-5, 1e-4],
                "dropout": [0.1, 0.2, 0.4]},
        "sgd": {"alpha": [1e-4, 1e-3, 1e-2],
                "penalty": ["l2", "elasticnet"],
                "loss": ["squared_error", "huber"],
                "learning_rate": ["optimal", "invscaling", "adaptive"],
                "eta0": [1e-3, 1e-2],
                "max_iter": [1000, 3000]},
        "tree": {"max_depth": [5, 10, 15, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 5],
                "criterion": ["squared_error", "absolute_error"]},
        "for": {"n_estimators": [100, 200],
                "max_depth": [10, 20, None],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
                "max_features": ["sqrt", 1.0]}
    }

    def __init__(self, config = None):
        self.config = config or {}
        self.database_path = self.config.get("database", {}).get("path", "./models_monitoring.db")
        self.data_dir = self.config.get("paths", {}).get("datasets_dir", "./datasets")
        self.ckp_folder = self.config.get("paths", {}).get("checkpoints_dir", "./checkpoints")
        self.DEVICE = self.config.get("system", {}).get("device", "cpu")
        self.lock = threading.RLock()
        self.param_grids = self.config.get("param_grids", self.param_grids)

        self.MODELS = {}
        self.DATASETS = {}
        self.DEF_MODEL = "BEST"
        self.abs_counter = 0
        self.ds_counter = 0
        self.CUR_DS = 0
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.ckp_folder, exist_ok=True)

        print(self._init_db())
        print(self._load_datasets_from_db())
        print(self._load_models_from_db())
        print("Initialization complete\n")

    def _init_db(self):
        with sqlite3.connect(self.database_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS Datasets_overview (
                    Id INTEGER PRIMARY KEY,
                    Data_path TEXT NOT NULL
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS Models_overview (
                    ID INTEGER PRIMARY KEY,
                    Type TEXT NOT NULL,
                    Best_score REAL,
                    Params_json TEXT NOT NULL
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS Checkpoints (
                    Timestamp TEXT NOT NULL,
                    Model_ID INTEGER NOT NULL,
                    Filename TEXT NOT NULL PRIMARY KEY,
                    Mean_absolute_error REAL NOT NULL,
                    Mean_squared_error REAL NOT NULL,
                    R2_score REAL NOT NULL,
                    Dataset_id INTEGER NOT NULL,
                    FOREIGN KEY (Model_ID) REFERENCES Models_overview(ID),
                    FOREIGN KEY (Dataset_id) REFERENCES Datasets_overview(Id)
                )
            """)
        return "Successfully initialized database\n"
    
    def _load_datasets_from_db(self):
        with sqlite3.connect(self.database_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            rows = conn.execute("""
                SELECT Id, Data_path
                FROM Datasets_overview
                ORDER BY Id
            """).fetchall()
        for ds_id, path in rows:
            if not os.path.exists(path):
                print(f"Warning: dataset file not found for Id={ds_id}: {path}\n")
                continue
            df = pd.read_csv(path)
            if df.shape[1] < 2:
                print(f"Warning: dataset Id={ds_id} has too few columns\n")
                continue
            y = df.iloc[:, -1].copy()
            X = df.iloc[:, :-1].copy()
            self.DATASETS[ds_id] = (X, y)
        if self.DATASETS:
            self.ds_counter = max(self.DATASETS.keys()) + 1
        else:
            self.ds_counter = 0
        return "Successfully initialized datasets\n"
    
    def _load_models_from_db(self):
        with sqlite3.connect(self.database_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            models = conn.execute("""
                SELECT ID, Type, Params_json
                FROM Models_overview
                ORDER BY ID
            """).fetchall()
            checkpoints = conn.execute("""
                SELECT c.Model_ID, c.Filename
                FROM Checkpoints c
                INNER JOIN (
                    SELECT Model_ID, MAX(Timestamp) AS max_ts
                    FROM Checkpoints
                    GROUP BY Model_ID
                ) last_cp
                ON c.Model_ID = last_cp.Model_ID
                AND c.Timestamp = last_cp.max_ts
            """).fetchall()
        checkpoint_map = {model_id: filename for model_id, filename in checkpoints}
        for model_id, model_type, params in models:
            kwargs = json.loads(params)
            model_obj = self._build_model_by_type(model_type, kwargs)
            ckp_name = checkpoint_map.get(model_id)
            ckp_path = os.path.join(self.ckp_folder, ckp_name) if ckp_name else None
            self.MODELS[model_id] = {"type": model_type, "model": model_obj, "ckp_path": ckp_path, "params": kwargs}
        if self.MODELS:
            self.abs_counter = max(self.MODELS.keys()) + 1
        else:
            self.abs_counter = 0
        return "Successfully initialized models\n"

    def _build_model_by_type(self, model_type: str, kwargs: dict[str]):
        if model_type == "nn":
            kwargs["DEVICE"] = self.DEVICE
            return MyModel(**kwargs)
        elif model_type == "sgd":
            return sklearn.linear_model.SGDRegressor(**kwargs)
        elif model_type == "tree":
            return sklearn.tree.DecisionTreeRegressor(**kwargs)
        elif model_type == "for":
            return sklearn.ensemble.RandomForestRegressor(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}\n")

    def _set_def_model_(self, NEW_DEF):
        with self.lock:
            if NEW_DEF != "BEST" and NEW_DEF not in list(self.MODELS.keys()):
                raise IndexError("Invalid model type!")
            self.DEF_MODEL = "BEST" if NEW_DEF == "BEST" else int(NEW_DEF)
        return f"Successfully set new default model for {NEW_DEF}\n"
    
    def _set_def_dataset_(self, NEW_DEF):
        with self.lock:
            NEW_DEF = int(NEW_DEF)
            if NEW_DEF not in list(self.DATASETS.keys()):
                raise IndexError("Invalid Dataset index!")
            self.CUR_DS = NEW_DEF
        return f"Successfully set current dataset to {NEW_DEF}\n"

    def get_model_overview(self, index):
        with self.lock:
            with sqlite3.connect(self.database_path) as con:
                cur = con.cursor()
                cur.execute('SELECT Type, Best_score, Params_json FROM Models_overview WHERE ID = ?', (index, ))
                data = cur.fetchone()
        if data is None: raise ValueError(f"Model with ID = {index} doesn't exist!\n")
        typ, sc, par = data
        print(f"Model ID: {index}\nModel Type: {typ}\nModel Best Score (MSE): {sc}\nModel params: {json.loads(par)}\n")

    def get_dataset_overview(self, index):
        with self.lock:
            if not index in list(self.DATASETS.keys()): raise IndexError(f"Dataset with ID = {index} doesn't exist!\n")
            X = self.DATASETS[index][0]
        print(f"Dataset ID: {index}\nDataset Shape: {X.shape}\nDataset Head:\n")
        print(tabulate(X.head(), headers="keys", tablefmt="grid", showindex=True))

    def get_data_from_file(self, data_path, table=None):
        extension = data_path.split(".")[-1].strip()
        if extension == "csv":
            df = pd.read_csv(data_path)
        elif extension == "json":
            df = pd.read_json(data_path)
        elif extension == "db":
            if not table:
                raise ValueError("Table value is empty!\n")
            else:
                con = sqlite3.connect(data_path)
                df = pd.read_sql_query(f'SELECT * FROM "{table}"', con)
                con.close()
        else:
            raise ValueError("This extenstion has not been supported yet\n")
        y, X = df[df.columns[-1]], df.drop(columns=[df.columns[-1]])
        return X, y

    def register_dataset(self, data_path, table=None, index=None):
        with self.lock:
            X, y = self.get_data_from_file(data_path, table)
            if index is not None:
                if index in list(self.DATASETS.keys()):
                    raise IndexError("Index already exists!\n")
                else:
                    self.DATASETS[index] = (X, y)
                    tmp = X.copy()
                    tmp["y"] = y
                    name = os.path.join(self.data_dir, f"Dataset_{index}.csv")
                    tmp.to_csv(name, index=False)
                    connection = sqlite3.connect(self.database_path)
                    cur = connection.cursor()
                    cur.execute('''INSERT INTO Datasets_overview (Id, Data_path) VALUES (?, ?)''', (index, name))
                    connection.commit()
                    connection.close()
                    return f"Successfully registered dataset with ID = {index}\n"
            else:
                while self.ds_counter in list(self.DATASETS.keys()): self.ds_counter += 1
                self.DATASETS[self.ds_counter] = (X, y)
                tmp = X.copy()
                tmp["y"] = y
                name = os.path.join(self.data_dir, f"Dataset_{self.ds_counter}.csv")
                tmp.to_csv(name, index=False)
                connection = sqlite3.connect(self.database_path)
                cur = connection.cursor()
                cur.execute('''INSERT INTO Datasets_overview (Id, Data_path) VALUES (?, ?)''', (self.ds_counter, name))
                connection.commit()
                connection.close()
                self.ds_counter += 1
                return f"Successfully registered dataset with ID = {self.ds_counter - 1}\n"

    def append_to_dataset(self, index, data_path, table=None):
        with self.lock:
            if not index in list(self.DATASETS.keys()):
                raise IndexError("Invalid index!\n")
            X, y = self.get_data_from_file(data_path, table)
            with sqlite3.connect(self.database_path) as con:
                cur = con.cursor()
                cur.execute('SELECT Data_path FROM Datasets_overview WHERE Id = ?', (index, ))
                data = cur.fetchone()
                if data is None: raise ValueError(f"No data found for Dataset with ID = {index}\n")
                data = data[0]
                df = pd.read_csv(data)
                if df.shape[1] != X.shape[1] + 1: raise ValueError("Cannot append DataFrames with incorrect shapes!\n")
                tmp = X.copy()
                tmp["y"] = y
                if set(df.columns) != set(tmp.columns): raise ValueError("Cannot append DataFrames with mismatching columns!\n")
                res = pd.concat([df, tmp], ignore_index=True)
                res.to_csv(data, index=False)
                self.DATASETS[index] = (pd.concat([self.DATASETS[index][0], X], ignore_index=True), pd.concat([self.DATASETS[index][1], y], ignore_index=True))
            return f"Successfully updated dataset with ID = {index}"


    def dataset_deletion(self, index):
        with self.lock:
            if not index in list(self.DATASETS.keys()): raise IndexError("Index doesn't exist!\n")
            con = sqlite3.connect(self.database_path)
            cur = con.cursor()
            cur.execute('''DELETE FROM Checkpoints WHERE Dataset_id = ?''', (index, ))
            cur.execute('''DELETE FROM Datasets_overview WHERE Id = ?''', (index, ))
            con.commit()
            con.close()
            del self.DATASETS[index]
        return f"Successfully deleted dataset with ID = {index}\n"

    def model_creation(self, model: str, index=None, kwargs: dict[str] = None):
        with self.lock:
            if kwargs is None: kwargs = {}
            if index is not None:
                if index in list(self.MODELS.keys()):
                    raise IndexError("Index already exists!\n")
            else:
                while self.abs_counter in list(self.MODELS.keys()): self.abs_counter += 1
                index = self.abs_counter
            connection = sqlite3.connect(self.database_path)
            cur = connection.cursor()
            if model == "nn":
                default_params = self.config.get("models", {}).get("nn", {})
                params_list = {
                    "lr": default_params.get("lr", 0.003),
                    "epoch": default_params.get("epoch", 10),
                    "in_features": kwargs.get("in_features", 9)
                }
                for param in params_list.keys():
                    if not param in list(kwargs.keys()):
                        kwargs[param] = params_list[param]
                kwargs.setdefault("DEVICE", self.DEVICE)
                self.MODELS[index] = {
                    "type": "nn",
                    "model": MyModel(**kwargs),
                    "ckp_path": None,
                    "params": kwargs
                }
                cur.execute('''INSERT INTO Models_overview (ID, Type, Params_json) VALUES (?, ?, ?)''', (index, self.MODELS[index]["type"], json.dumps(kwargs)))
            elif model == "sgd":
                # params_list = {"alpha": 0.001, "max_iter": 1000, "learning_rate": "invscaling", "loss": "squared_error", "penalty": "l2"}
                default_params = self.config.get("models", {}).get("sgd", {})
                params_list = {
                    "alpha": default_params.get("alpha", 0.001),
                    "max_iter": default_params.get("max_iter", 1000),
                    "learning_rate": default_params.get("learning_rate", "invscaling"),
                    "loss": default_params.get("loss", "squared_error"),
                    "penalty": default_params.get("penalty", "l2")
                }
                for param in params_list.keys():
                    if not param in list(kwargs.keys()):
                        kwargs[param] = params_list[param]
                self.MODELS[index] = {
                    "type": "sgd",
                    "model": sklearn.linear_model.SGDRegressor(**kwargs),
                    "ckp_path": None,
                    "params": kwargs
                }
                cur.execute('''INSERT INTO Models_overview (ID, Type, Params_json) VALUES (?, ?, ?)''', (index, self.MODELS[index]["type"], json.dumps(kwargs)))
            elif model == "tree": ##Заменить на ARIMA после Beta
                # params_list = {"criterion": "squared_error", "splitter": "best", "max_depth": 10}
                default_params = self.config.get("models", {}).get("tree", {})
                params_list = {
                    "criterion": default_params.get("criterion", "squared_error"),
                    "splitter": default_params.get("splitter", "best"),
                    "max_depth": default_params.get("max_depth", 10)
                }
                for param in params_list.keys():
                    if not param in list(kwargs.keys()):
                        kwargs[param] = params_list[param]
                self.MODELS[index] = {
                    "type": "tree",
                    "model": sklearn.tree.DecisionTreeRegressor(**kwargs),
                    "ckp_path": None,
                    "params": kwargs
                }
                cur.execute('''INSERT INTO Models_overview (ID, Type, Params_json) VALUES (?, ?, ?)''', (index, self.MODELS[index]["type"], json.dumps(kwargs)))
            elif model == "for":
                # params_list = {"criterion": "squared_error", "n_estimators": 50, "max_depth": 10}
                default_params = self.config.get("models", {}).get("for", {})
                params_list = {
                    "criterion": default_params.get("criterion", "squared_error"),
                    "n_estimators": default_params.get("n_estimators", 50),
                    "max_depth": default_params.get("max_depth", 10)
                }
                for param in params_list.keys():
                    if not param in list(kwargs.keys()):
                        kwargs[param] = params_list[param]
                self.MODELS[index] = {
                    "type": "for",
                    "model": sklearn.ensemble.RandomForestRegressor(**kwargs),
                    "ckp_path": None,
                    "params": kwargs
                }
                cur.execute('''INSERT INTO Models_overview (ID, Type, Params_json) VALUES (?, ?, ?)''', (index, self.MODELS[index]["type"], json.dumps(kwargs)))
            else:
                connection.close()
                raise ValueError("Unknown model type!")
            connection.commit()
            connection.close()
        return f"Successfully registered model with ID = {index}\n"

    def model_deletion(self, index):
        with self.lock:
            if not index in list(self.MODELS.keys()): raise IndexError("Index doesn't exist!\n")
            con = sqlite3.connect(self.database_path)
            cur = con.cursor()
            cur.execute('''DELETE FROM Checkpoints WHERE Model_ID = ?''', (index, ))
            cur.execute('''DELETE FROM Models_overview WHERE ID = ?''', (index, ))
            con.commit()
            con.close()
            del self.MODELS[index]
        return f"Successfully deleted model with ID = {index}\n"

    def _get_model_info(self, index):
        with self.lock:
            if not int(index) in list(self.MODELS.keys()):
                raise IndexError("Invalid index\n")
            with sqlite3.connect(self.database_path) as con:
                df = pd.read_sql_query(f'SELECT * FROM Checkpoints WHERE Model_ID = ?', con, params=(int(index), ))
        return df

    def check_for_drift(self, index):
        data = self._get_model_info(index)
        tmp = data.copy()
        tmp["Timestamp"] = pd.to_datetime(tmp["Timestamp"])
        print("Detecting drift for datasets...")
        for d_id, group in tmp.groupby("Dataset_id"):
            group = group.sort_values("Timestamp").reset_index(drop=True)
            mse = group["Mean_squared_error"].astype(float)
            if len(group) < 6:
                print(f"ID: {d_id}: Impossible to detect with this few rows\n")
            else:
                last_mse = mse.iloc[-1]
                best_mse = mse.min()
                mean = mse.iloc[-6:-1].mean()
                drift_index = 0
                if last_mse > 1.5 * best_mse: drift_index += 1
                if last_mse > 1.2 * mean: drift_index += 1
                if drift_index == 2:
                    print(f"Dataset ID: {d_id}: High chance of model or data drift, consider inspecting dataset\n")
                elif drift_index == 1:
                    print(f"Dataset ID: {d_id}: Small chance of model or data drift, keep observing\n")
                else:
                    print(f"Dataset ID: {d_id}: No chance of model or data drift :)\n")
        return "Diagnostics completed\n"
    
    def print_model_info(self, index):
        df = self._get_model_info(index).sort_values("Timestamp").reset_index(drop=True)
        print(f"Model with ID = {index}:\n")
        print(f"Model type: {self.MODELS[index]['type']}\n")
        print(f"Model hyperparameters: {self.MODELS[index]['params']}\n")
        print(f"Current checkpoint path: {self.MODELS[index]['ckp_path']}\n")
        print(f"Checkpoints number: {df.shape[0]}\n")
        print(tabulate(df, headers="keys", tablefmt="grid", showindex=True))

    def select_model_version(self, index, version):
        with self.lock:
            df = self._get_model_info(index).sort_values("Timestamp").reset_index(drop=True)
            if not 0 <= version < len(df):
                raise IndexError("Invalid index!\n")
            self.MODELS[index]["ckp_path"] = os.path.join(self.ckp_folder, df.iloc[version]["Filename"])
            return f"Checkpoint {version} has been chosen for a model with ID {index}\n"

    def fit_model(self, index, X_train, y_train, anew=False, save_metrics=None):
        if index == "BEST": index = self.return_best()
        sample_interval = self.config.get("monitor", {}).get("sample_interval", 0.1)
        with self.lock: monitor = Monitor(device=self.DEVICE if self.MODELS[index]["type"] == "nn" else "cpu", sample_interval=sample_interval)
        monitor.start()
        try:
            if self.MODELS[index]["type"] == "nn":
                with self.lock:
                    model = self.MODELS[index]["model"]
                    if self.MODELS[index]["ckp_path"] != None and not anew:
                        model.load_checkpoint(self.MODELS[index]["ckp_path"])
                ds_train = MyDataSet(X_train, y_train)
                batch_size = self.config.get("training", {}).get("batch_size", 128)
                num_workers = self.config.get("training", {}).get("num_workers", 2)
                dl_train = DataLoader(
                    ds_train,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=(self.DEVICE != "cpu"),
                    persistent_workers=(num_workers > 0),
                    drop_last=True,
                )
                model.fit(dl_train)
                with self.lock:
                    self.MODELS[index]["model"] = model
                    filename = str(index) + "_" + f"{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}" + ".pt"
                    model.save_checkpoint(os.path.join(self.ckp_folder, filename))
                    self.MODELS[index]["ckp_path"] = os.path.join(self.ckp_folder, filename)
            elif self.MODELS[index]["type"] in ["sgd", "for", "tree"]:
                with self.lock:
                    model = self.MODELS[index]["model"]
                    if self.MODELS[index]["ckp_path"] != None and not anew:
                        model = joblib.load(self.MODELS[index]["ckp_path"])
                model.fit(X_train, y_train)
                with self.lock:
                    self.MODELS[index]["model"] = model
                    filename = str(index) + "_" + f"{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}" + ".pkl"
                    joblib.dump(model, os.path.join(self.ckp_folder, filename))
                    self.MODELS[index]["ckp_path"] = os.path.join(self.ckp_folder, filename)
            else:
                raise TypeError("Invalid model type!\n")
        finally:
            monitor.stop()
            
        stats = monitor.summary()
        print("Время обучения, сек: ", stats["train_time_sec"], "\n")
        if self.MODELS[index]["type"] == "nn":
            print("\nПик GPU, Мб: ", stats["peak_gpu_mb"], "\n")
        else:
            print("\nВремя на CPU, сек: ", stats["cpu_time_sec"], "\nПик RAM, Мб: ", stats["peak_ram_mb"], "\n")
        if save_metrics:
            if self.MODELS[index]["type"] == "nn":
                path = self.save_graphics(index, stats["ram_history"], monitor.sample_interval, a2=stats["gpu_history"])
                print(f"Graph for RAM and GPU saved to {path}\n")
            else:
                path = self.save_graphics(index, stats["ram_history"], monitor.sample_interval, cpu=True)
                print(f"Graph for RAM saved to {path}\n")
        return filename

    def predict(self, index, X_test):
        with self.lock:
            if index == "BEST": index = self.return_best()
            if self.MODELS[index]["type"] == "nn":
                model = self.MODELS[index]["model"]
                if X_test.shape[1] != model.in_features: raise ValueError("Error: Net and Dataset aren't in same shape!\n")
                if self.MODELS[index]["ckp_path"] != None:
                    model.load_checkpoint(self.MODELS[index]["ckp_path"])
                model.eval()
                with torch.no_grad():
                    X_test_tensor = torch.as_tensor(X_test, dtype=torch.float32).float().to(self.DEVICE)
                    return model(X_test_tensor).squeeze(-1).cpu().numpy()
            elif self.MODELS[index]["type"] in ["sgd", "for", "tree"]:
                model = self.MODELS[index]["model"]
                if self.MODELS[index]["ckp_path"] != None:
                    model = joblib.load(self.MODELS[index]["ckp_path"])
                return model.predict(X_test)
            else:
                return False

    def evaluate_model(self, index, X, y, params=None, metrics=True):
        sample_interval = self.config.get("monitor", {}).get("sample_interval", 0.1)
        with self.lock: monitor = Monitor(device=self.DEVICE if self.MODELS[index]["type"] == "nn" else "cpu", sample_interval=sample_interval)
        monitor.start()
        try:
            if params is None: params = self.MODELS[index]["params"]
            if self.MODELS[index]["type"] == "nn" and X.shape[1] != params["in_features"]: raise ValueError("Error: Net and Dataset aren't in same shape\n")
            preds = self.cv_nn(self.MODELS[index]["model"], X, y, model_kwargs=params) if self.MODELS[index]["type"] == "nn" else self.cv_sklearn(self.MODELS[index]["type"], X, y, model_kwargs=params)
        finally:
            monitor.stop()
        
        stats = monitor.summary()
        print("Время кросс-валидации, сек: ", stats["train_time_sec"], "\n")
        if metrics:
            if self.MODELS[index]["type"] == "nn":
                print("\nПик GPU при кросс-валидации, Мб: ", stats["peak_gpu_mb"], "\n")
            else:
                print("\nВремя на CPU при кросс-валидации, сек: ", stats["cpu_time_sec"], "\nПик RAM при кросс-валидации, Мб: ", stats["peak_ram_mb"], "\n")
        return (False, False, False) if preds is False else preds

    def cv_nn(self, model, X, y, random_state=42, model_kwargs=None):
        n_splits = self.config.get("cv", {}).get("n_splits", 5)
        batch_size = self.config.get("training", {}).get("batch_size", 128)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        model_kwargs_tmp = {} if model_kwargs is None else copy.deepcopy(model_kwargs)
        model_kwargs_tmp["in_features"] = X.shape[1]
        fold_metrics = []
        sample_interval = self.config.get("monitor", {}).get("sample_interval", 0.1)
        for fold_id, (train_idx, valid_idx) in enumerate(kf.split(X), start=1):
            fold_monitor = Monitor(device=self.DEVICE, sample_interval=sample_interval)
            fold_monitor.start()
            try:
                X_train, X_valid = X[train_idx], X[valid_idx]
                y_train, y_valid = y[train_idx], y[valid_idx]
                train_ds = MyDataSet(X_train, y_train)

                train_dl = DataLoader(
                    train_ds,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=0,
                    pin_memory=(self.DEVICE != "cpu"),
                    drop_last=True,
                )
                model_kwargs_tmp["DEVICE"] = self.DEVICE
                model = MyModel(**model_kwargs_tmp)
                model.fit(train_dl)
                model.eval()
                with torch.no_grad():
                    X_valid_tensor = torch.as_tensor(X_valid, dtype=torch.float32, device=model.DEVICE)
                    preds = model(X_valid_tensor).squeeze(-1).cpu().numpy()
                mae = mean_absolute_error(y_valid, preds)
                mse = mean_squared_error(y_valid, preds)
                r2 = r2_score(y_valid, preds)
                fold_metrics.append({
                    "fold": fold_id,
                    "mae": mae,
                    "mse": mse,
                    "r2": r2,
                })
            finally:
                fold_monitor.stop()
                print(f"Fold_ID: {fold_id}; Время фолда, сек: {fold_monitor.summary()['train_time_sec']}\n")

        metrics_df = pd.DataFrame(fold_metrics)
        return (metrics_df["mae"].mean(), metrics_df["mse"].mean(), metrics_df["r2"].mean())

    def cv_sklearn(self, mtype, X, y, shuffle=True, random_state=42, model_kwargs=None):
        n_splits = self.config.get("cv", {}).get("n_splits", 5)
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        X = pd.DataFrame(X).reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)
        fold_metrics = []
        sample_interval = self.config.get("monitor", {}).get("sample_interval", 0.1)
        for fold_id, (train_idx, valid_idx) in enumerate(kf.split(X), start=1):
            fold_monitor = Monitor(device="cpu", sample_interval=sample_interval)
            fold_monitor.start()
            try:
                X_train = X.iloc[train_idx]
                X_valid = X.iloc[valid_idx]
                y_train = y.iloc[train_idx]
                y_valid = y.iloc[valid_idx]
                fold_model = self.builders[mtype](**model_kwargs)
                fold_model.fit(X_train, y_train)
                preds = fold_model.predict(X_valid)

                mae = mean_absolute_error(y_valid, preds)
                mse = mean_squared_error(y_valid, preds)
                r2 = r2_score(y_valid, preds)

                fold_metrics.append({
                    "fold": fold_id,
                    "mae": mae,
                    "mse": mse,
                    "r2": r2,
                })
            finally:
                fold_monitor.stop()
                print(f"Fold_ID: {fold_id}; Время фолда, сек: {fold_monitor.summary()['train_time_sec']}\n")

        metrics_df = pd.DataFrame(fold_metrics)
        return (metrics_df["mae"].mean(), metrics_df["mse"].mean(), metrics_df["r2"].mean())

    def update_model(self, mode, X, y, logging=True, metrics=True, save_metrics=False):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        if mode == "all":
            print("=" * 60, "\n")
            for index in list(self.MODELS.keys()):
                mae, mse, r2 = 1e10, 1e10, 1e10
                best_params = {}
                for i, el in enumerate(product(*list(self.param_grids[self.MODELS[index]["type"]].values())), start=1):
                    params = dict(zip(list(self.param_grids[self.MODELS[index]["type"]].keys()), el))
                    if self.MODELS[index]["type"] == "nn": params["in_features"] = X_train.shape[1]
                    cur_mae, cur_mse, cur_r2 = self.evaluate_model(index, X_train, y_train, params=params, metrics=metrics)
                    print("Iteration: ", i, "; MSE: ", cur_mse, "\n")
                    if cur_mse < mse:
                        mae, mse, r2 = cur_mae, cur_mse, cur_r2
                        best_params = params
                with self.lock:
                    if self.MODELS[index]['type'] == 'nn': 
                        best_params["DEVICE"] = self.DEVICE
                        best_params["in_features"] = X_train.shape[1]
                    self.MODELS[index]["params"] = best_params
                    with sqlite3.connect(self.database_path) as con:
                        cur = con.cursor()
                        cur.execute('UPDATE Models_overview SET Params_json = ? WHERE ID = ?', (json.dumps(best_params), index))
                        con.commit()
                    self.MODELS[index]["model"] = self.builders[self.MODELS[index]["type"]](**best_params)
                ckp_name = self.fit_model(index, X_train, y_train, save_metrics=save_metrics)
                print("Лучшие параметры: ", best_params, "\nЛучший скор (Best MSE): ", mse, "\tMAE: ", mae, "\tR2_score: ", r2, "\n")
                if not ckp_name: continue
                preds = self.predict(index, X_test)
                mae, mse, r2 = mean_absolute_error(y_test, preds), mean_squared_error(y_test, preds), r2_score(y_test, preds)
                print(f"Метрики на тестовом наборе:\n")
                with self.lock:
                    if logging:
                        new_row = (f"{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}", index, ckp_name, mae, mse, r2, self.CUR_DS)
                        self.update_ckp_db(new_row)
                print("MODEL_ID: ", index, "\tMean_absolute_score: ", mae, "\tMean_squared_score: ", mse, "\tR2_score: ", r2, "\n")
            print("=" * 60, "\n")
        elif mode == "BEST" or type(mode) == int:
            print("=" * 60, "\n")
            index = self.return_best() if mode == "BEST" else mode
            mae, mse, r2 = 1e10, 1e10, 1e10
            best_params = {}
            for i, el in enumerate(product(*list(self.param_grids[self.MODELS[index]["type"]].values())), start=1):
                params = dict(zip(list(self.param_grids[self.MODELS[index]["type"]].keys()), el))
                if self.MODELS[index]["type"] == "nn": params["in_features"] = X_train.shape[1]
                cur_mae, cur_mse, cur_r2 = self.evaluate_model(index, X_train, y_train, params=params, metrics=metrics)
                print("Iteration: ", i, "; MSE: ", cur_mse, "\n")
                if cur_mse < mse:
                    mae, mse, r2 = cur_mae, cur_mse, cur_r2
                    best_params = params
            with self.lock:
                if self.MODELS[index]['type'] == 'nn':
                    best_params["DEVICE"] = self.DEVICE
                    best_params["in_features"] = X_train.shape[1]
                self.MODELS[index]["params"] = best_params
                with sqlite3.connect(self.database_path) as con:
                        cur = con.cursor()
                        cur.execute('UPDATE Models_overview SET Params_json = ? WHERE ID = ?', (json.dumps(best_params), index))
                        con.commit()
                self.MODELS[index]["model"] = self.builders[self.MODELS[index]["type"]](**best_params)
            ckp_name = self.fit_model(index, X_train, y_train, save_metrics=save_metrics)
            print("Лучшие параметры: ", best_params, "\nЛучший скор (Best MSE): ", mse, "\tMAE: ", mae, "\tR2_score: ", r2, "\n")
            if not ckp_name: raise ValueError("Unexpected model type!\n")
            preds = self.predict(index, X_test)
            mae, mse, r2 = mean_absolute_error(y_test, preds), mean_squared_error(y_test, preds), r2_score(y_test, preds)
            print(f"Метрики на тестовом наборе:\n")
            with self.lock:
                if logging:
                    new_row = (f"{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}", index, ckp_name, mae, mse, r2, self.CUR_DS)
                    self.update_ckp_db(new_row)
            print("MODEL_ID: ", index, "\tMean_absolute_score: ", mae, "\tMean_squared_score: ", mse, "\tR2_score: ", r2, "\n")
            print("=" * 60)
        else: raise ValueError("Unexpected evaluating mode!\n")

    def return_best(self):
        con = sqlite3.connect(self.database_path)
        cur = con.cursor()
        cur.execute('''SELECT ID
                    FROM Models_overview
                    WHERE Best_score IS NOT NULL
                    AND Best_score = (
                        SELECT MIN(Best_score)
                        FROM Models_overview
                        WHERE Best_score IS NOT NULL)''')
        data = cur.fetchone()
        con.close()
        if data is None: raise ValueError("Database is empty!\n")
        return int(data[0])

    def update_ckp_db(self, row: tuple):
        connection = sqlite3.connect(self.database_path)
        cursor = connection.cursor()
        cursor.execute('''
            INSERT INTO Checkpoints (Timestamp, Model_ID, Filename, Mean_absolute_error, Mean_squared_error, R2_score, Dataset_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
        
        ''', row)
        connection.commit()
        connection.close()
        return f"Successfully updated information about checkpoint {row[2]}\n"

    def save_model(self, index):
        with self.lock:
            if index == "BEST":
                index = self.return_best()
            if not type(index) == int:
                raise IndexError("Invalid model index!\n")
            model = self.MODELS[index]["model"]
            name = f"{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}_{index}"
            path = os.path.join(self.ckp_folder, name)
            if self.MODELS[index]["type"] == "nn":
                torch.save(model.state_dict(), path + ".pt")
                self.MODELS[index]["ckp_path"] = path + ".pt"
            else:
                joblib.dump(model, path + ".pkl")
                self.MODELS[index]["ckp_path"] = path + ".pkl"
        return f"Successfully saved model {index}. Path: {path + '.pt' if self.MODELS[index]['type'] == 'nn' else path + '.pkl'}\n"

    def save_graphics(self, index, a1, timestamp, cpu=True, a2=None):
        x_ax1 = [i * timestamp for i in range(len(a1))]
        name = f"{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}" + f"_Model{index}.png"
        if a2 is None:
            plt.figure()
            plt.plot(x_ax1, a1)
            plt.title("Выделенная память на CPU от времени выполнения процесса" if cpu else "Выделенная видеопамять от времени выполнения процесса")
            plt.xlabel("Прошедшее время, сек")
            plt.ylabel("Выделено на CPU, Мб" if cpu else "Выделено на GPU, Мб")
            plt.savefig(os.path.join(self.ckp_folder, name))
        else:
            x_ax2 = [i * timestamp for i in range(len(a2))]
            _, axs = plt.subplots(1, 2)
            axs[0].plot(x_ax1, a1)
            axs[1].plot(x_ax2, a2)
            axs[0].set_title("Выделенная память на CPU от времени выполнения процесса")
            axs[1].set_title("Выделенная видеопамять от времени выполнения процесса")
            axs[0].set_xlabel("Прошедшее время, сек")
            axs[1].set_xlabel("Прошедшее время, сек")
            axs[0].set_ylabel("Выделено на CPU, Мб")
            axs[1].set_ylabel("Выделено на GPU, Мб")
            plt.tight_layout()
            plt.savefig(os.path.join(self.ckp_folder, name))
        plt.close()
        return os.path.join(self.ckp_folder, name)
    
    def explain_model(self, index): #Ещё в бете, потом будет SHAP и графики для деревьев
        with self.lock:
            if index == "BEST": index = self.return_best()
            if index not in self.MODELS: raise ValueError(f"Model with ID = {index} not found!\n")
            if self.CUR_DS not in self.DATASETS: raise ValueError("Dataset is not selected!\n")
            model_info = self.MODELS[index]
            model_type = model_info["type"]
            model = model_info["model"]
            X, y = self.DATASETS[self.CUR_DS]
        rand_idx = np.random.randint(0, len(X))
        x_row = X.iloc[[rand_idx]]
        y_true = y.iloc[rand_idx]
        pred = self.predict(index, x_row)[0]
        print("=" * 60)
        print(f"Model ID: {index} | Type: {model_type}")
        print(f"Sample index: {rand_idx}")
        print(f"True value: {y_true}")
        print(f"Prediction: {pred:.5f}")
        print("=" * 60)

        if model_type == "sgd":
            print("Linear model explanation (coefficients):\n")
            coefs = model.coef_
            contributions = []
            for i, col in enumerate(x_row.columns):
                val = x_row.iloc[0, i]
                contrib = val * coefs[i]
                contributions.append((col, val, contrib))
            contributions.sort(key=lambda x: abs(x[2]), reverse=True)
            for col, val, contrib in contributions[:5]: print(f"{col}: value={val:.4f}, contribution={contrib:.4f}")
        elif model_type in ["tree", "for"]:
            print("Feature importance explanation:\n")
            importances = model.feature_importances_
            pairs = list(zip(x_row.columns, importances))
            pairs.sort(key=lambda x: x[1], reverse=True)
            for col, imp in pairs[:5]:
                val = x_row[col].values[0]
                print(f"{col}: value={val:.4f}, importance={imp:.4f}")
        elif model_type == "nn":
            print("NN sensitivity analysis (approx):\n")
            base_pred = pred
            contributions = []
            for col in x_row.columns:
                x_modified = x_row.copy()
                x_modified[col] *= 1.05
                new_pred = self.predict(index, x_modified)[0]
                effect = new_pred - base_pred
                contributions.append((col, x_row[col].values[0], effect))
            contributions.sort(key=lambda x: abs(x[2]), reverse=True)
            for col, val, effect in contributions[:5]:
                print(f"{col}: value={val:.4f}, effect={effect:.4f}")
        else:
            print("Unknown model type, no explanation available")

        print("=" * 60)
        return "Explanation completed\n"
    
    def auto_select_and_predict(self, dataset_id):
        with self.lock:
            if dataset_id not in self.DATASETS:
                raise ValueError(f"Dataset {dataset_id} not found!\n")
            X, y = self.DATASETS[dataset_id]

        n_rows, n_cols = X.shape
        if n_cols > 20:
            priority = ["nn", "for", "sgd", "tree"]
        elif n_rows < 5000:
            priority = ["tree", "for", "sgd", "nn"]
        else:
            priority = ["sgd", "for", "tree", "nn"]
        print(f"Dataset shape: {X.shape}")
        print(f"Model priority: {priority}\n")

        best_model_id = None
        best_score = float("inf")
        for mtype in priority:
            for idx, info in self.MODELS.items():
                if info["type"] != mtype:
                    continue
                try:
                    df = self._get_model_info(idx)
                    if df.empty:
                        continue
                    mse = df["Mean_squared_error"].astype(float).min()
                    if mse < best_score:
                        best_score = mse
                        best_model_id = idx
                except:
                    continue
            if best_model_id is not None:
                break
        if best_model_id is None:
            print("No trained models found, cannot select\n")
            return None
        print(f"Selected model ID: {best_model_id} (type={self.MODELS[best_model_id]['type']})")
        print(f"Best historical MSE: {best_score}\n")
        preds = self.predict(best_model_id, X)
        mae = mean_absolute_error(y, preds)
        mse = mean_squared_error(y, preds)
        r2 = r2_score(y, preds)
        print("Evaluation on dataset:")
        print(f"MAE: {mae}")
        print(f"MSE: {mse}")
        print(f"R2: {r2}\n")
        return {
            "model_id": best_model_id,
            "type": self.MODELS[best_model_id]["type"],
            "mae": mae,
            "mse": mse,
            "r2": r2
        }