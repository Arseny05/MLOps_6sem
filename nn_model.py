import sklearn
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
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
import time
import threading
import psutil
import matplotlib.pyplot as plt
from tabulate import tabulate
import traceback
import cmd
import copy
import yaml


class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, X_train, y_train):
        self.X = X_train
        self.y = y_train

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        x = self.X.iloc[index] if hasattr(self.X, "iloc") else self.X[index]
        y = self.y.iloc[index] if hasattr(self.y, "iloc") else self.y[index]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class MyModel(nn.Module):
    def __init__(self, lr = 0.003, epoch = 10, in_features = 9, weight_decay=1e-5, dropout=0.4, DEVICE="cpu"):
        super().__init__()
        
        self.in_features = in_features
        self.ep = epoch
        self.model = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(16, momentum=0.2),
            nn.Dropout(dropout),

            nn.Linear(in_features=16, out_features=32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(32, momentum=0.2),
            nn.Dropout(dropout),

            nn.Linear(in_features=32, out_features=32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(32, momentum=0.2),
            nn.Dropout(dropout),

            nn.Linear(in_features=32, out_features=16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(16, momentum=0.2),
            nn.Dropout(dropout),

            nn.Linear(in_features=16, out_features=8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(8, momentum=0.2),
            nn.Dropout(dropout),

            nn.Linear(in_features=8, out_features=1),
        ).to(DEVICE)
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.loss = torch.nn.MSELoss().to(DEVICE)
        self.DEVICE = DEVICE


    def forward(self, x):
        return self.model(x)
    

    def fit(self, train_dl: DataLoader):
        self.train()
        for e in range(self.ep):
            progress_train = tqdm(
                total=len(train_dl),
                desc=f"Epoch {e}",
                leave=False,
            )
            loss_stat = []
            for batch, gt_batch in train_dl:
                batch = batch.to(self.DEVICE)
                gt_batch = gt_batch.to(self.DEVICE)
                self.optim.zero_grad(set_to_none=True)
                preds = self.forward(batch)
                gt_batch = gt_batch.float().view(-1, 1)
                loss = self.loss(preds, gt_batch)
                loss.backward()
                self.optim.step()

                loss_stat.append(loss.detach())
                progress_train.update()
            progress_train.close()
            loss_stat = torch.stack(loss_stat).mean()
            print(
                f"Epoch {e},",
                f"train_loss: {loss_stat.item():.8f}",
            )
        self.eval()
        
    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)
        
    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path, map_location=self.DEVICE), strict=True)