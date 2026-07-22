import torch
from torch.utils.data import Dataset
import numpy as np


class StockDataset(Dataset):
    def __init__(self, df, features, stock_map, seq_len=20):

        self.features = features
        self.seq_len = seq_len
        self.stock_map = stock_map

        self.X_seq = []
        self.X_flat = []
        self.y = []
        self.s = []

        self.create_sequences(df)

    # -----------------------
    # CREATE SEQUENCES (STOCK-WISE)
    # -----------------------
    def create_sequences(self, df):

        for stock_name, stock_df in df.groupby('stock'):

            stock_df = stock_df.sort_values('Date').reset_index(drop=True)

            stock_id = self.stock_map.get(stock_name, 0)

            values = stock_df[self.features].values
            targets = stock_df['target'].values

            for i in range(len(stock_df) - self.seq_len):

                seq = values[i:i+self.seq_len]
                flat = values[i+self.seq_len-1]

                target = targets[i+self.seq_len]

                self.X_seq.append(seq)
                self.X_flat.append(flat)
                self.y.append(target)
                self.s.append(stock_id)

    # -----------------------
    def __len__(self):
        return len(self.y)

    # -----------------------
    def __getitem__(self, idx):
        return (
            torch.tensor(self.X_seq[idx], dtype=torch.float32),
            torch.tensor(self.X_flat[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32).unsqueeze(0),
            torch.tensor(self.s[idx], dtype=torch.long)
        )