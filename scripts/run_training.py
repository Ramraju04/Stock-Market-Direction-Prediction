import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.dataset import StockDataset
from src.model import AdvancedStockModel
from src.train import train_model, save_model
from src.evaluate import evaluate


# -----------------------
# LOAD DATA
# -----------------------
train_df = pd.read_csv("data/processed/train.csv")
val_df = pd.read_csv("data/processed/val.csv")


# -----------------------
# FEATURES (UPDATED)
# -----------------------
features = [
    'Open','High','Low','Close','Volume','VIX',
    'hl_range','candle_body','log_return',
    'momentum_5','momentum_10',
    'SMA_5','SMA_20',
    'return_lag1','return_lag2','return_lag3',

    # 🔥 IMPORTANT FEATURES
    'RSI','rolling_volatility','volume_change'
]


# -----------------------
# STOCK MAP
# -----------------------
all_stocks = pd.concat([train_df['stock'], val_df['stock']]).unique()
stock_map = {s: i for i, s in enumerate(all_stocks)}


# -----------------------
# DATASET
# -----------------------
train_dataset = StockDataset(train_df, features, stock_map)
val_dataset = StockDataset(val_df, features, stock_map)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


# -----------------------
# MODEL
# -----------------------
model = AdvancedStockModel(
    input_size=len(features),
    hidden_size=64,
    num_stocks=len(stock_map)
)


# -----------------------
# TRAIN
# -----------------------
print("\n🚀 Training started...\n")
train_model(model, train_loader, val_loader, epochs=20)


# -----------------------
# LOAD BEST MODEL
# -----------------------
model.load_state_dict(torch.load("models/best_model.pth"))


# -----------------------
# EVALUATE
# -----------------------
print("\n🔍 Evaluating...")
mse, acc, f_acc, cov, _, _ = evaluate(model, val_loader)

print("\n📊 FINAL:")
print(f"Accuracy: {acc*100:.2f}%")
print(f"Filtered Accuracy: {f_acc*100:.2f}%")
print(f"Coverage: {cov:.2f}")


# -----------------------
# SAVE
# -----------------------
import pickle
scalers = {}  # optional for now

save_model(model, scalers, stock_map)

print("\n✅ DONE")