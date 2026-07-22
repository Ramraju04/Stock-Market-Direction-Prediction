import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.dataset import StockDataset
from src.model import AdvancedStockModel
from src.evaluate import evaluate


# -----------------------
# LOAD TEST DATA
# -----------------------
test_df = pd.read_csv("data/processed/test.csv")


# -----------------------
# FEATURES (SAME AS TRAIN)
# -----------------------
features = [
    'Open','High','Low','Close','Volume','VIX',
    'hl_range','candle_body','log_return',
    'momentum_5','momentum_10',
    'SMA_5','SMA_20',
    'return_lag1','return_lag2','return_lag3',
    'RSI','rolling_volatility','volume_change'
]


# -----------------------
# LOAD STOCK MAP
# -----------------------
import pickle

with open("models/stock_map.pkl", "rb") as f:
    stock_map = pickle.load(f)


# -----------------------
# DATASET
# -----------------------
test_dataset = StockDataset(test_df, features, stock_map)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# -----------------------
# LOAD MODEL
# -----------------------
model = AdvancedStockModel(
    input_size=len(features),
    hidden_size=64,
    num_stocks=len(stock_map)
)

model.load_state_dict(torch.load("models/best_model.pth"))


# -----------------------
# EVALUATE
# -----------------------
print("\n🔍 Testing on unseen data...\n")

mse, acc, f_acc, cov, preds, targets = evaluate(model, test_loader)

print("\n📊 TEST RESULTS")
print(f"Direction Accuracy: {acc*100:.2f}%")
print(f"Filtered Accuracy: {f_acc*100:.2f}%")
print(f"Coverage: {cov:.2f}")