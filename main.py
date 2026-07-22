import os
import pandas as pd
import torch
from torch.utils.data import DataLoader

# preprocessing
from scripts.run_preprocessing import main as run_preprocessing

# core modules
from src.dataset import StockDataset
from src.model import AdvancedStockModel
from src.train import train_model, save_model
from src.evaluate import evaluate


# -----------------------
# TRAINING PIPELINE
# -----------------------
def run_training():

    print("\n Loading processed data...\n")

    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")

    # -----------------------
    # FEATURES (FINAL)
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
    # STOCK MAP
    # -----------------------
    all_stocks = pd.concat([train_df['stock'], val_df['stock']]).unique()
    stock_map = {s: i for i, s in enumerate(all_stocks)}

    print("Stocks:", len(stock_map))

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
    print("\n🚀 Training...\n")
    train_model(model, train_loader, val_loader, epochs=20)

    # -----------------------
    # LOAD BEST MODEL
    # -----------------------
    model.load_state_dict(torch.load("models/best_model.pth"))

    # -----------------------
    # EVALUATE
    # -----------------------
    print("\n🔍 Evaluating...\n")
    mse, acc, f_acc, cov, _, _ = evaluate(model, val_loader)

    print("\n📊 FINAL RESULTS")
    print(f"Direction Accuracy: {acc*100:.2f}%")
    print(f"Filtered Accuracy: {f_acc*100:.2f}%")
    print(f"Coverage: {cov:.2f}")

    # -----------------------
    # SAVE MODEL
    # -----------------------
    save_model(model, scalers={}, stock_map=stock_map)

    print("\n✅ PIPELINE COMPLETE")


# -----------------------
# MAIN
# -----------------------
if __name__ == "__main__":

    print("\n🔧 STEP 1: PREPROCESSING")
    run_preprocessing()

    print("\n🔧 STEP 2: TRAINING")
    run_training()