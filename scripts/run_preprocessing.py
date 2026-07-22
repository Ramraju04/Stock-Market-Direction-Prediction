import pandas as pd
import numpy as np
import os

from src.prepare_data import load_data, remove_leakage


# -----------------------
# STRONG TARGET CREATION
# -----------------------
def create_target(df):

    df_list = []

    for stock, stock_df in df.groupby('stock'):

        stock_df = stock_df.copy()

        # Next day return
        raw_return = (stock_df['Open'].shift(-1) - stock_df['Open']) / stock_df['Open']

        # 🔥 REMOVE NOISE (VERY IMPORTANT)
        threshold = 0.005   # 0.2%

        stock_df['target'] = raw_return

        # Direction label
        stock_df['direction'] = np.where(raw_return > threshold, 1,
                                  np.where(raw_return < -threshold, 0, -1))

        # Keep only strong signals
        stock_df = stock_df[stock_df['direction'] != -1]

        df_list.append(stock_df)

    df = pd.concat(df_list).reset_index(drop=True)

    print("After target filtering:", df.shape)
    return df


# -----------------------
# FEATURE ENGINEERING
# -----------------------
def create_features(df):

    df_list = []

    for stock, stock_df in df.groupby('stock'):

        stock_df = stock_df.copy()

        stock_df['hl_range'] = (stock_df['High'] - stock_df['Low']) / stock_df['Open']
        stock_df['candle_body'] = (stock_df['Close'] - stock_df['Open']) / stock_df['Open']
        stock_df['log_return'] = np.log(stock_df['Close'] / stock_df['Open'])

        stock_df['momentum_5'] = stock_df['Close'].pct_change(5)
        stock_df['momentum_10'] = stock_df['Close'].pct_change(10)

        stock_df['SMA_5'] = stock_df['Close'].rolling(5).mean()
        stock_df['SMA_20'] = stock_df['Close'].rolling(20).mean()

        stock_df['return_lag1'] = stock_df['Close'].pct_change(1)
        stock_df['return_lag2'] = stock_df['Close'].pct_change(2)
        stock_df['return_lag3'] = stock_df['Close'].pct_change(3)

        # 🔥 IMPORTANT EXTRA FEATURES
        stock_df['rolling_volatility'] = stock_df['Close'].pct_change().rolling(10).std()
        stock_df['RSI'] = compute_rsi(stock_df['Close'])
        stock_df['volume_change'] = stock_df['Volume'].pct_change()

        df_list.append(stock_df)

    df = pd.concat(df_list).reset_index(drop=True)

    print("Features created")
    return df


# -----------------------
# RSI FUNCTION
# -----------------------
def compute_rsi(series, period=14):
    delta = series.diff()

    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()

    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))

    return rsi


# -----------------------
# CLEAN DATA
# -----------------------
def clean_data(df):
    df = df.dropna().reset_index(drop=True)
    print("After cleaning:", df.shape)
    return df


# -----------------------
# SPLIT DATA
# -----------------------
def split_data(df):

    train_list, val_list, test_list = [], [], []

    for stock, stock_df in df.groupby('stock'):

        stock_df = stock_df.sort_values('Date').reset_index(drop=True)

        n = len(stock_df)

        train_end = int(0.7 * n)
        val_end = int(0.85 * n)

        train_list.append(stock_df.iloc[:train_end])
        val_list.append(stock_df.iloc[train_end:val_end])
        test_list.append(stock_df.iloc[val_end:])

    train = pd.concat(train_list).reset_index(drop=True)
    val = pd.concat(val_list).reset_index(drop=True)
    test = pd.concat(test_list).reset_index(drop=True)

    print("Train:", train.shape)
    print("Val:", val.shape)
    print("Test:", test.shape)

    return train, val, test


# -----------------------
# MAIN PIPELINE
# -----------------------
def main():

    df = load_data(r"data\raw\final_stock_dataset.csv")

    df = remove_leakage(df)

    df = create_target(df)

    df = create_features(df)

    df = clean_data(df)

    train, val, test = split_data(df)

    os.makedirs("data/processed", exist_ok=True)

    train.to_csv("data/processed/train.csv", index=False)
    val.to_csv("data/processed/val.csv", index=False)
    test.to_csv("data/processed/test.csv", index=False)

    print("\n✅ Preprocessing complete")


if __name__ == "__main__":
    main()