import pandas as pd


# -----------------------
# LOAD DATA
# -----------------------
def load_data(path):
    df = pd.read_csv(path)

    print("Original Shape:", df.shape)

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['stock', 'Date']).reset_index(drop=True)

    return df


# -----------------------
# REMOVE LEAKAGE
# -----------------------
def remove_leakage(df):
    leak_cols = ['next_open', 'next_day_return']

    for col in leak_cols:
        if col in df.columns:
            df = df.drop(columns=[col])

    print("Leakage removed")
    return df