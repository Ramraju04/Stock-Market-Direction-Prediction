import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score


# -----------------------
# LOAD DATA
# -----------------------
train_df = pd.read_csv("data/processed/train.csv")
val_df = pd.read_csv("data/processed/val.csv")
test_df = pd.read_csv("data/processed/test.csv")


# -----------------------
# FEATURES
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
# PREPARE DATA
# -----------------------
def prepare(df):
    X = df[features].values
    y = (df['target'] > 0).astype(int).values
    return X, y


X_train, y_train = prepare(train_df)
X_val, y_val = prepare(val_df)
X_test, y_test = prepare(test_df)


# -----------------------
# MODEL
# -----------------------
model = lgb.LGBMClassifier(
    n_estimators=800,
    learning_rate=0.03,
    max_depth=-1,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)


# -----------------------
# TRAIN
# -----------------------
print("\n🚀 Training LightGBM...\n")

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='logloss'
)


# -----------------------
# EVALUATION FUNCTION
# -----------------------
def evaluate(X, y, name):

    probs = model.predict_proba(X)[:, 1]

    # normal prediction
    preds = (probs > 0.5).astype(int)
    acc = accuracy_score(y, preds)

    # 🔥 confidence filtering
    confidence = np.abs(probs - 0.5)
    mask = confidence > 0.1

    if mask.sum() > 0:
        f_acc = accuracy_score(y[mask], preds[mask])
    else:
        f_acc = 0

    coverage = mask.mean()

    print(f"\n📊 {name} RESULTS")
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"Filtered Accuracy: {f_acc*100:.2f}%")
    print(f"Coverage: {coverage:.2f}")
import joblib
import os

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/lightgbm_model.pkl")

print("✅ LightGBM model saved")
# -----------------------
# RESULTS
# -----------------------
evaluate(X_train, y_train, "TRAIN")
evaluate(X_val, y_val, "VALIDATION")
evaluate(X_test, y_test, "TEST")