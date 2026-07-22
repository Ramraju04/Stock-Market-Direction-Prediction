import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


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
    X = df[features]
    y = (df['target'] > 0).astype(int)
    return X, y


X_train, y_train = prepare(train_df)
X_val, y_val = prepare(val_df)
X_test, y_test = prepare(test_df)


# -----------------------
# MODEL
# -----------------------
model = CatBoostClassifier(
    iterations=800,
    depth=6,
    learning_rate=0.05,
    loss_function='Logloss',
    eval_metric='Accuracy',
    verbose=100
)


# -----------------------
# TRAIN
# -----------------------
print("\n🚀 Training CatBoost...\n")

model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    use_best_model=True
)


# -----------------------
# SAVE MODEL
# -----------------------
os.makedirs("models", exist_ok=True)
model.save_model("models/catboost_model.cbm")

print("✅ Model saved")


# -----------------------
# EVALUATION FUNCTION
# -----------------------
def evaluate(X, y, name):

    probs = model.predict_proba(X)[:, 1]
    preds = (probs > 0.5).astype(int)

    acc = accuracy_score(y, preds)

    # confidence filtering
    confidence = np.abs(probs - 0.5)
    mask = confidence > 0.1

    if mask.sum() > 0:
        f_acc = accuracy_score(y[mask], preds[mask])
    else:
        f_acc = 0

    coverage = mask.mean()

    print(f"\n {name} RESULTS")
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"Filtered Accuracy: {f_acc*100:.2f}%")
    print(f"Coverage: {coverage:.2f}")

    return probs, preds


# -----------------------
# RUN EVALUATION
# -----------------------
train_probs, train_preds = evaluate(X_train, y_train, "TRAIN")
val_probs, val_preds = evaluate(X_val, y_val, "VALIDATION")
test_probs, test_preds = evaluate(X_test, y_test, "TEST")


# -----------------------
# GRAPH 1: FEATURE IMPORTANCE
# -----------------------
importances = model.get_feature_importance()

plt.figure()
plt.barh(features, importances)
plt.xlabel("Importance")
plt.title("Feature Importance (CatBoost)")
plt.tight_layout()
plt.savefig("models/feature_importance.png")
plt.show()


# -----------------------
# GRAPH 2: PREDICTION CONFIDENCE
# -----------------------
confidence = np.abs(test_probs - 0.5)

plt.figure()
plt.hist(confidence, bins=50)
plt.title("Prediction Confidence Distribution")
plt.xlabel("Confidence")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("models/confidence_distribution.png")
plt.show()


# -----------------------
#  GRAPH 3: CONFUSION MATRIX
# -----------------------
cm = confusion_matrix(y_test, test_preds)

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("Actual")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
plt.savefig("models/confusion_matrix.png")
plt.show()


print("\n All graphs saved in models/")