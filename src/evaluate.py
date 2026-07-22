import torch
import numpy as np


def evaluate(model, dataloader):

    model.eval()

    reg_preds = []
    cls_preds = []
    targets = []

    # -----------------------
    # COLLECT PREDICTIONS
    # -----------------------
    with torch.no_grad():
        for x_seq, x_flat, y, s in dataloader:

            reg_out, cls_out = model(x_seq, x_flat, s)

            reg_preds.extend(reg_out.detach().cpu().numpy())
            cls_preds.extend(cls_out.detach().cpu().numpy())
            targets.extend(y.detach().cpu().numpy())

    reg_preds = np.array(reg_preds).flatten()
    cls_preds = np.array(cls_preds).flatten()
    targets = np.array(targets).flatten()

    # -----------------------
    # REGRESSION METRIC
    # -----------------------
    mse = np.mean((reg_preds - targets) ** 2)

    # -----------------------
    #  CLASSIFICATION (MAIN)
    # -----------------------
    probs = 1 / (1 + np.exp(-cls_preds))  # sigmoid

    pred_dir = (probs > 0.5).astype(int)
    true_dir = (targets > 0).astype(int)

    accuracy = np.mean(pred_dir == true_dir)

    # -----------------------
    # CONFIDENCE FILTER
    # -----------------------
    confidence = np.abs(probs - 0.5)

    threshold = 0.05   #  tune (0.05 → 0.2)

    mask = confidence > threshold

    if mask.sum() > 0:
        filtered_acc = np.mean(pred_dir[mask] == true_dir[mask])
    else:
        filtered_acc = 0.0

    coverage = np.mean(mask)

    # -----------------------
    # DEBUG INFO
    # -----------------------
    print("\nDEBUG:")
    print("Prob min:", probs.min())
    print("Prob max:", probs.max())
    print("Confidence mean:", confidence.mean())

    # -----------------------
    # RESULTS
    # -----------------------
    print("\n📊 Evaluation Results")
    print(f"MSE: {mse:.6f}")
    print(f"Direction Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Filtered Accuracy: {filtered_acc:.4f} ({filtered_acc*100:.2f}%)")
    print(f"Coverage: {coverage:.2f}")
    print(f"Confidence Threshold: {threshold}")

    return mse, accuracy, filtered_acc, coverage, reg_preds, targets