import torch
import torch.optim as optim
import torch.nn as nn
import os
import pickle


# -----------------------
# TRAIN FUNCTION
# -----------------------
def train_model(model, train_loader, val_loader, epochs=20):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 🔥 OPTIMIZER (GOOD DEFAULT)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)

    # -----------------------
    # LOSSES
    # -----------------------
    criterion_reg = nn.MSELoss()

    # 🔥 HANDLE CLASS IMBALANCE
    pos_weight = torch.tensor([2.0]).to(device)
    criterion_cls = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # -----------------------
    # SCHEDULER
    # -----------------------
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    best_val_loss = float('inf')

    for epoch in range(epochs):

        # -----------------------
        # TRAIN
        # -----------------------
        model.train()
        train_loss = 0

        for x_seq, x_flat, y, s in train_loader:

            x_seq = x_seq.to(device)
            x_flat = x_flat.to(device)
            y = y.to(device)
            s = s.to(device)

            optimizer.zero_grad()

            # FORWARD
            reg_pred, cls_pred = model(x_seq, x_flat, s)

            # -----------------------
            # LOSSES
            # -----------------------
            loss_reg = criterion_reg(reg_pred, y)

            # direction label
            direction = (y > 0).float()

            loss_cls = criterion_cls(cls_pred, direction)

            #  FINAL LOSS (IMPORTANT)
            loss = loss_reg + 1.2 * loss_cls

            loss.backward()

            #  PREVENT EXPLODING GRADIENTS
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # -----------------------
        # VALIDATION
        # -----------------------
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for x_seq, x_flat, y, s in val_loader:

                x_seq = x_seq.to(device)
                x_flat = x_flat.to(device)
                y = y.to(device)
                s = s.to(device)

                reg_pred, cls_pred = model(x_seq, x_flat, s)

                loss_reg = criterion_reg(reg_pred, y)
                direction = (y > 0).float()
                loss_cls = criterion_cls(cls_pred, direction)

                loss = loss_reg + 0.7 * loss_cls

                val_loss += loss.item()

        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        # -----------------------
        # SAVE BEST MODEL
        # -----------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/best_model.pth")

        print(f"\nEpoch {epoch+1}")
        print(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")


# -----------------------
# SAVE MODEL
# -----------------------
def save_model(model, scalers, stock_map):

    os.makedirs("models", exist_ok=True)

    torch.save(model.state_dict(), "models/model.pth")

    with open("models/scalers.pkl", "wb") as f:
        pickle.dump(scalers, f)

    with open("models/stock_map.pkl", "wb") as f:
        pickle.dump(stock_map, f)

    print("✅ Model, scalers & stock map saved")