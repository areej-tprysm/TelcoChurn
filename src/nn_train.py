import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.pipeline import NUM_COLS, get_data
from src.train_boosting import boost_pipeline

EPOCHS = 100
BATCH_SIZE = 32


def make_model(n_inputs, dropout):
    return nn.Sequential(
        nn.Linear(n_inputs, 256),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(128, 1),
    )


def train(X_tr, y_tr, X_val, y_val, dropout, weight_decay):
    torch.manual_seed(42)
    model = make_model(X_tr.shape[1], dropout)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_decay)
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)

    train_losses = []
    val_losses = []
    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        with torch.no_grad():
            train_loss = loss_fn(model(X_tr), y_tr).item()
            val_loss = loss_fn(model(X_val), y_val).item()
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            print("epoch", epoch, "| train loss", round(train_loss, 4), "| val loss", round(val_loss, 4))

    return model, train_losses, val_losses


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(X)).numpy().ravel()


def plot_losses(over_train, over_val, fix_train, fix_val, path):
    plt.figure(figsize=(11, 4))

    plt.subplot(1, 2, 1)
    plt.plot(over_train, label="train")
    plt.plot(over_val, label="val")
    plt.title("No dropout, no weight decay")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(fix_train, label="train")
    plt.plot(fix_val, label="val")
    plt.title("Dropout 0.5 + weight decay 0.001")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()

    plt.savefig(path, bbox_inches="tight")
    plt.close()


def main():
    os.makedirs("docs/plots", exist_ok=True)

    X_train, X_test, y_train, y_test = get_data()
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
    cat_cols = [c for c in X_train.columns if c not in NUM_COLS]

    pre = ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])
    X_tr_t = torch.tensor(pre.fit_transform(X_tr), dtype=torch.float32)
    X_val_t = torch.tensor(pre.transform(X_val), dtype=torch.float32)
    X_test_t = torch.tensor(pre.transform(X_test), dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr.to_numpy(), dtype=torch.float32).view(-1, 1)
    y_val_t = torch.tensor(y_val.to_numpy(), dtype=torch.float32).view(-1, 1)

    print("Run 1: no dropout, no weight decay")
    over_model, over_train, over_val = train(X_tr_t, y_tr_t, X_val_t, y_val_t, dropout=0.0, weight_decay=0.0)
    print("best val loss:", round(min(over_val), 4), "at epoch", int(np.argmin(over_val)))

    print()
    print("Run 2: dropout 0.5, weight decay 0.001")
    fix_model, fix_train, fix_val = train(X_tr_t, y_tr_t, X_val_t, y_val_t, dropout=0.5, weight_decay=0.001)
    print("best val loss:", round(min(fix_val), 4), "at epoch", int(np.argmin(fix_val)))

    plot_losses(over_train, over_val, fix_train, fix_val, "docs/plots/nn_loss_curves.png")
    print("saved docs/plots/nn_loss_curves.png")

    print()
    print("Compare on test set (PR-AUC)")
    over_pr = average_precision_score(y_test, predict(over_model, X_test_t))
    fix_pr = average_precision_score(y_test, predict(fix_model, X_test_t))

    cat = boost_pipeline("CatBoost", NUM_COLS + ["charge_per_month", "new_customer"], cat_cols)
    cat.fit(X_tr, y_tr)
    cat_pr = average_precision_score(y_test, cat.predict_proba(X_test)[:, 1])

    print("MLP (overfit):", round(over_pr, 3))
    print("MLP (dropout + weight decay):", round(fix_pr, 3))
    print("CatBoost (Day 4):", round(cat_pr, 3))


if __name__ == "__main__":
    main()