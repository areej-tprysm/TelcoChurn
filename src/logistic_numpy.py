import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from src.data import clean_data
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def log_loss(y, p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

def fit(X, y, lr=0.1, epochs=2000):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for epoch in range(epochs):
        p = sigmoid(X @ w + b)
        error = p - y
        dw = X.T @ error / n
        db = error.mean()
        w = w - lr * dw
        b = b - lr * db
        if epoch % 200 == 0:
            print(epoch, round(log_loss(y, p), 4))
    return w, b


def predict_proba(X, w, b):
    return sigmoid(X @ w + b)


def predict(X, w, b):
    return (predict_proba(X, w, b) >= 0.5).astype(int)


if __name__ == "__main__":
    raw = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df = clean_data(raw)
    X = pd.get_dummies(df.drop(columns=["customerID", "Churn"]), drop_first=True, dtype=int)
    y = df["Churn"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    w, b = fit(X_train, y_train)
    my_acc = accuracy_score(y_test, predict(X_test, w, b))

    sk = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    sk_acc = accuracy_score(y_test, sk.predict(X_test))

    print("my model accuracy:", round(my_acc, 4))
    print("sklearn accuracy: ", round(sk_acc, 4))

    