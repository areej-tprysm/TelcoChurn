import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import clean_data

DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
N_TRIALS = 50
COST_FN = 5
COST_FP = 1


def get_data():
    df = clean_data(pd.read_csv(DATA_PATH))
    X = df.drop(columns=["customerID", "Churn"])
    y = df["Churn"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def build_pipeline(cat_cols, C=1.0, class_weight=None):
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])
    model = LogisticRegression(C=C, class_weight=class_weight, max_iter=1000)
    return Pipeline([("pre", pre), ("model", model)])


def tune(X, y, cat_cols):
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    def objective(trial):
        C = trial.suggest_float("C", 0.001, 100, log=True)
        class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])
        pipe = build_pipeline(cat_cols, C, class_weight)
        scores = cross_val_score(pipe, X, y, cv=inner_cv, scoring="average_precision")
        return scores.mean()

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS)
    return study


def nested_cv(X, y, cat_cols):
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in outer_cv.split(X, y):
        X_tr = X.iloc[train_idx]
        y_tr = y.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]

        study = tune(X_tr, y_tr, cat_cols)
        best = study.best_params
        pipe = build_pipeline(cat_cols, best["C"], best["class_weight"])
        pipe.fit(X_tr, y_tr)
        probs = pipe.predict_proba(X_val)[:, 1]
        score = average_precision_score(y_val, probs)
        scores.append(score)
        print("outer fold pr-auc:", round(score, 3), "best params:", best)
    return np.array(scores)


def total_cost(y, probs, threshold):
    pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return COST_FN * fn + COST_FP * fp


def pick_threshold(y, probs):
    best_threshold = 0.5
    best_cost = 10 ** 9
    for t in np.arange(0.01, 1.0, 0.01):
        cost = total_cost(y, probs, t)
        if cost < best_cost:
            best_cost = cost
            best_threshold = t
    return best_threshold


def plot_calibration(y, probs, path):
    frac_pos, mean_pred = calibration_curve(y, probs, n_bins=10)
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "--", label="perfect calibration")
    plt.plot(mean_pred, frac_pos, "o-", label="logistic regression")
    plt.xlabel("predicted probability")
    plt.ylabel("actual churn rate")
    plt.title("Calibration curve (test set)")
    plt.legend()
    plt.savefig(path)
    plt.close()


def main():
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    X_train, X_test, y_train, y_test = get_data()
    cat_cols = [c for c in X_train.columns if c not in NUM_COLS]

    scores = nested_cv(X_train, y_train, cat_cols)
    print("nested cv pr-auc:", round(scores.mean(), 3), "±", round(scores.std(), 3))

    study = tune(X_train, y_train, cat_cols)
    best = study.best_params
    print("final best params:", best)
    print("final best cv pr-auc:", round(study.best_value, 3))
    study.trials_dataframe().to_csv("docs/optuna_trials.csv", index=False)

    final = build_pipeline(cat_cols, best["C"], best["class_weight"])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_probs = cross_val_predict(final, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    threshold = pick_threshold(y_train, train_probs)
    print("best threshold:", round(threshold, 2))

    final.fit(X_train, y_train)
    test_probs = final.predict_proba(X_test)[:, 1]
    plot_calibration(y_test, test_probs, "docs/calibration_curve.png")

    for t in [0.5, threshold]:
        pred = (test_probs >= t).astype(int)
        print("test threshold", round(t, 2), "cost:", total_cost(y_test, test_probs, t), "recall:", round(recall_score(y_test, pred), 3))


if __name__ == "__main__":
    main()