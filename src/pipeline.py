import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import clean_data

DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
SCORING = "average_precision"
N_TRIALS = 50


def get_data():
    df = clean_data(pd.read_csv(DATA_PATH))
    X = df.drop(columns=["customerID", "Churn"])
    y = df["Churn"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def build_pipeline(cat_cols):
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])
    model = LogisticRegression(max_iter=1000)
    return Pipeline([("pre", pre), ("model", model)])


def set_best(pipe, params):
    return clone(pipe).set_params(model__C=params["C"], model__class_weight=params["class_weight"])


def tune(pipe, X, y):
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    def objective(trial):
        params = {
            "C": trial.suggest_float("C", 1e-3, 100, log=True),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        }
        model = set_best(pipe, params)
        return cross_val_score(model, X, y, cv=inner_cv, scoring=SCORING).mean()

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS)
    return study


def nested_cv(pipe, X, y):
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for i, (tr, va) in enumerate(outer_cv.split(X, y), 1):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y.iloc[tr], y.iloc[va]
        study = tune(pipe, X_tr, y_tr)
        model = set_best(pipe, study.best_params).fit(X_tr, y_tr)
        score = average_precision_score(y_va, model.predict_proba(X_va)[:, 1])
        scores.append(score)
        print(f"outer fold {i}: pr-auc = {score:.3f}, best params = {study.best_params}")
    return np.array(scores)


def main():
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    X_train, X_test, y_train, y_test = get_data()
    cat_cols = [c for c in X_train.columns if c not in NUM_COLS]
    pipe = build_pipeline(cat_cols)

    scores = nested_cv(pipe, X_train, y_train)
    print(f"nested cv pr-auc: {scores.mean():.3f} ± {scores.std():.3f}")

    study = tune(pipe, X_train, y_train)
    print("final best params:", study.best_params)
    print(f"final best cv pr-auc: {study.best_value:.3f}")
    study.trials_dataframe().to_csv("docs/optuna_trials.csv", index=False)
    print("saved docs/optuna_trials.csv")


if __name__ == "__main__":
    main()