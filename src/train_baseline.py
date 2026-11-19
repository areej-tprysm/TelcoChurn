import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.data import clean_data

DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
SCORING = {
    "accuracy": "accuracy",
    "recall": "recall",
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
}


def get_data():
    df = clean_data(pd.read_csv(DATA_PATH))
    X = df.drop(columns=["customerID", "Churn"])
    y = df["Churn"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def make_pipeline(model, num_cols, cat_cols):
    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])
    return Pipeline([("pre", pre), ("model", model)])


def run_cv(model, X, y, cv):
    scores = cross_validate(model, X, y, cv=cv, scoring=SCORING)
    return {m: (scores[f"test_{m}"].mean(), scores[f"test_{m}"].std()) for m in SCORING}


def print_table(results):
    print("| Model | Accuracy | Recall | F1 | ROC AUC |")
    print("|---|---|---|---|---|")
    for name, r in results.items():
        cells = [f"{r[m][0]:.3f} ± {r[m][1]:.3f}" for m in SCORING]
        print(f"| {name} | " + " | ".join(cells) + " |")


def main():
    X_train, X_test, y_train, y_test = get_data()
    cat_cols = [c for c in X_train.columns if c not in NUM_COLS]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "Dummy (most frequent)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "k-NN (k=5)": KNeighborsClassifier(n_neighbors=5),
    }
    results = {}
    for name, model in models.items():
        results[name] = run_cv(make_pipeline(model, NUM_COLS, cat_cols), X_train, y_train, cv)

    print("Baselines: stratified 5-fold CV on the train set")
    print_table(results)

    leak_results = {"No leakage (pipeline)": results["Logistic Regression"]}

    X_encoded = pd.get_dummies(X_train, drop_first=True, dtype=int)
    X_scaled_early = StandardScaler().fit_transform(X_encoded)
    leak_results["Leak 1: scaled before CV"] = run_cv(
        LogisticRegression(max_iter=1000), X_scaled_early, y_train, cv)

    rng = np.random.default_rng(42)
    X_leaky = X_train.copy()
    X_leaky["leaky_feature"] = y_train + rng.normal(0, 0.3, len(y_train))
    leak_results["Leak 2: leaky feature"] = run_cv(
        make_pipeline(LogisticRegression(max_iter=1000), NUM_COLS + ["leaky_feature"], cat_cols),
        X_leaky, y_train, cv)

    print()
    print("Leakage experiment (Logistic Regression)")
    print_table(leak_results)


if __name__ == "__main__":
    main()