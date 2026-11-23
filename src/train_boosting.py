import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from src.features import NEW_FEATURES, add_features
from src.pipeline import NUM_COLS, build_pipeline, get_data, pick_threshold

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def make_model(name):
    if name == "LightGBM":
        return LGBMClassifier(random_state=42, verbose=-1)
    return CatBoostClassifier(random_seed=42, verbose=0, allow_writing_files=False)


def boost_pipeline(name, num_cols, cat_cols):
    pre = ColumnTransformer([
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ], verbose_feature_names_out=False)
    return Pipeline([
        ("features", FunctionTransformer(add_features)),
        ("pre", pre),
        ("model", make_model(name)),
    ])


def cv_score(pipe, X, y):
    scores = cross_val_score(pipe, X, y, cv=CV, scoring="average_precision")
    return scores.mean(), scores.std()


def get_threshold(pipe, X, y):
    probs = cross_val_predict(pipe, X, y, cv=CV, method="predict_proba")[:, 1]
    return pick_threshold(y, probs)


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("docs/plots", exist_ok=True)

    X_train, X_test, y_train, y_test = get_data()
    cat_cols = [c for c in X_train.columns if c not in NUM_COLS]

    print("Step 1: compare models (5-fold CV, PR-AUC)")
    lr = build_pipeline(cat_cols, C=1.96, class_weight=None)
    lr_mean, lr_std = cv_score(lr, X_train, y_train)
    print("Logistic Regression (Day 3):", round(lr_mean, 3), "±", round(lr_std, 3))

    lgbm_mean, lgbm_std = cv_score(boost_pipeline("LightGBM", NUM_COLS, cat_cols), X_train, y_train)
    print("LightGBM:", round(lgbm_mean, 3), "±", round(lgbm_std, 3))

    cat_mean, cat_std = cv_score(boost_pipeline("CatBoost", NUM_COLS, cat_cols), X_train, y_train)
    print("CatBoost:", round(cat_mean, 3), "±", round(cat_std, 3))

    if lgbm_mean > cat_mean:
        best_boost = "LightGBM"
        base_score = lgbm_mean
    else:
        best_boost = "CatBoost"
        base_score = cat_mean

    print()
    print("Step 2: test new features on", best_boost, "(base:", round(base_score, 3), ")")
    kept = []
    for f in NEW_FEATURES:
        mean, std = cv_score(boost_pipeline(best_boost, NUM_COLS + [f], cat_cols), X_train, y_train)
        if mean > base_score:
            kept.append(f)
            print(f, round(mean, 3), "- keep")
        else:
            print(f, round(mean, 3), "- drop")

    num_cols = NUM_COLS + kept
    final_boost = boost_pipeline(best_boost, num_cols, cat_cols)
    boost_mean, boost_std = cv_score(final_boost, X_train, y_train)
    print("all kept features together:", round(boost_mean, 3), "±", round(boost_std, 3))

    if boost_mean < base_score:
        print("together they are worse than base, so no new features are used")
        kept = []
        num_cols = NUM_COLS
        final_boost = boost_pipeline(best_boost, num_cols, cat_cols)
        boost_mean = base_score
    print("kept features:", kept)

    print()
    print("Step 3: final boosted model vs Day 3")
    print(best_boost, ":", round(boost_mean, 3))
    print("Logistic Regression (Day 3):", round(lr_mean, 3))

    if boost_mean > lr_mean:
        best_name = best_boost
        best_pipe = final_boost
        best_cv = boost_mean
        best_features = num_cols + cat_cols
    else:
        best_name = "Logistic Regression"
        best_pipe = lr
        best_cv = lr_mean
        best_features = NUM_COLS + cat_cols

    threshold = get_threshold(best_pipe, X_train, y_train)
    best_pipe.fit(X_train, y_train)
    joblib.dump(best_pipe, "models/best_model.joblib")
    config = {
        "model": best_name,
        "features": best_features,
        "threshold": round(float(threshold), 2),
        "cv_pr_auc": round(float(best_cv), 3),
        "params": best_pipe.named_steps["model"].get_params(),
    }
    with open("models/best_model_config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)
    print("best model:", best_name, "| threshold:", round(threshold, 2), "| saved to models/")

    print()
    print("Step 4: SHAP for", best_boost)
    boost_threshold = get_threshold(final_boost, X_train, y_train)
    final_boost.fit(X_train, y_train)
    probs = final_boost.predict_proba(X_test)[:, 1]
    preds = (probs >= boost_threshold).astype(int)
    actual = y_test.to_numpy()

    fp = np.where((preds == 1) & (actual == 0))[0][0]
    fn = np.where((preds == 0) & (actual == 1))[0][0]
    tp = np.where((preds == 1) & (actual == 1))[0][0]

    pre = final_boost.named_steps["pre"]
    X_test_ready = pd.DataFrame(pre.transform(add_features(X_test)), columns=pre.get_feature_names_out())
    explainer = shap.TreeExplainer(final_boost.named_steps["model"])
    shap_values = explainer(X_test_ready)
    if shap_values.values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    shap.plots.beeswarm(shap_values, show=False)
    plt.savefig("docs/plots/shap_beeswarm.png", bbox_inches="tight")
    plt.close()

    for label, i in [("false_positive", fp), ("false_negative", fn), ("true_positive", tp)]:
        shap.plots.waterfall(shap_values[i], show=False)
        plt.savefig("docs/plots/shap_" + label + ".png", bbox_inches="tight")
        plt.close()
        print(label, "| test row:", i, "| churn prob:", round(probs[i], 2), "| actual:", actual[i])

    print("threshold used for SHAP model:", round(boost_threshold, 2))
    print("plots saved to docs/plots/")


if __name__ == "__main__":
    main()