import os

import matplotlib.pyplot as plt
import pandas as pd
import umap
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.data import clean_data

DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
K = 4
PROFILE_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen",
                "month_to_month", "fiber", "Churn"]
SHOW_COLS = ["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Contract",
             "InternetService", "PhoneService", "PaymentMethod", "segment", "Churn", "anomaly_score"]


def get_features(df):
    X = df.drop(columns=["customerID", "Churn"])
    X = pd.get_dummies(X, drop_first=True, dtype=int)
    return StandardScaler().fit_transform(X)


def show_silhouette(X):
    for k in range(2, 9):
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
        score = silhouette_score(X, labels)
        print("k =", k, "silhouette =", round(score, 3))


def main():
    os.makedirs("docs/plots", exist_ok=True)
    df = clean_data(pd.read_csv(DATA_PATH))
    X = get_features(df)

    print("Lab 1: k-means segments")
    show_silhouette(X)
    print("chosen k:", K)
    df["segment"] = KMeans(n_clusters=K, n_init=10, random_state=42).fit_predict(X)

    df["month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)
    df["fiber"] = (df["InternetService"] == "Fiber optic").astype(int)
    profile = df.groupby("segment")[PROFILE_COLS].mean().round(2)
    profile["customers"] = df["segment"].value_counts().sort_index()
    print(profile.to_string())
    profile.to_csv("docs/segment_profiles.csv")

    print()
    print("Lab 2: UMAP plot")
    points = umap.UMAP(n_components=2, random_state=42).fit_transform(X)
    plt.figure(figsize=(7, 6))
    plt.scatter(points[:, 0], points[:, 1], c=df["segment"], cmap="tab10", s=5)
    plt.colorbar(label="segment")
    plt.title("Customer segments (UMAP)")
    plt.savefig("docs/plots/umap_segments.png", bbox_inches="tight")
    plt.close()
    print("saved docs/plots/umap_segments.png")

    print()
    print("Lab 3: Isolation Forest")
    iso = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    iso.fit(X)
    df["anomaly_score"] = iso.decision_function(X)
    df["is_anomaly"] = iso.predict(X) == -1
    print("flagged accounts:", df["is_anomaly"].sum())

    top10 = df.sort_values("anomaly_score").head(10)[SHOW_COLS]
    print(top10.to_string(index=False))
    top10.to_csv("docs/top10_anomalies.csv", index=False)


if __name__ == "__main__":
    main()