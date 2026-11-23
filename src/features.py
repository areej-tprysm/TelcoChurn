NEW_FEATURES = ["charge_per_month", "num_services", "new_customer", "monthly_x_contract"]
SERVICE_COLS = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]


def add_features(X):
    X = X.copy()
    X["charge_per_month"] = X["TotalCharges"] / X["tenure"].replace(0, 1)
    X["num_services"] = (X[SERVICE_COLS] == "Yes").sum(axis=1)
    X["new_customer"] = (X["tenure"] <= 6).astype(int)
    X["monthly_x_contract"] = X["MonthlyCharges"] * (X["Contract"] == "Month-to-month")
    return X