import pandas as pd


def load_data(path):
    """Load the churn CSV with explicit dtypes."""
    dtypes = {
        "customerID": "string",
        "SeniorCitizen": "int64",
        "tenure": "int64",
        "MonthlyCharges": "float64",
        "TotalCharges": "string",   # has blanks, fixed in clean_data
    }
    cat_cols = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
                "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
                "PaperlessBilling", "PaymentMethod", "Churn"]
    for col in cat_cols:
        dtypes[col] = "category"
    return pd.read_csv(path, dtype=dtypes)


def clean_data(data):
    """Fix the problems found in EDA."""
    data = data.copy()
    # 1. TotalCharges blank -> 0 (new customers), then number
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce").fillna(0)
    # 2. "No internet service" / "No phone service" -> "No"
    cols = ["MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies"]
    data[cols] = data[cols].replace({"No internet service": "No", "No phone service": "No"})
    # 3. Churn Yes/No -> 1/0
    data["Churn"] = data["Churn"].map({"No": 0, "Yes": 1})
    # 4. remove duplicate customers
    data = data.drop_duplicates(subset="customerID")
    return data