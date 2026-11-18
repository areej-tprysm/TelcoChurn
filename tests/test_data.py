import pandas as pd

from src.data import clean_data, load_data

PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
raw = pd.read_csv(PATH)
clean = clean_data(raw)


def test_load_data_dtypes():
    df = load_data(PATH)
    assert df.shape == (7043, 21)
    assert df["tenure"].dtype == "int64"
    assert df["Contract"].dtype == "category"


def test_total_charges_fixed():
    assert clean["TotalCharges"].dtype == "float64"
    blank = raw["TotalCharges"].str.strip() == ""
    assert (clean.loc[blank, "TotalCharges"] == 0).all()


def test_no_service_values_removed():
    assert "No internet service" not in clean["OnlineSecurity"].unique().tolist()
    assert "No phone service" not in clean["MultipleLines"].unique().tolist()


def test_churn_is_0_1():
    assert set(clean["Churn"].unique()) == {0, 1}


def test_no_duplicate_customers():
    assert clean["customerID"].duplicated().sum() == 0