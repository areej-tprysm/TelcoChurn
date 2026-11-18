# Day 1 EDA Summary: Telco Customer Churn

## Data
7043 customers, 21 columns. Target: Churn (Yes/No).

## Key concepts
- Vectorization: instead of using a loop, numpy does the operation on the whole array at once, that's why it is faster. Broadcasting is when numpy uses a smaller array (like the mean of each column) on all the rows by itself.
- **pandas vs Polars:** pandas is the commonly used. Polars is faster and lazy (runs the query at the end). For our 7043 rows pandas is fine.
- **Timestamps:** this dataset has no date columns, so there are no date issues and no time-based split.
- **Train/serve skew:** the same `clean_data()` should be used for training and for new data, so both are cleaned the same way.

## EDA findings
| Check | Finding |
|---|---|
| Missing | 0 NaN, but 11 blank TotalCharges (All of them had tenure = 0, so they are new customers.) |
| Duplicates | 0 |
| Distributions | TotalCharges right-skewed (0.96) |
| Outliers | 0 with IQR (train data) |
| Class balance | 26.5% churn (imbalanced) |
| Correlations | tenure and TotalCharges strongly related |
| Inconsistent categories | "No internet service" / "No phone service" same as "No" |
| Impossible values | none |

## Cleaning (clean_data)
1. TotalCharges blanks -> 0
2. "No internet/phone service" -> "No"
3. Churn -> 1/0
4. Duplicate customers removed

A unit test for each in `tests/test_data.py`.

## 3 hypotheses
1. Month-to-month customers churn more than yearly contracts.
2. New customers (low tenure) churn more.
3. Fiber optic and electronic check customers churn more.

## Target leakage candidates
- customerID: unique per customer, the model can memorise it.


- **MCAR vs MNAR:** MCAR means values are missing randomly, so filling with mean or median is fine. 
- MNAR means they are missing because of some reason, so filling them simply can be wrong. In this data TotalCharges was missing because the customers were new, so I filled 0.
- Columns that could leak: customerID, and maybe tenure and TotalCharges.