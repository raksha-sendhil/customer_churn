"""
preprocess.py — Load and clean the E-Commerce churn dataset.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def load_data(path: str = "data/E_Commerce_Dataset.xlsx") -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="E Comm")
    return df


def preprocess(df: pd.DataFrame):
    """
    Clean, encode, and split features/target.
    Returns X (DataFrame), y (Series), feature_names (list).
    """
    df = df.copy()

    # Drop ID — not a predictive feature
    if "CustomerID" in df.columns:
        df = df.drop(columns=["CustomerID"])

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    num_cols = [c for c in num_cols if c != "Churn"]
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    # Numeric → median imputation (pandas 2.x compatible)
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    # Categorical → mode imputation
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Encode categorical columns
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    # Final NaN check — drop any remaining
    df = df.dropna()

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    return X, y, X.columns.tolist()
