"""
predict.py — Run churn predictions on new data using the saved model.

Usage:
    python predict.py --input data/new_customers.xlsx --output outputs/predictions.csv
    python predict.py --input data/new_customers.xlsx          # prints to stdout
"""

import argparse
import joblib
import pandas as pd
from preprocess import preprocess

MODEL_PATH = "models/random_forest_churn.pkl"


def load_model(path: str = MODEL_PATH):
    return joblib.load(path)


def predict(df: pd.DataFrame, model) -> pd.DataFrame:
    """
    Expects df to have the same columns as the training data
    (CustomerID is optional; Churn column will be ignored if present).
    """
    # Keep CustomerID for output if present
    ids = df["CustomerID"].values if "CustomerID" in df.columns else range(len(df))

    X, _, _ = preprocess(df.assign(Churn=0))  # dummy Churn column for preprocessor

    probs  = model.predict_proba(X)[:, 1]
    labels = model.predict(X)

    result = pd.DataFrame({
        "CustomerID":   ids,
        "ChurnProb":    probs.round(4),
        "ChurnPredicted": labels,
    })
    result["Risk"] = pd.cut(
        result["ChurnProb"],
        bins=[0, 0.3, 0.6, 1.0],
        labels=["Low", "Medium", "High"],
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict customer churn")
    parser.add_argument("--input",  required=True,  help="Path to input Excel/CSV file")
    parser.add_argument("--output", required=False, help="Path to save predictions CSV")
    args = parser.parse_args()

    # Load data
    if args.input.endswith(".csv"):
        df_new = pd.read_csv(args.input)
    else:
        df_new = pd.read_excel(args.input, sheet_name="E Comm")

    model = load_model()
    predictions = predict(df_new, model)

    if args.output:
        predictions.to_csv(args.output, index=False)
        print(f"Predictions saved → {args.output}")
    else:
        print(predictions.to_string(index=False))
