import argparse
import joblib
import numpy as np
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Batch Inference for NZ Loan Defaults")
    parser.add_argument("--model", type=str, default="loan_default_model.joblib", help="Path to serialized joblib artifact")
    parser.add_argument("--input", type=str, required=True, help="Path to new applications CSV file")
    args = parser.parse_args()

    # Load model package
    print(f"Loading model artifact: {args.model}")
    package = joblib.load(args.model)
    pipeline = package["pipeline"]
    threshold = package["threshold"]
    expected_features = package["features"]

    # Load and process new loans
    print(f"Loading incoming applications: {args.input}")
    df = pd.read_csv(args.input)

    # Re-apply feature engineering on dates if they are present
    for col in ["application_date", "issue_date", "first_payment_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    if "issue_date" in df.columns and "application_date" in df.columns:
        df["days_app_to_issue"] = (df["issue_date"] - df["application_date"]).dt.days
    if "first_payment_date" in df.columns and "issue_date" in df.columns:
        df["days_issue_to_payment"] = (df["first_payment_date"] - df["issue_date"]).dt.days

    # Ensure we align exact feature list
    missing_cols = [c for col in expected_features if col not in df.columns]
    if missing_cols:
         raise ValueError(f"Input file lacks expected model features: {missing_cols}")

    X = df[expected_features]

    # Score
    probabilities = pipeline.predict_proba(X)[:, 1]
    predictions = np.where(probabilities >= threshold, "High risk", "Low risk")

    results = df.copy()
    results["default_probability"] = probabilities.round(4)
    results["prediction_status"] = predictions

    # Output target columns
    output_cols = ["loan_id", "age", "annual_income", "default_probability", "prediction_status"]
    existing_output_cols = [col for col in output_cols if col in results.columns]

    print("
--- Sample Predictions Output ---")
    print(results[existing_output_cols].head())

if __name__ == "__main__":
    main()
