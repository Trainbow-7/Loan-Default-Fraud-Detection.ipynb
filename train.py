import argparse
import glob
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve

def main():
    parser = argparse.ArgumentParser(description="Train NZ Loan Default Predictor Pipeline")
    parser.add_argument("--data", type=str, required=True, help="Path to the loans CSV dataset")
    parser.add_argument("--output", type=str, default="loan_default_model.joblib", help="Filename to export model")
    args = parser.parse_args()

    print(f"Loading data from: {args.data}")
    df = pd.read_csv(args.data)

    # 1. Feature Engineering
    for col in ["application_date", "issue_date", "first_payment_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    if "issue_date" in df.columns and "application_date" in df.columns:
        df["days_app_to_issue"] = (df["issue_date"] - df["application_date"]).dt.days
    if "first_payment_date" in df.columns and "issue_date" in df.columns:
        df["days_issue_to_payment"] = (df["first_payment_date"] - df["issue_date"]).dt.days

    TARGET = "default"
    ID_COLS = ["loan_id"]
    LEAKY_COLS = ["fraud", "application_date", "issue_date", "first_payment_date"]

    X = df.drop(columns=[TARGET] + ID_COLS + LEAKY_COLS, errors="ignore")
    y = df[TARGET].astype(int)

    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 2. Pipeline setup
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric_features),
        ("cat", categorical_pipe, categorical_features),
    ])

    best_model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    pipeline = Pipeline([("prep", preprocessor), ("model", best_model)])

    # 3. Cross-Validation Threshold Tuning on Training
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print("Tuning decision threshold using 5-fold cross-validation...")
    oof_proba = cross_val_predict(pipeline, X_train, y_train, cv=cv, method="predict_proba")[:, 1]

    prec, rec, thr = precision_recall_curve(y_train, oof_proba)
    f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
    best_threshold = float(thr[np.argmax(f1s)])
    print(f"Optimal threshold found: {best_threshold:.3f} (CV F1 = {f1s.max():.3f})")

    # 4. Final Train
    print("Training final model pipeline on entire training split...")
    pipeline.fit(X_train, y_train)

    # 5. Serialize Artifacts
    joblib.dump({
        "pipeline": pipeline,
        "threshold": best_threshold,
        "features": X.columns.tolist()
    }, args.output)
    print(f"Successfully saved full deployment bundle to: {args.output}")

if __name__ == "__main__":
    main()
