# NZ Bank Loan Default Predictor

This repository contains an end-to-end Machine Learning pipeline to predict loan defaults for New Zealand bank loan applicants using borrower demographics, financial, and digital behavioral attributes.

## Project Workflow
1. **Data Preprocessing & Feature Engineering**: Calculates days between application, issue, and payment dates to prevent massive categorical expansion. Removes high-cardinality raw dates and leaky flags (e.g., target-related metrics like fraud).
2. **Balanced Training**: Implements robust preprocessing pipelines and applies class balancing (`class_weight='balanced'`) to handle extreme default class imbalance (~0.96% default rate).
3. **Decision Threshold Tuning**: Optimizes the probability classification threshold using cross-validated Out-Of-Fold F1 score (rather than defaulting to 0.5).
4. **Deployment Export**: Saves the entire processor, optimal threshold, and estimator as a unified `loan_default_model.joblib` artifact.

## Directory Structure
```
├── train.py              # Script to build, tune, and save the pipeline
├── predict.py            # Script to run batch inference on new loan applicants
├── requirements.txt       # Production dependencies
└── README.md              # Project documentation
```

## Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
Place your synthetic bank loans CSV in the root directory or configure its path, then run:
```bash
python train.py --data nz_bank_loans_synthetic_with_dates.csv
```
This will train the pipeline, optimize the classification cutoff, and export `loan_default_model.joblib`.

### 3. Run Inference / Predictions
To make predictions on new applications:
```bash
python predict.py --model loan_default_model.joblib --input sample_loans.csv
```
