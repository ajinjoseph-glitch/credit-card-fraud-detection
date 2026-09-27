# Credit Card Fraud Detection with Anomaly Models

This project demonstrates anomaly detection for credit card fraud detection using:

- Isolation Forest
- Local Outlier Factor (LOF)

The dataset used in the example is the popular Kaggle credit card fraud dataset.

## Features

- Easy setup with Python and sklearn
- Baseline anomaly detection models
- Evaluation metrics for fraud detection
- Ready for extension with autoencoders or XGBoost

## Repository structure

```text
credit-card-fraud-detection/
├── README.md
├── requirements.txt
├── .gitignore
└── src/
    └── anomaly_fraud_detection.py
```

## Setup

1. Clone the repository.
2. Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Download the Kaggle dataset and save it as `creditcard.csv` in the project root.

Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Run

```bash
python src/anomaly_fraud_detection.py --data creditcard.csv
```

## Output

The script prints:

- class distribution
- confusion matrix
- precision, recall, F1-score, and support for each model

## Notes

- Fraud cases are rare, so accuracy can be misleading.
- For fraud detection, precision/recall and F1-score are more important.
- This is a baseline anomaly-detection implementation; you can extend it with:
  - autoencoders
  - SMOTE
  - supervised models like XGBoost
  - threshold tuning

## Example

```python
from sklearn.ensemble import IsolationForest

model = IsolationForest(contamination=0.0017, random_state=42)
model.fit(X_train)

pred = model.predict(X_test)
pred = [1 if x == -1 else 0 for x in pred]
```
