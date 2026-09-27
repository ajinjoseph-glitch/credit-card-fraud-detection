import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor


def load_dataset(dataset_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load the dataset and split into features and target."""
    df = pd.read_csv(dataset_path)

    if "Class" not in df.columns:
        raise ValueError("Dataset must contain a 'Class' column for fraud labels.")

    X = df.drop(columns=["Class"])
    y = df["Class"]
    return X, y


def convert_anomaly_labels(predictions: np.ndarray) -> np.ndarray:
    """Convert anomaly model labels (-1 for anomaly, 1 for normal) to fraud labels (1 for fraud, 0 for normal)."""
    return np.where(predictions == -1, 1, 0)


def evaluate_model(name: str, predictions: np.ndarray, y_true: pd.Series) -> None:
    """Print confusion matrix and classification report."""
    print(f"\n{name}")
    print("-" * 50)
    print(confusion_matrix(y_true, predictions))
    print(classification_report(y_true, predictions, zero_division=0))


def train_isolation_forest(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> np.ndarray:
    """Train and predict using Isolation Forest."""
    contamination = y_train.mean()
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
    )
    model.fit(X_train)
    pred = model.predict(X_test)
    return convert_anomaly_labels(pred)


def train_local_outlier_factor(X_test: pd.DataFrame, y_test: pd.Series) -> np.ndarray:
    """Train and predict using Local Outlier Factor."""
    contamination = y_test.mean() if len(y_test) > 0 else 0.001
    model = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
    pred = model.fit_predict(X_test)
    return convert_anomaly_labels(pred)


def main() -> None:
    parser = argparse.ArgumentParser(description="Credit card fraud detection using anomaly models.")
    parser.add_argument("--data", type=str, required=True, help="Path to the creditcard dataset CSV file.")
    args = parser.parse_args()

    dataset_path = Path(args.data)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    X, y = load_dataset(str(dataset_path))

    print("Fraud cases:", int(y.sum()))
    print("Non-fraud cases:", int(len(y) - y.sum()))
    print("Fraud ratio:", round(float(y.mean()), 6))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    iso_predictions = train_isolation_forest(X_train, X_test, y_train, y_test)
    evaluate_model("Isolation Forest", iso_predictions, y_test)

    lof_predictions = train_local_outlier_factor(X_test, y_test)
    evaluate_model("Local Outlier Factor", lof_predictions, y_test)


if __name__ == "__main__":
    main()
