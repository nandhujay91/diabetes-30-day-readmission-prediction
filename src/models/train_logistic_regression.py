from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.pipeline import Pipeline

from .preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    build_preprocessor,
)


def main():

    project_path = Path(__file__).resolve().parents[2]
    data_folder = project_path / "data" / "processed"

    train = pd.read_csv(data_folder / "train_model_input.csv")
    original_train = pd.read_csv(data_folder / "train.csv")

    feature_columns = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

    X = train[feature_columns]
    y = train["readmitted_30_days"]
    groups = original_train["patient_nbr"]

    model = Pipeline(
        [
            ("preprocessing", build_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    scoring = {
        "pr_auc": "average_precision",
        "roc_auc": "roc_auc",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
    }

    results = cross_validate(
        model,
        X,
        y,
        groups=groups,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1,
    )

    print("\nLOGISTIC REGRESSION - CROSS VALIDATION")

    for metric in scoring:
        scores = results[f"test_{metric}"]

        print(f"{metric.upper()}: {scores.mean():.4f} +/- {scores.std():.4f}")


if __name__ == "__main__":
    main()
