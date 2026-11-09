from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from .preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    build_preprocessor,
)


def main():

    project_path = Path(__file__).resolve().parents[2]
    data_folder = project_path / "data" / "processed"
    report_folder = project_path / "reports" / "model_evaluation"

    report_folder.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(data_folder / "train_model_input.csv")
    patient_ids = pd.read_csv(data_folder / "train.csv")["patient_nbr"]

    features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

    X = train[features]
    y = train["readmitted_30_days"]

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

    probabilities = np.zeros(len(train))

    print("\nGenerating cross-validation predictions...")

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y, patient_ids), start=1):
        fold_model = clone(model)

        fold_model.fit(X.iloc[train_idx], y.iloc[train_idx])

        probabilities[valid_idx] = fold_model.predict_proba(X.iloc[valid_idx])[:, 1]

        print(f"Fold {fold} completed.")

    print(
        "\nOut-of-fold PR-AUC:",
        round(average_precision_score(y, probabilities), 4),
    )

    results = []

    for threshold in np.arange(0.10, 0.91, 0.05):
        predictions = (probabilities >= threshold).astype(int)

        results.append(
            {
                "threshold": round(threshold, 2),
                "precision": round(precision_score(y, predictions, zero_division=0), 4),
                "recall": round(recall_score(y, predictions, zero_division=0), 4),
                "f1_score": round(f1_score(y, predictions, zero_division=0), 4),
                "flagged_percentage": round(predictions.mean() * 100, 2),
                "flagged_encounters": int(predictions.sum()),
            }
        )

    results = pd.DataFrame(results)

    results.to_csv(
        report_folder / "threshold_analysis.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "actual": y,
            "predicted_probability": probabilities,
        }
    ).to_csv(
        report_folder / "oof_predictions.csv",
        index=False,
    )

    print("\nTHRESHOLD ANALYSIS")
    print(results.to_string(index=False))

    print("\nResults saved in:", report_folder)


if __name__ == "__main__":
    main()
