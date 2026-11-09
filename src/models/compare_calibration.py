from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from .preprocessing import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, build_preprocessor


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

    cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    results = []

    for name, class_weight in [
        ("balanced", "balanced"),
        ("unweighted", None),
    ]:
        model = Pipeline(
            [
                ("preprocessing", build_preprocessor()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight=class_weight,
                        max_iter=1000,
                        random_state=42,
                    ),
                ),
            ]
        )

        probabilities = np.zeros(len(train))

        print(f"\nTraining {name} model...")

        for fold, (train_idx, valid_idx) in enumerate(
            cv.split(X, y, patient_ids), start=1
        ):
            model_fold = clone(model)

            model_fold.fit(X.iloc[train_idx], y.iloc[train_idx])

            probabilities[valid_idx] = model_fold.predict_proba(X.iloc[valid_idx])[:, 1]

            print(f"Fold {fold} completed.")

        pr_auc = average_precision_score(y, probabilities)
        roc_auc = roc_auc_score(y, probabilities)
        brier = brier_score_loss(y, probabilities)

        results.append(
            {
                "model": name,
                "pr_auc": pr_auc,
                "roc_auc": roc_auc,
                "brier_score": brier,
                "mean_predicted_score": probabilities.mean(),
            }
        )

        predictions = pd.DataFrame(
            {
                "actual": y,
                "predicted_probability": probabilities,
            }
        )

        predictions.to_csv(
            report_folder / f"{name}_oof_predictions.csv",
            index=False,
        )

        predictions["risk_group"] = pd.qcut(
            predictions["predicted_probability"],
            q=10,
            duplicates="drop",
        )

        calibration = (
            predictions.groupby(
                "risk_group",
                observed=True,
            )
            .agg(
                encounters=("actual", "count"),
                actual_readmission_rate=("actual", "mean"),
                average_predicted_score=("predicted_probability", "mean"),
            )
            .reset_index()
        )

        calibration["actual_readmission_rate"] *= 100
        calibration["average_predicted_score"] *= 100

        calibration.to_csv(
            report_folder / f"{name}_calibration.csv",
            index=False,
        )

        print(f"{name} model completed.")

    results = pd.DataFrame(results)

    results.to_csv(
        report_folder / "logistic_regression_comparison.csv",
        index=False,
    )

    print("\nMODEL COMPARISON")
    print(results.round(4).to_string(index=False))

    print("\nResults saved in:", report_folder)


if __name__ == "__main__":
    main()
