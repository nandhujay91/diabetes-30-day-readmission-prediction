from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
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

    model = Pipeline(
        [
            ("preprocessing", build_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=12,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1,
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

    print("\nTraining Random Forest...")

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y, patient_ids), start=1):
        fold_model = clone(model)

        fold_model.fit(X.iloc[train_idx], y.iloc[train_idx])

        probabilities[valid_idx] = fold_model.predict_proba(X.iloc[valid_idx])[:, 1]

        print(f"Fold {fold} completed.")

    pr_auc = average_precision_score(y, probabilities)
    roc_auc = roc_auc_score(y, probabilities)
    brier = brier_score_loss(y, probabilities)

    print("\nRANDOM FOREST RESULTS")
    print("PR-AUC:", round(pr_auc, 4))
    print("ROC-AUC:", round(roc_auc, 4))
    print("Brier Score:", round(brier, 4))

    predictions = pd.DataFrame(
        {
            "actual": y,
            "predicted_probability": probabilities,
        }
    )

    predictions.to_csv(
        report_folder / "random_forest_oof_predictions.csv",
        index=False,
    )

    results = pd.DataFrame(
        [
            {
                "model": "random_forest",
                "pr_auc": pr_auc,
                "roc_auc": roc_auc,
                "brier_score": brier,
            }
        ]
    )

    results.to_csv(
        report_folder / "random_forest_results.csv",
        index=False,
    )

    print("\nResults saved in:", report_folder)


if __name__ == "__main__":
    main()
