from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedGroupKFold,
)
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
    artifact_folder = project_path / "artifacts"

    report_folder.mkdir(parents=True, exist_ok=True)
    artifact_folder.mkdir(parents=True, exist_ok=True)

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
                LGBMClassifier(
                    random_state=42,
                    n_jobs=1,
                    verbosity=-1,
                ),
            ),
        ]
    )

    parameters = {
        "classifier__n_estimators": [150, 200, 300],
        "classifier__max_depth": [3, 4, 5, -1],
        "classifier__learning_rate": [0.03, 0.05, 0.08],
        "classifier__num_leaves": [15, 20, 31],
        "classifier__min_child_samples": [20, 30, 50],
        "classifier__subsample": [0.8, 1.0],
        "classifier__colsample_bytree": [0.8, 1.0],
        "classifier__reg_lambda": [1, 5, 10],
    }

    cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=parameters,
        n_iter=12,
        scoring="average_precision",
        cv=cv,
        random_state=42,
        n_jobs=1,
        verbose=1,
        refit=True,
        return_train_score=False,
    )

    print("\nStarting LightGBM hyperparameter tuning...")

    search.fit(X, y, groups=patient_ids)

    print("\nBEST PARAMETERS")

    for name, value in search.best_params_.items():
        print(f"{name}: {value}")

    print("\nBEST CROSS-VALIDATION PR-AUC:")
    print(round(search.best_score_, 4))

    results = pd.DataFrame(search.cv_results_)

    results = results.sort_values("rank_test_score")

    results.to_csv(
        report_folder / "lightgbm_tuning_results.csv",
        index=False,
    )

    pd.DataFrame([search.best_params_]).to_csv(
        report_folder / "lightgbm_best_parameters.csv",
        index=False,
    )

    joblib.dump(
        search.best_estimator_,
        artifact_folder / "lightgbm_tuned_pipeline.joblib",
    )

    print("\nTuning results saved in:", report_folder)
    print("Tuned model saved in:", artifact_folder)


if __name__ == "__main__":
    main()
