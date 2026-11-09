from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedGroupKFold,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

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
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=1,
                ),
            ),
        ]
    )

    parameters = {
        "classifier__n_estimators": [150, 200, 300],
        "classifier__max_depth": [3, 4, 5],
        "classifier__learning_rate": [0.03, 0.05, 0.08],
        "classifier__min_child_weight": [1, 3, 5],
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

    print("\nStarting XGBoost hyperparameter tuning...")

    search.fit(X, y, groups=patient_ids)

    print("\nBEST PARAMETERS")

    for name, value in search.best_params_.items():
        print(f"{name}: {value}")

    print("\nBEST CROSS-VALIDATION PR-AUC:")
    print(round(search.best_score_, 4))

    results = pd.DataFrame(search.cv_results_)

    results = results.sort_values(by="rank_test_score")

    results.to_csv(
        report_folder / "xgboost_tuning_results.csv",
        index=False,
    )

    best_parameters = pd.DataFrame([search.best_params_])

    best_parameters.to_csv(
        report_folder / "xgboost_best_parameters.csv",
        index=False,
    )

    joblib.dump(
        search.best_estimator_,
        artifact_folder / "xgboost_tuned_pipeline.joblib",
    )

    print("\nTuning results saved in:", report_folder)
    print("Tuned model saved in:", artifact_folder)


if __name__ == "__main__":
    main()
