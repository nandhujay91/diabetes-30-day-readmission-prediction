from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    build_preprocessor,
)


def main():

    project_path = Path(__file__).resolve().parents[2]

    data_folder = project_path / "data" / "processed"
    artifact_folder = project_path / "artifacts"

    artifact_folder.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(data_folder / "train_model_input.csv")

    feature_columns = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

    X_train = train[feature_columns]
    y_train = train["readmitted_30_days"]

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

    print("\nTraining Logistic Regression...")

    model.fit(X_train, y_train)

    model_path = artifact_folder / "logistic_regression_pipeline.joblib"

    joblib.dump(model, model_path)

    print("\nTRAINING COMPLETED")
    print("Training encounters:", len(X_train))
    print("Original predictors:", len(feature_columns))
    print("Positive encounters:", int(y_train.sum()))
    print("Negative encounters:", int((y_train == 0).sum()))
    print("Model saved:", model_path)


if __name__ == "__main__":
    main()
