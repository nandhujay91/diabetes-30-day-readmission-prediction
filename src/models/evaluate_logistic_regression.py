from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def main():

    project_path = Path(__file__).resolve().parents[2]

    data_folder = project_path / "data" / "processed"
    artifact_folder = project_path / "artifacts"

    test = pd.read_csv(data_folder / "test_model_input.csv")

    model = joblib.load(artifact_folder / "logistic_regression_pipeline.joblib")

    feature_columns = test.drop(columns=["readmitted_30_days"]).columns

    X_test = test[feature_columns]
    y_test = test["readmitted_30_days"]

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("\nLOGISTIC REGRESSION - TEST RESULTS")

    print("PR-AUC:", round(average_precision_score(y_test, probabilities), 4))

    print("ROC-AUC:", round(roc_auc_score(y_test, probabilities), 4))

    print("Precision:", round(precision_score(y_test, predictions, zero_division=0), 4))

    print("Recall:", round(recall_score(y_test, predictions, zero_division=0), 4))

    print("F1-score:", round(f1_score(y_test, predictions, zero_division=0), 4))

    print("\nCONFUSION MATRIX")
    print(confusion_matrix(y_test, predictions))

    print("\nCLASSIFICATION REPORT")
    print(classification_report(y_test, predictions, zero_division=0))


if __name__ == "__main__":
    main()
