from pathlib import Path

import pandas as pd
from sklearn.metrics import brier_score_loss


def main():

    project_path = Path(__file__).resolve().parents[2]

    prediction_file = (
        project_path / "reports" / "model_evaluation" / "oof_predictions.csv"
    )

    report_folder = project_path / "reports" / "model_evaluation"

    predictions = pd.read_csv(prediction_file)

    actual = predictions["actual"]
    probability = predictions["predicted_probability"]

    predictions["risk_group"] = pd.qcut(probability, q=10, duplicates="drop")

    calibration = (
        predictions.groupby("risk_group", observed=True)
        .agg(
            encounters=("actual", "count"),
            actual_readmission_rate=("actual", "mean"),
            average_predicted_score=("predicted_probability", "mean"),
        )
        .reset_index()
    )

    calibration["actual_readmission_rate"] *= 100
    calibration["average_predicted_score"] *= 100

    brier_score = brier_score_loss(actual, probability)

    calibration.to_csv(report_folder / "calibration_analysis.csv", index=False)

    print("\nCALIBRATION ANALYSIS")
    print(calibration.round(2).to_string(index=False))

    print("\nBrier Score:", round(brier_score, 4))

    print("\nResults saved in:", report_folder)


if __name__ == "__main__":
    main()
