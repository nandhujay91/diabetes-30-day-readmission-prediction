from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports" / "model_evaluation"

PREDICTION_FILE = REPORTS / "tuned_model_oof_predictions.csv"


# Load LightGBM out-of-fold predictions
df = pd.read_csv(PREDICTION_FILE)

df = df[df["model"] == "LightGBM"].copy()

actual = df["actual"].astype(int)
probabilities = df["predicted_probability"]


# Test different probability thresholds
thresholds = np.arange(0.05, 0.96, 0.05)

results = []

for threshold in thresholds:
    predicted = (probabilities >= threshold).astype(int)

    true_positive = ((actual == 1) & (predicted == 1)).sum()
    false_positive = ((actual == 0) & (predicted == 1)).sum()
    false_negative = ((actual == 1) & (predicted == 0)).sum()

    precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive) > 0
        else 0
    )

    recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative) > 0
        else 0
    )

    f1_score = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    flagged = predicted.sum()

    results.append(
        {
            "threshold": round(threshold, 2),
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "flagged_encounters": flagged,
            "flagged_percentage": flagged / len(df) * 100,
            "true_positives": true_positive,
            "false_positives": false_positive,
            "false_negatives": false_negative,
        }
    )


# Create and save threshold results
threshold_results = pd.DataFrame(results)

threshold_results.to_csv(
    REPORTS / "lightgbm_threshold_analysis.csv",
    index=False,
)


# Display results
print("\nLIGHTGBM THRESHOLD ANALYSIS")
print(threshold_results.round(4).to_string(index=False))

print("\nTOTAL ENCOUNTERS:", len(df))
print("ACTUAL READMISSIONS:", actual.sum())
print(
    "ACTUAL READMISSION RATE:",
    round(actual.mean() * 100, 2),
    "%",
)

print("\nThreshold analysis completed.")
print("Results saved successfully.")
