from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import brier_score_loss

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports" / "model_evaluation"

PREDICTION_FILE = REPORTS / "tuned_model_oof_predictions.csv"


# Load out-of-fold predictions
df = pd.read_csv(PREDICTION_FILE)

calibration_results = []
summary_results = []

plt.figure(figsize=(8, 6))

for model_name, model_data in df.groupby("model"):
    actual = model_data["actual"]
    probabilities = model_data["predicted_probability"]

    # Calculate Brier score
    brier = brier_score_loss(actual, probabilities)

    summary_results.append(
        {
            "model": model_name,
            "brier_score": brier,
            "mean_predicted_probability": probabilities.mean(),
            "actual_readmission_rate": actual.mean(),
        }
    )

    # Divide predictions into 10 groups
    model_data = model_data.copy()

    model_data["risk_group"] = pd.qcut(
        model_data["predicted_probability"],
        q=10,
        duplicates="drop",
    )

    # Compare predicted and actual readmission rates
    calibration_table = (
        model_data.groupby("risk_group", observed=True)
        .agg(
            patient_count=("actual", "size"),
            mean_predicted_risk=("predicted_probability", "mean"),
            actual_readmission_rate=("actual", "mean"),
        )
        .reset_index()
    )

    calibration_table["model"] = model_name

    calibration_results.append(calibration_table)

    # Draw calibration curve
    plt.plot(
        calibration_table["mean_predicted_risk"],
        calibration_table["actual_readmission_rate"],
        marker="o",
        label=model_name,
    )

    print(f"\n{model_name}")
    print(f"Brier score: {brier:.4f}")

# Reference line for perfect calibration
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect calibration",
)

plt.xlabel("Mean predicted readmission probability")
plt.ylabel("Actual readmission rate")
plt.title("Calibration Comparison: XGBoost vs LightGBM")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

# Save calibration chart
plt.savefig(
    REPORTS / "tuned_model_calibration.png",
    dpi=300,
)
plt.close()


# Save calibration tables
calibration_table = pd.concat(
    calibration_results,
    ignore_index=True,
)

summary = pd.DataFrame(summary_results)

calibration_table.to_csv(
    REPORTS / "tuned_model_calibration.csv",
    index=False,
)

summary.to_csv(
    REPORTS / "tuned_model_calibration_summary.csv",
    index=False,
)


print("\nCALIBRATION SUMMARY")
print(summary.round(4).to_string(index=False))

print("\nCALIBRATION ANALYSIS COMPLETED")
print("Reports saved successfully.")
