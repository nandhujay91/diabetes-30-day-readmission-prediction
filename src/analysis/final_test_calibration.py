from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

ROOT = Path(__file__).resolve().parents[2]

FEATURES_PATH = ROOT / "data/processed/test_features.csv"
TEST_PATH = ROOT / "data/processed/test.csv"
MODEL_PATH = ROOT / "artifacts/lightgbm_tuned_pipeline.joblib"

OUTPUT_DIR = ROOT / "reports/model_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PLOT_PATH = OUTPUT_DIR / "lightgbm_test_calibration.png"
TABLE_PATH = OUTPUT_DIR / "lightgbm_test_calibration.csv"


# Load test data and saved model
features = pd.read_csv(FEATURES_PATH)
test = pd.read_csv(TEST_PATH)
model = joblib.load(MODEL_PATH)


# Check that the test data matches
if len(features) != len(test):
    raise ValueError("Test files have different row counts.")

if not np.array_equal(
    features["encounter_id"].to_numpy(), test["encounter_id"].to_numpy()
):
    raise ValueError("Encounter IDs do not match.")

y_true = test["readmitted_30_days"].to_numpy()

if not np.array_equal(features["readmitted_30_days"].to_numpy(), y_true):
    raise ValueError("Target values do not match.")


# Select model features
numerical_features = [
    "time_in_hospital",
    "lab_procedures_per_day",
    "medications_per_day",
    "procedures_per_day",
    "long_hospital_stay",
    "number_inpatient",
    "number_outpatient",
    "number_emergency",
    "medication_count",
    "medication_changed_count",
    "medication_up_count",
    "medication_down_count",
    "insulin_used",
    "medication_regimen_changed",
]

categorical_features = [
    "age_group",
    "admission_type_group",
    "admission_source_group",
    "discharge_destination_group",
    "primary_diagnosis_group",
    "secondary_diagnosis_group",
    "additional_diagnosis_group",
    "glucose_test_result",
    "hba1c_test_result",
    "medical_specialty",
    "payer_code",
]

model_features = numerical_features + categorical_features

missing_features = [
    column for column in model_features if column not in features.columns
]

if missing_features:
    raise ValueError(f"Missing model features: {missing_features}")


# Generate test probabilities
X_test = features[model_features]

probabilities = model.predict_proba(X_test)[:, 1]

if not np.isfinite(probabilities).all():
    raise ValueError("Invalid prediction probabilities found.")


# Calculate overall calibration metrics
brier_score = brier_score_loss(y_true, probabilities)

mean_predicted = probabilities.mean()
actual_rate = y_true.mean()

print("\nFINAL TEST CALIBRATION ANALYSIS")
print("=" * 65)

print(f"Test encounters:          {len(y_true)}")
print(f"Mean predicted risk:      {mean_predicted:.4f}")
print(f"Actual readmission rate:  {actual_rate:.4f}")
print(f"Brier score:               {brier_score:.4f}")


# Divide predictions into 10 equal-sized groups
calibration_data = pd.DataFrame(
    {"actual": y_true, "predicted_probability": probabilities}
)

calibration_data["probability_group"] = pd.qcut(
    calibration_data["predicted_probability"], q=10, labels=False, duplicates="drop"
)

calibration_results = (
    calibration_data.groupby("probability_group", observed=True)
    .agg(
        encounters=("actual", "size"),
        mean_predicted_probability=("predicted_probability", "mean"),
        actual_readmission_rate=("actual", "mean"),
    )
    .reset_index()
)


# Calculate the difference between predicted and actual rates
calibration_results["calibration_difference"] = (
    calibration_results["mean_predicted_probability"]
    - calibration_results["actual_readmission_rate"]
)


# Save calibration results
calibration_results.to_csv(TABLE_PATH, index=False)


# Plot calibration curve
plt.figure(figsize=(8, 6))

plt.plot(
    calibration_results["mean_predicted_probability"],
    calibration_results["actual_readmission_rate"],
    marker="o",
    label="LightGBM",
)

plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")

plt.xlabel("Mean predicted probability")
plt.ylabel("Actual readmission rate")
plt.title("LightGBM Calibration - Independent Test Set")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(PLOT_PATH, dpi=300)
plt.close()


# Display results
print("\nCALIBRATION BY PROBABILITY GROUP")
print("-" * 65)
print(calibration_results.round(4).to_string(index=False))

print(f"\nCalibration table saved: {TABLE_PATH}")
print(f"Calibration plot saved: {PLOT_PATH}")

print("\nCalibration analysis completed successfully.")
