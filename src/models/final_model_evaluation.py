from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[2]

FEATURES_PATH = ROOT / "data/processed/test_features.csv"
TEST_PATH = ROOT / "data/processed/test.csv"
MODEL_PATH = ROOT / "artifacts/lightgbm_tuned_pipeline.joblib"

OUTPUT_DIR = ROOT / "reports/model_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.15


# Load test data and saved model
features = pd.read_csv(FEATURES_PATH)
test = pd.read_csv(TEST_PATH)
model = joblib.load(MODEL_PATH)


# Check test data
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


# Generate test predictions
X_test = features[model_features]

probabilities = model.predict_proba(X_test)[:, 1]

if not np.isfinite(probabilities).all():
    raise ValueError("Invalid prediction probabilities found.")

if not np.isin(y_true, [0, 1]).all():
    raise ValueError("Target must contain only 0 and 1.")


# Apply the selected analytical threshold
y_pred = (probabilities >= THRESHOLD).astype(int)


# Calculate evaluation metrics
pr_auc = average_precision_score(y_true, probabilities)
roc_auc = roc_auc_score(y_true, probabilities)
brier = brier_score_loss(y_true, probabilities)

precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

flagged = int(y_pred.sum())
total = len(y_true)
flagged_rate = flagged / total


# Print evaluation results
print("\nFINAL LIGHTGBM MODEL EVALUATION")
print("=" * 65)

print(f"Test encounters:       {total}")
print(f"Actual readmissions:   {int(y_true.sum())}")
print(f"Readmission rate:      {y_true.mean():.4f}")

print("\nRANKING AND PROBABILITY METRICS")
print("-" * 65)
print(f"PR-AUC:                {pr_auc:.4f}")
print(f"ROC-AUC:               {roc_auc:.4f}")
print(f"Brier score:           {brier:.4f}")

print("\nCLASSIFICATION METRICS")
print("-" * 65)
print(f"Threshold:             {THRESHOLD:.2f}")
print(f"Precision:             {precision:.4f}")
print(f"Recall:                {recall:.4f}")
print(f"F1-score:              {f1:.4f}")

print("\nCONFUSION MATRIX")
print("-" * 65)
print(f"True Negatives:        {tn}")
print(f"False Positives:       {fp}")
print(f"False Negatives:       {fn}")
print(f"True Positives:        {tp}")

print("\nHOSPITAL FOLLOW-UP CAPACITY")
print("-" * 65)
print(f"Flagged encounters:    {flagged}")
print(f"Flagged percentage:    {flagged_rate:.2%}")

print("\nCLASSIFICATION REPORT")
print("-" * 65)
print(
    classification_report(
        y_true, y_pred, target_names=["No readmission", "Readmission"], zero_division=0
    )
)


# Save summary metrics
summary = pd.DataFrame(
    [
        {
            "model": "LightGBM",
            "test_encounters": total,
            "actual_readmissions": int(y_true.sum()),
            "readmission_rate": y_true.mean(),
            "threshold": THRESHOLD,
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "brier_score": brier,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
            "flagged_encounters": flagged,
            "flagged_rate": flagged_rate,
        }
    ]
)

output_path = OUTPUT_DIR / "final_lightgbm_evaluation.csv"

summary.to_csv(output_path, index=False)

print(f"\nResults saved: {output_path}")
print("\nFinal evaluation completed successfully.")
