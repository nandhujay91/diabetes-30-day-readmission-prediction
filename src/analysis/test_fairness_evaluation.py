from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

FEATURES_PATH = ROOT / "data/processed/test_features.csv"
TEST_PATH = ROOT / "data/processed/test.csv"
MODEL_PATH = ROOT / "artifacts/lightgbm_tuned_pipeline.joblib"

OUTPUT_DIR = ROOT / "reports/model_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.15
BOOTSTRAP_ITERATIONS = 500
RANDOM_SEED = 42


# Load test data and the saved model
features = pd.read_csv(FEATURES_PATH)
test = pd.read_csv(TEST_PATH)
model = joblib.load(MODEL_PATH)

if len(features) != len(test):
    raise ValueError("Test files have different row counts.")

if not np.array_equal(
    features["encounter_id"].to_numpy(), test["encounter_id"].to_numpy()
):
    raise ValueError("Encounter order does not match.")

if not np.array_equal(
    features["readmitted_30_days"].to_numpy(), test["readmitted_30_days"].to_numpy()
):
    raise ValueError("Target values do not match.")


# Select the model features
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


# Generate predictions on the independent test set
X_test = features[model_features]

probabilities = model.predict_proba(X_test)[:, 1]

evaluation = pd.DataFrame(
    {
        "patient_nbr": test["patient_nbr"].to_numpy(),
        "actual": test["readmitted_30_days"].to_numpy(),
        "probability": probabilities,
        "age_group": features["age_group"].fillna("Unknown").astype(str),
        "gender": features["gender_group"].fillna("Unknown").astype(str),
        "race": features["race_group"].fillna("Unknown").astype(str),
    }
)

evaluation["predicted"] = (evaluation["probability"] >= THRESHOLD).astype(int)


def calculate_metrics(group):

    actual = group["actual"]
    predicted = group["predicted"]

    tp = int(((actual == 1) & (predicted == 1)).sum())
    fp = int(((actual == 0) & (predicted == 1)).sum())
    tn = int(((actual == 0) & (predicted == 0)).sum())
    fn = int(((actual == 1) & (predicted == 0)).sum())

    total = len(group)

    return {
        "readmission_rate": actual.sum() / total,
        "recall": tp / (tp + fn) if tp + fn else np.nan,
        "false_positive_rate": fp / (fp + tn) if fp + tn else np.nan,
        "precision": tp / (tp + fp) if tp + fp else np.nan,
    }


# Calculate fairness metrics for each demographic group
demographic_groups = ["age_group", "gender", "race"]

original_results = []

for attribute in demographic_groups:
    for category, group in evaluation.groupby(attribute):
        metrics = calculate_metrics(group)

        original_results.append(
            {
                "attribute": attribute,
                "group": category,
                "encounters": len(group),
                "unique_patients": group["patient_nbr"].nunique(),
                **metrics,
            }
        )


# Prepare patient-level bootstrap
patient_rows = {
    patient_id: positions
    for patient_id, positions in evaluation.groupby("patient_nbr").indices.items()
}

patient_ids = np.array(list(patient_rows))

rng = np.random.default_rng(RANDOM_SEED)

bootstrap_results = []

for iteration in range(BOOTSTRAP_ITERATIONS):
    sampled_patients = rng.choice(patient_ids, size=len(patient_ids), replace=True)

    sampled_rows = np.concatenate(
        [patient_rows[patient_id] for patient_id in sampled_patients]
    )

    sample = evaluation.iloc[sampled_rows]

    for attribute in demographic_groups:
        for category, group in sample.groupby(attribute):
            metrics = calculate_metrics(group)

            bootstrap_results.append(
                {
                    "iteration": iteration,
                    "attribute": attribute,
                    "group": category,
                    **metrics,
                }
            )


# Add patient-level 95% confidence intervals
bootstrap_df = pd.DataFrame(bootstrap_results)

results = []

for original in original_results:
    row = original.copy()

    group_results = bootstrap_df[
        (bootstrap_df["attribute"] == original["attribute"])
        & (bootstrap_df["group"] == original["group"])
    ]

    for metric in ["readmission_rate", "recall", "false_positive_rate", "precision"]:
        values = group_results[metric].dropna()

        if len(values) < BOOTSTRAP_ITERATIONS * 0.8:
            lower = np.nan
            upper = np.nan
        else:
            lower = values.quantile(0.025)
            upper = values.quantile(0.975)

        row[f"{metric}_lower_95"] = lower
        row[f"{metric}_upper_95"] = upper

    results.append(row)


# Save test fairness results
results_df = pd.DataFrame(results)

output_path = OUTPUT_DIR / "lightgbm_test_fairness_metrics.csv"

results_df.to_csv(output_path, index=False)

print("\nINDEPENDENT TEST-SET FAIRNESS RESULTS")
print("-" * 70)
print(f"Test encounters: {len(evaluation)}")
print(f"Unique patients: {len(patient_ids)}")
print(f"Readmission rate: {evaluation['actual'].mean():.4f}")
print(f"Threshold: {THRESHOLD}")
print(f"Bootstrap iterations: {BOOTSTRAP_ITERATIONS}")

print("\n")
print(results_df.round(4).to_string(index=False))

print(f"\nResults saved: {output_path}")
print("\nAnalysis completed successfully.")
