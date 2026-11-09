from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

features = pd.read_csv(ROOT / "data/processed/train_features.csv")
train = pd.read_csv(ROOT / "data/processed/train.csv")
predictions = pd.read_csv(
    ROOT / "reports/model_evaluation/tuned_model_oof_predictions.csv"
)

output_dir = ROOT / "reports/model_evaluation"
output_dir.mkdir(parents=True, exist_ok=True)

threshold = 0.15
min_group_size = 30
bootstrap_iterations = 500
random_seed = 42


# Keep LightGBM predictions
predictions = predictions[predictions["model"].str.lower() == "lightgbm"].copy()

if predictions.empty:
    raise ValueError("No LightGBM predictions found.")

if not (len(features) == len(train) == len(predictions)):
    raise ValueError("The files have different row counts.")

if not np.array_equal(
    predictions["actual"].to_numpy(), train["readmitted_30_days"].to_numpy()
):
    raise ValueError("Target values do not match.")

if train["patient_nbr"].isna().any():
    raise ValueError("Missing patient IDs found.")

probabilities = predictions["predicted_probability"]

if probabilities.isna().any() or not probabilities.between(0, 1).all():
    raise ValueError("Invalid predicted probabilities found.")


# Prepare evaluation data
evaluation = pd.DataFrame(
    {
        "patient_nbr": train["patient_nbr"].to_numpy(),
        "actual": predictions["actual"].to_numpy(),
        "probability": probabilities.to_numpy(),
        "age_group": features["age_group"].fillna("Unknown").astype(str),
        "gender": features["gender_group"].fillna("Unknown").astype(str),
        "race": features["race_group"].fillna("Unknown").astype(str),
    }
)

evaluation["predicted"] = (evaluation["probability"] >= threshold).astype(int)


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


# Calculate original group results
demographic_groups = ["age_group", "gender", "race"]

original_results = []

for attribute in demographic_groups:
    for category, group in evaluation.groupby(attribute):
        if len(group) < min_group_size:
            continue

        metrics = calculate_metrics(group)

        original_results.append(
            {
                "attribute": attribute,
                "group": category,
                "encounters": len(group),
                **metrics,
            }
        )


# Prepare patient-level sampling
patient_ids = evaluation["patient_nbr"].unique()

patient_rows = {
    patient_id: positions
    for patient_id, positions in evaluation.groupby("patient_nbr").indices.items()
}

rng = np.random.default_rng(random_seed)

bootstrap_results = []

# Resample patients and retain all their encounters
for iteration in range(bootstrap_iterations):
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


# Calculate 95% bootstrap confidence intervals
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

        if len(values) < bootstrap_iterations * 0.8:
            lower = np.nan
            upper = np.nan
        else:
            lower = values.quantile(0.025)
            upper = values.quantile(0.975)

        row[f"{metric}_lower_95"] = lower
        row[f"{metric}_upper_95"] = upper

    results.append(row)


# Save results
results_df = pd.DataFrame(results)

output_path = output_dir / "lightgbm_patient_bootstrap_confidence_intervals.csv"

results_df.to_csv(output_path, index=False)

print("\nPATIENT-LEVEL BOOTSTRAP RESULTS")
print("-" * 70)
print(f"Bootstrap iterations: {bootstrap_iterations}")
print(f"Unique patients: {len(patient_ids)}")
print(f"Threshold: {threshold}")

print("\n")
print(results_df.round(4).to_string(index=False))

print(f"\nResults saved: {output_path}")
print("\nAnalysis completed successfully.")
