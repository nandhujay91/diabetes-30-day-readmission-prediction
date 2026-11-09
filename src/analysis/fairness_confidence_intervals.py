from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

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


def get_confidence_interval(successes, total):

    if total == 0:
        return np.nan, np.nan

    return proportion_confint(count=successes, nobs=total, alpha=0.05, method="wilson")


# Keep LightGBM predictions
predictions = predictions[predictions["model"].str.lower() == "lightgbm"].copy()

if predictions.empty:
    raise ValueError("No LightGBM predictions found.")

# Check that the data matches
if not (len(features) == len(train) == len(predictions)):
    raise ValueError("The files have different row counts.")

if not np.array_equal(
    predictions["actual"].to_numpy(), train["readmitted_30_days"].to_numpy()
):
    raise ValueError("Actual target values do not match.")

probabilities = predictions["predicted_probability"]

if probabilities.isna().any() or not probabilities.between(0, 1).all():
    raise ValueError("Invalid predicted probabilities found.")

# Prepare the evaluation data
evaluation = pd.DataFrame(
    {
        "actual": predictions["actual"].to_numpy(),
        "probability": probabilities.to_numpy(),
    }
)

evaluation["predicted"] = (evaluation["probability"] >= threshold).astype(int)

demographic_groups = {
    "age_group": "age_group",
    "gender": "gender_group",
    "race": "race_group",
}

for name, column in demographic_groups.items():
    if column not in features.columns:
        raise ValueError(f"Missing feature: {column}")

    evaluation[name] = features[column].fillna("Unknown").astype(str).to_numpy()


# Calculate metrics for each demographic group
results = []

for attribute in demographic_groups:
    for category, group in evaluation.groupby(attribute):
        if len(group) < min_group_size:
            continue

        actual = group["actual"]
        predicted = group["predicted"]

        tp = int(((actual == 1) & (predicted == 1)).sum())
        fp = int(((actual == 0) & (predicted == 1)).sum())
        tn = int(((actual == 0) & (predicted == 0)).sum())
        fn = int(((actual == 1) & (predicted == 0)).sum())

        total = len(group)

        metrics = {
            "readmission_rate": (int(actual.sum()), total),
            "recall": (tp, tp + fn),
            "false_positive_rate": (fp, fp + tn),
            "precision": (tp, tp + fp),
        }

        row = {"attribute": attribute, "group": category, "encounters": total}

        for metric, (successes, denominator) in metrics.items():
            if denominator == 0:
                estimate = np.nan
                lower = np.nan
                upper = np.nan
            else:
                estimate = successes / denominator

                lower, upper = get_confidence_interval(successes, denominator)

            row[metric] = estimate
            row[f"{metric}_lower_95"] = lower
            row[f"{metric}_upper_95"] = upper

        results.append(row)


# Save the results
results_df = pd.DataFrame(results)

if results_df.empty:
    raise ValueError("No fairness results were generated.")

output_path = output_dir / "lightgbm_fairness_confidence_intervals.csv"

results_df.to_csv(output_path, index=False)

print("\nFAIRNESS RESULTS WITH 95% CONFIDENCE INTERVALS")
print("-" * 65)
print(results_df.round(4).to_string(index=False))

print(f"\nResults saved: {output_path}")
print("\nAnalysis completed successfully.")
