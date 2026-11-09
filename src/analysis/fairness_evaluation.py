from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score

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


# Keep LightGBM predictions
predictions = predictions[predictions["model"].str.lower() == "lightgbm"].copy()

if predictions.empty:
    raise ValueError("No LightGBM predictions found.")

# Check that the data matches
if not (len(features) == len(train) == len(predictions)):
    raise ValueError("The files have different numbers of rows.")

if not np.array_equal(
    predictions["actual"].to_numpy(), train["readmitted_30_days"].to_numpy()
):
    raise ValueError("The actual target values do not match.")

probabilities = predictions["predicted_probability"]

if probabilities.isna().any() or not probabilities.between(0, 1).all():
    raise ValueError("Invalid predicted probabilities found.")

print("Data loaded and validated successfully.")


# Prepare demographic information
groups = {"age_group": "age_group", "gender": "gender_group", "race": "race_group"}

for column in groups.values():
    if column not in features.columns:
        raise ValueError(f"Column not found in training features: {column}")

evaluation = pd.DataFrame(
    {
        "actual": predictions["actual"].to_numpy(),
        "probability": probabilities.to_numpy(),
    }
)

for name, column in groups.items():
    evaluation[name] = features[column].fillna("Unknown").astype(str).to_numpy()

evaluation["predicted"] = (evaluation["probability"] >= threshold).astype(int)


def evaluate_groups(data, group_column):

    results = []

    for group_name, group in data.groupby(group_column):
        if len(group) < min_group_size:
            print(f"Skipping {group_name}: not enough encounters.")
            continue

        actual = group["actual"]
        predicted = group["predicted"]

        tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()

        results.append(
            {
                "attribute": group_column,
                "group": group_name,
                "encounters": len(group),
                "readmissions": int(actual.sum()),
                "readmission_rate": actual.mean(),
                "precision": precision_score(actual, predicted, zero_division=0),
                "recall": recall_score(actual, predicted, zero_division=0),
                "false_positive_rate": fp / (fp + tn) if fp + tn else np.nan,
                "false_negative_rate": fn / (fn + tp) if fn + tp else np.nan,
                "flagged_rate": predicted.mean(),
            }
        )

    return results


# Compare model performance across demographic groups
all_results = []

for attribute in groups:
    print(f"\nFAIRNESS RESULTS: {attribute.upper()}")
    print("-" * 60)

    results = evaluate_groups(evaluation, attribute)
    all_results.extend(results)

    if results:
        print(pd.DataFrame(results).round(4).to_string(index=False))


# Save the results
fairness_results = pd.DataFrame(all_results)

if fairness_results.empty:
    raise ValueError("No fairness results were generated.")

fairness_results.to_csv(output_dir / "lightgbm_fairness_metrics.csv", index=False)


# Create comparison charts
metrics = ["recall", "precision", "false_positive_rate"]

for metric in metrics:
    for attribute in groups:
        data = fairness_results[fairness_results["attribute"] == attribute].sort_values(
            "group"
        )

        plt.figure(figsize=(9, 5))

        plt.bar(data["group"], data[metric])

        plt.title(f"{metric.replace('_', ' ').title()} by {attribute}")
        plt.xlabel(attribute.replace("_", " ").title())
        plt.ylabel(metric.replace("_", " ").title())
        plt.xticks(rotation=45, ha="right")

        plt.tight_layout()

        plt.savefig(
            output_dir / f"lightgbm_fairness_{attribute}_{metric}.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()


print("\nFairness evaluation completed.")
print(f"Threshold: {threshold}")
print(f"Total encounters: {len(evaluation):,}")
print(f"Results saved in: {output_dir}")
