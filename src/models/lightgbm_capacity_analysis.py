from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports" / "model_evaluation"

PREDICTION_FILE = REPORTS / "tuned_model_oof_predictions.csv"


# Load LightGBM out-of-fold predictions
df = pd.read_csv(PREDICTION_FILE)

df = df[df["model"] == "LightGBM"].copy()

df = df.sort_values(
    "predicted_probability",
    ascending=False,
).reset_index(drop=True)

total_patients = len(df)
total_readmissions = df["actual"].sum()

# Follow-up capacity scenarios
capacities = [5, 10, 15, 20, 25, 30]

results = []

for capacity in capacities:
    # Select the highest-risk patients
    number_selected = round(total_patients * capacity / 100)

    selected = df.head(number_selected)

    true_positives = selected["actual"].sum()
    false_positives = number_selected - true_positives
    false_negatives = total_readmissions - true_positives

    precision = true_positives / number_selected if number_selected > 0 else 0

    recall = true_positives / total_readmissions if total_readmissions > 0 else 0

    results.append(
        {
            "follow_up_capacity": capacity,
            "patients_selected": number_selected,
            "precision": precision,
            "recall": recall,
            "true_positives": int(true_positives),
            "false_positives": int(false_positives),
            "false_negatives": int(false_negatives),
            "flagged_percentage": number_selected / total_patients * 100,
        }
    )


# Save capacity analysis
capacity_results = pd.DataFrame(results)

capacity_results.to_csv(
    REPORTS / "lightgbm_capacity_analysis.csv",
    index=False,
)


# Display results
print("\nLIGHTGBM CAPACITY ANALYSIS")

print("\nTOTAL ENCOUNTERS:", total_patients)
print("ACTUAL READMISSIONS:", total_readmissions)
print(
    "READMISSION RATE:",
    round(total_readmissions / total_patients * 100, 2),
    "%",
)

print("\nCAPACITY RESULTS")

print(capacity_results.round(4).to_string(index=False))

print("\nCapacity analysis completed.")
print("Results saved successfully.")
