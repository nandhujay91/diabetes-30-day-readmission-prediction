from pathlib import Path

import pandas as pd

project_path = Path(__file__).resolve().parents[2]
train_path = project_path / "data" / "processed" / "train_features.csv"

df = pd.read_csv(train_path)

feature_decisions = {
    "identifiers": [
        "encounter_id",
        "patient_nbr",
    ],
    "target": [
        "readmitted",
        "readmitted_30_days",
    ],
    "historical_features": [
        "number_inpatient",
        "number_outpatient",
        "number_emergency",
        "total_previous_visits",
        "has_previous_inpatient_visit",
        "has_previous_outpatient_visit",
        "has_previous_emergency_visit",
    ],
    "prediction_time_review": [
        "discharge_disposition_id",
        "discharge_destination_group",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "diag_1",
        "diag_2",
        "diag_3",
        "max_glu_serum",
        "A1Cresult",
    ],
    "fairness_review": [
        "race",
        "race_group",
        "gender",
        "gender_group",
    ],
}

print("\nMODEL FEATURE DECISIONS")

for group, features in feature_decisions.items():
    available = [feature for feature in features if feature in df.columns]

    print(f"\n{group.upper()}")
    for feature in available:
        print("-", feature)

    missing = set(features) - set(available)

    if missing:
        print("Not found:", sorted(missing))

print("\nFEATURE DECISION REVIEW COMPLETED")
