from pathlib import Path

import pandas as pd

project_path = Path(__file__).resolve().parents[2]

file_path = project_path / "data" / "processed" / "train_visits.csv"

df = pd.read_csv(file_path)

features = [
    "total_previous_visits",
    "has_previous_inpatient_visit",
    "has_previous_outpatient_visit",
    "has_previous_emergency_visit",
]

for feature in features:
    print(f"\n{feature}")
    print(df[feature].value_counts().sort_index())
    print("Missing values:", df[feature].isna().sum())

print("\nVisit history summary:")
print(df[features].describe())
