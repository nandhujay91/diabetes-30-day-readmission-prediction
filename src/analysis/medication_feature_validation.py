from pathlib import Path

import pandas as pd

project_path = Path(__file__).resolve().parents[2]

file_path = project_path / "data" / "processed" / "train_medication.csv"

df = pd.read_csv(file_path)

features = [
    "medication_count",
    "medication_changed_count",
    "medication_up_count",
    "medication_down_count",
    "insulin_used",
    "medication_regimen_changed",
]

for feature in features:
    print(f"\n{feature}")
    print(df[feature].value_counts().sort_index())
    print("Missing values:", df[feature].isna().sum())

print("\nMedication feature summary:")
print(df[features].describe())
