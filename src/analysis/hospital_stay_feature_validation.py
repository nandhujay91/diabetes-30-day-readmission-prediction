from pathlib import Path

import pandas as pd

project_path = Path(__file__).resolve().parents[2]

file_path = project_path / "data" / "processed" / "train_hospital_stay.csv"

df = pd.read_csv(file_path)

features = [
    "lab_procedures_per_day",
    "medications_per_day",
    "procedures_per_day",
    "long_hospital_stay",
]

for feature in features:
    print(f"\n{feature}")

    print(df[feature].describe())

    print("Missing values:", df[feature].isna().sum())

print("\nLong hospital stay distribution:")
print(df["long_hospital_stay"].value_counts().sort_index())
