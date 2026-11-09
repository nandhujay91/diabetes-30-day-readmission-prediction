from pathlib import Path

import pandas as pd

project_path = Path(__file__).resolve().parents[2]
file_path = project_path / "data" / "processed" / "train_diagnosis.csv"

df = pd.read_csv(file_path)

diagnosis_features = [
    "primary_diagnosis_group",
    "secondary_diagnosis_group",
    "additional_diagnosis_group",
]

for feature in diagnosis_features:
    print(f"\n{feature}")
    print(df[feature].value_counts(dropna=False))

    print("Missing values:", df[feature].isna().sum())
    print("Unique groups:", df[feature].nunique())
