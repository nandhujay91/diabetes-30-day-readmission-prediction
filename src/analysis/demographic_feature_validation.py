from pathlib import Path

import pandas as pd

project_path = Path(__file__).resolve().parents[2]

file_path = project_path / "data" / "processed" / "train_demographic.csv"

df = pd.read_csv(file_path)

features = [
    "age_group",
    "gender_group",
    "race_group",
]

for feature in features:
    print(f"\n{feature}")

    print(df[feature].value_counts(dropna=False))

    print("Missing values:", df[feature].isna().sum())

    print("Unique categories:", df[feature].nunique())
