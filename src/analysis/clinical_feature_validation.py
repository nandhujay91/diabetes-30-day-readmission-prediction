from pathlib import Path

import pandas as pd

project_path = Path(__file__).resolve().parents[2]

file_path = project_path / "data" / "processed" / "train_clinical.csv"

df = pd.read_csv(file_path)

features = [
    "glucose_test_result",
    "glucose_test_available",
    "hba1c_test_result",
    "hba1c_test_available",
]

for feature in features:
    print(f"\n{feature}")
    print(df[feature].value_counts(dropna=False))
    print("Missing values:", df[feature].isna().sum())
