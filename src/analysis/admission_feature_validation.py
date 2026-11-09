from pathlib import Path

import pandas as pd

project_path = Path(__file__).resolve().parents[2]

file_path = project_path / "data" / "processed" / "train_admission.csv"

df = pd.read_csv(file_path)

features = {
    "admission_type_id": "admission_type_group",
    "admission_source_id": "admission_source_group",
    "discharge_disposition_id": "discharge_destination_group",
}

for source, target in features.items():
    print(f"\n{target}")

    print("\nCategory distribution:")
    print(df[target].value_counts(dropna=False))

    print("\nOriginal ID mapping:")
    print(
        df.groupby([source, target], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(source)
        .to_string(index=False)
    )

    print("\nMissing values:", df[target].isna().sum())

    print("Unique groups:", df[target].nunique())
