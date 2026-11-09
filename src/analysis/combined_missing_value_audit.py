from pathlib import Path

import pandas as pd

project_path = Path(__file__).resolve().parents[2]
data_folder = project_path / "data" / "processed"

train = pd.read_csv(data_folder / "train_features.csv")
test = pd.read_csv(data_folder / "test_features.csv")


for name, df in [("TRAIN", train), ("TEST", test)]:
    print(f"\n{'=' * 50}")
    print(f"{name} MISSING VALUE REPORT")
    print("=" * 50)

    missing = df.isna().sum()
    missing = missing[missing > 0]

    report = pd.DataFrame(
        {
            "missing_count": missing,
            "missing_percentage": (missing / len(df) * 100).round(2),
        }
    )

    print(report.sort_values("missing_count", ascending=False))

    print("\nTotal missing cells:", df.isna().sum().sum())
    print("Columns with missing values:", len(missing))


print("\nMISSING VALUE AUDIT COMPLETED")
