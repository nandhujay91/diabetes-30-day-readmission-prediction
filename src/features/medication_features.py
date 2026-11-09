from pathlib import Path

import pandas as pd

medication_columns = [
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
]


def create_medication_features(df):

    df["medication_count"] = (
        df[medication_columns].isin(["Steady", "Up", "Down"]).sum(axis=1)
    )

    df["medication_changed_count"] = (
        df[medication_columns].isin(["Up", "Down"]).sum(axis=1)
    )

    df["medication_up_count"] = (df[medication_columns] == "Up").sum(axis=1)

    df["medication_down_count"] = (df[medication_columns] == "Down").sum(axis=1)

    df["insulin_used"] = (df["insulin"] != "No").astype(int)

    df["medication_regimen_changed"] = (df["change"] == "Ch").astype(int)

    return df


def main():

    project_path = Path(__file__).resolve().parents[2]

    for file in ["train.csv", "test.csv"]:
        input_path = project_path / "data" / "processed" / file
        df = pd.read_csv(input_path)

        df = create_medication_features(df)

        output_file = file.replace(".csv", "_medication.csv")
        output_path = project_path / "data" / "processed" / output_file

        df.to_csv(output_path, index=False)

        print(f"\nProcessed: {file}")
        print(f"Rows: {len(df)}")
        print("Medication features created successfully.")

        print(
            df[
                [
                    "medication_count",
                    "medication_changed_count",
                    "medication_up_count",
                    "medication_down_count",
                    "insulin_used",
                    "medication_regimen_changed",
                ]
            ].head()
        )

        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
