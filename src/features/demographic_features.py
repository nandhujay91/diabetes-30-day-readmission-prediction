from pathlib import Path

import pandas as pd


def create_demographic_features(df):

    df["age_group"] = df["age"].fillna("Unknown")

    df["gender_group"] = (
        df["gender"].replace({"Unknown/Invalid": "Unknown"}).fillna("Unknown")
    )

    df["race_group"] = df["race"].fillna("Unknown")

    return df


def main():

    project_path = Path(__file__).resolve().parents[2]

    for file in ["train.csv", "test.csv"]:
        input_path = project_path / "data" / "processed" / file
        df = pd.read_csv(input_path)

        df = create_demographic_features(df)

        output_file = file.replace(".csv", "_demographic.csv")
        output_path = project_path / "data" / "processed" / output_file

        df.to_csv(output_path, index=False)

        print(f"\nProcessed: {file}")
        print(f"Rows: {len(df)}")
        print("Demographic features created successfully.")

        print(df[["age_group", "gender_group", "race_group"]].head())

        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
