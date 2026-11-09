from pathlib import Path

import pandas as pd


def create_hospital_stay_features(df):

    df["lab_procedures_per_day"] = (
        df["num_lab_procedures"] / df["time_in_hospital"]
    ).round(2)

    df["medications_per_day"] = (df["num_medications"] / df["time_in_hospital"]).round(
        2
    )

    df["procedures_per_day"] = (df["num_procedures"] / df["time_in_hospital"]).round(2)

    df["long_hospital_stay"] = (df["time_in_hospital"] > 7).astype(int)

    return df


def main():

    project_path = Path(__file__).resolve().parents[2]

    for file in ["train.csv", "test.csv"]:
        input_path = project_path / "data" / "processed" / file
        df = pd.read_csv(input_path)

        df = create_hospital_stay_features(df)

        output_file = file.replace(".csv", "_hospital_stay.csv")
        output_path = project_path / "data" / "processed" / output_file

        df.to_csv(output_path, index=False)

        print(f"\nProcessed: {file}")
        print(f"Rows: {len(df)}")
        print("Hospital stay features created successfully.")

        print(
            df[
                [
                    "lab_procedures_per_day",
                    "medications_per_day",
                    "procedures_per_day",
                    "long_hospital_stay",
                ]
            ].head()
        )

        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
