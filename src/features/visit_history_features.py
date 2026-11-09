from pathlib import Path

import pandas as pd


def create_visit_features(df):

    df["total_previous_visits"] = (
        df["number_inpatient"] + df["number_outpatient"] + df["number_emergency"]
    )

    df["has_previous_inpatient_visit"] = (df["number_inpatient"] > 0).astype(int)

    df["has_previous_outpatient_visit"] = (df["number_outpatient"] > 0).astype(int)

    df["has_previous_emergency_visit"] = (df["number_emergency"] > 0).astype(int)

    return df


def main():

    project_path = Path(__file__).resolve().parents[2]

    for file in ["train.csv", "test.csv"]:
        input_path = project_path / "data" / "processed" / file
        df = pd.read_csv(input_path)

        df = create_visit_features(df)

        output_file = file.replace(".csv", "_visits.csv")
        output_path = project_path / "data" / "processed" / output_file

        df.to_csv(output_path, index=False)

        print(f"\nProcessed: {file}")
        print(f"Rows: {len(df)}")
        print("Visit history features created successfully.")

        print(
            df[
                [
                    "total_previous_visits",
                    "has_previous_inpatient_visit",
                    "has_previous_outpatient_visit",
                    "has_previous_emergency_visit",
                ]
            ].head()
        )

        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
