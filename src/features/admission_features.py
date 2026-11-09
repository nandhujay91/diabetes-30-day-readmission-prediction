from pathlib import Path

import pandas as pd


def create_admission_features(df):

    admission_type = {
        1: "Emergency_or_Urgent",
        2: "Emergency_or_Urgent",
        3: "Elective",
        4: "Newborn",
        5: "Other_or_Unknown",
        6: "Other_or_Unknown",
        7: "Trauma_Center",
        8: "Other_or_Unknown",
    }

    admission_source = {
        1: "Referral",
        2: "Referral",
        3: "Referral",
        4: "Transfer",
        5: "Transfer",
        6: "Transfer",
        7: "Emergency_Room",
        8: "Other",
        9: "Unknown",
        10: "Transfer",
        11: "Newborn_Birth",
        14: "Newborn_Birth",
        17: "Unmapped_ID_17",
        18: "Transfer",
        19: "Transfer",
        20: "Other_or_Unknown",
        22: "Transfer",
        25: "Transfer",
    }

    discharge_destination = {
        1: "Home",
        2: "Hospital",
        3: "Care_Facility",
        4: "Care_Facility",
        5: "Care_Facility",
        6: "Home_Health",
        7: "Against_Medical_Advice",
        8: "Home_Health",
        9: "Hospital",
        10: "Other",
        11: "Expired",
        12: "Other",
        13: "Hospice",
        14: "Hospice",
        15: "Care_Facility",
        16: "Other",
        17: "Other",
        18: "Unknown",
        19: "Expired",
        20: "Expired",
        21: "Expired",
        22: "Care_Facility",
        23: "Care_Facility",
        24: "Care_Facility",
        25: "Unknown",
        26: "Unknown",
        27: "Care_Facility",
        28: "Care_Facility",
        29: "Hospital",
        30: "Care_Facility",
    }

    df["admission_type_group"] = (
        df["admission_type_id"].map(admission_type).fillna("Unknown")
    )

    df["admission_source_group"] = (
        df["admission_source_id"].map(admission_source).fillna("Unknown")
    )

    df["discharge_destination_group"] = (
        df["discharge_disposition_id"].map(discharge_destination).fillna("Unknown")
    )

    return df


def main():

    project_path = Path(__file__).resolve().parents[2]

    for file in ["train.csv", "test.csv"]:
        input_path = project_path / "data" / "processed" / file
        df = pd.read_csv(input_path)

        df = create_admission_features(df)

        output_file = file.replace(".csv", "_admission.csv")
        output_path = project_path / "data" / "processed" / output_file

        df.to_csv(output_path, index=False)

        print(f"\nProcessed: {file}")
        print(f"Rows: {len(df)}")
        print("Admission features created successfully.")
        print(
            df[
                [
                    "admission_type_group",
                    "admission_source_group",
                    "discharge_destination_group",
                ]
            ].head()
        )

        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
