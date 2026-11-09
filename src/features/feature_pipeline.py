from pathlib import Path

import pandas as pd

from .admission_features import create_admission_features
from .clinical_features import create_clinical_features
from .demographic_features import create_demographic_features
from .diagnosis_features import group_diagnosis
from .hospital_stay_features import create_hospital_stay_features
from .medication_features import create_medication_features
from .visit_history_features import create_visit_features


def create_all_features(df):

    # Diagnosis features
    df["primary_diagnosis_group"] = df["diag_1"].apply(group_diagnosis)
    df["secondary_diagnosis_group"] = df["diag_2"].apply(group_diagnosis)
    df["additional_diagnosis_group"] = df["diag_3"].apply(group_diagnosis)

    # Other feature groups
    df = create_medication_features(df)
    df = create_visit_features(df)
    df = create_admission_features(df)
    df = create_demographic_features(df)
    df = create_hospital_stay_features(df)
    df = create_clinical_features(df)

    return df


def main():

    project_path = Path(__file__).resolve().parents[2]
    input_folder = project_path / "data" / "processed"

    for file in ["train.csv", "test.csv"]:
        input_path = input_folder / file
        df = pd.read_csv(input_path)

        original_rows = len(df)

        df = create_all_features(df)

        output_file = file.replace(".csv", "_features.csv")
        output_path = input_folder / output_file

        df.to_csv(output_path, index=False)

        print(f"\nProcessed: {file}")
        print(f"Original rows: {original_rows}")
        print(f"Final rows: {len(df)}")
        print(f"Total columns: {len(df.columns)}")
        print("All feature groups combined successfully.")
        print(f"Saved: {output_path}")

    print("\nFeature pipeline completed successfully.")


if __name__ == "__main__":
    main()
