from pathlib import Path

import pandas as pd


def create_clinical_features(df):

    # Glucose test features
    df["glucose_test_result"] = df["max_glu_serum"].fillna("Not_Tested")

    df["glucose_test_available"] = (df["max_glu_serum"].notna()).astype(int)

    # HbA1c test features
    df["hba1c_test_result"] = df["A1Cresult"].fillna("Not_Tested")

    df["hba1c_test_available"] = (df["A1Cresult"].notna()).astype(int)

    return df


def main():

    project_path = Path(__file__).resolve().parents[2]

    for file in ["train.csv", "test.csv"]:
        input_path = project_path / "data" / "processed" / file
        df = pd.read_csv(input_path)

        df = create_clinical_features(df)

        output_file = file.replace(".csv", "_clinical.csv")
        output_path = project_path / "data" / "processed" / output_file

        df.to_csv(output_path, index=False)

        print(f"\nProcessed: {file}")
        print(f"Rows: {len(df)}")
        print("Clinical features created successfully.")

        print(
            df[
                [
                    "glucose_test_result",
                    "glucose_test_available",
                    "hba1c_test_result",
                    "hba1c_test_available",
                ]
            ].head()
        )

        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
