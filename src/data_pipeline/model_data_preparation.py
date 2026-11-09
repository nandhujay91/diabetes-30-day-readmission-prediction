from pathlib import Path

import pandas as pd


def prepare_model_data(df):

    numerical_features = [
        "time_in_hospital",
        "lab_procedures_per_day",
        "medications_per_day",
        "procedures_per_day",
        "long_hospital_stay",
        "number_inpatient",
        "number_outpatient",
        "number_emergency",
        "medication_count",
        "medication_changed_count",
        "medication_up_count",
        "medication_down_count",
        "insulin_used",
        "medication_regimen_changed",
    ]

    categorical_features = [
        "age_group",
        "admission_type_group",
        "admission_source_group",
        "discharge_destination_group",
        "primary_diagnosis_group",
        "secondary_diagnosis_group",
        "additional_diagnosis_group",
        "glucose_test_result",
        "hba1c_test_result",
        "medical_specialty",
        "payer_code",
    ]

    target = "readmitted_30_days"

    feature_columns = numerical_features + categorical_features

    missing_columns = [
        column for column in feature_columns + [target] if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    X = df[feature_columns].copy()
    y = df[target].copy()

    return X, y, numerical_features, categorical_features


def main():

    project_path = Path(__file__).resolve().parents[2]
    data_folder = project_path / "data" / "processed"

    for file in ["train_features.csv", "test_features.csv"]:
        df = pd.read_csv(data_folder / file)

        X, y, numerical, categorical = prepare_model_data(df)

        output_name = file.replace("_features.csv", "_model_input.csv")

        X["readmitted_30_days"] = y

        output_path = data_folder / output_name
        X.to_csv(output_path, index=False)

        print(f"\nProcessed: {file}")
        print("Rows:", len(X))
        print("Predictor columns:", len(numerical) + len(categorical))
        print("Numerical features:", len(numerical))
        print("Categorical features:", len(categorical))
        print("Target distribution:")
        print(y.value_counts().sort_index())
        print("Saved:", output_path)


if __name__ == "__main__":
    main()
