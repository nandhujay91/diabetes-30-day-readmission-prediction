from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERICAL_FEATURES = [
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

CATEGORICAL_FEATURES = [
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


def build_preprocessor():

    numerical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("numerical", numerical_pipeline, NUMERICAL_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    return preprocessor


def main():

    project_path = Path(__file__).resolve().parents[2]
    data_folder = project_path / "data" / "processed"

    train = pd.read_csv(data_folder / "train_model_input.csv")
    test = pd.read_csv(data_folder / "test_model_input.csv")

    X_train = train[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    X_test = test[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]

    preprocessor = build_preprocessor()

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print("\nPREPROCESSING COMPLETED")
    print("Training shape:", X_train_processed.shape)
    print("Testing shape:", X_test_processed.shape)
    print("Preprocessing fitted on training data only.")


if __name__ == "__main__":
    main()
