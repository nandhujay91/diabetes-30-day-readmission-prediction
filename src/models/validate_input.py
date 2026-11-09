import numpy as np
import pandas as pd

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

REQUIRED_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


def validate_input(data):

    if isinstance(data, dict):
        data = pd.DataFrame([data])

    elif isinstance(data, list):
        data = pd.DataFrame(data)

    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "Input must be a dictionary, list of dictionaries, or DataFrame."
        )

    if data.empty:
        raise ValueError("Input data is empty.")

    missing = [column for column in REQUIRED_FEATURES if column not in data.columns]

    if missing:
        raise ValueError(f"Missing required features: {missing}")

    data = data[REQUIRED_FEATURES].copy()

    for column in NUMERICAL_FEATURES:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

        if data[column].isna().any():
            raise ValueError(f"Invalid or missing numerical values in: {column}")

        if not np.isfinite(data[column]).all():
            raise ValueError(f"Non-finite values found in: {column}")

    for column in CATEGORICAL_FEATURES:
        if data[column].isna().any():
            raise ValueError(f"Missing categorical values in: {column}")

        data[column] = data[column].astype(str)

    return data
