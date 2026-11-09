from pathlib import Path

import pandas as pd

project_path = Path(__file__).resolve().parents[2]
data_folder = project_path / "data" / "processed"

train = pd.read_csv(data_folder / "train_features.csv")
test = pd.read_csv(data_folder / "test_features.csv")

print("\nDATASET SHAPES")
print("Train:", train.shape)
print("Test:", test.shape)

print("\nDUPLICATE COLUMNS")
print("Train:", train.columns.duplicated().sum())
print("Test:", test.columns.duplicated().sum())

print("\nMISSING VALUES")
print("Train:", train.isna().sum().sum())
print("Test:", test.isna().sum().sum())

print("\nDUPLICATE ENCOUNTERS")
print("Train:", train["encounter_id"].duplicated().sum())
print("Test:", test["encounter_id"].duplicated().sum())

print("\nTARGET DISTRIBUTION")
print(train["readmitted_30_days"].value_counts())
print(test["readmitted_30_days"].value_counts())

print("\nENGINEERED FEATURES")

engineered_features = [
    "primary_diagnosis_group",
    "secondary_diagnosis_group",
    "additional_diagnosis_group",
    "medication_count",
    "medication_changed_count",
    "medication_up_count",
    "medication_down_count",
    "insulin_used",
    "medication_regimen_changed",
    "total_previous_visits",
    "has_previous_inpatient_visit",
    "has_previous_outpatient_visit",
    "has_previous_emergency_visit",
    "admission_type_group",
    "admission_source_group",
    "discharge_destination_group",
    "age_group",
    "gender_group",
    "race_group",
    "lab_procedures_per_day",
    "medications_per_day",
    "procedures_per_day",
    "long_hospital_stay",
    "glucose_test_result",
    "glucose_test_available",
    "hba1c_test_result",
    "hba1c_test_available",
]

for feature in engineered_features:
    print(feature, ":", feature in train.columns)

print("\nVALIDATION COMPLETED")
