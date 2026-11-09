from pathlib import Path

import pandas as pd

project_path = Path(__file__).resolve().parents[2]
file_path = project_path / "data" / "processed" / "train_features.csv"

df = pd.read_csv(file_path)

excluded_columns = [
    "encounter_id",
    "patient_nbr",
    "readmitted",
    "readmitted_30_days",
]

review_columns = [
    "discharge_disposition_id",
    "discharge_destination_group",
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "diag_1",
    "diag_2",
    "diag_3",
    "max_glu_serum",
    "A1Cresult",
    "medical_specialty",
    "payer_code",
]

print("\nIDENTIFIERS AND TARGETS")
for column in excluded_columns:
    print(column, ":", column in df.columns)

print("\nFEATURES REQUIRING PREDICTION-TIME REVIEW")
for column in review_columns:
    print(column, ":", column in df.columns)

print("\nTOTAL COLUMNS:", len(df.columns))
print("TOTAL ROWS:", len(df))

print("\nFINAL MODEL INPUT AUDIT COMPLETED")
