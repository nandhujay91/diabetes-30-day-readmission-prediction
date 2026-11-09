from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "diabetes_eligible.csv"


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    return pd.read_csv(DATA_PATH, low_memory=False)


def analyse_patients(data):
    encounters_per_patient = data.groupby("patient_nbr").size()

    total_encounters = len(data)
    unique_patients = data["patient_nbr"].nunique()

    patients_with_multiple_encounters = (encounters_per_patient > 1).sum()

    patients_with_one_encounters = (encounters_per_patient == 1).sum()

    print("\nELIGIBLE COHORT SUMMARY")
    print(f"Total encounters: {total_encounters}")
    print(f"Unique patients: {unique_patients}")
    print(f"Patients with one encounter: {patients_with_one_encounters}")
    print(
        "Patients with multiple encounters:",
        patients_with_multiple_encounters,
    )
    print(
        "Maximum encounters per patient:",
        encounters_per_patient.max(),
    )

    print("\nENCOUNTERS PER PATIENT DISTRIBUTION")

    distribution = (
        encounters_per_patient.value_counts()
        .sort_index()
        .rename_axis("number_of_encounters")
        .reset_index(name="number_of_patients")
    )

    print(distribution.to_string(index=False))


def main():
    data = load_data()
    analyse_patients(data)


if __name__ == "__main__":
    main()
