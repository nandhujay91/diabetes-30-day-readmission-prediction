import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "diabetes_eligible.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed"

TARGET_COLUMN = "readmitted_30_days"
GROUP_COLUMN = "patient_nbr"


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    return pd.read_csv(DATA_PATH, low_memory=False)


def split_data(data):
    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    X = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN]
    groups = data[GROUP_COLUMN]

    train_indices, test_indices = next(splitter.split(X, y, groups))

    train_data = data.iloc[train_indices].copy()
    test_data = data.iloc[test_indices].copy()

    return train_data, test_data


def validate_split(train_data, test_data):
    train_patients = set(train_data[GROUP_COLUMN])
    test_patients = set(test_data[GROUP_COLUMN])

    overlapping_patients = train_patients.intersection(test_patients)

    if overlapping_patients:
        raise ValueError("Patient leakage detected between train and test.")

    if set(train_data.index).intersection(test_data.index):
        raise ValueError("Encounter overlap detected between train and test.")

    logger.info("Patient-level split validation passed.")


def print_split_summary(train_data, test_data):
    print("\nTRAINING DATA")
    print(f"Encounters: {len(train_data)}")
    print(f"Unique patients: {train_data[GROUP_COLUMN].nunique()}")
    print(
        "30-day readmission rate:",
        f"{train_data[TARGET_COLUMN].mean() * 100:.2f}%",
    )

    print("\nTEST DATA")
    print(f"Encounters: {len(test_data)}")
    print(f"Unique patients: {test_data[GROUP_COLUMN].nunique()}")
    print(
        "30-day readmission rate:",
        f"{test_data[TARGET_COLUMN].mean() * 100:.2f}%",
    )

    print("\nSPLIT SUMMARY")
    print(
        f"Training percentage: {len(train_data) / (len(train_data) + len(test_data)) * 100:.2f}%"
    )
    print(
        f"Testing percentage: {len(test_data) / (len(train_data) + len(test_data)) * 100:.2f}%"
    )


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    data = load_data()

    train_data, test_data = split_data(data)

    validate_split(train_data, test_data)

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    train_data.to_csv(OUTPUT_PATH / "train.csv", index=False)
    test_data.to_csv(OUTPUT_PATH / "test.csv", index=False)

    print_split_summary(train_data, test_data)

    print("\nTRAIN AND TEST DATA SAVED SUCCESSFULLY")


if __name__ == "__main__":
    main()
