import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "interim" / "diabetes_cleaned.csv"


# These columns identify records or patients.
IDENTIFIER_COLUMNS = [
    "encounter_id",
    "patient_nbr",
]

# These columns contain the outcome we want to predict.
TARGET_COLUMNS = [
    "readmitted",
    "readmitted_30_days",
]


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    return pd.read_csv(DATA_PATH, low_memory=False)


def audit_features(data):
    identifiers = [column for column in IDENTIFIER_COLUMNS if column in data.columns]

    targets = [column for column in TARGET_COLUMNS if column in data.columns]

    features = [
        column for column in data.columns if column not in identifiers + targets
    ]

    print("\nDATASET SUMMARY")
    print(f"Total columns: {len(data.columns)}")
    print(f"Identifier columns: {len(identifiers)}")
    print(f"Target columns: {len(targets)}")
    print(f"Potential feature columns: {len(features)}")

    print("\nIDENTIFIER COLUMNS")
    print(identifiers)

    print("\nTARGET COLUMNS")
    print(targets)

    print("\nPOTENTIAL MODEL FEATURES")
    for column in features:
        print(column)

    return features


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    data = load_data()

    # Recreate the binary target for this audit.
    data["readmitted_30_days"] = (
        data["readmitted"].map({"NO": 0, ">30": 0, "<30": 1}).astype("int8")
    )

    audit_features(data)

    print("\nFEATURE AUDIT COMPLETED")


if __name__ == "__main__":
    main()
