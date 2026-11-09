import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "interim" / "diabetes_cleaned.csv"


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    return pd.read_csv(DATA_PATH, low_memory=False)


def create_readmission_target(data):
    target_mapping = {
        "NO": 0,
        ">30": 0,
        "<30": 1,
    }

    data = data.copy()

    data["readmitted_30_days"] = data["readmitted"].map(target_mapping)

    if data["readmitted_30_days"].isna().any():
        raise ValueError("Unexpected or missing readmission values found.")

    data["readmitted_30_days"] = data["readmitted_30_days"].astype("int8")

    return data


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    data = load_data()
    data = create_readmission_target(data)

    print("\nBINARY TARGET DISTRIBUTION")

    distribution = (
        data["readmitted_30_days"]
        .value_counts()
        .sort_index()
        .rename_axis("readmitted_30_days")
        .reset_index(name="encounter_count")
    )

    distribution["percentage"] = (
        distribution["encounter_count"] / len(data) * 100
    ).round(2)

    print(distribution.to_string(index=False))

    print("\nORIGINAL TARGET VALUES")
    print(data["readmitted"].value_counts())

    print("\nTARGET CREATION COMPLETED")


if __name__ == "__main__":
    main()
