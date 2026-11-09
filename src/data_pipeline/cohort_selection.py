import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = PROJECT_ROOT / "data" / "interim" / "diabetes_cleaned.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "diabetes_eligible.csv"

DEATH_DISPOSITIONS = [11, 19, 20, 21]


def load_data():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_PATH}")

    return pd.read_csv(INPUT_PATH, low_memory=False)


def select_eligible_encounters(data):
    data = data.copy()

    data["readmitted_30_days"] = data["readmitted"].map({"NO": 0, ">30": 0, "<30": 1})

    if data["readmitted_30_days"].isna().any():
        raise ValueError("Unexpected readmission values found.")

    eligible_data = data[
        ~data["discharge_disposition_id"].isin(DEATH_DISPOSITIONS)
    ].copy()

    eligible_data["readmitted_30_days"] = eligible_data["readmitted_30_days"].astype(
        "int8"
    )

    return eligible_data


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    data = load_data()
    eligible_data = select_eligible_encounters(data)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    eligible_data.to_csv(OUTPUT_PATH, index=False)

    print("\nCOHORT SELECTION COMPLETED")
    print(f"Original encounters: {len(data)}")
    print(f"Eligible encounters: {len(eligible_data)}")
    print(f"Excluded encounters: {len(data) - len(eligible_data)}")

    print("\nBINARY TARGET DISTRIBUTION")
    print(eligible_data["readmitted_30_days"].value_counts().sort_index())

    print(
        "\n30-day readmission rate:",
        f"{eligible_data['readmitted_30_days'].mean() * 100:.2f}%",
    )

    print(f"\nSaved dataset: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
