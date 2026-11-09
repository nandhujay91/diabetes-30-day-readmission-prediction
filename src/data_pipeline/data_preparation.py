"""Data preparation workflow for the diabetes readmission project."""

import logging
from pathlib import Path

from src.data_pipeline.data_cleaning import clean_diabetes_data
from src.data_pipeline.data_loader import load_diabetes_data

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "data" / "interim" / "diabetes_cleaned.csv"


def prepare_diabetes_dataset():
    """Load and clean the original diabetes dataset."""

    logger.info("Starting dataset preparation.")

    raw_data = load_diabetes_data()
    cleaned_data = clean_diabetes_data(raw_data)

    logger.info(
        "Prepared dataset: %s rows, %s columns",
        cleaned_data.shape[0],
        cleaned_data.shape[1],
    )

    return cleaned_data


def main():
    """Run the preparation workflow and save the interim dataset."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    prepared_data = prepare_diabetes_dataset()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    prepared_data.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\nDATA PREPARATION COMPLETED")
    print("Saved file:", OUTPUT_PATH)
    print("Dataset shape:", prepared_data.shape)


if __name__ == "__main__":
    main()
