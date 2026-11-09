"""Data loading functions for the diabetes readmission project."""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

DIABETES_DATA_FILE = RAW_DATA_DIR / "diabetic_data.csv"
ID_MAPPING_FILE = RAW_DATA_DIR / "IDS_mapping.csv"


def load_csv(file_path: Path) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame."""

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    logger.info("Loading %s", file_path.name)

    dataframe = pd.read_csv(
        file_path,
        low_memory=False,
        na_values=["?"],
    )

    logger.info(
        "Loaded %s rows and %s columns",
        dataframe.shape[0],
        dataframe.shape[1],
    )

    return dataframe


def load_diabetes_data() -> pd.DataFrame:
    """Load the original diabetes hospital dataset."""

    return load_csv(DIABETES_DATA_FILE)


def load_id_mapping() -> pd.DataFrame:
    """Load the hospital ID mapping reference dataset."""

    return load_csv(ID_MAPPING_FILE)
