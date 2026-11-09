"""Structural validation for the diabetes readmission dataset."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


REQUIRED_COLUMNS = {
    "encounter_id",
    "patient_nbr",
    "race",
    "gender",
    "age",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
    "diag_1",
    "diag_2",
    "diag_3",
    "A1Cresult",
    "max_glu_serum",
    "readmitted",
}


def validate_dataset_not_empty(
    dataframe: pd.DataFrame,
) -> None:
    """Ensure the dataset contains records."""

    if dataframe.empty:
        raise ValueError("Dataset contains no records.")

    logger.info("Dataset is not empty.")


def validate_required_columns(
    dataframe: pd.DataFrame,
) -> None:
    """Verify that all required columns are available."""

    missing_columns = REQUIRED_COLUMNS - set(dataframe.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    logger.info("Required column validation passed.")


def validate_unique_encounters(
    dataframe: pd.DataFrame,
) -> None:
    """Verify that encounter identifiers are unique."""

    if dataframe["encounter_id"].isna().any():
        raise ValueError("Encounter IDs contain missing values.")

    if dataframe["encounter_id"].duplicated().any():
        raise ValueError("Duplicate encounter IDs detected.")

    logger.info("Encounter uniqueness validation passed.")


def validate_dataset(
    dataframe: pd.DataFrame,
) -> None:
    """Execute all structural validation checks."""

    validate_dataset_not_empty(dataframe)
    validate_required_columns(dataframe)
    validate_unique_encounters(dataframe)

    logger.info("All structural validation checks passed.")
