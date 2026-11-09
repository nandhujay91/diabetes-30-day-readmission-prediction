"""Initial data cleaning functions for the diabetes readmission project."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


# Columns identified as uninformative during data profiling.
COLUMNS_TO_DROP = [
    "weight",
    "examide",
    "citoglipton",
]


# Missing values will be represented explicitly.
CATEGORICAL_COLUMNS = [
    "race",
    "medical_specialty",
    "payer_code",
]


def remove_uninformative_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Remove columns with insufficient analytical value."""

    columns_to_drop = [
        column for column in COLUMNS_TO_DROP if column in dataframe.columns
    ]

    cleaned_data = dataframe.drop(
        columns=columns_to_drop,
    ).copy()

    logger.info("Removed columns: %s", columns_to_drop)

    return cleaned_data


def handle_categorical_missing_values(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Represent missing categorical values explicitly."""

    cleaned_data = dataframe.copy()

    for column in CATEGORICAL_COLUMNS:
        if column in cleaned_data.columns:
            cleaned_data[column] = cleaned_data[column].fillna("Unknown")

    logger.info("Categorical missing-value handling completed.")

    return cleaned_data


def clean_diabetes_data(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Apply initial data-cleaning transformations."""

    logger.info("Starting initial data cleaning.")

    cleaned_data = remove_uninformative_columns(dataframe)

    cleaned_data = handle_categorical_missing_values(cleaned_data)

    logger.info(
        "Initial cleaning completed. Shape: %s",
        cleaned_data.shape,
    )

    return cleaned_data
