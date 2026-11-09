"""Data quality assessment for the diabetes readmission dataset."""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORT_DIR = PROJECT_ROOT / "reports" / "data_quality"


def generate_column_quality_report(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Generate quality metrics for every dataset column."""

    total_rows = len(dataframe)

    report = pd.DataFrame(
        {
            "column_name": dataframe.columns,
            "data_type": [str(dataframe[column].dtype) for column in dataframe.columns],
            "missing_count": [
                dataframe[column].isna().sum() for column in dataframe.columns
            ],
            "unique_count": [
                dataframe[column].nunique(dropna=True) for column in dataframe.columns
            ],
        }
    )

    report["missing_percentage"] = (report["missing_count"] / total_rows * 100).round(2)

    report["is_constant"] = report["unique_count"] <= 1

    return report.sort_values(
        by="missing_percentage",
        ascending=False,
    ).reset_index(drop=True)


def generate_dataset_quality_summary(
    dataframe: pd.DataFrame,
) -> dict:
    """Calculate overall dataset quality metrics."""

    summary = {
        "total_rows": len(dataframe),
        "total_columns": len(dataframe.columns),
        "exact_duplicate_rows": int(dataframe.duplicated().sum()),
        "duplicate_encounter_ids": int(dataframe["encounter_id"].duplicated().sum()),
        "total_missing_values": int(dataframe.isna().sum().sum()),
        "columns_with_missing_values": int(dataframe.isna().any().sum()),
        "constant_columns": int((dataframe.nunique(dropna=True) <= 1).sum()),
    }

    return summary


def generate_readmission_distribution(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate the distribution of readmission categories."""

    distribution = (
        dataframe["readmitted"]
        .value_counts(dropna=False)
        .rename_axis("readmission_status")
        .reset_index(name="encounter_count")
    )

    distribution["percentage"] = (
        distribution["encounter_count"] / len(dataframe) * 100
    ).round(2)

    return distribution


def save_quality_reports(
    column_report: pd.DataFrame,
    readmission_report: pd.DataFrame,
) -> None:
    """Save data quality reports to the reports directory."""

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    column_report.to_csv(
        REPORT_DIR / "column_quality_report.csv",
        index=False,
    )

    readmission_report.to_csv(
        REPORT_DIR / "readmission_distribution.csv",
        index=False,
    )

    logger.info(
        "Data quality reports saved to %s",
        REPORT_DIR,
    )


def run_data_quality_assessment(
    dataframe: pd.DataFrame,
) -> dict:
    """Execute the complete data quality assessment."""

    logger.info("Starting data quality assessment.")

    summary = generate_dataset_quality_summary(dataframe)

    column_report = generate_column_quality_report(dataframe)

    readmission_report = generate_readmission_distribution(dataframe)

    save_quality_reports(
        column_report,
        readmission_report,
    )

    logger.info("Data quality assessment completed.")

    return {
        "summary": summary,
        "column_report": column_report,
        "readmission_report": readmission_report,
    }
