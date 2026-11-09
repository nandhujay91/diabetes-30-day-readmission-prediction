import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "interim" / "diabetes_cleaned.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "business_insights"


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    return pd.read_csv(DATA_PATH, low_memory=False)


def analyse_readmission(data):
    distribution = data["readmitted"].value_counts().reset_index()
    distribution.columns = ["readmission_status", "encounter_count"]

    distribution["percentage"] = (
        distribution["encounter_count"] / len(data) * 100
    ).round(2)

    return distribution


def analyse_patient_visits(data):
    visits = data.groupby("patient_nbr").size()

    summary = {
        "total_encounters": len(data),
        "unique_patients": data["patient_nbr"].nunique(),
        "patients_with_multiple_visits": (visits > 1).sum(),
        "patients_with_one_visit": (visits == 1).sum(),
        "maximum_visits_by_patient": visits.max(),
    }

    visit_distribution = (
        visits.value_counts()
        .sort_index()
        .rename_axis("number_of_visits")
        .reset_index(name="number_of_patients")
    )

    return summary, visit_distribution


def save_reports(readmission_report, visit_report):
    REPORT_PATH.mkdir(parents=True, exist_ok=True)

    readmission_report.to_csv(
        REPORT_PATH / "target_distribution.csv",
        index=False,
    )

    visit_report.to_csv(
        REPORT_PATH / "patient_encounter_frequency.csv",
        index=False,
    )

    logger.info("Analysis reports saved successfully.")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    data = load_data()

    readmission_report = analyse_readmission(data)
    patient_summary, visit_report = analyse_patient_visits(data)

    save_reports(readmission_report, visit_report)

    print("\nREADMISSION DISTRIBUTION")
    print(readmission_report.to_string(index=False))

    print("\nPATIENT VISIT SUMMARY")

    for metric, value in patient_summary.items():
        print(f"{metric}: {value}")

    print("\nPATIENT VISIT DISTRIBUTION")
    print(visit_report.to_string(index=False))


if __name__ == "__main__":
    main()
