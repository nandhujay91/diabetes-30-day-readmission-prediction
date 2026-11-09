from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "interim" / "diabetes_cleaned.csv"

MAPPING_PATH = (
    PROJECT_ROOT / "data" / "interim" / "mappings" / "discharge_disposition_id.csv"
)

REPORT_PATH = PROJECT_ROOT / "reports" / "business_insights"


def load_data():
    data = pd.read_csv(DATA_PATH, low_memory=False)
    mapping = pd.read_csv(MAPPING_PATH)

    return data, mapping


def analyse_readmission_by_discharge(data, mapping):
    data = data.copy()

    data["readmitted_30_days"] = data["readmitted"].map({"NO": 0, ">30": 0, "<30": 1})

    description_map = mapping.set_index("id")["description"]

    data["discharge_description"] = data["discharge_disposition_id"].map(
        description_map
    )

    data["discharge_description"] = data["discharge_description"].fillna(
        "Unknown / Unmapped"
    )

    summary = (
        data.groupby(
            ["discharge_disposition_id", "discharge_description"],
            dropna=False,
        )
        .agg(
            encounters=("readmitted_30_days", "size"),
            readmitted_within_30_days=("readmitted_30_days", "sum"),
            readmission_rate=("readmitted_30_days", "mean"),
        )
        .reset_index()
    )

    summary["readmission_rate"] = (summary["readmission_rate"] * 100).round(2)

    summary = summary.sort_values(
        "readmission_rate",
        ascending=False,
    )

    return summary


def main():
    data, mapping = load_data()

    summary = analyse_readmission_by_discharge(data, mapping)

    REPORT_PATH.mkdir(parents=True, exist_ok=True)

    summary.to_csv(
        REPORT_PATH / "readmission_by_discharge.csv",
        index=False,
    )

    print("\n30-DAY READMISSION BY DISCHARGE OUTCOME")
    print(summary.to_string(index=False))

    print("\nOVERALL 30-DAY READMISSION RATE:")
    print(f"{data['readmitted'].eq('<30').mean() * 100:.2f}%")


if __name__ == "__main__":
    main()
