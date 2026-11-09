from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "interim" / "diabetes_cleaned.csv"

# Special discharge categories that need separate investigation.
DEATH_DISPOSITIONS = [11, 19, 20, 21]
HOSPICE_DISPOSITIONS = [13, 14]


def load_data():
    return pd.read_csv(DATA_PATH, low_memory=False)


def analyse_eligibility(data):
    data = data.copy()

    data["readmitted_30_days"] = data["readmitted"].map({"NO": 0, ">30": 0, "<30": 1})

    data["discharge_group"] = "Other"

    data.loc[
        data["discharge_disposition_id"].isin(DEATH_DISPOSITIONS),
        "discharge_group",
    ] = "Death-related"

    data.loc[
        data["discharge_disposition_id"].isin(HOSPICE_DISPOSITIONS),
        "discharge_group",
    ] = "Hospice"

    summary = (
        data.groupby("discharge_group")
        .agg(
            encounters=("readmitted_30_days", "size"),
            readmissions=("readmitted_30_days", "sum"),
            readmission_rate=("readmitted_30_days", "mean"),
        )
        .reset_index()
    )

    summary["readmission_rate"] = (summary["readmission_rate"] * 100).round(2)

    print("\nDISCHARGE GROUP ANALYSIS")
    print(summary.to_string(index=False))

    excluded = data["discharge_disposition_id"].isin(
        DEATH_DISPOSITIONS + HOSPICE_DISPOSITIONS
    )

    eligible_data = data.loc[~excluded]

    print("\nPOPULATION COMPARISON")
    print(f"Original encounters: {len(data)}")
    print(f"Death/hospice encounters: {excluded.sum()}")
    print(f"Remaining encounters: {len(eligible_data)}")

    print(
        "Remaining 30-day readmission rate:",
        f"{eligible_data['readmitted_30_days'].mean() * 100:.2f}%",
    )


def main():
    data = load_data()
    analyse_eligibility(data)


if __name__ == "__main__":
    main()
