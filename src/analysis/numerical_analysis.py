from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parents[2]

data_path = project_root / "data" / "processed" / "train.csv"
report_path = project_root / "reports" / "data_quality" / "numerical_analysis.csv"

numerical_features = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_diagnoses",
    "number_inpatient",
    "number_outpatient",
    "number_emergency",
]


def main():
    data = pd.read_csv(data_path, low_memory=False)

    report = data[numerical_features].describe().T

    report["missing_count"] = data[numerical_features].isna().sum()
    report["zero_count"] = (data[numerical_features] == 0).sum()

    report["outlier_count"] = 0

    for column in numerical_features:
        q1 = data[column].quantile(0.25)
        q3 = data[column].quantile(0.75)
        iqr = q3 - q1

        lower_limit = q1 - 1.5 * iqr
        upper_limit = q3 + 1.5 * iqr

        report.loc[column, "outlier_count"] = (
            (data[column] < lower_limit) | (data[column] > upper_limit)
        ).sum()

    report = report.round(2)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_path)

    print("\nNUMERICAL FEATURE ANALYSIS")
    print("-" * 45)
    print(report.to_string())

    print(f"\nReport saved: {report_path}")
    print("\nNumerical analysis completed successfully.")


if __name__ == "__main__":
    main()
