from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parents[2]

data_path = project_root / "data" / "processed" / "train.csv"
report_path = project_root / "reports" / "data_quality" / "categorical_analysis.csv"

numerical_features = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_diagnoses",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
]

excluded_columns = [
    "encounter_id",
    "patient_nbr",
    "readmitted",
    "readmitted_30_days",
]


def main():
    data = pd.read_csv(data_path, low_memory=False)

    categorical_features = [
        column
        for column in data.columns
        if column not in numerical_features + excluded_columns
    ]

    results = []

    for column in categorical_features:
        counts = data[column].value_counts(dropna=False)
        percentages = data[column].value_counts(dropna=False, normalize=True) * 100

        for category, count in counts.items():
            results.append(
                {
                    "feature": column,
                    "category": category,
                    "count": count,
                    "percentage": round(percentages[category], 2),
                }
            )

    report = pd.DataFrame(results)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_path, index=False)

    print("\nCATEGORICAL FEATURE ANALYSIS")
    print("-" * 45)

    print(f"Categorical features analysed: {len(categorical_features)}")
    print(f"Total category records: {len(report)}")

    print("\nNumber of categories per feature:")
    print(report.groupby("feature")["category"].count().to_string())

    print("\nMost frequent categories:")
    for column in categorical_features:
        print(f"\n{column}")
        print(
            report[report["feature"] == column]
            .nlargest(5, "count")[["category", "count", "percentage"]]
            .to_string(index=False)
        )

    print(f"\nReport saved: {report_path}")
    print("\nCategorical analysis completed successfully.")


if __name__ == "__main__":
    main()
