from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parents[2]

data_path = project_root / "data" / "processed" / "train.csv"
report_path = (
    project_root
    / "reports"
    / "business_insights"
    / "numerical_readmission_analysis.csv"
)

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

    results = []

    for feature in numerical_features:
        for readmission, group in data.groupby("readmitted_30_days"):
            results.append(
                {
                    "feature": feature,
                    "readmitted_30_days": readmission,
                    "encounters": len(group),
                    "mean": round(group[feature].mean(), 2),
                    "median": round(group[feature].median(), 2),
                    "std": round(group[feature].std(), 2),
                }
            )

    report = pd.DataFrame(results)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_path, index=False)

    print("\nNUMERICAL FEATURES VS READMISSION")
    print("-" * 50)

    for feature in numerical_features:
        print(f"\n{feature}")
        print(report[report["feature"] == feature].to_string(index=False))

    print(f"\nReport saved: {report_path}")
    print("\nAnalysis completed successfully.")


if __name__ == "__main__":
    main()
