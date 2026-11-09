from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parents[2]

data_path = project_root / "data" / "processed" / "train.csv"
report_path = (
    project_root / "reports" / "business_insights" / "missing_value_analysis.csv"
)

features = [
    "max_glu_serum",
    "A1Cresult",
]


def main():
    data = pd.read_csv(data_path, low_memory=False)

    results = []

    for feature in features:
        missing_count = data[feature].isna().sum()
        missing_percentage = data[feature].isna().mean() * 100

        data["missing_status"] = (
            data[feature]
            .isna()
            .map(
                {
                    True: "Missing",
                    False: "Available",
                }
            )
        )

        summary = (
            data.groupby("missing_status")["readmitted_30_days"]
            .agg(
                encounters="count",
                readmitted="sum",
                readmission_rate="mean",
            )
            .reset_index()
        )

        summary.insert(0, "feature", feature)
        summary["readmission_rate"] = (summary["readmission_rate"] * 100).round(2)

        results.append(summary)

        print(f"\n{feature}")
        print(f"Missing values: {missing_count}")
        print(f"Missing percentage: {missing_percentage:.2f}%")
        print(summary.to_string(index=False))

    report = pd.concat(results, ignore_index=True)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_path, index=False)

    print(f"\nReport saved: {report_path}")
    print("\nAnalysis completed successfully.")


if __name__ == "__main__":
    main()
