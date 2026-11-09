from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parents[2]

data_path = project_root / "data" / "processed" / "train.csv"
report_path = (
    project_root
    / "reports"
    / "business_insights"
    / "categorical_readmission_analysis.csv"
)

categorical_features = [
    "race",
    "gender",
    "age",
    "admission_type_id",
    "admission_source_id",
    "discharge_disposition_id",
    "payer_code",
    "medical_specialty",
    "diag_1",
    "diag_2",
    "diag_3",
    "max_glu_serum",
    "A1Cresult",
    "metformin",
    "glimepiride",
    "glipizide",
    "glyburide",
    "pioglitazone",
    "rosiglitazone",
    "insulin",
    "change",
    "diabetesMed",
]


def main():
    data = pd.read_csv(data_path, low_memory=False)

    results = []

    for feature in categorical_features:
        summary = (
            data.groupby(feature, dropna=False)["readmitted_30_days"]
            .agg(encounters="count", readmitted="sum", readmission_rate="mean")
            .reset_index()
        )

        summary = summary.rename(columns={feature: "category"})
        summary.insert(0, "feature", feature)

        summary["readmission_rate"] = (summary["readmission_rate"] * 100).round(2)

        results.append(summary)

    report = pd.concat(results, ignore_index=True)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_path, index=False)

    print("\nCATEGORICAL FEATURES VS READMISSION")
    print("-" * 50)

    for feature in categorical_features:
        print(f"\n{feature}")

        result = report[report["feature"] == feature]
        result = result.sort_values("readmission_rate", ascending=False)

        print(result.head(10).to_string(index=False))

    print(f"\nTotal category records: {len(report)}")
    print(f"Report saved: {report_path}")
    print("\nAnalysis completed successfully.")


if __name__ == "__main__":
    main()
