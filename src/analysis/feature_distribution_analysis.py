from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parents[2]

data_path = project_root / "data" / "processed" / "train.csv"

report_path = (
    project_root / "reports" / "data_quality" / "feature_distribution_summary.csv"
)


def main():
    data = pd.read_csv(data_path, low_memory=False)

    # Remove identifiers and target columns from feature analysis.
    excluded_columns = [
        "encounter_id",
        "patient_nbr",
        "readmitted",
        "readmitted_30_days",
    ]

    features = data.drop(columns=excluded_columns)

    # These columns contain category IDs or grouped categories,
    # even though they are stored as numbers.
    categorical_columns = [
        "age",
        "admission_type_id",
        "discharge_disposition_id",
        "admission_source_id",
    ]

    results = []

    for column in features.columns:
        values = features[column]

        if column in categorical_columns or values.dtype == "object":
            feature_type = "Categorical"
        else:
            feature_type = "Numerical"

        top_values = values.value_counts(dropna=False).head(5)

        top_values_text = ", ".join(
            f"{value}: {count}" for value, count in top_values.items()
        )

        results.append(
            {
                "feature": column,
                "feature_type": feature_type,
                "data_type": str(values.dtype),
                "unique_values": values.nunique(dropna=True),
                "missing_count": values.isna().sum(),
                "missing_percentage": round(values.isna().mean() * 100, 2),
                "most_common_values": top_values_text,
            }
        )

    report = pd.DataFrame(results)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_path, index=False)

    print("\nFEATURE DISTRIBUTION ANALYSIS")
    print("-" * 45)

    print("\nFeature count by type:")
    print(report["feature_type"].value_counts())

    print("\nFeatures with missing values:")
    print(
        report.loc[
            report["missing_count"] > 0,
            ["feature", "missing_count", "missing_percentage"],
        ].to_string(index=False)
    )

    print("\nFeatures with only one unique value:")
    print(
        report.loc[
            report["unique_values"] <= 1, ["feature", "unique_values"]
        ].to_string(index=False)
    )

    print(f"\nReport saved: {report_path}")
    print("\nFeature distribution analysis completed.")


if __name__ == "__main__":
    main()
