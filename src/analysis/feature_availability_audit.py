from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parents[2]

data_path = project_root / "data" / "processed" / "diabetes_eligible.csv"
report_path = (
    project_root / "reports" / "data_quality" / "feature_availability_audit.csv"
)


def main():
    data = pd.read_csv(data_path, low_memory=False)

    identifiers = ["encounter_id", "patient_nbr"]
    targets = ["readmitted", "readmitted_30_days"]

    audit_results = []

    for column in data.columns:
        if column in identifiers:
            category = "Identifier"
            decision = "Exclude"

        elif column in targets:
            category = "Target"
            decision = "Exclude"

        else:
            category = "Candidate feature"
            decision = "Review"

        audit_results.append(
            {
                "column": column,
                "category": category,
                "decision": decision,
                "data_type": str(data[column].dtype),
                "missing_count": data[column].isna().sum(),
                "missing_percentage": round(data[column].isna().mean() * 100, 2),
                "unique_values": data[column].nunique(),
            }
        )

    audit = pd.DataFrame(audit_results)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(report_path, index=False)

    print("\nFEATURE AVAILABILITY AUDIT")
    print("-" * 40)

    print("\nColumn classification:")
    print(audit["category"].value_counts())

    print("\nColumns excluded from model inputs:")
    print(audit[audit["decision"] == "Exclude"]["column"].to_list())

    print("\nCandidate features:")
    print(audit[audit["decision"] == "Review"]["column"].to_list())

    print("\nMissing values:")
    print(
        audit.loc[
            (audit["decision"] == "Review") & (audit["missing_count"] > 0),
            ["column", "missing_count", "missing_percentage"],
        ].to_string(index=False)
    )

    print(f"\nReport saved: {report_path}")
    print("\nFeature audit completed successfully.")


if __name__ == "__main__":
    main()
