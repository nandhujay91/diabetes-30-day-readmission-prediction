from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parents[2]

data_path = project_root / "data" / "processed" / "train.csv"
association_path = (
    project_root / "reports" / "business_insights" / "feature_association_analysis.csv"
)
report_path = (
    project_root / "reports" / "business_insights" / "feature_selection_analysis.csv"
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
    association = pd.read_csv(association_path)

    features = numerical_features + categorical_features
    results = []

    for feature in features:
        missing = data[feature].isna().mean() * 100

        test = association[association["feature"] == feature]

        p_value = test["adjusted_p_value"].iloc[0]
        significant = test["significant"].iloc[0]

        results.append(
            {
                "feature": feature,
                "feature_type": "Numerical"
                if feature in numerical_features
                else "Categorical",
                "missing_percentage": round(missing, 2),
                "adjusted_p_value": p_value,
                "significant": significant,
            }
        )

    report = pd.DataFrame(results)

    report["selection_status"] = "Review"

    report.loc[
        (report["missing_percentage"] < 50) & report["significant"], "selection_status"
    ] = "Candidate"

    report.loc[report["missing_percentage"] >= 50, "selection_status"] = (
        "Investigate missingness"
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_path, index=False)

    print("\nFEATURE SELECTION ANALYSIS")
    print("-" * 50)
    print(report.to_string(index=False))

    print("\nSelection summary:")
    print(report["selection_status"].value_counts())

    print(f"\nReport saved: {report_path}")
    print("\nAnalysis completed successfully.")


if __name__ == "__main__":
    main()
