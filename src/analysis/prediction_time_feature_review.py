from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parents[2]

data_path = project_root / "data" / "processed" / "train.csv"
report_path = (
    project_root
    / "reports"
    / "business_insights"
    / "prediction_time_feature_review.csv"
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

fairness_features = ["race", "gender"]
missing_features = ["max_glu_serum", "A1Cresult"]

conditional_features = [
    "discharge_disposition_id",
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_diagnoses",
    "diag_1",
    "diag_2",
    "diag_3",
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

    features = numerical_features + categorical_features
    results = []

    for feature in features:
        if feature in fairness_features:
            decision = "Fairness analysis"
        elif feature in missing_features:
            decision = "Investigate missingness"
        elif feature in conditional_features:
            decision = "Prediction-time review"
        else:
            decision = "Candidate"

        results.append(
            {
                "feature": feature,
                "feature_type": "Numerical"
                if feature in numerical_features
                else "Categorical",
                "missing_percentage": round(data[feature].isna().mean() * 100, 2),
                "availability": decision,
            }
        )

    report = pd.DataFrame(results)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_path, index=False)

    print("\nPREDICTION-TIME FEATURE REVIEW")
    print("-" * 50)
    print(report.to_string(index=False))

    print("\nAvailability summary:")
    print(report["availability"].value_counts())

    print(f"\nReport saved: {report_path}")
    print("\nAnalysis completed successfully.")


if __name__ == "__main__":
    main()
