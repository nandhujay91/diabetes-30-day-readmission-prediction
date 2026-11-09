from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parents[2]
data_path = project_root / "data" / "processed" / "train.csv"
report_path = project_root / "reports" / "data_quality" / "feature_type_analysis.csv"


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

feature_groups = {
    "Demographics": ["race", "gender", "age"],
    "Admission": [
        "admission_type_id",
        "admission_source_id",
        "discharge_disposition_id",
    ],
    "Hospital stay": [
        "time_in_hospital",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_diagnoses",
    ],
    "Previous visits": ["number_outpatient", "number_emergency", "number_inpatient"],
    "Clinical": ["diag_1", "diag_2", "diag_3", "max_glu_serum", "A1Cresult"],
    "Medication": [
        "metformin",
        "repaglinide",
        "nateglinide",
        "chlorpropamide",
        "glimepiride",
        "acetohexamide",
        "glipizide",
        "glyburide",
        "tolbutamide",
        "pioglitazone",
        "rosiglitazone",
        "acarbose",
        "miglitol",
        "troglitazone",
        "tolazamide",
        "insulin",
        "glyburide-metformin",
        "glipizide-metformin",
        "glimepiride-pioglitazone",
        "metformin-rosiglitazone",
        "metformin-pioglitazone",
        "change",
        "diabetesMed",
    ],
    "Healthcare": ["payer_code", "medical_specialty"],
}


def get_group(column):
    for group, columns in feature_groups.items():
        if column in columns:
            return group
    return "Not classified"


def main():
    data = pd.read_csv(data_path, low_memory=False)

    excluded = ["encounter_id", "patient_nbr", "readmitted", "readmitted_30_days"]
    features = data.drop(columns=excluded)

    report = pd.DataFrame(
        {
            "feature": features.columns,
            "feature_group": [get_group(col) for col in features.columns],
            "feature_type": [
                "Numerical" if col in numerical_features else "Categorical"
                for col in features.columns
            ],
            "missing_count": features.isna().sum().values,
            "missing_percentage": (features.isna().mean() * 100).round(2).values,
            "unique_values": features.nunique().values,
        }
    )

    if (report["feature_group"] == "Not classified").any():
        raise ValueError("Some features are not classified.")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_path, index=False)

    print("\nFEATURE TYPE ANALYSIS")
    print(report.groupby(["feature_group", "feature_type"]).size())
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
