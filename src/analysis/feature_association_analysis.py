from pathlib import Path

import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu
from statsmodels.stats.multitest import multipletests

project_root = Path(__file__).resolve().parents[2]

data_path = project_root / "data" / "processed" / "train.csv"
report_path = (
    project_root / "reports" / "business_insights" / "feature_association_analysis.csv"
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

    results = []

    for feature in numerical_features:
        group_0 = data.loc[data["readmitted_30_days"] == 0, feature]
        group_1 = data.loc[data["readmitted_30_days"] == 1, feature]

        statistic, p_value = mannwhitneyu(group_0, group_1)

        results.append(
            {
                "feature": feature,
                "feature_type": "Numerical",
                "test": "Mann-Whitney U",
                "statistic": statistic,
                "p_value": p_value,
                "effect_size": None,
            }
        )

    for feature in categorical_features:
        table = pd.crosstab(data[feature], data["readmitted_30_days"])

        if table.shape[0] < 2 or table.shape[1] < 2:
            continue

        chi2, p_value, _, _ = chi2_contingency(table)

        total = table.to_numpy().sum()
        rows, columns = table.shape

        effect_size = (chi2 / (total * min(rows - 1, columns - 1))) ** 0.5

        results.append(
            {
                "feature": feature,
                "feature_type": "Categorical",
                "test": "Chi-square",
                "statistic": chi2,
                "p_value": p_value,
                "effect_size": effect_size,
            }
        )

    report = pd.DataFrame(results)

    report["adjusted_p_value"] = multipletests(report["p_value"], method="fdr_bh")[1]

    report["significant"] = report["adjusted_p_value"] < 0.05

    report = report.sort_values("adjusted_p_value")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_path, index=False)

    print("\nFEATURE ASSOCIATION ANALYSIS")
    print("-" * 50)

    print(report.head(20).to_string(index=False))

    print("\nTotal features tested:", len(report))
    print("Significant features:", report["significant"].sum())

    print(f"\nReport saved: {report_path}")
    print("\nAnalysis completed successfully.")


if __name__ == "__main__":
    main()
