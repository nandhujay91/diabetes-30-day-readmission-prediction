from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parents[2]

data_path = project_root / "data" / "processed" / "train.csv"
report_path = (
    project_root / "reports" / "business_insights" / "diagnosis_group_analysis.csv"
)

diagnosis_features = ["diag_1", "diag_2", "diag_3"]


def group_diagnosis(code):
    if pd.isna(code):
        return "Unknown"

    code = str(code).strip()

    if code.startswith("E"):
        return "External causes (E-codes)"

    if code.startswith("V"):
        return "Supplementary factors (V-codes)"

    try:
        code = float(code)
    except ValueError:
        return "Other"

    if 1 <= code <= 139:
        return "Infectious diseases"
    elif 140 <= code <= 239:
        return "Neoplasms"
    elif 240 <= code <= 279:
        return "Endocrine and metabolic diseases"
    elif 280 <= code <= 289:
        return "Blood diseases"
    elif 290 <= code <= 319:
        return "Mental disorders"
    elif 320 <= code <= 389:
        return "Nervous system and sensory diseases"
    elif 390 <= code <= 459:
        return "Circulatory diseases"
    elif 460 <= code <= 519:
        return "Respiratory diseases"
    elif 520 <= code <= 579:
        return "Digestive diseases"
    elif 580 <= code <= 629:
        return "Genitourinary diseases"
    elif 630 <= code <= 679:
        return "Pregnancy-related conditions"
    elif 680 <= code <= 709:
        return "Skin diseases"
    elif 710 <= code <= 739:
        return "Musculoskeletal diseases"
    elif 740 <= code <= 759:
        return "Congenital conditions"
    elif 760 <= code <= 779:
        return "Perinatal conditions"
    elif 780 <= code <= 799:
        return "Symptoms and abnormal findings"
    elif 800 <= code <= 999:
        return "Injury and poisoning"

    return "Other"


def main():
    data = pd.read_csv(data_path, low_memory=False)

    results = []

    for feature in diagnosis_features:
        data["diagnosis_group"] = data[feature].apply(group_diagnosis)

        summary = (
            data.groupby("diagnosis_group")["readmitted_30_days"]
            .agg(encounters="count", readmitted="sum", readmission_rate="mean")
            .reset_index()
        )

        summary = summary.rename(columns={"diagnosis_group": "category"})

        summary.insert(0, "diagnosis_feature", feature)

        summary["readmission_rate"] = (summary["readmission_rate"] * 100).round(2)

        results.append(summary)

        print(f"\n{feature}")
        print(summary.sort_values("encounters", ascending=False).to_string(index=False))

    report = pd.concat(results, ignore_index=True)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_path, index=False)

    print(f"\nReport saved: {report_path}")
    print("\nAnalysis completed successfully.")


if __name__ == "__main__":
    main()
