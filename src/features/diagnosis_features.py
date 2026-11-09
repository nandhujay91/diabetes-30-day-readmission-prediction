from pathlib import Path

import pandas as pd


def group_diagnosis(code):
    if pd.isna(code):
        return "Unknown"

    code = str(code).strip()

    if code.startswith("V"):
        return "V-codes"
    if code.startswith("E"):
        return "E-codes"

    try:
        code = float(code)
    except ValueError:
        return "Unknown"

    if 1 <= code < 140:
        return "Infectious diseases"
    elif 140 <= code < 240:
        return "Neoplasms"
    elif 240 <= code < 280:
        return "Endocrine and metabolic diseases"
    elif 280 <= code < 290:
        return "Blood diseases"
    elif 290 <= code < 320:
        return "Mental disorders"
    elif 320 <= code < 390:
        return "Nervous and sensory diseases"
    elif 390 <= code < 460:
        return "Circulatory diseases"
    elif 460 <= code < 520:
        return "Respiratory diseases"
    elif 520 <= code < 580:
        return "Digestive diseases"
    elif 580 <= code < 630:
        return "Genitourinary diseases"
    elif 630 <= code < 680:
        return "Pregnancy-related diseases"
    elif 680 <= code < 710:
        return "Skin diseases"
    elif 710 <= code < 740:
        return "Musculoskeletal diseases"
    elif 740 <= code < 760:
        return "Congenital diseases"
    elif 760 <= code < 780:
        return "Perinatal conditions"
    elif 780 <= code < 800:
        return "Symptoms and abnormal findings"
    elif 800 <= code < 1000:
        return "Injury and poisoning"

    return "Other"


def main():
    project_path = Path(__file__).resolve().parents[2]
    input_files = ["train.csv", "test.csv"]

    for file in input_files:
        file_path = project_path / "data" / "processed" / file
        df = pd.read_csv(file_path)

        df["primary_diagnosis_group"] = df["diag_1"].apply(group_diagnosis)
        df["secondary_diagnosis_group"] = df["diag_2"].apply(group_diagnosis)
        df["additional_diagnosis_group"] = df["diag_3"].apply(group_diagnosis)

        output_file = file.replace(".csv", "_diagnosis.csv")
        output_path = project_path / "data" / "processed" / output_file

        df.to_csv(output_path, index=False)

        print(f"\nProcessed: {file}")
        print(f"Rows: {len(df)}")
        print("Diagnosis groups created successfully.")
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
