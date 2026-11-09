from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data/processed/train_model_input.csv"
MODEL_PATH = ROOT / "artifacts/lightgbm_tuned_pipeline.joblib"
OUTPUT_DIR = ROOT / "reports/model_evaluation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load training data and model
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["readmitted_30_days"])

pipeline = joblib.load(MODEL_PATH)

preprocessor = pipeline.named_steps["preprocessing"]
model = pipeline.named_steps["classifier"]

# Use a sample of training data
X_sample = X.sample(n=min(1000, len(X)), random_state=42)

# Transform data using the fitted preprocessing pipeline
X_transformed = preprocessor.transform(X_sample)

feature_names = preprocessor.get_feature_names_out()

# Calculate SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_transformed)

# Select positive class for binary classification
if isinstance(shap_values, list):
    shap_values = shap_values[1]
elif shap_values.ndim == 3:
    shap_values = shap_values[:, :, 1]

# Map encoded features back to original features
numerical_features = [
    "time_in_hospital",
    "lab_procedures_per_day",
    "medications_per_day",
    "procedures_per_day",
    "long_hospital_stay",
    "number_inpatient",
    "number_outpatient",
    "number_emergency",
    "medication_count",
    "medication_changed_count",
    "medication_up_count",
    "medication_down_count",
    "insulin_used",
    "medication_regimen_changed",
]

categorical_features = [
    "age_group",
    "admission_type_group",
    "admission_source_group",
    "discharge_destination_group",
    "primary_diagnosis_group",
    "secondary_diagnosis_group",
    "additional_diagnosis_group",
    "glucose_test_result",
    "hba1c_test_result",
    "medical_specialty",
    "payer_code",
]

feature_groups = {}

for i, encoded_name in enumerate(feature_names):
    original_name = encoded_name.split("__", 1)[-1]

    if original_name in numerical_features:
        group_name = original_name
    else:
        group_name = None

        for column in categorical_features:
            if original_name.startswith(column + "_"):
                group_name = column
                break

    if group_name is None:
        print(f"Unmapped feature: {encoded_name}")
        continue

    feature_groups.setdefault(group_name, []).append(i)

# Combine SHAP contributions within each original feature
grouped_results = []

for group_name, indices in feature_groups.items():
    grouped_shap = shap_values[:, indices].sum(axis=1)

    importance = abs(grouped_shap).mean()

    grouped_results.append({"feature": group_name, "mean_abs_shap": importance})

importance_df = pd.DataFrame(grouped_results)

importance_df = importance_df.sort_values("mean_abs_shap", ascending=False)

# Save results
output_csv = OUTPUT_DIR / "lightgbm_grouped_shap_importance.csv"
importance_df.to_csv(output_csv, index=False)

# Plot grouped feature importance
plt.figure(figsize=(10, 8))

plt.barh(importance_df["feature"], importance_df["mean_abs_shap"])

plt.xlabel("Mean absolute grouped SHAP value")
plt.ylabel("Original feature")
plt.title("LightGBM - Grouped Feature Importance")

plt.gca().invert_yaxis()
plt.tight_layout()

output_plot = OUTPUT_DIR / "lightgbm_grouped_shap_importance.png"
plt.savefig(output_plot, dpi=300, bbox_inches="tight")
plt.close()

print("\nGROUPED SHAP FEATURE IMPORTANCE")
print(importance_df.to_string(index=False))

print(f"\nSaved CSV: {output_csv}")
print(f"Saved plot: {output_plot}")
