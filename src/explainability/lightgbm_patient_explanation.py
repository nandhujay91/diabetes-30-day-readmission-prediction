from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = ROOT / "data/processed/train_model_input.csv"
TRAIN_PATH = ROOT / "data/processed/train.csv"
MODEL_PATH = ROOT / "artifacts/lightgbm_tuned_pipeline.joblib"
OUTPUT_DIR = ROOT / "reports/model_evaluation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load data and trained model
X = pd.read_csv(INPUT_PATH)
train = pd.read_csv(TRAIN_PATH)

pipeline = joblib.load(MODEL_PATH)

# Check that both files have the same row order
if len(X) != len(train):
    raise ValueError("Training files have different row counts.")

if not X["readmitted_30_days"].equals(train["readmitted_30_days"]):
    raise ValueError("Target order does not match between training files.")

X = X.drop(columns=["readmitted_30_days"])

# Select a reproducible sample of training encounters
sample_size = min(1000, len(X))
sample_indices = X.sample(n=sample_size, random_state=42).index

X_sample = X.loc[sample_indices]

# Select the highest predicted-risk encounter in this sample
probabilities = pipeline.predict_proba(X_sample)[:, 1]

selected_position = np.argmax(probabilities)
selected_index = sample_indices[selected_position]

X_patient = X.loc[[selected_index]]
patient_info = train.loc[selected_index]

predicted_probability = probabilities[selected_position]

# Get model components
preprocessor = pipeline.named_steps["preprocessing"]
model = pipeline.named_steps["classifier"]

# Transform the selected encounter
X_transformed = preprocessor.transform(X_patient)

if sparse.issparse(X_transformed):
    X_transformed = X_transformed.toarray()

feature_names = preprocessor.get_feature_names_out()

# Calculate SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_transformed)

# Select positive class
if isinstance(shap_values, list):
    shap_values = shap_values[1]
elif shap_values.ndim == 3:
    shap_values = shap_values[:, :, 1]

shap_values = shap_values[0]

# Get expected model output
base_value = explainer.expected_value

if isinstance(base_value, (list, np.ndarray)):
    base_value = np.asarray(base_value).flatten()[-1]

base_value = float(base_value)

# Create explanation table
explanation = pd.DataFrame(
    {
        "feature": feature_names,
        "feature_value": X_transformed[0],
        "shap_value": shap_values,
    }
)

explanation["impact"] = np.where(
    explanation["shap_value"] > 0, "Increases prediction", "Decreases prediction"
)

explanation["absolute_impact"] = explanation["shap_value"].abs()

explanation = explanation.sort_values("absolute_impact", ascending=False)

# Save complete explanation
output_csv = OUTPUT_DIR / "lightgbm_patient_explanation.csv"
explanation.to_csv(output_csv, index=False)

# Display encounter information
print("\nPATIENT EXPLANATION")
print("-" * 50)

print(f"Encounter ID: {patient_info['encounter_id']}")
print(f"Patient ID: {patient_info['patient_nbr']}")
print(f"Actual readmission: {patient_info['readmitted_30_days']}")
print(f"Predicted probability: {predicted_probability:.2%}")

print("\nTOP 5 FEATURES INCREASING PREDICTION")
print("-" * 50)

positive = explanation[explanation["shap_value"] > 0].nlargest(5, "shap_value")

print(positive[["feature", "shap_value"]].to_string(index=False))

print("\nTOP 5 FEATURES DECREASING PREDICTION")
print("-" * 50)

negative = explanation[explanation["shap_value"] < 0].nsmallest(5, "shap_value")

print(negative[["feature", "shap_value"]].to_string(index=False))

# Create SHAP waterfall chart
shap_explanation = shap.Explanation(
    values=shap_values,
    base_values=base_value,
    data=X_transformed[0],
    feature_names=feature_names,
)

plt.figure(figsize=(10, 8))
shap.plots.waterfall(shap_explanation, max_display=15, show=False)

plt.title("LightGBM - Individual Patient Explanation")
plt.tight_layout()

output_plot = OUTPUT_DIR / "lightgbm_patient_explanation.png"

plt.savefig(output_plot, dpi=300, bbox_inches="tight")

plt.close()

print(f"\nSaved CSV: {output_csv}")
print(f"Saved chart: {output_plot}")
