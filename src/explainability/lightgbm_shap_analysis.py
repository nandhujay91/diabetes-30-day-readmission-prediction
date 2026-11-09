from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.models.preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports" / "model_evaluation"

REPORTS.mkdir(parents=True, exist_ok=True)


# Load the trained LightGBM model
pipeline = joblib.load(ARTIFACTS / "lightgbm_tuned_pipeline.joblib")

df = pd.read_csv(DATA / "train_model_input.csv")

X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]

# Select a sample for SHAP analysis
X_sample = X.sample(
    n=min(1000, len(X)),
    random_state=42,
)

# Use the preprocessing already fitted in the saved pipeline
preprocessor = pipeline.named_steps["preprocessing"]
model = pipeline.named_steps["classifier"]

X_processed = preprocessor.transform(X_sample)

if hasattr(X_processed, "toarray"):
    X_processed = X_processed.toarray()

feature_names = preprocessor.get_feature_names_out()

X_processed = pd.DataFrame(
    X_processed,
    columns=feature_names,
    index=X_sample.index,
)

print("Calculating SHAP values...")

# Explain the positive class: 30-day readmission
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_processed)

if isinstance(shap_values, list):
    shap_values = shap_values[1]
elif shap_values.ndim == 3:
    shap_values = shap_values[:, :, 1]

# Calculate average absolute SHAP importance
importance = pd.DataFrame(
    {
        "feature": feature_names,
        "mean_absolute_shap": np.abs(shap_values).mean(axis=0),
    }
)

importance = importance.sort_values(
    "mean_absolute_shap",
    ascending=False,
)

# Save feature importance
importance.to_csv(
    REPORTS / "lightgbm_shap_importance.csv",
    index=False,
)

# Save global feature importance chart
shap.summary_plot(
    shap_values,
    X_processed,
    plot_type="bar",
    max_display=20,
    show=False,
)

plt.title("LightGBM - Feature Importance")
plt.tight_layout()

plt.savefig(
    REPORTS / "lightgbm_shap_importance.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()

# Save detailed SHAP summary chart
shap.summary_plot(
    shap_values,
    X_processed,
    max_display=20,
    show=False,
)

plt.title("LightGBM - SHAP Summary")
plt.tight_layout()

plt.savefig(
    REPORTS / "lightgbm_shap_summary.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()

# Display the most influential features
print("\nTOP 20 FEATURES")

print(importance.head(20).round(5).to_string(index=False))

print("\nSHAP analysis completed.")
print("Results saved in:", REPORTS)
