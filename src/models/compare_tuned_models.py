from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.models.preprocessing import build_preprocessor

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports" / "model_evaluation"


# Load training data
model_data = pd.read_csv(DATA / "train_model_input.csv")
train_data = pd.read_csv(DATA / "train.csv")

# Check that both files contain the same training encounters
if len(model_data) != len(train_data):
    raise ValueError("Training files have different row counts.")

if not model_data["readmitted_30_days"].equals(train_data["readmitted_30_days"]):
    raise ValueError("Target values do not match between training files.")

X = model_data.drop(columns=["readmitted_30_days"])
y = model_data["readmitted_30_days"].astype(int)
groups = train_data["patient_nbr"]


# Load the best parameters
xgb_params = pd.read_csv(REPORTS / "xgboost_best_parameters.csv").iloc[0].to_dict()

lgbm_params = pd.read_csv(REPORTS / "lightgbm_best_parameters.csv").iloc[0].to_dict()


# Remove the classifier__ prefix
xgb_params = {
    key.replace("classifier__", ""): value for key, value in xgb_params.items()
}

lgbm_params = {
    key.replace("classifier__", ""): value for key, value in lgbm_params.items()
}


# Convert integer parameters
xgb_params["n_estimators"] = int(xgb_params["n_estimators"])
xgb_params["max_depth"] = int(xgb_params["max_depth"])
xgb_params["min_child_weight"] = int(xgb_params["min_child_weight"])

lgbm_params["n_estimators"] = int(lgbm_params["n_estimators"])
lgbm_params["max_depth"] = int(lgbm_params["max_depth"])
lgbm_params["num_leaves"] = int(lgbm_params["num_leaves"])
lgbm_params["min_child_samples"] = int(lgbm_params["min_child_samples"])


# Create both tuned models
models = {
    "XGBoost": XGBClassifier(
        **xgb_params,
        random_state=42,
        eval_metric="logloss",
        n_jobs=1,
    ),
    "LightGBM": LGBMClassifier(
        **lgbm_params,
        random_state=42,
        verbosity=-1,
        n_jobs=1,
    ),
}


# Five-fold patient-grouped cross-validation
cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

preprocessor = build_preprocessor()

fold_results = []
all_predictions = []


for model_name, classifier in models.items():
    print(f"\nTraining {model_name}")

    oof_predictions = np.zeros(len(X))

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y, groups), start=1):
        pipeline = Pipeline(
            [
                ("preprocessor", clone(preprocessor)),
                ("classifier", clone(classifier)),
            ]
        )

        pipeline.fit(
            X.iloc[train_idx],
            y.iloc[train_idx],
        )

        probabilities = pipeline.predict_proba(X.iloc[valid_idx])[:, 1]

        oof_predictions[valid_idx] = probabilities

        fold_results.append(
            {
                "model": model_name,
                "fold": fold,
                "pr_auc": average_precision_score(y.iloc[valid_idx], probabilities),
                "roc_auc": roc_auc_score(y.iloc[valid_idx], probabilities),
                "brier_score": brier_score_loss(y.iloc[valid_idx], probabilities),
            }
        )

        print(f"Fold {fold} completed")

    all_predictions.append(
        pd.DataFrame(
            {
                "model": model_name,
                "actual": y.to_numpy(),
                "predicted_probability": oof_predictions,
            }
        )
    )


# Combine predictions from both models
predictions = pd.concat(all_predictions, ignore_index=True)
fold_results = pd.DataFrame(fold_results)


# Calculate overall performance
summary = []

for model_name, result in predictions.groupby("model"):
    actual = result["actual"]
    probabilities = result["predicted_probability"]

    predicted = (probabilities >= 0.5).astype(int)

    summary.append(
        {
            "model": model_name,
            "pr_auc": average_precision_score(actual, probabilities),
            "roc_auc": roc_auc_score(actual, probabilities),
            "brier_score": brier_score_loss(actual, probabilities),
            "precision": precision_score(actual, predicted, zero_division=0),
            "recall": recall_score(actual, predicted, zero_division=0),
        }
    )


comparison = pd.DataFrame(summary).sort_values(
    "pr_auc",
    ascending=False,
)


# Save comparison results
comparison.to_csv(
    REPORTS / "tuned_model_comparison.csv",
    index=False,
)

fold_results.to_csv(
    REPORTS / "tuned_model_fold_results.csv",
    index=False,
)

predictions.to_csv(
    REPORTS / "tuned_model_oof_predictions.csv",
    index=False,
)


# Display results
print("\nTUNED MODEL COMPARISON")
print(comparison.round(4).to_string(index=False))

print("\nFOLD-LEVEL RESULTS")
print(fold_results.round(4).to_string(index=False))

print("\nAll comparison reports saved successfully.")
