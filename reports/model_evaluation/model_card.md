# Model Card: Diabetes 30-Day Hospital Readmission Prediction

## 1. Model Overview

| Item | Description |
|---|---|
| Project | Diabetes 30-Day Hospital Readmission Prediction |
| Model | LightGBM Classifier |
| Problem type | Binary classification |
| Prediction target | Readmission within 30 days |
| Positive class | 1 — Readmitted within 30 days |
| Negative class | 0 — Not readmitted within 30 days |
| Selected model | Tuned LightGBM |
| Prediction threshold | 0.15 (analytical reference) |
| Dataset | UCI Diabetes 130-US Hospitals (1999–2008) |

## 2. Business Objective

The objective is to identify diabetic patients who may be at higher risk of hospital readmission within 30 days after discharge.

The model is intended to support hospital teams in prioritising follow-up activities and reviewing discharge planning.

It is designed as a decision-support tool, not as a replacement for clinical judgement.

## 3. Dataset Description

**Source:** UCI Diabetes 130-US Hospitals for Years 1999–2008.

The dataset contains hospital encounter records for patients with diabetes.

Original dataset:
- 101,766 encounters
- 50 columns
- 71,518 unique patients

Target distribution:
- No readmission: 54,864 encounters
- Readmission after 30 days: 35,545 encounters
- Readmission within 30 days: 11,357 encounters

The target was converted into a binary classification variable:

- `<30` = 1
- `NO` and `>30` = 0

## 4. Data Preparation

The following steps were performed:

- Validated encounter identifiers and required columns.
- Checked missing values and duplicate records.
- Removed constant columns.
- Removed the weight column due to approximately 97% missing values.
- Preserved missing clinical test results where appropriate.
- Excluded encounters with death-related discharge dispositions from the eligible cohort.
- Retained other unknown and unmapped categories.

Final eligible cohort:
- 100,114 encounters
- 70,439 unique patients

## 5. Train-Test Strategy

A patient-level group-aware split was used to prevent encounters from the same patient appearing in both training and test sets.

| Dataset | Encounters | Unique patients |
|---|---:|---:|
| Training | 80,090 | 56,417 |
| Testing | 20,024 | 14,022 |

The test set was kept separate from model training and hyperparameter selection.

Group-aware cross-validation was used during model development.

## 6. Feature Engineering

A total of 27 engineered features were created.

Feature groups include:

- Diagnosis categories
- Medication usage and changes
- Previous inpatient, outpatient and emergency visits
- Admission and discharge information
- Demographic categories
- Hospital stay and procedure measures
- Glucose and HbA1c test-result indicators

The final model uses 25 predictors:

- 14 numerical features
- 11 categorical features

Race and gender were excluded from model predictors and retained for fairness evaluation.

## 7. Model Development

The following models were evaluated:

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM

LightGBM achieved the highest provisional cross-validation PR-AUC and was selected for further evaluation.

Best LightGBM parameters:

| Parameter | Value |
|---|---:|
| Learning rate | 0.08 |
| Number of estimators | 200 |
| Maximum depth | 5 |
| Number of leaves | 15 |
| Minimum child samples | 20 |
| Subsample | 0.8 |
| Column sample by tree | 0.8 |
| L2 regularisation | 1 |

## 8. Independent Test Performance

The final model was evaluated on 20,024 independent test encounters.

| Metric | Result |
|---|---:|
| PR-AUC | 0.2145 |
| ROC-AUC | 0.6708 |
| Brier score | 0.0963 |
| Precision | 0.2140 |
| Recall | 0.3935 |
| F1-score | 0.2772 |
| Accuracy | 0.77 |

### Confusion Matrix

| | Predicted negative | Predicted positive |
|---|---:|---:|
| Actual negative | 14,468 | 3,284 |
| Actual positive | 1,378 | 894 |

At the analytical threshold of 0.15:

- 4,178 encounters were flagged.
- 20.86% of test encounters were flagged.
- 894 actual readmissions were identified.
- 1,378 actual readmissions were missed.

## 9. Probability Calibration

Independent test calibration results:

| Metric | Result |
|---|---:|
| Mean predicted probability | 0.1123 |
| Actual readmission rate | 0.1135 |
| Brier score | 0.0963 |

The predicted and actual overall readmission rates were close.

Calibration was also examined using 10 equal-sized probability groups.

No immediate recalibration was applied. This finding does not establish that the probabilities will remain calibrated in a different hospital or a more recent population.

## 10. Explainability

SHAP was used to understand model behaviour.

Important features included:

- Previous inpatient visits
- Discharge destination
- Primary diagnosis group
- Payer code
- Medical specialty
- Length of hospital stay
- Medication count
- Age group

Previous inpatient visits and discharge destination had particularly high grouped SHAP importance.

SHAP values describe how features contribute to model predictions. They do not establish causal relationships.

## 11. Fairness Evaluation

Fairness was explored across:

- Age groups
- Gender groups
- Race groups

The evaluation included:

- Recall
- Precision
- False-positive rate
- Readmission rate
- Wilson confidence intervals
- Patient-level bootstrap confidence intervals

Independent test results showed differences in recall and false-positive rates across demographic groups.

These differences require further investigation. Small group sizes also result in wider uncertainty intervals.

No claim of fairness or absence of bias is made.

## 12. Intended Use

Potential research and analytical uses:

- Identifying encounters with higher predicted readmission risk.
- Supporting investigation of discharge planning.
- Prioritising follow-up resources in a research setting.
- Understanding factors associated with model predictions.

The model is not approved for autonomous clinical decisions or direct patient-care deployment.

## 13. Limitations and Risks

1. The dataset covers historical hospital encounters from 1999–2008.
2. Healthcare practices and patient populations may have changed.
3. The model has not been externally validated using contemporary hospital data.
4. The model does not identify all patients who will be readmitted.
5. False-positive predictions may increase follow-up workload.
6. Performance varies across demographic groups.
7. Some clinical measurements have substantial missingness.
8. The prediction target is based on recorded readmission information and may not capture every readmission outside the observed data.
9. Calibration and fairness may change when the model is applied to new populations.

## 14. Deployment Status

**Current status:** Research and development.

The model has been trained, tuned, evaluated and explained. Independent test evaluation and preliminary fairness analysis are complete.

Production deployment, external validation, monitoring and clinical approval remain outstanding.

## 15. Artifacts

| Artifact | Location |
|---|---|
| Saved LightGBM model | `artifacts/lightgbm_tuned_pipeline.joblib` |
| Final test evaluation | `reports/model_evaluation/final_lightgbm_evaluation.csv` |
| Test calibration | `reports/model_evaluation/lightgbm_test_calibration.csv` |
| Test fairness | `reports/model_evaluation/lightgbm_test_fairness_metrics.csv` |
| Grouped SHAP importance | `reports/model_evaluation/lightgbm_grouped_shap_importance.csv` |
| Patient bootstrap fairness | `reports/model_evaluation/lightgbm_patient_bootstrap_confidence_intervals.csv` |

---

**Documentation status:** Initial model card prepared.

**Important:** This model card describes a research model. It does not represent a clinically validated or production-approved healthcare system.