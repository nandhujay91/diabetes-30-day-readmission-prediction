import pytest

from src.models.predict import predict_readmission


def get_valid_input():
    """Return one valid patient encounter with all required model features."""

    return {
        # Numerical features
        "time_in_hospital": 5,
        "lab_procedures_per_day": 3.5,
        "medications_per_day": 2.4,
        "procedures_per_day": 0.2,
        "long_hospital_stay": 0,
        "number_inpatient": 1,
        "number_outpatient": 0,
        "number_emergency": 1,
        "medication_count": 10,
        "medication_changed_count": 2,
        "medication_up_count": 1,
        "medication_down_count": 1,
        "insulin_used": 1,
        "medication_regimen_changed": 1,
        # Categorical features
        "age_group": "[60-70)",
        "admission_type_group": "Emergency",
        "admission_source_group": "Emergency",
        "discharge_destination_group": "Home",
        "primary_diagnosis_group": "Circulatory",
        "secondary_diagnosis_group": "Diabetes",
        "additional_diagnosis_group": "Other",
        "glucose_test_result": "Not_Tested",
        "hba1c_test_result": "Not_Tested",
        "medical_specialty": "Unknown",
        "payer_code": "Unknown",
    }


def test_valid_input_prediction():
    """Check that a valid patient input produces a prediction."""

    data = get_valid_input()

    result = predict_readmission(data)

    assert len(result) == 1
    assert "predicted_probability" in result.columns
    assert "predicted_status" in result.columns

    probability = result.loc[0, "predicted_probability"]

    assert 0 <= probability <= 1
    assert result.loc[0, "predicted_status"] in [
        "High Risk",
        "Lower Risk",
    ]


def test_missing_feature():
    """Check that missing required features are rejected."""

    data = get_valid_input()
    del data["age_group"]

    with pytest.raises(ValueError, match="Missing required features"):
        predict_readmission(data)


def test_invalid_numerical_value():
    """Check that invalid numerical values are rejected."""

    data = get_valid_input()
    data["time_in_hospital"] = "invalid"

    with pytest.raises(
        ValueError,
        match="Invalid or missing numerical values",
    ):
        predict_readmission(data)


def test_missing_categorical_value():
    """Check that missing categorical values are rejected."""

    data = get_valid_input()
    data["age_group"] = None

    with pytest.raises(
        ValueError,
        match="Missing categorical values",
    ):
        predict_readmission(data)


def test_multiple_encounters():
    """Check that multiple patient encounters can be predicted."""

    data = get_valid_input()

    second_encounter = get_valid_input()
    second_encounter["time_in_hospital"] = 8
    second_encounter["number_inpatient"] = 3

    multiple_encounters = [data, second_encounter]

    result = predict_readmission(multiple_encounters)

    assert len(result) == 2
    assert result["predicted_probability"].between(0, 1).all()
    assert result["predicted_status"].isin(["High Risk", "Lower Risk"]).all()
