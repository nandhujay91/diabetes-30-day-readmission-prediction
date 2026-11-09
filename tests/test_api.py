from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def get_valid_input():
    return {
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


def test_home_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Diabetes Readmission Prediction API is running."
    }


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "model": "LightGBM",
    }


def test_prediction_endpoint():
    response = client.post("/predict", json=get_valid_input())

    assert response.status_code == 200

    result = response.json()["predictions"]

    assert len(result) == 1
    assert 0 <= result[0]["predicted_probability"] <= 1
    assert result[0]["predicted_status"] in [
        "High Risk",
        "Lower Risk",
    ]


def test_prediction_missing_feature():
    data = get_valid_input()
    del data["age_group"]

    response = client.post("/predict", json=data)

    assert response.status_code == 422
    assert "Missing required features" in response.json()["detail"]


def test_multiple_patient_predictions():
    first_patient = get_valid_input()

    second_patient = get_valid_input()
    second_patient["time_in_hospital"] = 8
    second_patient["number_inpatient"] = 3

    response = client.post(
        "/predict",
        json=[first_patient, second_patient],
    )

    assert response.status_code == 200

    predictions = response.json()["predictions"]

    assert len(predictions) == 2

    for prediction in predictions:
        assert 0 <= prediction["predicted_probability"] <= 1
        assert prediction["predicted_status"] in [
            "High Risk",
            "Lower Risk",
        ]


def test_api_configuration():
    from config.settings import settings

    response = client.get("/openapi.json")

    assert response.status_code == 200

    api_info = response.json()["info"]

    assert api_info["title"] == settings.APP_NAME
    assert api_info["version"] == settings.APP_VERSION
