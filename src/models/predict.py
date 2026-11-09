import joblib
import pandas as pd

from config.settings import settings
from src.models.validate_input import validate_input

if not settings.MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {settings.MODEL_PATH}")


model = joblib.load(settings.MODEL_PATH)


def predict_readmission(data):

    input_data = validate_input(data)

    probabilities = model.predict_proba(input_data)[:, 1]

    results = pd.DataFrame(
        {
            "predicted_probability": probabilities,
            "predicted_status": [
                "High Risk"
                if probability >= settings.PREDICTION_THRESHOLD
                else "Lower Risk"
                for probability in probabilities
            ],
        }
    )

    return results
