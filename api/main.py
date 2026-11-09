import logging
import time
from typing import Any

from fastapi import FastAPI, HTTPException, Request

from config.logging_config import setup_logging
from config.settings import settings
from src.models.predict import predict_readmission

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "Predicts the risk of 30-day hospital readmission for diabetic patients."
    ),
    version=settings.APP_VERSION,
)


@app.middleware("http")
async def log_requests(request: Request, call_next):

    start_time = time.perf_counter()

    try:
        response = await call_next(request)

        duration = time.perf_counter() - start_time

        logger.info(
            "%s %s | Status: %s | Duration: %.3f seconds",
            request.method,
            request.url.path,
            response.status_code,
            duration,
        )

        return response

    except Exception:
        logger.exception(
            "Request failed: %s %s",
            request.method,
            request.url.path,
        )
        raise


@app.get("/")
def home():
    return {"message": f"{settings.APP_NAME} is running."}


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": "LightGBM",
    }


@app.post("/predict")
def predict(data: dict[str, Any] | list[dict[str, Any]]):

    try:
        results = predict_readmission(data)

        logger.info(
            "Prediction completed successfully | Encounters: %d",
            len(results),
        )

        return {"predictions": results.to_dict(orient="records")}

    except (ValueError, TypeError) as error:
        logger.warning("Prediction input rejected: %s", error)

        raise HTTPException(
            status_code=422,
            detail=str(error),
        )

    except Exception:
        logger.exception("Prediction failed due to an internal error")

        raise HTTPException(
            status_code=500,
            detail="Internal prediction error.",
        )
