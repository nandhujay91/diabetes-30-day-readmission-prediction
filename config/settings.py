from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    APP_NAME: str = "Diabetes Readmission Prediction API"

    APP_VERSION: str = "1.0.0"

    API_HOST: str = "127.0.0.1"

    API_PORT: int = 8000

    MODEL_PATH: Path = ROOT_DIR / "artifacts" / "lightgbm_tuned_pipeline.joblib"

    PREDICTION_THRESHOLD: float = 0.15

    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
