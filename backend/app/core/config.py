from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    APP_ENV: str = "development"
    APP_NAME: str = "FraudGuard AI API"
    APP_VERSION: str = "1.0.0"
    API_V1_PREFIX: str = "/api/v1"

    DATABASE_URL: str = (
        "postgresql+psycopg://fraudguard:fraudguard_dev_password@localhost:5432/fraudguard"
    )
    DIRECT_DATABASE_URL: str | None = None

    MODEL_ARTIFACT_PATH: str = "ml/artifacts/1.0.0/pipeline.joblib"
    MODEL_FEATURE_SCHEMA_VERSION: str = "creditcard-2023-v1"

    API_AUTH_MODE: str = "local_private"  # "local_private" or "jwt"
    ANALYST_API_TOKEN: str = "fg-dev-analyst-token-change-me"
    ADMIN_API_TOKEN: str = "fg-dev-admin-token-change-me"
    JWT_SECRET_KEY: str = "fg-dev-jwt-secret-not-for-production"
    JWT_ALGORITHM: str = "HS256"

    CSV_DATA_DIR: str = "."
    STREAMLIT_API_BASE_URL: str = "http://localhost:8000/api/v1"
    LOG_LEVEL: str = "INFO"

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parents[3]


@lru_cache
def get_settings() -> Settings:
    return Settings()

