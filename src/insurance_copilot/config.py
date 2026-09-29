from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "development"
    log_level: str = "INFO"

    database_path: Path = Path("data/runtime/complaints.db")

    checkpoint_database_path: Path = Path("data/runtime/langgraph_checkpoints.sqlite")

    mlflow_tracking_uri: str = "sqlite:///data/runtime/mlflow.db"
    mlflow_experiment_name: str = "insureassist-local"

    llm_provider: str = "gemini"
    llm_api_key: str | None = None
    llm_model: str | None = None
    embedding_model: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
