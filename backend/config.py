import os
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_ARTIFACTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "model_artifacts")
)


class Settings(BaseSettings):
    artifacts_dir: str = DEFAULT_ARTIFACTS_DIR
    medium_risk_ratio: float = 0.5

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
