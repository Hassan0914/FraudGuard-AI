from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    artifacts_dir: str = "./backend/model_artifacts"
    medium_risk_ratio: float = 0.5

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
