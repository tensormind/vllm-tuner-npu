from __future__ import annotations

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Logging
    ENV_MODE: str = "production"
    LOG_LEVEL: str = ""
    JSON_LOGS: bool = False
    COLORIZE: bool = True

    # HuggingFace
    HF_TOKEN: SecretStr | None = None

    # Vantage Cloud Pricing API
    VANTAGE_API_TOKEN: SecretStr | None = None

    @property
    def resolved_log_level(self) -> str:
        if self.LOG_LEVEL:
            return self.LOG_LEVEL
        return "INFO" if self.ENV_MODE.lower() == "production" else "DEBUG"


settings = Settings()
