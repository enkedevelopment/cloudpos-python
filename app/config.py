from functools import lru_cache
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Epose AI Scanner API"
    app_version: str = "1.0.0"
    environment: str = "production"
    log_level: str = "INFO"

    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    max_upload_bytes: int = 15 * 1024 * 1024
    min_upload_bytes: int = 1024
    target_min_dimension: int = 640
    max_image_dimension: int = 1920
    jpeg_quality: int = 92
    max_concurrent_scans: int = 4

    google_application_credentials: str | None = None

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3.5:4b"
    ollama_timeout_seconds: float = 180

    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3-flash"
    gemini_timeout_seconds: float = 120

    web_search_timeout_seconds: float = 20
    web_search_retries: int = 3

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    def install_google_credentials_env(self) -> None:
        if self.google_application_credentials:
            import os

            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.google_application_credentials


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    for local_credentials in (Path("api.json"), Path("../api.json")):
        if not settings.google_application_credentials and local_credentials.exists():
            settings.google_application_credentials = str(local_credentials.resolve())
    settings.install_google_credentials_env()
    return settings
