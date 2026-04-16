"""Application configuration loaded from environment variables / .env file."""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the tool-calling agent.

    All values are read from environment variables or the .env file.
    Override any value by setting the corresponding environment variable.
    """

    google_api_key: str
    data_dir: Path = Path("./data")
    gemini_model: str = "gemini-2.5-flash"
    max_tokens: int = 2048
    tool_timeout_seconds: float = 5.0
    tool_max_retries: int = 3
    tool_retry_delay_seconds: float = 0.5
    max_agent_iterations: int = 10
    log_level: str = "INFO"
    log_tool_calls: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
