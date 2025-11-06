"""
Application Settings and Configuration

Centralized configuration management using Pydantic Settings.
"""

from functools import lru_cache
from typing import Literal
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # Environment
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"
    API_VERSION: str = "v0.1"
    DEBUG: bool = False

    # OpenAI Configuration
    OPENAI_API_KEY: str = Field(..., description="OpenAI API key")
    OPENAI_MODEL: str = Field(default="gpt-4.1", description="OpenAI model name")
    OPENAI_TEMPERATURE: float = Field(default=0.3, ge=0.0, le=2.0)
    OPENAI_MAX_TOKENS: int = Field(default=2048, ge=1, le=4096)

    # Vector Store Configuration
    DEFAULT_VECTOR_STORE_NAME: str = "knowledge_base"
    MAX_NUM_RESULTS: int = Field(default=5, ge=1, le=20)

    # File Upload Configuration
    MAX_FILE_SIZE_MB: int = Field(default=50, ge=1, le=200)
    ALLOWED_EXTENSIONS: list[str] = Field(default=[".pdf"])
    UPLOAD_DIR: str = "data/uploads"

    # Processing Configuration
    MAX_WORKERS: int = Field(default=10, ge=1, le=50)
    MAX_TURNS: int = Field(default=10, ge=1, le=20)

    # Session Configuration
    SESSION_PREFIX: str = "session"
    SESSION_DB_PATH: str = "data/sessions"

    # API Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = Field(default=60, ge=1)

    # CORS Configuration
    CORS_ORIGINS: list[str] = Field(
        default=[
            "http://localhost:3000",
            "http://localhost:8501",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:8501",
        ]
    )

    # Logging Configuration
    LOG_LEVEL: str = Field(default="INFO")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    @field_validator("OPENAI_API_KEY")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Validate OpenAI API key is not empty."""
        if not v or v.strip() == "":
            raise ValueError("OPENAI_API_KEY cannot be empty")
        return v

    @field_validator("MAX_FILE_SIZE_MB")
    @classmethod
    def validate_file_size(cls, v: int) -> int:
        """Validate file size is reasonable."""
        if v > 200:
            raise ValueError("MAX_FILE_SIZE_MB cannot exceed 200MB")
        return v

    @property
    def max_file_size_bytes(self) -> int:
        """Convert max file size to bytes."""
        return self.MAX_FILE_SIZE_MB * 1024 * 1024

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENVIRONMENT == "production"

    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.ENVIRONMENT == "development"


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Uses lru_cache to ensure settings are loaded only once.
    """
    return Settings()
