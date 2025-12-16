"""
Application configuration settings.
Uses pydantic-settings to load and validate environment variables.
"""

from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """

    # Application Configuration
    APP_NAME: str = "skin_lesion_ai_prototype_backend"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Model Paths
    MODEL_A_PATH: str = "saved_models/model_a.h5"
    MODEL_B_PATH: str = "saved_models/model_b.h5"
    MODEL_C_PATH: str = "saved_models/model_c.pkl"

    # Model Configuration
    MODEL_A_WEIGHT: float = 0.5
    MODEL_C_WEIGHT: float = 0.5

    # Image Processing
    IMAGE_SIZE: int = 224
    IMAGE_CHANNELS: int = 3

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/app.log"

    # CORS
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173"]

    class Config:
        env_file = ".env"
        case_sensitive = True


# Create global settings instance
settings = Settings()
