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
    MODEL_A_PATH: str = "saved_models/model_a_densenet121_10k_final.pth"
    MODEL_B_PATH: str = "saved_models/model_b_resnet50_50k_final.pth"
    MODEL_C_PATH: str = "saved_models/model_c_xgb_4k.pkl"

    # Model Configuration
    MODEL_A_THRESHOLD: float = 0.5  # Decision threshold for Model A
    MODEL_A_WEIGHT: float = 0.5
    MODEL_C_WEIGHT: float = 0.5

    # Model B Normalization Statistics (from training)
    MODEL_B_DIAM_MEAN: float = 3.942394540316884
    MODEL_B_DIAM_STD: float = 1.7737857888083763
    MODEL_B_FEATURE_MEANS: List[float] = [
        20.013025, 28.290689, 34.819284, 54.598683, 42.351409,
        8.617704, 19.139193, 1.068769, 5.046094, 1.356797,
        -8.941774, 9.488028, 7.551469, 2.545446, 3.084826,
        11.920692, 2.725545, 0.307373
    ]
    MODEL_B_FEATURE_STDS: List[float] = [
        3.995662, 5.315793, 5.720525, 5.568372, 10.883513,
        10.126484, 5.440581, 0.766017, 2.651096, 2.225372,
        3.508010, 3.509122, 2.426656, 1.197536, 2.059287,
        6.045630, 1.751081, 0.125826
    ]

    # Image Processing
    IMAGE_SIZE: int = 224
    IMAGE_CHANNELS: int = 3

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/app.log"

    # CORS
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173"]

    # MongoDB Configuration
    MONGODB_URI: str = "mongodb://localhost:27017"
    MONGODB_DB_NAME: str = "skin_lesion_triage_db"

    class Config:
        env_file = ".env"
        case_sensitive = True


# Create global settings instance
settings = Settings()
