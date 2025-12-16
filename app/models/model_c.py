"""
Model C: Random Forest tabular classifier for malignancy prediction.

This module handles loading and inference for Model C, which predicts malignancy
probability using features extracted by Model B combined with clinical metadata.

Model characteristics:
- Algorithm: Random Forest Classifier
- Input: 18 features from Model B + clinical metadata (age, sex, location, diameter)
- Output: Probability of malignancy [0, 1]
- Features are one-hot encoded for categorical variables
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Optional, Dict
from app.core.logger import logger
from app.core.config import settings


class ModelCLoader:
    """
    Singleton class to load and manage Model C (Random Forest classifier).
    """

    _instance: Optional['ModelCLoader'] = None
    _model = None
    _feature_names: list = None

    def __new__(cls):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super(ModelCLoader, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize model loader (called once)."""
        logger.info("Initializing Model C...")
        self._load_model()
        logger.info("Model C loaded successfully")

    def _load_model(self):
        """
        Load Random Forest model from pickle file.

        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model loading fails
        """
        model_path = settings.MODEL_C_PATH

        # Check if model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model C file not found at: {model_path}. "
                f"Please ensure the model file exists at this location."
            )

        try:
            logger.info(f"Loading Model C from: {model_path}")

            # Load the artifact (contains model + feature names)
            artifact = joblib.load(model_path)

            if isinstance(artifact, dict):
                # Artifact format: {"model": rf_model, "features": feature_names}
                self._model = artifact["model"]
                self._feature_names = artifact["features"]
                logger.info(f"Loaded model with {len(self._feature_names)} features")
            else:
                # Just the model without feature names
                self._model = artifact
                logger.warning("Model loaded without feature names")

            logger.info(f"Model C loaded: {type(self._model).__name__}")

        except Exception as e:
            logger.error(f"Failed to load Model C: {str(e)}")
            raise RuntimeError(f"Failed to load Model C: {str(e)}")

    def get_model(self):
        """Get the loaded model instance."""
        if self._model is None:
            raise RuntimeError("Model C not initialized")
        return self._model

    def get_feature_names(self) -> list:
        """Get the feature names expected by the model."""
        if self._feature_names is None:
            logger.warning("Feature names not available")
            return []
        return self._feature_names.copy()


def prepare_features_for_model_c(
    extracted_features: np.ndarray,
    age: int,
    sex: str,
    location: str,
    diameter: float
) -> pd.DataFrame:
    """
    Prepare feature vector for Model C.

    Combines Model B features with clinical metadata and applies
    one-hot encoding for categorical variables.

    Args:
        extracted_features: 18 features from Model B
        age: Patient age in years
        sex: Patient sex ("male" or "female")
        location: Lesion location
        diameter: Lesion diameter in mm

    Returns:
        DataFrame with features ready for Model C

    Raises:
        ValueError: If inputs are invalid
    """
    # Validate Model B features
    if len(extracted_features) != 18:
        raise ValueError(
            f"Expected 18 features from Model B, got {len(extracted_features)}"
        )

    # Feature names from Model B
    model_b_feature_names = [
        "tbp_lv_A", "tbp_lv_B", "tbp_lv_C", "tbp_lv_H", "tbp_lv_L",
        "tbp_lv_areaMM2", "tbp_lv_area_perim_ratio", "tbp_lv_color_std_mean",
        "tbp_lv_deltaA", "tbp_lv_deltaB", "tbp_lv_deltaL", "tbp_lv_deltaLB",
        "tbp_lv_deltaLBnorm", "tbp_lv_minorAxisMM", "tbp_lv_norm_color",
        "tbp_lv_perimeterMM", "tbp_lv_stdL", "tbp_lv_symm_2axis"
    ]

    # Create base dataframe with Model B features + clinical inputs
    data = {}

    # Add Model B features
    for i, feat_name in enumerate(model_b_feature_names):
        data[feat_name] = [float(extracted_features[i])]

    # Add clinical metadata
    data["age_approx"] = [int(age)]
    data["clin_size_long_diam_mm"] = [float(diameter)]
    data["sex"] = [sex.lower()]
    data["tbp_lv_location_simple"] = [location.lower()]

    df = pd.DataFrame(data)

    # Apply one-hot encoding (matching training)
    # drop_first=True to avoid dummy variable trap
    df_encoded = pd.get_dummies(
        df,
        columns=["sex", "tbp_lv_location_simple"],
        drop_first=True
    )

    logger.debug(f"Prepared {len(df_encoded.columns)} features for Model C")

    return df_encoded


def predict_with_model_c(
    extracted_features: np.ndarray,
    age: int,
    sex: str,
    location: str,
    diameter: float
) -> float:
    """
    Predict malignancy probability using Model C.

    This function:
    1. Prepares features from Model B + metadata
    2. Applies one-hot encoding
    3. Makes prediction with Random Forest
    4. Returns probability

    Args:
        extracted_features: 18 features from Model B
        age: Patient age in years
        sex: Patient sex ("male" or "female")
        location: Lesion location
        diameter: Lesion diameter in mm

    Returns:
        Probability of malignancy [0, 1]

    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If prediction fails

    Example:
        >>> features = extract_features_with_model_b(image, diameter)
        >>> probability = predict_with_model_c(features, 45, "female", "back", 6.5)
        >>> print(f"Model C probability: {probability:.3f}")
    """
    try:
        # Get model
        loader = ModelCLoader()
        model = loader.get_model()
        expected_features = loader.get_feature_names()

        # Prepare features
        df_features = prepare_features_for_model_c(
            extracted_features, age, sex, location, diameter
        )

        # Align features with training order
        # This ensures features are in the same order as during training
        if expected_features:
            # Reorder columns to match training
            missing_cols = set(expected_features) - set(df_features.columns)
            extra_cols = set(df_features.columns) - set(expected_features)

            # Add missing columns with zeros
            for col in missing_cols:
                df_features[col] = 0

            # Remove extra columns
            df_features = df_features[expected_features]

            logger.debug(f"Aligned {len(df_features.columns)} features")
        else:
            logger.warning("No expected feature names, using as-is")

        # Make prediction
        probability = model.predict_proba(df_features)[:, 1][0]

        logger.info(f"Model C prediction: {probability:.4f}")

        # Validate output range
        if not (0.0 <= probability <= 1.0):
            logger.warning(f"Probability out of range: {probability}")
            probability = np.clip(probability, 0.0, 1.0)

        return float(probability)

    except Exception as e:
        logger.error(f"Model C prediction failed: {str(e)}")
        raise RuntimeError(f"Model C prediction failed: {str(e)}")


def validate_model_c_exists() -> bool:
    """
    Check if Model C file exists at the configured path.

    Returns:
        True if model file exists, False otherwise
    """
    model_path = settings.MODEL_C_PATH
    exists = os.path.exists(model_path)

    if exists:
        logger.info(f"Model C file found at: {model_path}")
    else:
        logger.warning(f"Model C file not found at: {model_path}")

    return exists


def get_model_c_info() -> dict:
    """
    Get information about Model C.

    Returns:
        Dictionary with model information
    """
    try:
        loader = ModelCLoader()
        model = loader.get_model()
        feature_names = loader.get_feature_names()

        info = {
            "model_name": "Random Forest Classifier",
            "model_path": settings.MODEL_C_PATH,
            "model_type": type(model).__name__,
            "num_features": len(feature_names) if feature_names else "unknown",
            "feature_names": feature_names[:10] if feature_names else [],  # First 10
            "input": "18 Model B features + age + sex + location + diameter",
            "output": "probability [0, 1]"
        }

        # Add Random Forest specific info if available
        if hasattr(model, 'n_estimators'):
            info["n_estimators"] = model.n_estimators
        if hasattr(model, 'max_depth'):
            info["max_depth"] = model.max_depth

        return info

    except Exception as e:
        logger.error(f"Failed to get Model C info: {str(e)}")
        return {"error": str(e)}
