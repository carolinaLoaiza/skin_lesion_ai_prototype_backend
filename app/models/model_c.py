"""
Model C: XGBoost tabular classifier for malignancy prediction.

This module handles loading and inference for Model C, which predicts malignancy
probability using features extracted by Model B combined with clinical metadata.

Model characteristics:
- Algorithm: XGBoost Classifier
- Input: 18 features from Model B + clinical metadata (age, sex, location, diameter)
- Output: Probability of malignancy [0, 1]
- Features are one-hot encoded for categorical variables
- NO normalization or scaling (raw features)
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Optional, Dict, List
from app.core.logger import logger
from app.core.config import settings
from app.utils.metadata_preprocessing import prepare_metadata_for_model_c

# Import SHAP for explainability
try:
    import shap
    SHAP_AVAILABLE = True
    logger.info("SHAP library loaded successfully")
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP library not available - explainability features disabled")


class ModelCLoader:
    """
    Singleton class to load and manage Model C (XGBoost classifier) with SHAP explainability.
    """

    _instance: Optional['ModelCLoader'] = None
    _model = None
    _feature_names: list = None
    _shap_explainer = None

    def __new__(cls):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super(ModelCLoader, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize model loader and SHAP explainer (called once)."""
        logger.info("Initializing Model C...")
        self._load_model()
        self._load_shap_explainer()
        logger.info("Model C loaded successfully")

    def _load_model(self):
        """
        Load XGBoost model from pickle file.

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

            # Load the XGBoost model
            self._model = joblib.load(model_path)

            # Try to get feature names from the model
            if hasattr(self._model, 'feature_names_in_'):
                self._feature_names = list(self._model.feature_names_in_)
                logger.info(f"Loaded model with {len(self._feature_names)} features")
            elif hasattr(self._model, 'get_booster'):
                # XGBoost Booster API
                try:
                    booster = self._model.get_booster()
                    self._feature_names = booster.feature_names
                    logger.info(f"Loaded model with {len(self._feature_names)} features from booster")
                except:
                    logger.warning("Could not extract feature names from XGBoost model")
                    self._feature_names = None
            else:
                logger.warning("Model loaded without feature names")
                self._feature_names = None

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

    def _load_shap_explainer(self):
        """
        Initialize SHAP TreeExplainer for the XGBoost model.
        This is called once during model initialization.
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available - skipping explainer initialization")
            self._shap_explainer = None
            return

        if self._model is None:
            logger.error("Cannot initialize SHAP explainer - model not loaded")
            self._shap_explainer = None
            return

        try:
            logger.info("Initializing SHAP TreeExplainer...")
            self._shap_explainer = shap.TreeExplainer(self._model)
            logger.info("SHAP TreeExplainer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize SHAP explainer: {str(e)}")
            self._shap_explainer = None

    def get_shap_explainer(self):
        """Get the SHAP explainer instance."""
        if not SHAP_AVAILABLE:
            raise RuntimeError("SHAP library not available - install with: pip install shap")
        if self._shap_explainer is None:
            raise RuntimeError("SHAP explainer not initialized")
        return self._shap_explainer


def prepare_features_for_model_c(
    extracted_features: np.ndarray,
    age: int,
    sex: str,
    location: str,
    diameter: float
) -> pd.DataFrame:
    """
    Prepare feature vector for Model C.

    Combines Model B features (18) with clinical metadata (10 after encoding)
    to create the complete 28-feature input for Model C.

    Args:
        extracted_features: 18 features from Model B
        age: Patient age in years
        sex: Patient sex ("male" or "female")
        location: Lesion location
        diameter: Lesion diameter in mm

    Returns:
        DataFrame with 28 features ready for Model C:
        - 18 Model B features
        - 10 metadata features (age, diameter, sex_male, 7 location columns)

    Raises:
        ValueError: If inputs are invalid
    """
    # Validate Model B features
    if len(extracted_features) != 18:
        raise ValueError(
            f"Expected 18 features from Model B, got {len(extracted_features)}"
        )

    # Feature names from Model B (must match training exactly)
    model_b_feature_names = [
        "tbp_lv_A", "tbp_lv_B", "tbp_lv_C", "tbp_lv_H", "tbp_lv_L",
        "tbp_lv_areaMM2", "tbp_lv_area_perim_ratio", "tbp_lv_color_std_mean",
        "tbp_lv_deltaA", "tbp_lv_deltaB", "tbp_lv_deltaL", "tbp_lv_deltaLB",
        "tbp_lv_deltaLBnorm", "tbp_lv_minorAxisMM", "tbp_lv_norm_color",
        "tbp_lv_perimeterMM", "tbp_lv_stdL", "tbp_lv_symm_2axis"
    ]

    # Create DataFrame with Model B features
    model_b_data = {}
    for i, feat_name in enumerate(model_b_feature_names):
        model_b_data[feat_name] = [float(extracted_features[i])]

    df_model_b = pd.DataFrame(model_b_data)

    # Prepare metadata using centralized preprocessing
    # This handles validation, normalization, and one-hot encoding
    df_metadata = prepare_metadata_for_model_c(age, sex, location, diameter)

    # Combine Model B features + metadata features
    # Concatenate horizontally (axis=1)
    df_combined = pd.concat([df_model_b, df_metadata], axis=1)

    logger.debug(f"Prepared {len(df_combined.columns)} features for Model C")

    return df_combined


def predict_with_model_c(
    extracted_features: np.ndarray,
    age: int,
    sex: str,
    location: str,
    diameter: float
) -> float:
    """
    Predict malignancy probability using Model C (XGBoost).

    This function:
    1. Prepares features from Model B + metadata
    2. Applies one-hot encoding
    3. Makes prediction with XGBoost
    4. Returns probability

    NOTE: No normalization or scaling is applied - XGBoost uses raw features.

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
            "model_name": "XGBoost Classifier",
            "model_path": settings.MODEL_C_PATH,
            "model_type": type(model).__name__,
            "num_features": len(feature_names) if feature_names else "unknown",
            "feature_names": feature_names[:10] if feature_names else [],  # First 10
            "input": "18 Model B features + age + sex + location + diameter",
            "output": "probability [0, 1]"
        }

        # Add XGBoost specific info if available
        if hasattr(model, 'n_estimators'):
            info["n_estimators"] = model.n_estimators
        if hasattr(model, 'max_depth'):
            info["max_depth"] = model.max_depth
        if hasattr(model, 'learning_rate'):
            info["learning_rate"] = model.learning_rate

        return info

    except Exception as e:
        logger.error(f"Failed to get Model C info: {str(e)}")
        return {"error": str(e)}


def explain_prediction_with_shap(
    extracted_features: np.ndarray,
    age: int,
    sex: str,
    location: str,
    diameter: float
) -> Dict:
    """
    Generate SHAP explanation for a Model C prediction.

    Returns SHAP values explaining how each feature contributed to the prediction.

    Args:
        extracted_features: 18 features from Model B
        age: Patient age in years
        sex: Patient sex ("male" or "female")
        location: Lesion location
        diameter: Lesion diameter in mm

    Returns:
        Dictionary with:
        - shap_values: List of SHAP values for each feature
        - feature_names: List of feature names
        - feature_values: List of actual feature values
        - base_value: Base prediction value (expected value)
        - prediction: Model prediction probability

    Raises:
        RuntimeError: If SHAP is not available or explainer not initialized

    Example:
        >>> features = extract_features_with_model_b(image, 6.5)
        >>> explanation = explain_prediction_with_shap(features, 45, "male", "Left Arm", 6.5)
        >>> for name, shap_val, feat_val in zip(explanation['feature_names'],
        ...                                       explanation['shap_values'],
        ...                                       explanation['feature_values']):
        ...     impact = "increases" if shap_val > 0 else "decreases"
        ...     print(f"{name} = {feat_val:.3f} -> {impact} risk by {abs(shap_val):.4f}")
    """
    if not SHAP_AVAILABLE:
        raise RuntimeError(
            "SHAP library not available. Install with: pip install shap"
        )

    try:
        # Get model components
        loader = ModelCLoader()
        model = loader.get_model()
        explainer = loader.get_shap_explainer()
        feature_names = loader.get_feature_names()

        # Prepare features (same as prediction)
        df_features = prepare_features_for_model_c(
            extracted_features, age, sex, location, diameter
        )

        # Align features with training order
        if feature_names:
            missing_cols = set(feature_names) - set(df_features.columns)
            for col in missing_cols:
                df_features[col] = 0
            df_features = df_features[feature_names]

        logger.info("Computing SHAP values...")

        # Calculate SHAP values
        shap_values = explainer.shap_values(df_features)

        # For binary classification, shap_values might be 2D (class 0, class 1)
        # We want class 1 (malignant)
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]  # Class 1 (malignant)

        # Get base value (expected value)
        if hasattr(explainer, 'expected_value'):
            base_value = explainer.expected_value
            # For binary classification, get class 1 base value
            if isinstance(base_value, (list, np.ndarray)) and len(base_value) == 2:
                base_value = float(base_value[1])
            else:
                base_value = float(base_value)
        else:
            base_value = 0.0

        # Get prediction probability
        prediction_proba = model.predict_proba(df_features)[:, 1][0]

        # Extract SHAP values for the single prediction
        if shap_values.ndim == 2:
            shap_values_single = shap_values[0]
        else:
            shap_values_single = shap_values

        # Get feature values
        feature_values = df_features.values[0].tolist()

        logger.info(f"SHAP explanation computed: {len(shap_values_single)} features")

        return {
            "shap_values": shap_values_single.tolist(),
            "feature_names": feature_names if feature_names else [f"feature_{i}" for i in range(len(shap_values_single))],
            "feature_values": feature_values,
            "base_value": base_value,
            "prediction": float(prediction_proba)
        }

    except Exception as e:
        logger.error(f"SHAP explanation failed: {str(e)}")
        raise RuntimeError(f"SHAP explanation failed: {str(e)}")
