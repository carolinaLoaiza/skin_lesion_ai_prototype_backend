"""
Model B: ResNet-50 feature extractor for skin lesion analysis.

This module handles loading and inference for Model B, which extracts 18
clinical features from dermoscopic images plus lesion diameter.

Model characteristics:
- Architecture: ResNet-50 with custom head
- Input: Preprocessed image (3, 224, 224) + diameter (normalized)
- Output: 18 predicted features (denormalized)
- Features include color, shape, and texture measurements
"""

import os
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from typing import Optional, Tuple
from app.core.config import settings
from app.core.logger import logger


# Feature names (18 features predicted by Model B)
FEATURE_NAMES = [
    "tbp_lv_A",
    "tbp_lv_B",
    "tbp_lv_C",
    "tbp_lv_H",
    "tbp_lv_L",
    "tbp_lv_areaMM2",
    "tbp_lv_area_perim_ratio",
    "tbp_lv_color_std_mean",
    "tbp_lv_deltaA",
    "tbp_lv_deltaB",
    "tbp_lv_deltaL",
    "tbp_lv_deltaLB",
    "tbp_lv_deltaLBnorm",
    "tbp_lv_minorAxisMM",
    "tbp_lv_norm_color",
    "tbp_lv_perimeterMM",
    "tbp_lv_stdL",
    "tbp_lv_symm_2axis"
]

# Geometric features (affected by diameter)
GEOMETRIC_FEATURES = [
    "tbp_lv_areaMM2",
    "tbp_lv_area_perim_ratio",
    "tbp_lv_minorAxisMM",
    "tbp_lv_perimeterMM",
    "tbp_lv_symm_2axis"
]


class LesionModelD_post(nn.Module):
    """
    ResNet-50 based model for predicting lesion features.

    Architecture matches training:
    - ResNet-50 backbone (pretrained on ImageNet)
    - Custom head for feature prediction
    - Post-processing with diameter for geometric features
    """

    def __init__(self, num_targets=18, geom_names=None, all_features=None):
        super().__init__()

        # Load ResNet-50 without final FC layer
        self.cnn = models.resnet50(weights=None)
        self.cnn.fc = nn.Identity()

        self.num_targets = num_targets
        self.geom_names = geom_names or GEOMETRIC_FEATURES
        self.all_features = all_features or FEATURE_NAMES

        # Indices of geometric features
        self.geom_idx = [self.all_features.index(f) for f in self.geom_names]

        # Initial prediction from image only
        self.fc_init = nn.Sequential(
            nn.Linear(2048, num_targets),
        )

        # Post-processing with diameter for geometric features
        self.fc_post = nn.Linear(len(self.geom_idx) + 1, len(self.geom_idx))

    def forward(self, x_img, x_diameter):
        """
        Forward pass.

        Args:
            x_img: Image tensor (batch, 3, 224, 224)
            x_diameter: Diameter tensor (batch, 1)

        Returns:
            Predicted features (batch, 18)
        """
        # Extract features from image
        x_feat = self.cnn(x_img)

        # Initial prediction of all features
        pred_init = self.fc_init(x_feat)

        # Post-process geometric features with diameter
        geom_pred = pred_init[:, self.geom_idx]
        geom_pred = self.fc_post(torch.cat([geom_pred, x_diameter], dim=1))

        # Combine with non-geometric features
        pred = pred_init.clone()
        pred[:, self.geom_idx] = geom_pred

        return pred


class ModelBLoader:
    """
    Singleton class to load and manage Model B (ResNet-50 feature extractor).
    """

    _instance: Optional['ModelBLoader'] = None
    _model: Optional[nn.Module] = None
    _device: torch.device = None

    def __new__(cls):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super(ModelBLoader, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize model loader (called once)."""
        # Use CPU for compatibility
        self._device = torch.device("cpu")
        logger.info(f"Model B will run on device: {self._device}")

        # Load model
        self._model = self._load_model()
        logger.info("Model B loaded successfully")

    def _create_model_architecture(self) -> nn.Module:
        """
        Create ResNet-50 feature extractor architecture.

        Returns:
            PyTorch model matching training architecture
        """
        model = LesionModelD_post(
            num_targets=18,
            geom_names=GEOMETRIC_FEATURES,
            all_features=FEATURE_NAMES
        )

        logger.debug("Created ResNet-50 feature extractor architecture")
        return model

    def _load_model(self) -> nn.Module:
        """
        Load model weights from checkpoint file.

        Returns:
            Loaded model in evaluation mode

        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model loading fails
        """
        model_path = settings.MODEL_B_PATH

        # Check if model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model B file not found at: {model_path}. "
                f"Please ensure the model file exists at this location."
            )

        try:
            # Create model architecture
            model = self._create_model_architecture()

            # Load weights
            logger.info(f"Loading Model B from: {model_path}")
            state_dict = torch.load(
                model_path,
                map_location=self._device,
                weights_only=True
            )

            model.load_state_dict(state_dict)

            # Set to evaluation mode
            model.eval()

            # Move to device
            model.to(self._device)

            logger.info(f"Model B loaded successfully on {self._device}")

            return model

        except Exception as e:
            logger.error(f"Failed to load Model B: {str(e)}")
            raise RuntimeError(f"Failed to load Model B: {str(e)}")

    def get_model(self) -> nn.Module:
        """Get the loaded model instance."""
        if self._model is None:
            raise RuntimeError("Model B not initialized")
        return self._model

    def get_device(self) -> torch.device:
        """Get the device the model is running on."""
        return self._device


def normalize_diameter(diameter: float) -> float:
    """
    Normalize diameter using training statistics.

    Args:
        diameter: Lesion diameter in millimeters

    Returns:
        Normalized diameter
    """
    # These values come from your training data
    # diam_mean and diam_std from your notebook
    diam_mean = settings.MODEL_B_DIAM_MEAN
    diam_std = settings.MODEL_B_DIAM_STD

    normalized = (diameter - diam_mean) / diam_std
    return float(normalized)


def denormalize_features(features_normalized: np.ndarray) -> np.ndarray:
    """
    Denormalize predicted features using training statistics.

    Args:
        features_normalized: Normalized features (18,)

    Returns:
        Denormalized features in original scale
    """
    # These come from your training data (y_means and y_stds)
    y_means = np.array(settings.MODEL_B_FEATURE_MEANS)
    y_stds = np.array(settings.MODEL_B_FEATURE_STDS)

    features_denormalized = features_normalized * y_stds + y_means
    return features_denormalized


def extract_features_with_model_b(
    image_array: np.ndarray,
    diameter: float
) -> np.ndarray:
    """
    Extract 18 clinical features using Model B.

    This function:
    1. Validates inputs
    2. Normalizes diameter
    3. Converts image to PyTorch tensor
    4. Performs forward pass
    5. Denormalizes output features
    6. Returns features as numpy array

    Args:
        image_array: Preprocessed image (3, 224, 224), float32
        diameter: Lesion diameter in millimeters

    Returns:
        Array of 18 extracted features in original scale

    Raises:
        ValueError: If input shape or values are invalid
        RuntimeError: If feature extraction fails

    Example:
        >>> from app.utils import preprocess_image_for_model_a
        >>> image_array = await preprocess_image_for_model_a(image_file)
        >>> features = extract_features_with_model_b(image_array, diameter=6.5)
        >>> print(f"Extracted {len(features)} features")
    """
    # Validate image input
    if not isinstance(image_array, np.ndarray):
        raise ValueError(f"Expected numpy array, got {type(image_array)}")

    expected_shape = (3, 224, 224)
    if image_array.shape != expected_shape:
        raise ValueError(
            f"Invalid image shape: {image_array.shape}. Expected {expected_shape}"
        )

    if image_array.dtype != np.float32:
        raise ValueError(f"Invalid dtype: {image_array.dtype}. Expected float32")

    # Validate diameter
    if not isinstance(diameter, (int, float)) or diameter <= 0:
        raise ValueError(f"Invalid diameter: {diameter}. Must be positive number")

    try:
        # Get model and device
        loader = ModelBLoader()
        model = loader.get_model()
        device = loader.get_device()

        # Normalize diameter
        diameter_norm = normalize_diameter(diameter)

        # Convert to PyTorch tensors
        image_tensor = torch.from_numpy(image_array).unsqueeze(0)  # (1, 3, 224, 224)
        diameter_tensor = torch.tensor([[diameter_norm]], dtype=torch.float32)  # (1, 1)

        # Move to device
        image_tensor = image_tensor.to(device)
        diameter_tensor = diameter_tensor.to(device)

        # Forward pass
        with torch.no_grad():
            features_norm = model(image_tensor, diameter_tensor)  # (1, 18)
            features_norm = features_norm.cpu().numpy().squeeze()  # (18,)

        # Denormalize features
        features = denormalize_features(features_norm)

        logger.info(f"Model B extracted {len(features)} features")
        logger.debug(f"Feature range: [{features.min():.3f}, {features.max():.3f}]")

        return features.astype(np.float32)

    except Exception as e:
        logger.error(f"Model B feature extraction failed: {str(e)}")
        raise RuntimeError(f"Model B feature extraction failed: {str(e)}")


def get_feature_names() -> list:
    """
    Get names of the 18 features extracted by Model B.

    Returns:
        List of feature names
    """
    return FEATURE_NAMES.copy()


def validate_model_b_exists() -> bool:
    """
    Check if Model B file exists at the configured path.

    Returns:
        True if model file exists, False otherwise
    """
    model_path = settings.MODEL_B_PATH
    exists = os.path.exists(model_path)

    if exists:
        logger.info(f"Model B file found at: {model_path}")
    else:
        logger.warning(f"Model B file not found at: {model_path}")

    return exists


def get_model_b_info() -> dict:
    """
    Get information about Model B.

    Returns:
        Dictionary with model information
    """
    try:
        loader = ModelBLoader()
        model = loader.get_model()
        device = loader.get_device()

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())

        return {
            "model_name": "ResNet-50 Feature Extractor",
            "model_path": settings.MODEL_B_PATH,
            "device": str(device),
            "total_parameters": total_params,
            "num_features": 18,
            "feature_names": FEATURE_NAMES,
            "geometric_features": GEOMETRIC_FEATURES,
            "input": "image (3, 224, 224) + diameter (mm)",
            "output": "18 clinical features"
        }
    except Exception as e:
        logger.error(f"Failed to get Model B info: {str(e)}")
        return {"error": str(e)}
