"""
Model A: DenseNet-121 for melanoma risk classification.

This module handles loading and inference for Model A, which is a DenseNet-121
model pretrained on ImageNet and fine-tuned for binary classification of
skin lesion malignancy risk.

Model characteristics:
- Architecture: DenseNet-121
- Input: Preprocessed image (3, 224, 224), float32
- Output: Probability of malignancy [0, 1]
- Backbone: Frozen (only final layer trained)
- Training: Binary classification with BCEWithLogitsLoss
"""

import os
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from typing import Optional
from app.core.config import settings
from app.core.logger import logger


class ModelALoader:
    """
    Singleton class to load and manage Model A (DenseNet-121).

    The model is loaded once and reused for all predictions to avoid
    reloading overhead on every request.
    """

    _instance: Optional['ModelALoader'] = None
    _model: Optional[nn.Module] = None
    _device: torch.device = None

    def __new__(cls):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super(ModelALoader, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize model loader (called once)."""
        # Determine device (CPU for compatibility)
        self._device = torch.device("cpu")
        logger.info(f"Model A will run on device: {self._device}")

        # Load model
        self._model = self._load_model()
        logger.info("Model A loaded successfully")

    def _create_model_architecture(self) -> nn.Module:
        """
        Create DenseNet-121 architecture with modified final layer.

        Must match exactly the architecture used during training:
        - DenseNet-121 pretrained on ImageNet
        - Final classifier layer replaced with Linear(num_features, 1)
        - Backbone frozen, only final layer trainable

        Returns:
            PyTorch model with correct architecture
        """
        # Load DenseNet-121 (pretrained=False since we'll load trained weights)
        model = models.densenet121(weights=None)

        # Get number of input features for the classifier
        in_features = model.classifier.in_features

        # Replace classifier with single output neuron
        model.classifier = nn.Linear(in_features, 1)

        logger.debug(f"Created DenseNet-121 architecture with {in_features} input features")

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
        model_path = settings.MODEL_A_PATH

        # Check if model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model A file not found at: {model_path}. "
                f"Please ensure the model file exists at this location."
            )

        try:
            # Create model architecture
            model = self._create_model_architecture()

            # Load checkpoint
            logger.info(f"Loading Model A from: {model_path}")
            checkpoint = torch.load(
                model_path,
                map_location=self._device,
                weights_only=False  # Load full checkpoint with metadata
            )

            # Extract model state dict
            if 'model_state_dict' in checkpoint:
                # Checkpoint includes training metadata
                state_dict = checkpoint['model_state_dict']
                epoch = checkpoint.get('epoch', 'unknown')
                val_auc_pr = checkpoint.get('val_auc_pr', 'unknown')
                logger.info(f"Loaded checkpoint from epoch {epoch}, val AUC-PR: {val_auc_pr}")
            else:
                # Checkpoint is just the state dict
                state_dict = checkpoint

            # Load weights into model
            model.load_state_dict(state_dict)

            # Set to evaluation mode
            model.eval()

            # Move to device
            model.to(self._device)

            logger.info(f"Model A loaded successfully on {self._device}")

            return model

        except Exception as e:
            logger.error(f"Failed to load Model A: {str(e)}")
            raise RuntimeError(f"Failed to load Model A: {str(e)}")

    def get_model(self) -> nn.Module:
        """
        Get the loaded model instance.

        Returns:
            Loaded PyTorch model in eval mode
        """
        if self._model is None:
            raise RuntimeError("Model A not initialized")
        return self._model

    def get_device(self) -> torch.device:
        """
        Get the device the model is running on.

        Returns:
            PyTorch device (cpu or cuda)
        """
        return self._device


def predict_with_model_a(image_array: np.ndarray) -> float:
    """
    Predict malignancy probability using Model A.

    This function:
    1. Validates input shape and dtype
    2. Converts numpy array to PyTorch tensor
    3. Adds batch dimension
    4. Performs forward pass
    5. Applies sigmoid activation
    6. Returns probability as float

    Args:
        image_array: Preprocessed image as numpy array
                    Shape: (3, 224, 224)
                    Dtype: float32
                    Normalized with ImageNet stats

    Returns:
        Probability of malignancy in range [0, 1]

    Raises:
        ValueError: If input shape or dtype is invalid
        RuntimeError: If prediction fails

    Example:
        >>> from app.utils import preprocess_image_for_model_a
        >>> image_array = await preprocess_image_for_model_a(image_file)
        >>> probability = predict_with_model_a(image_array)
        >>> print(f"Malignancy probability: {probability:.3f}")
    """
    # Validate input
    if not isinstance(image_array, np.ndarray):
        raise ValueError(f"Expected numpy array, got {type(image_array)}")

    expected_shape = (3, 224, 224)
    if image_array.shape != expected_shape:
        raise ValueError(
            f"Invalid input shape: {image_array.shape}. "
            f"Expected {expected_shape}. "
            f"Make sure image is preprocessed with preprocess_image_for_model_a()"
        )

    if image_array.dtype != np.float32:
        raise ValueError(f"Invalid dtype: {image_array.dtype}. Expected float32")

    try:
        # Get model and device
        loader = ModelALoader()
        model = loader.get_model()
        device = loader.get_device()

        # Convert to PyTorch tensor
        image_tensor = torch.from_numpy(image_array)

        # Add batch dimension: (3, 224, 224) -> (1, 3, 224, 224)
        image_tensor = image_tensor.unsqueeze(0)

        # Move to device
        image_tensor = image_tensor.to(device)

        # Forward pass (no gradient computation needed)
        with torch.no_grad():
            logits = model(image_tensor)  # Shape: (1, 1)

            # Apply sigmoid to get probability
            probability = torch.sigmoid(logits)

            # Extract scalar value
            probability = probability.item()

        logger.info(f"Model A prediction: {probability:.4f}")

        # Validate output range
        if not (0.0 <= probability <= 1.0):
            logger.warning(f"Probability out of range: {probability}")
            probability = np.clip(probability, 0.0, 1.0)

        return float(probability)

    except Exception as e:
        logger.error(f"Model A prediction failed: {str(e)}")
        raise RuntimeError(f"Model A prediction failed: {str(e)}")


def apply_threshold(probability: float, threshold: Optional[float] = None) -> int:
    """
    Apply threshold to probability to get binary classification.

    Args:
        probability: Predicted probability [0, 1]
        threshold: Decision threshold (default from config)

    Returns:
        Binary prediction: 0 (benign) or 1 (malignant)

    Example:
        >>> prob = 0.35
        >>> prediction = apply_threshold(prob, threshold=0.5)
        >>> print(prediction)  # 0 (benign)
    """
    if threshold is None:
        threshold = settings.MODEL_A_THRESHOLD

    return 1 if probability >= threshold else 0


def validate_model_exists() -> bool:
    """
    Check if Model A file exists at the configured path.

    Returns:
        True if model file exists, False otherwise
    """
    model_path = settings.MODEL_A_PATH
    exists = os.path.exists(model_path)

    if exists:
        logger.info(f"Model A file found at: {model_path}")
    else:
        logger.warning(f"Model A file not found at: {model_path}")

    return exists


def get_model_info() -> dict:
    """
    Get information about the loaded model.

    Returns:
        Dictionary with model information
    """
    try:
        loader = ModelALoader()
        model = loader.get_model()
        device = loader.get_device()

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        return {
            "model_name": "DenseNet-121",
            "model_path": settings.MODEL_A_PATH,
            "device": str(device),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "threshold": settings.MODEL_A_THRESHOLD,
            "input_size": (3, 224, 224),
            "output": "probability [0, 1]"
        }
    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}")
        return {"error": str(e)}
