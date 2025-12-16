"""
Image preprocessing utilities for skin lesion classification models.

This module provides functions to preprocess dermoscopic images according to
the exact specifications used during model training (validation/test transforms).

Important: Preprocessing must match the validation pipeline used during training.
No data augmentation is applied during inference.
"""

import io
import numpy as np
from PIL import Image
from typing import Union, Tuple
from fastapi import UploadFile
from app.core.logger import logger


# ImageNet normalization statistics (used during model training)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Model input size
MODEL_INPUT_SIZE = 224


def validate_image(image: Image.Image) -> None:
    """
    Validate that the image meets basic requirements.

    Args:
        image: PIL Image object

    Raises:
        ValueError: If image is invalid or corrupted
    """
    if image is None:
        raise ValueError("Image is None")

    if image.size[0] == 0 or image.size[1] == 0:
        raise ValueError(f"Invalid image dimensions: {image.size}")

    # Check if image is too small
    min_size = 50
    if image.size[0] < min_size or image.size[1] < min_size:
        logger.warning(f"Image is very small: {image.size}. May affect prediction quality.")


async def load_image_from_upload(image_file: UploadFile) -> Image.Image:
    """
    Load image from FastAPI UploadFile.

    Args:
        image_file: FastAPI UploadFile object

    Returns:
        PIL Image object in RGB format

    Raises:
        ValueError: If image cannot be loaded or is corrupted
    """
    try:
        # Read bytes from upload
        image_bytes = await image_file.read()

        # Open image
        image = Image.open(io.BytesIO(image_bytes))

        # Validate
        validate_image(image)

        # Convert to RGB (handles grayscale, RGBA, etc.)
        image = image.convert("RGB")

        logger.info(f"Loaded image: {image.size}, mode: {image.mode}, format: {image.format}")
        return image

    except Exception as e:
        logger.error(f"Failed to load image: {str(e)}")
        raise ValueError(f"Invalid or corrupted image file: {str(e)}")


def resize_image(image: Image.Image, size: int = MODEL_INPUT_SIZE) -> Image.Image:
    """
    Resize image to specified size using bilinear interpolation.

    Args:
        image: PIL Image object
        size: Target size (width and height)

    Returns:
        Resized PIL Image
    """
    # Use BILINEAR interpolation (same as Albumentations cv2.INTER_LINEAR default)
    resized = image.resize((size, size), Image.BILINEAR)
    logger.debug(f"Resized image from {image.size} to {resized.size}")
    return resized


def normalize_image(image_array: np.ndarray) -> np.ndarray:
    """
    Normalize image using ImageNet statistics.

    Applies the normalization: (pixel_value / 255.0 - mean) / std

    Args:
        image_array: Numpy array of shape (H, W, C) with values in [0, 255]

    Returns:
        Normalized array of shape (H, W, C) with dtype float32
    """
    # Scale to [0, 1]
    image_array = image_array.astype(np.float32) / 255.0

    # Apply ImageNet normalization
    normalized = (image_array - IMAGENET_MEAN) / IMAGENET_STD

    return normalized.astype(np.float32)


def convert_to_chw_format(image_array: np.ndarray) -> np.ndarray:
    """
    Convert image from (H, W, C) to (C, H, W) format for PyTorch.

    Args:
        image_array: Numpy array of shape (H, W, C)

    Returns:
        Numpy array of shape (C, H, W)
    """
    # Transpose from (H, W, C) to (C, H, W)
    chw_array = np.transpose(image_array, (2, 0, 1))
    return chw_array


async def preprocess_image_for_model_a(
    image_file: UploadFile,
    target_size: int = MODEL_INPUT_SIZE
) -> np.ndarray:
    """
    Preprocess image for Model A (DenseNet-121) inference.

    Replicates the validation transforms from training:
    1. Load image from bytes
    2. Convert to RGB
    3. Resize to 224x224 (bilinear interpolation)
    4. Normalize with ImageNet statistics
    5. Convert to (C, H, W) format
    6. Return as float32 numpy array

    This function does NOT apply data augmentation (flips, rotations, etc.)
    as those are only used during training.

    Args:
        image_file: FastAPI UploadFile containing the image
        target_size: Target size for resizing (default: 224)

    Returns:
        Preprocessed image as numpy array of shape (3, 224, 224) with dtype float32

    Raises:
        ValueError: If image is invalid or preprocessing fails
    """
    try:
        # Step 1: Load image from upload
        image = await load_image_from_upload(image_file)

        # Step 2: Resize to target size
        image = resize_image(image, size=target_size)

        # Step 3: Convert to numpy array
        image_array = np.array(image)  # Shape: (H, W, C), dtype: uint8

        # Step 4: Normalize with ImageNet statistics
        image_array = normalize_image(image_array)  # Shape: (H, W, C), dtype: float32

        # Step 5: Convert to (C, H, W) format for PyTorch
        image_array = convert_to_chw_format(image_array)  # Shape: (C, H, W), dtype: float32

        logger.info(f"Image preprocessed successfully. Final shape: {image_array.shape}, dtype: {image_array.dtype}")

        return image_array

    except Exception as e:
        logger.error(f"Image preprocessing failed: {str(e)}")
        raise ValueError(f"Failed to preprocess image: {str(e)}")


def preprocess_image_from_path(image_path: str, target_size: int = MODEL_INPUT_SIZE) -> np.ndarray:
    """
    Preprocess image from file path (for testing purposes).

    Args:
        image_path: Path to image file
        target_size: Target size for resizing (default: 224)

    Returns:
        Preprocessed image as numpy array of shape (3, 224, 224) with dtype float32

    Raises:
        ValueError: If image cannot be loaded or preprocessing fails
    """
    try:
        # Load image
        image = Image.open(image_path)
        validate_image(image)
        image = image.convert("RGB")

        # Resize
        image = resize_image(image, size=target_size)

        # Convert to numpy array
        image_array = np.array(image)

        # Normalize
        image_array = normalize_image(image_array)

        # Convert to (C, H, W)
        image_array = convert_to_chw_format(image_array)

        logger.info(f"Image from path preprocessed. Shape: {image_array.shape}, dtype: {image_array.dtype}")

        return image_array

    except Exception as e:
        logger.error(f"Failed to preprocess image from path {image_path}: {str(e)}")
        raise ValueError(f"Failed to preprocess image: {str(e)}")


def validate_preprocessed_image(image_array: np.ndarray) -> bool:
    """
    Validate that preprocessed image meets expected specifications.

    Args:
        image_array: Preprocessed image array

    Returns:
        True if valid, raises ValueError otherwise

    Raises:
        ValueError: If image doesn't meet specifications
    """
    # Check shape
    expected_shape = (3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
    if image_array.shape != expected_shape:
        raise ValueError(f"Invalid shape: {image_array.shape}, expected {expected_shape}")

    # Check dtype
    if image_array.dtype != np.float32:
        raise ValueError(f"Invalid dtype: {image_array.dtype}, expected float32")

    # Check value range (normalized values should be roughly in [-3, 3] range)
    min_val, max_val = image_array.min(), image_array.max()
    if min_val < -5 or max_val > 5:
        logger.warning(f"Unusual value range: [{min_val:.2f}, {max_val:.2f}]. Image may be corrupted.")

    logger.debug("Preprocessed image validation passed")
    return True
