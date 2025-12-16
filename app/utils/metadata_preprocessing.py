"""
Metadata preprocessing utilities for clinical features.

This module provides functions to preprocess and encode clinical metadata
(age, sex, location, diameter) for use in Model C (tabular classifier).

The preprocessing must match the encoding used during model training.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from app.core.logger import logger


# Valid anatomical locations (normalized to lowercase with underscores)
VALID_LOCATIONS = [
    "head", "neck", "trunk", "upper_extremity", "lower_extremity",
    "abdomen", "back", "chest", "arm", "leg", "hand", "foot", "face"
]

# Sex encoding
SEX_ENCODING = {
    "male": 0,
    "female": 1
}


def normalize_location(location: str) -> str:
    """
    Normalize location string to standardized format.

    Args:
        location: Raw location string from user input

    Returns:
        Normalized location string (lowercase with underscores)

    Raises:
        ValueError: If location is not valid
    """
    # Convert to lowercase and replace spaces with underscores
    normalized = location.lower().strip().replace(" ", "_")

    # Validate
    if normalized not in VALID_LOCATIONS:
        raise ValueError(
            f"Invalid location '{location}'. Must be one of: {', '.join(VALID_LOCATIONS)}"
        )

    return normalized


def encode_sex(sex: str) -> int:
    """
    Encode sex as binary integer.

    Args:
        sex: Sex string ("male" or "female")

    Returns:
        Encoded sex (0 for male, 1 for female)

    Raises:
        ValueError: If sex is not valid
    """
    sex_lower = sex.lower().strip()

    if sex_lower not in SEX_ENCODING:
        raise ValueError(f"Invalid sex '{sex}'. Must be 'male' or 'female'")

    return SEX_ENCODING[sex_lower]


def encode_location_onehot(location: str) -> np.ndarray:
    """
    Encode location as one-hot vector.

    Args:
        location: Normalized location string

    Returns:
        One-hot encoded vector of length len(VALID_LOCATIONS)
    """
    # Normalize first
    location = normalize_location(location)

    # Create one-hot vector
    onehot = np.zeros(len(VALID_LOCATIONS), dtype=np.float32)
    index = VALID_LOCATIONS.index(location)
    onehot[index] = 1.0

    logger.debug(f"Location '{location}' encoded to index {index}")
    return onehot


def normalize_age(age: int, min_age: int = 0, max_age: int = 120) -> float:
    """
    Normalize age to [0, 1] range.

    Args:
        age: Age in years
        min_age: Minimum valid age (default: 0)
        max_age: Maximum valid age (default: 120)

    Returns:
        Normalized age in [0, 1]

    Raises:
        ValueError: If age is out of valid range
    """
    if age < min_age or age > max_age:
        raise ValueError(f"Age {age} is out of valid range [{min_age}, {max_age}]")

    normalized = (age - min_age) / (max_age - min_age)
    return float(normalized)


def normalize_diameter(diameter: float, min_diameter: float = 0.1, max_diameter: float = 50.0) -> float:
    """
    Normalize diameter to reasonable range.

    Args:
        diameter: Diameter in millimeters
        min_diameter: Minimum valid diameter (default: 0.1 mm)
        max_diameter: Maximum valid diameter (default: 50 mm)

    Returns:
        Normalized diameter

    Raises:
        ValueError: If diameter is out of valid range
    """
    if diameter <= 0:
        raise ValueError(f"Diameter must be positive, got {diameter}")

    if diameter < min_diameter or diameter > max_diameter:
        logger.warning(
            f"Diameter {diameter} mm is outside typical range [{min_diameter}, {max_diameter}] mm"
        )

    # Simple min-max normalization
    normalized = (diameter - min_diameter) / (max_diameter - min_diameter)
    # Clip to [0, 1] in case of outliers
    normalized = np.clip(normalized, 0.0, 1.0)

    return float(normalized)


def preprocess_metadata(
    age: int,
    sex: str,
    location: str,
    diameter: float
) -> Dict[str, any]:
    """
    Preprocess all clinical metadata.

    Args:
        age: Patient age in years
        sex: Patient sex ("male" or "female")
        location: Lesion location
        diameter: Lesion diameter in millimeters

    Returns:
        Dictionary containing preprocessed features:
        - age_normalized: float in [0, 1]
        - sex_encoded: int (0 or 1)
        - location_normalized: str (normalized format)
        - location_onehot: np.ndarray (one-hot encoding)
        - diameter_normalized: float in [0, 1]

    Raises:
        ValueError: If any input is invalid
    """
    try:
        # Normalize age
        age_norm = normalize_age(age)

        # Encode sex
        sex_enc = encode_sex(sex)

        # Normalize and encode location
        location_norm = normalize_location(location)
        location_onehot = encode_location_onehot(location_norm)

        # Normalize diameter
        diameter_norm = normalize_diameter(diameter)

        preprocessed = {
            "age_normalized": age_norm,
            "sex_encoded": sex_enc,
            "location_normalized": location_norm,
            "location_onehot": location_onehot,
            "diameter_normalized": diameter_norm
        }

        logger.info(
            f"Metadata preprocessed: age={age}→{age_norm:.3f}, "
            f"sex={sex}→{sex_enc}, location={location}→{location_norm}, "
            f"diameter={diameter}→{diameter_norm:.3f}"
        )

        return preprocessed

    except Exception as e:
        logger.error(f"Metadata preprocessing failed: {str(e)}")
        raise ValueError(f"Failed to preprocess metadata: {str(e)}")


def create_feature_vector_for_model_c(
    extracted_features: np.ndarray,
    age: int,
    sex: str,
    location: str,
    diameter: float
) -> np.ndarray:
    """
    Create complete feature vector for Model C.

    Combines:
    - 18 features extracted by Model B
    - Clinical metadata (age, sex, location, diameter)

    Args:
        extracted_features: Array of 18 features from Model B
        age: Patient age in years
        sex: Patient sex
        location: Lesion location
        diameter: Lesion diameter in mm

    Returns:
        Complete feature vector as numpy array (float32)

    Raises:
        ValueError: If inputs are invalid
    """
    # Validate extracted features
    if len(extracted_features) != 18:
        raise ValueError(f"Expected 18 features from Model B, got {len(extracted_features)}")

    # Preprocess metadata
    metadata = preprocess_metadata(age, sex, location, diameter)

    # Combine features
    # Order: [18 model B features, age, sex, diameter, location_onehot (13 values)]
    feature_vector = np.concatenate([
        extracted_features.flatten(),  # 18 features
        [metadata["age_normalized"]],  # 1 feature
        [metadata["sex_encoded"]],     # 1 feature
        [metadata["diameter_normalized"]],  # 1 feature
        metadata["location_onehot"]    # 13 features
    ]).astype(np.float32)

    logger.info(f"Created feature vector for Model C: shape={feature_vector.shape}, dtype={feature_vector.dtype}")

    return feature_vector


def validate_metadata(age: int, sex: str, location: str, diameter: float) -> bool:
    """
    Validate clinical metadata before preprocessing.

    Args:
        age: Patient age
        sex: Patient sex
        location: Lesion location
        diameter: Lesion diameter

    Returns:
        True if all metadata is valid

    Raises:
        ValueError: If any metadata is invalid
    """
    # Validate age
    if not isinstance(age, int) or age < 0 or age > 120:
        raise ValueError(f"Invalid age: {age}. Must be integer between 0 and 120")

    # Validate sex
    if sex.lower() not in ["male", "female"]:
        raise ValueError(f"Invalid sex: {sex}. Must be 'male' or 'female'")

    # Validate location
    location_normalized = location.lower().strip().replace(" ", "_")
    if location_normalized not in VALID_LOCATIONS:
        raise ValueError(
            f"Invalid location: {location}. Must be one of: {', '.join(VALID_LOCATIONS)}"
        )

    # Validate diameter
    if not isinstance(diameter, (int, float)) or diameter <= 0:
        raise ValueError(f"Invalid diameter: {diameter}. Must be positive number")

    logger.debug("Metadata validation passed")
    return True
