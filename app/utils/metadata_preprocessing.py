"""
Metadata preprocessing utilities for clinical features.

This module provides centralized preprocessing for clinical metadata
(age, sex, location, diameter) used by Models B and C.

IMPORTANT: The preprocessing must match exactly what was done during training.
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from app.core.logger import logger


# Valid anatomical locations (from training data)
# These are the exact categories used during Model C training
VALID_LOCATIONS = [
    "Head & Neck",
    "Left Arm",
    "Left Leg",
    "Right Arm",
    "Right Leg",
    "Torso Back",
    "Torso Front",
    "Unknown"
]

# Categories dropped by get_dummies(drop_first=True) during training
# When drop_first=True, pandas drops the FIRST category alphabetically
DROPPED_LOCATION = "Head & Neck"  # First alphabetically
DROPPED_SEX = "female"  # First alphabetically between "female" and "male"


def normalize_location(location: str) -> str:
    """
    Normalize and validate location string.

    Args:
        location: Raw location string from user input

    Returns:
        Normalized location string matching training categories

    Raises:
        ValueError: If location is not valid
    """
    # Handle common variations and normalize
    location_map = {
        # Exact matches (case-insensitive)
        "head & neck": "Head & Neck",
        "head and neck": "Head & Neck",
        "head": "Head & Neck",
        "neck": "Head & Neck",
        "face": "Head & Neck",

        "left arm": "Left Arm",
        "left_arm": "Left Arm",

        "left leg": "Left Leg",
        "left_leg": "Left Leg",

        "right arm": "Right Arm",
        "right_arm": "Right Arm",

        "right leg": "Right Leg",
        "right_leg": "Right Leg",

        "torso back": "Torso Back",
        "torso_back": "Torso Back",
        "back": "Torso Back",

        "torso front": "Torso Front",
        "torso_front": "Torso Front",
        "front": "Torso Front",
        "chest": "Torso Front",
        "abdomen": "Torso Front",

        "unknown": "Unknown",
    }

    location_lower = location.lower().strip()

    if location_lower in location_map:
        normalized = location_map[location_lower]
        logger.debug(f"Location '{location}' normalized to '{normalized}'")
        return normalized

    # Check if it's already in correct format
    for valid_loc in VALID_LOCATIONS:
        if location.strip() == valid_loc:
            return valid_loc

    raise ValueError(
        f"Invalid location '{location}'. Valid options: {', '.join(VALID_LOCATIONS)}"
    )


def validate_sex(sex: str) -> str:
    """
    Validate and normalize sex.

    Args:
        sex: Sex string

    Returns:
        Normalized sex ("male" or "female")

    Raises:
        ValueError: If sex is not valid
    """
    sex_lower = sex.lower().strip()

    if sex_lower not in ["male", "female"]:
        raise ValueError(f"Invalid sex '{sex}'. Must be 'male' or 'female'")

    return sex_lower


def validate_age(age: int) -> int:
    """
    Validate age.

    Args:
        age: Age in years

    Returns:
        Validated age

    Raises:
        ValueError: If age is invalid
    """
    if not isinstance(age, int) or age < 0 or age > 120:
        raise ValueError(f"Invalid age: {age}. Must be integer between 0 and 120")

    return age


def validate_diameter(diameter: float) -> float:
    """
    Validate diameter.

    Args:
        diameter: Diameter in millimeters

    Returns:
        Validated diameter

    Raises:
        ValueError: If diameter is invalid
    """
    if not isinstance(diameter, (int, float)) or diameter <= 0:
        raise ValueError(f"Invalid diameter: {diameter}. Must be positive number")

    return float(diameter)


def prepare_metadata_for_model_c(
    age: int,
    sex: str,
    location: str,
    diameter: float
) -> pd.DataFrame:
    """
    Prepare metadata for Model C using exact training preprocessing.

    This function creates a DataFrame with one-hot encoded categorical variables
    that exactly matches the encoding used during Model C training.

    Process:
    1. Validate inputs
    2. Create DataFrame with raw values
    3. Apply pd.get_dummies() with drop_first=True (matching training)

    After get_dummies with drop_first=True:
    - sex: creates "sex_male" (drops "female")
    - location: creates 7 columns (drops "Head & Neck")

    Args:
        age: Patient age in years
        sex: Patient sex ("male" or "female")
        location: Lesion anatomical location
        diameter: Lesion diameter in millimeters

    Returns:
        DataFrame with encoded features: age_approx, clin_size_long_diam_mm,
        sex_male, and 7 location columns (total 10 metadata features)

    Raises:
        ValueError: If any input is invalid
    """
    # Validate inputs
    age = validate_age(age)
    sex = validate_sex(sex)
    location = normalize_location(location)
    diameter = validate_diameter(diameter)

    # Manual one-hot encoding to match training exactly
    # This ensures we get the same columns as pd.get_dummies() with drop_first=True

    # Start with numerical features
    encoded_data = {
        "age_approx": [age],
        "clin_size_long_diam_mm": [diameter],
    }

    # Sex encoding: drop_first=True drops "female", keeps "sex_male"
    # sex_male = 1 if sex is "male", 0 if sex is "female"
    encoded_data["sex_male"] = [1 if sex == "male" else 0]

    # Location encoding: drop_first=True drops "Head & Neck" alphabetically
    # Only create columns for non-dropped categories
    location_columns = {
        "Left Arm": "tbp_lv_location_simple_Left Arm",
        "Left Leg": "tbp_lv_location_simple_Left Leg",
        "Right Arm": "tbp_lv_location_simple_Right Arm",
        "Right Leg": "tbp_lv_location_simple_Right Leg",
        "Torso Back": "tbp_lv_location_simple_Torso Back",
        "Torso Front": "tbp_lv_location_simple_Torso Front",
        "Unknown": "tbp_lv_location_simple_Unknown"
    }

    # Create all location columns with 0, then set the correct one to 1
    for loc_value, col_name in location_columns.items():
        encoded_data[col_name] = [1 if location == loc_value else 0]

    # If location is "Head & Neck" (dropped), all location columns are 0
    # This is correct behavior for drop_first=True

    df_encoded = pd.DataFrame(encoded_data)

    logger.debug(f"Metadata encoded into {len(df_encoded.columns)} columns")
    logger.debug(f"  sex_male={df_encoded['sex_male'].values[0]}")
    logger.debug(f"  location='{location}' (dropped='Head & Neck')")

    return df_encoded


def normalize_diameter_for_model_b(diameter: float, mean: float, std: float) -> float:
    """
    Normalize diameter for Model B using training statistics.

    Args:
        diameter: Diameter in millimeters
        mean: Mean from training data
        std: Standard deviation from training data

    Returns:
        Normalized diameter (z-score)
    """
    diameter = validate_diameter(diameter)
    normalized = (diameter - mean) / std
    return float(normalized)


def get_valid_locations() -> List[str]:
    """
    Get list of valid location values.

    Returns:
        List of valid location strings
    """
    return VALID_LOCATIONS.copy()


def get_valid_sex_values() -> List[str]:
    """
    Get list of valid sex values.

    Returns:
        List of valid sex strings
    """
    return ["male", "female"]
