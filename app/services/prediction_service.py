"""
Prediction pipeline service.

This module orchestrates the complete prediction pipeline using all 3 models:
1. Model A: DenseNet-121 image classifier
2. Model B: ResNet-50 feature extractor
3. Model C: XGBoost classifier

Returns individual probabilities from Model A and Model C for separate display in frontend.
"""

import numpy as np
from typing import Dict
from fastapi import UploadFile

from app.core.logger import logger
from app.utils.image_preprocessing import preprocess_image_for_model_a
from app.models import (
    predict_with_model_a,
    extract_features_with_model_b,
    predict_with_model_c,
)


async def run_full_prediction_pipeline(
    image_file: UploadFile,
    age: int,
    sex: str,
    location: str,
    diameter: float
) -> Dict:
    """
    Execute the complete prediction pipeline using all 3 models.

    Pipeline steps:
    1. Preprocess image for models
    2. Model A: Direct image -> probability
    3. Model B: Image + diameter -> 18 clinical features
    4. Model C: Features + metadata -> probability
    5. Return individual probabilities (no combining)

    Args:
        image_file: Uploaded image file
        age: Patient age in years
        sex: Patient sex ("male" or "female")
        location: Lesion anatomical location
        diameter: Lesion diameter in millimeters

    Returns:
        Dictionary with:
        - model_a_probability: Model A (DenseNet-121) prediction [0, 1]
        - model_c_probability: Model C (XGBoost) prediction [0, 1]
        - extracted_features: 18 features from Model B (ResNet-50)
        - metadata: Input metadata used for prediction

    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If pipeline fails

    Example:
        >>> result = await run_full_prediction_pipeline(
        ...     image_file=file,
        ...     age=45,
        ...     sex="female",
        ...     location="Torso Back",
        ...     diameter=6.5
        ... )
        >>> print(f"Model A: {result['model_a_probability']:.2%}")
        >>> print(f"Model C: {result['model_c_probability']:.2%}")
    """
    logger.info("Starting prediction pipeline")
    logger.info(f"Metadata: age={age}, sex={sex}, location={location}, diameter={diameter}mm")

    try:
        # STEP 1: Preprocess image
        logger.info("Step 1: Preprocessing image")
        image_array = await preprocess_image_for_model_a(image_file)
        logger.info(f"Image preprocessed: shape={image_array.shape}, dtype={image_array.dtype}")

        # STEP 2: Model A prediction
        logger.info("Step 2: Running Model A (DenseNet-121)")
        probability_a = predict_with_model_a(image_array)
        logger.info(f"Model A probability: {probability_a:.4f}")

        # STEP 3: Model B feature extraction
        logger.info("Step 3: Running Model B (ResNet-50) feature extraction")
        extracted_features = extract_features_with_model_b(image_array, diameter)
        logger.info(f"Model B extracted {len(extracted_features)} features")

        # STEP 4: Model C prediction
        logger.info("Step 4: Running Model C (XGBoost)")
        probability_c = predict_with_model_c(
            extracted_features,
            age,
            sex,
            location,
            diameter
        )
        logger.info(f"Model C probability: {probability_c:.4f}")

        # Prepare response - return individual probabilities without combining
        result = {
            "model_a_probability": round(probability_a, 4),
            "model_c_probability": round(probability_c, 4),
            "extracted_features": extracted_features.tolist(),
            "metadata": {
                "age": age,
                "sex": sex,
                "location": location,
                "diameter": diameter
            }
        }

        logger.info("Pipeline completed successfully")
        return result

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise RuntimeError(f"Prediction pipeline failed: {str(e)}")
