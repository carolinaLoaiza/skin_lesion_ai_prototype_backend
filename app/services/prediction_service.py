"""
Prediction pipeline service.

This module orchestrates the complete prediction pipeline using all 3 models:
1. Model A: DenseNet-121 image classifier
2. Model B: ResNet-50 feature extractor
3. Model C: Random Forest classifier

The final prediction combines Model A and Model C probabilities using
a weighted average.
"""

import numpy as np
from typing import Dict
from fastapi import UploadFile

from app.core.config import settings
from app.core.logger import logger
from app.utils.image_preprocessing import preprocess_image_for_model_a
from app.models import (
    predict_with_model_a,
    extract_features_with_model_b,
    predict_with_model_c,
)


def combine_predictions(
    probability_a: float,
    probability_c: float,
    weight_a: float = None,
    weight_c: float = None
) -> float:
    """
    Combine predictions from Model A and Model C using weighted average.

    Args:
        probability_a: Probability from Model A [0, 1]
        probability_c: Probability from Model C [0, 1]
        weight_a: Weight for Model A (default from settings)
        weight_c: Weight for Model C (default from settings)

    Returns:
        Combined probability [0, 1]

    Example:
        >>> prob_a = 0.6
        >>> prob_c = 0.7
        >>> final = combine_predictions(prob_a, prob_c, 0.5, 0.5)
        >>> print(f"Final probability: {final:.3f}")
        Final probability: 0.650
    """
    # Use settings if weights not provided
    if weight_a is None:
        weight_a = settings.MODEL_A_WEIGHT
    if weight_c is None:
        weight_c = settings.MODEL_C_WEIGHT

    # Normalize weights to sum to 1
    total_weight = weight_a + weight_c
    weight_a_norm = weight_a / total_weight
    weight_c_norm = weight_c / total_weight

    # Weighted average
    combined = (probability_a * weight_a_norm) + (probability_c * weight_c_norm)

    logger.info(
        f"Combined predictions: A={probability_a:.4f} (w={weight_a_norm:.2f}), "
        f"C={probability_c:.4f} (w={weight_c_norm:.2f}) -> {combined:.4f}"
    )

    return float(combined)


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
    5. Combine Model A and Model C predictions
    6. Return comprehensive results

    Args:
        image_file: Uploaded image file
        age: Patient age in years
        sex: Patient sex ("male" or "female")
        location: Lesion anatomical location
        diameter: Lesion diameter in millimeters

    Returns:
        Dictionary with:
        - final_probability: Combined prediction [0, 1]
        - model_a_probability: Model A prediction
        - model_c_probability: Model C prediction
        - extracted_features: 18 features from Model B
        - risk_category: "low", "medium", or "high"

    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If pipeline fails

    Example:
        >>> result = await run_full_prediction_pipeline(
        ...     image_file=file,
        ...     age=45,
        ...     sex="female",
        ...     location="back",
        ...     diameter=6.5
        ... )
        >>> print(f"Risk: {result['risk_category']}")
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
        logger.info("Step 4: Running Model C (Random Forest)")
        probability_c = predict_with_model_c(
            extracted_features,
            age,
            sex,
            location,
            diameter
        )
        logger.info(f"Model C probability: {probability_c:.4f}")

        # STEP 5: Combine predictions
        logger.info("Step 5: Combining predictions")
        final_probability = combine_predictions(probability_a, probability_c)
        logger.info(f"Final combined probability: {final_probability:.4f}")

        # STEP 6: Determine risk category
        risk_category = _determine_risk_category(final_probability)
        logger.info(f"Risk category: {risk_category}")

        # Prepare response
        result = {
            "final_probability": round(final_probability, 4),
            "model_a_probability": round(probability_a, 4),
            "model_c_probability": round(probability_c, 4),
            "extracted_features": extracted_features.tolist(),
            "risk_category": risk_category,
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


def _determine_risk_category(probability: float) -> str:
    """
    Determine risk category based on malignancy probability.

    Args:
        probability: Malignancy probability [0, 1]

    Returns:
        Risk category: "low", "medium", or "high"
    """
    if probability < 0.3:
        return "low"
    elif probability < 0.7:
        return "medium"
    else:
        return "high"
