"""
Prediction API endpoints.
Handles requests for skin lesion malignancy risk prediction.
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.schemas.prediction import PredictionRequest, PredictionResponse, ExplanationResponse, FeatureContribution
from app.services.prediction_service import run_full_prediction_pipeline
from app.core.logger import logger
from app.models import extract_features_with_model_b, explain_prediction_with_shap
from app.utils.image_preprocessing import preprocess_image_for_model_a


router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
async def predict_lesion(
    image: UploadFile = File(..., description="Skin lesion image file"),
    age: int = Form(..., description="Patient age in years"),
    sex: str = Form(..., description="Patient sex (male/female)"),
    location: str = Form(..., description="Lesion location on body"),
    diameter: float = Form(..., description="Lesion diameter in millimeters"),
):
    """
    Predict malignancy risk for a skin lesion.

    Processes the input image and clinical metadata through multiple AI models:
    - Model A: Deep learning image classifier
    - Model B: Feature extractor
    - Model C: Tabular classifier using extracted features and metadata

    Args:
        image: Uploaded image file of the skin lesion
        age: Patient's age in years
        sex: Patient's biological sex
        location: Anatomical location of the lesion
        diameter: Lesion diameter in millimeters

    Returns:
        PredictionResponse: Contains final probability and detailed model outputs

    Raises:
        HTTPException: If prediction fails or inputs are invalid
    """
    try:
        logger.info(f"Received prediction request - age: {age}, sex: {sex}, location: {location}, diameter: {diameter}")

        # Validate image file
        if not image.content_type or not image.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {image.content_type}. Please upload an image file."
            )

        # Run full prediction pipeline
        result = await run_full_prediction_pipeline(
            image_file=image,
            age=age,
            sex=sex,
            location=location,
            diameter=diameter
        )

        # Return validated response
        return PredictionResponse(**result)

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@router.post("/explain", response_model=ExplanationResponse)
async def explain_prediction(
    image: UploadFile = File(..., description="Skin lesion image file"),
    age: int = Form(..., description="Patient age in years"),
    sex: str = Form(..., description="Patient sex (male/female)"),
    location: str = Form(..., description="Lesion location on body"),
    diameter: float = Form(..., description="Lesion diameter in millimeters"),
):
    """
    Explain Model C prediction using SHAP values.

    Generates SHAP (SHapley Additive exPlanations) values that explain how each
    feature contributed to the Model C prediction for this specific patient.

    Args:
        image: Uploaded image file of the skin lesion
        age: Patient's age in years
        sex: Patient's biological sex
        location: Anatomical location of the lesion
        diameter: Lesion diameter in millimeters

    Returns:
        ExplanationResponse: Contains SHAP values for all features, sorted by importance

    Raises:
        HTTPException: If explanation generation fails or SHAP is not available
    """
    try:
        logger.info(f"Received explanation request - age: {age}, sex: {sex}, location: {location}, diameter: {diameter}")

        # Validate image file
        if not image.content_type or not image.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {image.content_type}. Please upload an image file."
            )

        # Preprocess image and extract features with Model B
        logger.info("Preprocessing image and extracting features...")
        image_array = await preprocess_image_for_model_a(image)
        extracted_features = extract_features_with_model_b(image_array, diameter)

        # Generate SHAP explanation
        logger.info("Generating SHAP explanation...")
        explanation = explain_prediction_with_shap(
            extracted_features,
            age,
            sex,
            location,
            diameter
        )

        # Format feature contributions and sort by absolute SHAP value
        feature_contributions = []
        for name, shap_val, feat_val in zip(
            explanation['feature_names'],
            explanation['shap_values'],
            explanation['feature_values']
        ):
            feature_contributions.append(
                FeatureContribution(
                    feature_name=name,
                    shap_value=shap_val,
                    feature_value=feat_val,
                    impact="increases" if shap_val > 0 else "decreases"
                )
            )

        # Sort by absolute SHAP value (most important first)
        feature_contributions.sort(key=lambda x: abs(x.shap_value), reverse=True)

        # Build response
        response_data = {
            "prediction": explanation['prediction'],
            "base_value": explanation['base_value'],
            "feature_contributions": feature_contributions,
            "metadata": {
                "age": age,
                "sex": sex,
                "location": location,
                "diameter": diameter
            }
        }

        return ExplanationResponse(**response_data)

    except RuntimeError as e:
        error_msg = str(e)
        if "SHAP library not available" in error_msg:
            logger.error("SHAP not available")
            raise HTTPException(
                status_code=503,
                detail="SHAP explainability feature not available. Please install SHAP library."
            )
        logger.error(f"Explanation failed: {error_msg}")
        raise HTTPException(status_code=500, detail=f"Explanation failed: {error_msg}")
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
