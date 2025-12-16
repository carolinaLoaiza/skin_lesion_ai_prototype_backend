"""
Prediction API endpoints.
Handles requests for skin lesion malignancy risk prediction.
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.schemas.prediction import PredictionRequest, PredictionResponse
from app.services.prediction_service import run_full_prediction_pipeline
from app.core.logger import logger


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
        if not image.content_type.startswith("image/"):
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
