"""
Prediction API endpoints.
Handles requests for skin lesion malignancy risk prediction.
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.schemas.prediction import PredictionRequest, PredictionResponse
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

        # TODO: Implement prediction pipeline
        # 1. Validate and preprocess image
        # 2. Validate and preprocess metadata
        # 3. Run Model A prediction
        # 4. Run Model B feature extraction
        # 5. Prepare input for Model C
        # 6. Run Model C prediction
        # 7. Combine predictions
        # 8. Return response

        # Placeholder response for initial structure
        return PredictionResponse(
            final_probability=0.0,
            model_a_probability=0.0,
            model_c_probability=0.0,
            extracted_features=[0.0] * 18,
            metadata={
                "age": age,
                "sex": sex,
                "location": location,
                "diameter": diameter
            }
        )

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
