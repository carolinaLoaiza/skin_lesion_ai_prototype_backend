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
from app.utils.feature_names import get_friendly_name, get_all_feature_mappings
from app.data.managers.analysis_manager import analysis_manager
from app.data.managers.storage_manager import storage_manager
from app.data.models.analysis import (
    AnalysisCaseCreate, ImageData, ClinicalData, ModelOutputs, ModelOutput,
    ShapAnalysis, ShapFeature, TemporalData
)
from datetime import datetime
from typing import Optional
import uuid


router = APIRouter()

# Model B feature names (18 features) - must match training order
MODEL_B_FEATURE_NAMES = [
    "tbp_lv_A", "tbp_lv_B", "tbp_lv_C", "tbp_lv_H", "tbp_lv_L",
    "tbp_lv_areaMM2", "tbp_lv_area_perim_ratio", "tbp_lv_color_std_mean",
    "tbp_lv_deltaA", "tbp_lv_deltaB", "tbp_lv_deltaL", "tbp_lv_deltaLB",
    "tbp_lv_deltaLBnorm", "tbp_lv_minorAxisMM", "tbp_lv_norm_color",
    "tbp_lv_perimeterMM", "tbp_lv_stdL", "tbp_lv_symm_2axis"
]


@router.post("/predict", response_model=PredictionResponse)
async def predict_lesion(
    image: UploadFile = File(..., description="Skin lesion image file"),
    age: int = Form(..., description="Patient age in years"),
    sex: str = Form(..., description="Patient sex (male/female)"),
    location: str = Form(..., description="Lesion location on body"),
    diameter: float = Form(..., description="Lesion diameter in millimeters"),
    patient_id: Optional[str] = Form(None, description="Optional patient ID to save analysis to database"),
    lesion_id: Optional[str] = Form(None, description="Optional lesion ID to save analysis to database"),
):
    """
    Predict malignancy risk for a skin lesion.

    Processes the input image and clinical metadata through multiple AI models:
    - Model A: Deep learning image classifier
    - Model B: Feature extractor
    - Model C: Tabular classifier using extracted features and metadata

    If patient_id and lesion_id are provided, the analysis will be automatically
    saved to the database and an analysis_id will be returned.

    Args:
        image: Uploaded image file of the skin lesion
        age: Patient's age in years
        sex: Patient's biological sex
        location: Anatomical location of the lesion
        diameter: Lesion diameter in millimeters
        patient_id: Optional patient ID (e.g., "PAT-001") to save analysis
        lesion_id: Optional lesion ID (e.g., "LES-001") to save analysis

    Returns:
        PredictionResponse: Contains probabilities, extracted features, and analysis_id if saved

    Raises:
        HTTPException: If prediction fails or inputs are invalid
    """
    try:
        logger.info(f"Received prediction request - age: {age}, sex: {sex}, location: {location}, diameter: {diameter}")

        if patient_id and lesion_id:
            logger.info(f"Will save to database - patient_id: {patient_id}, lesion_id: {lesion_id}")

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

        # If patient_id and lesion_id provided, save to database
        analysis_id = None
        if patient_id and lesion_id:
            try:
                # Generate unique analysis ID
                timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
                analysis_id = f"AN-{timestamp}-{str(uuid.uuid4())[:8]}"

                # Reset file pointer before saving (it was read during prediction)
                await image.seek(0)

                # Save uploaded image using StorageManager
                image_info = await storage_manager.save_uploaded_image(
                    file=image,
                    analysis_id=analysis_id,
                    patient_id=patient_id
                )

                # Generate SHAP explanation for the analysis
                shap_explanation = explain_prediction_with_shap(
                    result['extracted_features'],
                    age,
                    sex,
                    location,
                    diameter
                )

                # Create ALL features list (28 features total) with new structure
                all_features = []
                for name, shap_val, feat_val in zip(
                    shap_explanation['feature_names'],
                    shap_explanation['shap_values'],
                    shap_explanation['feature_values']
                ):
                    all_features.append(
                        ShapFeature(
                            feature=name,
                            display_name=get_friendly_name(name),
                            value=float(feat_val),
                            shap_value=float(shap_val),
                            impact="increases" if shap_val > 0 else "decreases"
                        )
                    )

                # Create Model B extracted features list
                model_b_features = []
                for i, feature_name in enumerate(MODEL_B_FEATURE_NAMES):
                    model_b_features.append({
                        "feature_name": feature_name,
                        "value": float(result['extracted_features'][i])
                    })

                # Create analysis case
                analysis_data = AnalysisCaseCreate(
                    analysis_id=analysis_id,
                    patient_id=patient_id,
                    lesion_id=lesion_id,
                    image=ImageData(
                        filename=image_info["filename"],
                        path=image_info["path"]
                    ),
                    clinical_data=ClinicalData(
                        age_at_capture=age,
                        lesion_size_mm=diameter
                    ),
                    model_outputs=ModelOutputs(
                        image_only_model=ModelOutput(
                            malignant_probability=result['model_a_probability']
                        ),
                        clinical_ml_model=ModelOutput(
                            malignant_probability=result['model_c_probability']
                        ),
                        extracted_features=model_b_features
                    ),
                    shap_analysis=ShapAnalysis(
                        prediction=float(shap_explanation['prediction']),
                        base_value=float(shap_explanation['base_value']),
                        features=all_features
                    ),
                    temporal_data=TemporalData(
                        capture_date=datetime.utcnow(),
                        days_since_first_observation=0  # Will be updated if needed
                    )
                )

                # Save to database
                await analysis_manager.create_analysis(analysis_data)
                logger.info(f"Analysis saved to database with ID: {analysis_id}")

            except Exception as e:
                logger.error(f"Failed to save analysis to database: {str(e)}")
                # Don't fail the request if DB save fails, just log it
                analysis_id = None

        # Add analysis_id to result
        result["analysis_id"] = analysis_id

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
                    display_name=get_friendly_name(name),
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


@router.get("/feature-names", response_model=dict)
async def get_feature_names():
    """
    Get mapping of all technical feature names to user-friendly display names.

    This endpoint provides a complete dictionary of feature name translations
    that the frontend can use to display feature information in a user-friendly way.

    Returns:
        dict: Mapping of technical feature names to display names

    Example response:
        {
            "tbp_lv_A": "Color Component A* (Inside Lesion)",
            "tbp_lv_areaMM2": "Lesion Area (mm²)",
            "age_approx": "Patient Age",
            ...
        }
    """
    try:
        return get_all_feature_mappings()
    except Exception as e:
        logger.error(f"Error retrieving feature names: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving feature names: {str(e)}")
