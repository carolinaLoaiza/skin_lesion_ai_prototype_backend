"""
Prediction API endpoints using Hugging Face Space.
Handles requests for skin lesion malignancy risk prediction by calling the HF Space API.
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.schemas.prediction import PredictionResponse, ExplanationResponse, FeatureContribution
from app.core.logger import logger
from app.utils.feature_names import get_friendly_name, get_all_feature_mappings
from app.utils.metadata_preprocessing import normalize_location
from app.data.managers.analysis_manager import analysis_manager
from app.data.managers.storage_manager import storage_manager
from app.data.models.analysis import (
    AnalysisCaseCreate, ImageData, ClinicalData, ModelOutputs, ModelOutput,
    ShapAnalysis, ShapFeature, TemporalData
)
from datetime import datetime
from typing import Optional, Dict, Any, List
import uuid
import requests
import json


router = APIRouter()

# Hugging Face Space configuration
HF_SPACE_URL = "https://carolinaloaiza-skin-lesion-inference.hf.space"
HF_UPLOAD_TIMEOUT = 30
HF_PREDICT_TIMEOUT = 10
HF_SSE_TIMEOUT = 120

# Model B feature names (18 features) - must match training order
MODEL_B_FEATURE_NAMES = [
    "tbp_lv_A", "tbp_lv_B", "tbp_lv_C", "tbp_lv_H", "tbp_lv_L",
    "tbp_lv_areaMM2", "tbp_lv_area_perim_ratio", "tbp_lv_color_std_mean",
    "tbp_lv_deltaA", "tbp_lv_deltaB", "tbp_lv_deltaL", "tbp_lv_deltaLB",
    "tbp_lv_deltaLBnorm", "tbp_lv_minorAxisMM", "tbp_lv_norm_color",
    "tbp_lv_perimeterMM", "tbp_lv_stdL", "tbp_lv_symm_2axis"
]


async def call_hugging_face_api(
    image_bytes: bytes,
    age: int,
    sex: str,
    location: str,
    diameter: float,
    include_shap: bool = True
) -> Dict[str, Any]:
    """
    Call Hugging Face Space API for prediction.

    Args:
        image_bytes: Image file bytes
        age: Patient age
        sex: Patient sex
        location: Lesion location
        diameter: Lesion diameter in mm
        include_shap: Whether to include SHAP explanation

    Returns:
        Dictionary with prediction results from HF Space

    Raises:
        HTTPException: If API call fails
    """
    try:
        # Step 1: Upload image to HF Space
        logger.info("Uploading image to Hugging Face Space...")

        files = {"files": ("image.jpg", image_bytes, "image/jpeg")}
        upload_response = requests.post(
            f"{HF_SPACE_URL}/gradio_api/upload",
            files=files,
            timeout=HF_UPLOAD_TIMEOUT
        )

        if upload_response.status_code != 200:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to upload image to HF Space: {upload_response.text}"
            )

        upload_data = upload_response.json()

        if not isinstance(upload_data, list) or len(upload_data) == 0:
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected upload response format: {upload_data}"
            )

        uploaded_file_path = upload_data[0]
        logger.info(f"Image uploaded successfully: {uploaded_file_path}")

        # Step 2: Prepare prediction payload
        image_obj = {
            "path": uploaded_file_path,
            "url": f"{HF_SPACE_URL}/gradio_api/file={uploaded_file_path}",
            "meta": {"_type": "gradio.FileData"}
        }

        payload = {
            "data": [
                image_obj,
                age,
                sex,
                location,
                diameter,
                include_shap
            ]
        }
        # Step 3: Queue prediction job
        logger.info("Queueing prediction job...")

        predict_response = requests.post(
            f"{HF_SPACE_URL}/gradio_api/call/predict",
            json=payload,
            timeout=HF_PREDICT_TIMEOUT
        )

        if predict_response.status_code != 200:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to queue prediction: {predict_response.text}"
            )

        event_data = predict_response.json()
        event_id = event_data.get("event_id")

        if not event_id:
            raise HTTPException(
                status_code=500,
                detail=f"No event_id in response: {event_data}"
            )

        logger.info(f"Prediction queued with event_id: {event_id}")

        # Step 4: Poll for results via Server-Sent Events
        logger.info("Waiting for prediction results...")

        sse_url = f"{HF_SPACE_URL}/gradio_api/call/predict/{event_id}"
        sse_response = requests.get(sse_url, stream=True, timeout=HF_SSE_TIMEOUT)

        if sse_response.status_code != 200:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to get prediction results: {sse_response.text}"
            )

        # Parse SSE stream
        for line in sse_response.iter_lines():
            if line:
                line_str = line.decode('utf-8')

                # SSE format: "data: {...}"
                if line_str.startswith('data: '):
                    data_str = line_str[6:]  # Remove "data: " prefix

                    try:
                        data = json.loads(data_str)

                        # Handle different response types
                        if isinstance(data, list) and len(data) > 0:
                            # This is the final result
                            prediction_json = data[0]

                            # Parse if it's a JSON string
                            if isinstance(prediction_json, str):
                                result = json.loads(prediction_json)
                            else:
                                result = prediction_json

                            logger.info("Prediction completed successfully")
                            return result

                    except json.JSONDecodeError:
                        # Skip non-JSON lines
                        continue

        # If we get here, no result was found
        raise HTTPException(
            status_code=500,
            detail="No prediction result received from HF Space"
        )

    except requests.exceptions.Timeout:
        logger.error("HF Space API request timeout")
        raise HTTPException(
            status_code=504,
            detail="Request to Hugging Face Space timed out"
        )
    except requests.exceptions.ConnectionError:
        logger.error("Failed to connect to HF Space")
        raise HTTPException(
            status_code=503,
            detail="Unable to connect to Hugging Face Space - service may be down"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error calling HF Space API: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get prediction from Hugging Face Space: {str(e)}"
        )


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
    Predict malignancy risk for a skin lesion using Hugging Face Space API.

    Processes the input image and clinical metadata through the deployed models on HF Space:
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
        # Normalize location before sending to HF Space
        location = normalize_location(location)  # "left leg" → "Left Leg"
        
        if patient_id and lesion_id:
            logger.info(f"Will save to database - patient_id: {patient_id}, lesion_id: {lesion_id}")

        # Validate image file
        if not image.content_type or not image.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {image.content_type}. Please upload an image file."
            )

        # Read image bytes
        image_bytes = await image.read()

        # Call Hugging Face Space API
        hf_result = await call_hugging_face_api(
            image_bytes=image_bytes,
            age=age,
            sex=sex,
            location=location,
            diameter=diameter,
            include_shap=bool(patient_id and lesion_id)  # Only request SHAP if saving to DB
        )

        # Check for errors in HF response
        if hf_result.get("status") == "error":
            raise HTTPException(
                status_code=500,
                detail=f"HF Space prediction failed: {hf_result.get('message', 'Unknown error')}"
            )

        # Extract results from HF response
        model_a_prob = hf_result.get("model_a_probability")
        model_c_prob = hf_result.get("model_c_probability")
        extracted_features = hf_result.get("extracted_features", [])
        metadata = hf_result.get("metadata", {})
        shap_data = hf_result.get("shap_explanation")

        # If patient_id and lesion_id provided, save to database
        analysis_id = None
        if patient_id and lesion_id:
            try:
                # Generate unique analysis ID
                timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
                analysis_id = f"AN-{timestamp}-{str(uuid.uuid4())[:8]}"

                # Reset file pointer before saving (it was read for HF API call)
                await image.seek(0)

                # Save uploaded image using StorageManager
                image_info = await storage_manager.save_uploaded_image(
                    file=image,
                    analysis_id=analysis_id,
                    patient_id=patient_id
                )

                # Process SHAP explanation from HF response
                all_features = []
                if shap_data and "all_contributors" in shap_data:
                    for contributor in shap_data["all_contributors"]:
                        all_features.append(
                            ShapFeature(
                                feature=contributor["feature_name"],
                                display_name=get_friendly_name(contributor["feature_name"]),
                                value=float(contributor["feature_value"]),
                                shap_value=float(contributor["shap_value"]),
                                impact=contributor["impact"]
                            )
                        )

                # Create Model B extracted features list
                model_b_features = []
                for i, feature_name in enumerate(MODEL_B_FEATURE_NAMES):
                    if i < len(extracted_features):
                        model_b_features.append({
                            "feature_name": feature_name,
                            "value": float(extracted_features[i])
                        })

                # Create analysis case
                analysis_data = AnalysisCaseCreate(
                    analysis_id=analysis_id,
                    patient_id=patient_id,
                    lesion_id=lesion_id,
                    image=ImageData(
                        filename=image_info["filename"],
                        path=image_info["path"],
                        content_type=image_info["content_type"],
                        data=image_info["data"]
                    ),
                    clinical_data=ClinicalData(
                        age_at_capture=age,
                        lesion_size_mm=diameter
                    ),
                    model_outputs=ModelOutputs(
                        image_only_model=ModelOutput(
                            malignant_probability=model_a_prob
                        ),
                        clinical_ml_model=ModelOutput(
                            malignant_probability=model_c_prob
                        ),
                        extracted_features=model_b_features
                    ),
                    shap_analysis=ShapAnalysis(
                        prediction=float(shap_data.get("prediction", model_c_prob) if shap_data else model_c_prob),
                        base_value=float(shap_data.get("base_value", 0.0) if shap_data else 0.0),
                        features=all_features
                    ),
                    temporal_data=TemporalData(
                        capture_date=datetime.utcnow(),
                        days_since_first_observation=0
                    )
                )

                # Save to database
                saved_analysis = await analysis_manager.create_analysis(analysis_data)
                logger.info(f"Analysis saved to database with ID: {analysis_id}")

            except Exception as e:
                logger.error(f"Failed to save analysis to database: {str(e)}")
                # Don't fail the request, just log the error
                analysis_id = None

        # Return response
        return PredictionResponse(
            model_a_probability=model_a_prob,
            model_c_probability=model_c_prob,
            extracted_features=extracted_features,
            metadata={
                "age": age,
                "sex": sex,
                "location": location,
                "diameter": diameter
            },
            analysis_id=analysis_id
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction request failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@router.get("/explain/{analysis_id}", response_model=ExplanationResponse)
async def get_explanation(analysis_id: str):
    """
    Get SHAP explanation for a previously saved analysis.

    This endpoint retrieves the SHAP explanation that was generated during prediction
    and saved to the database.

    Args:
        analysis_id: The unique analysis ID from a previous prediction

    Returns:
        ExplanationResponse: Contains SHAP values and feature contributions

    Raises:
        HTTPException: If analysis not found or retrieval fails
    """
    try:
        logger.info(f"Retrieving explanation for analysis: {analysis_id}")

        # Get analysis from database
        analysis = await analysis_manager.get_analysis(analysis_id)

        if not analysis:
            raise HTTPException(
                status_code=404,
                detail=f"Analysis not found: {analysis_id}"
            )

        # Extract SHAP data from Pydantic object
        shap_analysis = analysis.shap_analysis
        all_features = shap_analysis.features
        clinical_data = analysis.clinical_data

        # Convert to response format
        feature_contributions = []
        for feature in all_features:
            feature_contributions.append(
                FeatureContribution(
                    feature_name=feature.feature,
                    display_name=feature.display_name,
                    feature_value=feature.value,
                    shap_value=feature.shap_value,
                    impact=feature.impact
                )
            )

        return ExplanationResponse(
            analysis_id=analysis_id,
            prediction=shap_analysis.prediction,
            base_value=shap_analysis.base_value,
            feature_contributions=feature_contributions,
            metadata={
                "age": clinical_data.age_at_capture,
                "diameter": clinical_data.lesion_size_mm
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve explanation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve explanation: {str(e)}"
        )


@router.get("/feature-names")
async def get_feature_names():
    """
    Get all feature names and their friendly display names.

    Returns:
        Dictionary mapping technical feature names to friendly display names
    """
    try:
        logger.info("Retrieving feature name mappings")
        return get_all_feature_mappings()
    except Exception as e:
        logger.error(f"Failed to get feature names: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get feature names: {str(e)}"
        )
