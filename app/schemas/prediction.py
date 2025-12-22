"""
Pydantic schemas for prediction request and response validation.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any
from app.utils.metadata_preprocessing import (
    normalize_location,
    validate_sex,
    get_valid_locations,
    get_valid_sex_values
)


class PredictionRequest(BaseModel):
    """
    Schema for prediction request validation.
    Note: Image is handled separately as UploadFile in the endpoint.
    """
    age: int = Field(..., ge=0, le=120, description="Patient age in years")
    sex: str = Field(..., description="Patient sex (male/female)")
    location: str = Field(..., description="Lesion location on body")
    diameter: float = Field(..., gt=0, description="Lesion diameter in millimeters")

    @field_validator("sex")
    @classmethod
    def validate_sex_field(cls, v: str) -> str:
        """Validate sex field accepts only male or female."""
        # Use centralized validation
        return validate_sex(v)

    @field_validator("location")
    @classmethod
    def validate_location_field(cls, v: str) -> str:
        """Validate and normalize location field."""
        # Use centralized validation and normalization
        return normalize_location(v)


class PredictionResponse(BaseModel):
    """
    Schema for prediction response.
    Contains individual probabilities from Model A and Model C.
    Frontend will display them separately without combining.
    """
    model_a_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model A (DenseNet-121) malignancy probability"
    )
    model_c_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model C (XGBoost) malignancy probability"
    )
    extracted_features: List[float] = Field(
        ...,
        min_length=18,
        max_length=18,
        description="18 features extracted by Model B (ResNet-50)"
    )
    metadata: Dict[str, Any] = Field(
        ...,
        description="Input metadata used for prediction"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "model_a_probability": 0.72,
                "model_c_probability": 0.58,
                "extracted_features": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                       1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
                "metadata": {
                    "age": 45,
                    "sex": "female",
                    "location": "Torso Back",
                    "diameter": 6.5
                }
            }
        }


class FeatureContribution(BaseModel):
    """Individual feature contribution to the prediction."""
    feature_name: str = Field(..., description="Name of the feature")
    shap_value: float = Field(..., description="SHAP value (contribution to prediction)")
    feature_value: float = Field(..., description="Actual value of the feature")
    impact: str = Field(..., description="'increases' or 'decreases' risk")


class ExplanationResponse(BaseModel):
    """
    Schema for SHAP explanation response.
    Explains how each feature contributed to Model C's prediction.
    """
    prediction: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model C prediction probability"
    )
    base_value: float = Field(
        ...,
        description="Base prediction value (expected output)"
    )
    feature_contributions: List[FeatureContribution] = Field(
        ...,
        description="SHAP values for each feature, sorted by importance"
    )
    metadata: Dict[str, Any] = Field(
        ...,
        description="Input metadata used for prediction"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "prediction": 0.0471,
                "base_value": 0.4238,
                "feature_contributions": [
                    {
                        "feature_name": "tbp_lv_norm_color",
                        "shap_value": -0.803790,
                        "feature_value": 3.0,
                        "impact": "decreases"
                    },
                    {
                        "feature_name": "tbp_lv_H",
                        "shap_value": -0.758406,
                        "feature_value": 55.0,
                        "impact": "decreases"
                    },
                    {
                        "feature_name": "tbp_lv_deltaA",
                        "shap_value": 0.527082,
                        "feature_value": 5.0,
                        "impact": "increases"
                    }
                ],
                "metadata": {
                    "age": 45,
                    "sex": "male",
                    "location": "Left Arm",
                    "diameter": 6.5
                }
            }
        }
