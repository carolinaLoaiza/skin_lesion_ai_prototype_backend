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
    Contains final probability and detailed outputs from each model.
    """
    final_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Final combined malignancy probability (0-1)"
    )
    model_a_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model A (deep learning) malignancy probability"
    )
    model_c_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model C (tabular) malignancy probability"
    )
    extracted_features: List[float] = Field(
        ...,
        min_length=18,
        max_length=18,
        description="18 features extracted by Model B"
    )
    risk_category: str = Field(
        ...,
        description="Risk category: low, medium, or high"
    )
    metadata: Dict[str, Any] = Field(
        ...,
        description="Input metadata used for prediction"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "final_probability": 0.65,
                "model_a_probability": 0.72,
                "model_c_probability": 0.58,
                "extracted_features": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                       1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
                "risk_category": "medium",
                "metadata": {
                    "age": 45,
                    "sex": "female",
                    "location": "back",
                    "diameter": 6.5
                }
            }
        }
