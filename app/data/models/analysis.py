"""
Pydantic models for Analysis Cases collection.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
from datetime import datetime


class ImageData(BaseModel):
    """Schema for image information."""
    filename: str = Field(..., description="Name of the image file")
    path: str = Field(..., description="Path to the stored image file")
    content_type: str = Field(default="image/jpeg", description="MIME type of the image")
    data: Optional[bytes] = Field(
        default=None,
        exclude=True,
        description="Binary image data stored in MongoDB (excluded from JSON serialization)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "filename": "lesion_day0.jpg",
                "path": "mongodb://AN-001",
                "content_type": "image/jpeg"
            }
        }
    }


class ClinicalData(BaseModel):
    """Schema for clinical data at time of capture."""
    age_at_capture: int = Field(..., ge=0, le=150, description="Patient's age at time of image capture")
    lesion_size_mm: float = Field(..., gt=0, description="Lesion size in millimeters at time of capture")

    @field_validator("age_at_capture")
    @classmethod
    def validate_age(cls, v: int) -> int:
        """Validate age is reasonable."""
        if v < 0 or v > 150:
            raise ValueError("age_at_capture must be between 0 and 150")
        return v

    @field_validator("lesion_size_mm")
    @classmethod
    def validate_size(cls, v: float) -> float:
        """Validate lesion size."""
        if v <= 0:
            raise ValueError("lesion_size_mm must be greater than 0")
        if v > 100.0:
            raise ValueError("lesion_size_mm should be less than 100mm (verify if correct)")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "age_at_capture": 45,
                "lesion_size_mm": 12.4
            }
        }
    }


class ModelOutput(BaseModel):
    """Schema for individual model output."""
    malignant_probability: float = Field(..., ge=0.0, le=1.0, description="Probability of malignancy (0-1)")

    @field_validator("malignant_probability")
    @classmethod
    def validate_probability(cls, v: float) -> float:
        """Validate probability is between 0 and 1."""
        if v < 0.0 or v > 1.0:
            raise ValueError("malignant_probability must be between 0.0 and 1.0")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "malignant_probability": 0.72
            }
        }
    }


class ModelBFeature(BaseModel):
    """Schema for individual Model B extracted feature."""
    feature_name: str = Field(..., description="Technical name of the extracted feature")
    value: float = Field(..., description="Extracted feature value")

    model_config = {
        "json_schema_extra": {
            "example": {
                "feature_name": "tbp_lv_norm_color",
                "value": 3.0
            }
        }
    }


class ModelOutputs(BaseModel):
    """Schema for all model outputs."""
    image_only_model: ModelOutput = Field(..., description="Output from Model A (DenseNet-121)")
    clinical_ml_model: ModelOutput = Field(..., description="Output from Model C (XGBoost)")
    extracted_features: List[Dict[str, Any]] = Field(..., description="18 features extracted by Model B (ResNet-50)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "image_only_model": {
                    "malignant_probability": 0.72
                },
                "clinical_ml_model": {
                    "malignant_probability": 0.65
                },
                "extracted_features": [
                    {"feature_name": "tbp_lv_norm_color", "value": 3.0},
                    {"feature_name": "tbp_lv_H", "value": 55.0}
                ]
            }
        }
    }


class ShapFeature(BaseModel):
    """Schema for individual SHAP feature contribution."""
    feature: str = Field(..., description="Technical feature name")
    display_name: str = Field(..., description="User-friendly display name")
    value: float = Field(..., description="Actual value of the feature")
    shap_value: float = Field(..., description="SHAP contribution value")
    impact: str = Field(..., description="'increases' or 'decreases' risk")

    model_config = {
        "json_schema_extra": {
            "example": {
                "feature": "tbp_lv_norm_color",
                "display_name": "Normalized Color Variation",
                "value": 3.0,
                "shap_value": -0.803790,
                "impact": "decreases"
            }
        }
    }


class ShapAnalysis(BaseModel):
    """Schema for SHAP explainability analysis."""
    prediction: float = Field(..., ge=0.0, le=1.0, description="Model C prediction probability")
    base_value: float = Field(..., description="Base value (expected model output)")
    features: List[ShapFeature] = Field(..., description="All features with SHAP contributions (28 total)")

    @field_validator("prediction")
    @classmethod
    def validate_prediction(cls, v: float) -> float:
        """Validate prediction probability."""
        if v < 0.0 or v > 1.0:
            raise ValueError("prediction must be between 0.0 and 1.0")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "prediction": 0.65,
                "base_value": 0.4238,
                "features": [
                    {
                        "feature": "tbp_lv_norm_color",
                        "display_name": "Normalized Color Variation",
                        "value": 3.0,
                        "shap_value": -0.803790,
                        "impact": "decreases"
                    },
                    {
                        "feature": "tbp_lv_H",
                        "display_name": "Hue Angle (Inside Lesion)",
                        "value": 55.0,
                        "shap_value": -0.758406,
                        "impact": "decreases"
                    }
                ]
            }
        }
    }


class TemporalData(BaseModel):
    """Schema for temporal/time-series information."""
    capture_date: datetime = Field(..., description="Date and time of image capture")
    days_since_first_observation: int = Field(..., ge=0, description="Days since first observation of this lesion")

    @field_validator("days_since_first_observation")
    @classmethod
    def validate_days(cls, v: int) -> int:
        """Validate days is non-negative."""
        if v < 0:
            raise ValueError("days_since_first_observation must be non-negative")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "capture_date": "2025-01-01T10:30:00Z",
                "days_since_first_observation": 0
            }
        }
    }


class AnalysisCaseCreate(BaseModel):
    """Schema for creating a new analysis case."""
    analysis_id: str = Field(..., description="Unique analysis identifier (e.g., AN-001)")
    patient_id: str = Field(..., description="Patient ID this analysis belongs to")
    lesion_id: str = Field(..., description="Lesion ID this analysis is for")
    image: ImageData = Field(..., description="Image data")
    clinical_data: ClinicalData = Field(..., description="Clinical data at time of capture")
    model_outputs: ModelOutputs = Field(..., description="Outputs from all models")
    shap_analysis: ShapAnalysis = Field(..., description="SHAP explainability analysis")
    temporal_data: TemporalData = Field(..., description="Temporal information")

    model_config = {
        "json_schema_extra": {
            "example": {
                "analysis_id": "AN-001",
                "patient_id": "PAT-001",
                "lesion_id": "LES-001",
                "image": {
                    "filename": "lesion_day0.jpg",
                    "path": "/uploads/lesion_day0.jpg"
                },
                "clinical_data": {
                    "age_at_capture": 45,
                    "lesion_size_mm": 12.4
                },
                "model_outputs": {
                    "image_only_model": {
                        "malignant_probability": 0.72
                    },
                    "clinical_ml_model": {
                        "malignant_probability": 0.65
                    },
                    "extracted_features": [
                        {"feature_name": "tbp_lv_norm_color", "value": 3.0},
                        {"feature_name": "tbp_lv_H", "value": 55.0}
                    ]
                },
                "shap_analysis": {
                    "prediction": 0.65,
                    "base_value": 0.4238,
                    "features": [
                        {
                            "feature": "tbp_lv_norm_color",
                            "display_name": "Normalized Color Variation",
                            "value": 3.0,
                            "shap_value": -0.803790,
                            "impact": "decreases"
                        }
                    ]
                },
                "temporal_data": {
                    "capture_date": "2025-01-01T10:30:00Z",
                    "days_since_first_observation": 0
                }
            }
        }
    }


class AnalysisCaseInDB(AnalysisCaseCreate):
    """Schema for analysis case stored in database (includes MongoDB _id and created_at)."""
    id: str = Field(alias="_id", description="MongoDB ObjectId as string")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp when analysis was created")

    model_config = {
        "populate_by_name": True,
        "arbitrary_types_allowed": True,
        "json_schema_extra": {
            "example": {
                "_id": "507f1f77bcf86cd799439013",
                "analysis_id": "AN-001",
                "patient_id": "PAT-001",
                "lesion_id": "LES-001",
                "image": {
                    "filename": "lesion_day0.jpg",
                    "path": "/uploads/lesion_day0.jpg"
                },
                "clinical_data": {
                    "age_at_capture": 45,
                    "lesion_size_mm": 12.4
                },
                "model_outputs": {
                    "image_only_model": {
                        "malignant_probability": 0.72
                    },
                    "clinical_ml_model": {
                        "malignant_probability": 0.65
                    },
                    "extracted_features": [
                        {"feature_name": "tbp_lv_norm_color", "value": 3.0},
                        {"feature_name": "tbp_lv_H", "value": 55.0}
                    ]
                },
                "shap_analysis": {
                    "prediction": 0.65,
                    "base_value": 0.4238,
                    "features": [
                        {
                            "feature": "tbp_lv_norm_color",
                            "display_name": "Normalized Color Variation",
                            "value": 3.0,
                            "shap_value": -0.803790,
                            "impact": "decreases"
                        }
                    ]
                },
                "temporal_data": {
                    "capture_date": "2025-01-01T10:30:00Z",
                    "days_since_first_observation": 0
                },
                "created_at": "2025-01-01T10:30:00Z"
            }
        }
    }


class AnalysisCaseUpdate(BaseModel):
    """Schema for updating an existing analysis case (all fields optional)."""
    image: Optional[ImageData] = None
    clinical_data: Optional[ClinicalData] = None
    model_outputs: Optional[ModelOutputs] = None
    shap_analysis: Optional[ShapAnalysis] = None
    temporal_data: Optional[TemporalData] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "clinical_data": {
                    "age_at_capture": 46,
                    "lesion_size_mm": 13.2
                }
            }
        }
    }


class AnalysisCaseResponse(BaseModel):
    """Schema for analysis case API responses."""
    id: str = Field(alias="_id", description="Analysis case's MongoDB ObjectId")
    analysis_id: str
    patient_id: str
    lesion_id: str
    image: ImageData
    clinical_data: ClinicalData
    model_outputs: ModelOutputs
    shap_analysis: ShapAnalysis
    temporal_data: TemporalData
    created_at: datetime

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "_id": "507f1f77bcf86cd799439013",
                "analysis_id": "AN-001",
                "patient_id": "PAT-001",
                "lesion_id": "LES-001",
                "image": {
                    "filename": "lesion_day0.jpg",
                    "path": "/uploads/lesion_day0.jpg"
                },
                "clinical_data": {
                    "age_at_capture": 45,
                    "lesion_size_mm": 12.4
                },
                "model_outputs": {
                    "image_only_model": {
                        "malignant_probability": 0.72
                    },
                    "clinical_ml_model": {
                        "malignant_probability": 0.65
                    },
                    "extracted_features": [
                        {"feature_name": "tbp_lv_norm_color", "value": 3.0},
                        {"feature_name": "tbp_lv_H", "value": 55.0}
                    ]
                },
                "shap_analysis": {
                    "prediction": 0.65,
                    "base_value": 0.4238,
                    "features": [
                        {
                            "feature": "tbp_lv_norm_color",
                            "display_name": "Normalized Color Variation",
                            "value": 3.0,
                            "shap_value": -0.803790,
                            "impact": "decreases"
                        }
                    ]
                },
                "temporal_data": {
                    "capture_date": "2025-01-01T10:30:00Z",
                    "days_since_first_observation": 0
                },
                "created_at": "2025-01-01T10:30:00Z"
            }
        }
    }
