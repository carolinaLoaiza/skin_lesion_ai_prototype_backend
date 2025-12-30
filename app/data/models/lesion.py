"""
Pydantic models for Lesion collection.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import datetime


class LesionCreate(BaseModel):
    """Schema for creating a new lesion."""
    lesion_id: str = Field(..., description="Unique lesion identifier (e.g., LES-001)")
    patient_id: str = Field(..., description="Patient ID this lesion belongs to")
    lesion_location: str = Field(..., min_length=1, description="Anatomical location of the lesion")
    initial_size_mm: float = Field(..., gt=0, description="Initial size of lesion in millimeters")

    @field_validator("lesion_location")
    @classmethod
    def validate_location(cls, v: str) -> str:
        """Validate and normalize lesion location."""
        # Common anatomical locations (can be extended)
        allowed_locations = [
            "head/neck", "torso", "back", "upper extremity", "lower extremity",
            "left arm", "right arm", "left leg", "right leg", "chest", "abdomen",
            "face", "scalp", "hand", "foot", "anterior torso", "posterior torso",
            "lateral torso", "neck", "shoulder", "forearm", "thigh", "calf"
        ]
        v_lower = v.lower()
        # Allow any location, but validate it's not empty
        if not v_lower.strip():
            raise ValueError("lesion_location cannot be empty")
        return v_lower

    @field_validator("initial_size_mm")
    @classmethod
    def validate_size(cls, v: float) -> float:
        """Validate lesion size is reasonable (0.1mm to 100mm)."""
        if v < 0.1:
            raise ValueError("initial_size_mm must be at least 0.1mm")
        if v > 100.0:
            raise ValueError("initial_size_mm must be less than 100mm (consider if this is correct)")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "lesion_id": "LES-001",
                "patient_id": "PAT-001",
                "lesion_location": "back",
                "initial_size_mm": 12.4
            }
        }
    }


class LesionInDB(LesionCreate):
    """Schema for lesion stored in database (includes MongoDB _id and created_at)."""
    id: str = Field(alias="_id", description="MongoDB ObjectId as string")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp when lesion was created")

    model_config = {
        "populate_by_name": True,
        "arbitrary_types_allowed": True,
        "json_schema_extra": {
            "example": {
                "_id": "507f1f77bcf86cd799439012",
                "lesion_id": "LES-001",
                "patient_id": "PAT-001",
                "lesion_location": "back",
                "initial_size_mm": 12.4,
                "created_at": "2025-01-01T12:00:00Z"
            }
        }
    }


class LesionUpdate(BaseModel):
    """Schema for updating an existing lesion (all fields optional)."""
    lesion_location: Optional[str] = None
    initial_size_mm: Optional[float] = Field(None, gt=0)

    @field_validator("lesion_location")
    @classmethod
    def validate_location(cls, v: Optional[str]) -> Optional[str]:
        """Validate lesion location if provided."""
        if v is None:
            return v
        v_lower = v.lower()
        if not v_lower.strip():
            raise ValueError("lesion_location cannot be empty")
        return v_lower

    @field_validator("initial_size_mm")
    @classmethod
    def validate_size(cls, v: Optional[float]) -> Optional[float]:
        """Validate lesion size if provided."""
        if v is None:
            return v
        if v < 0.1:
            raise ValueError("initial_size_mm must be at least 0.1mm")
        if v > 100.0:
            raise ValueError("initial_size_mm must be less than 100mm")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "lesion_location": "upper back",
                "initial_size_mm": 13.2
            }
        }
    }


class LesionResponse(BaseModel):
    """Schema for lesion API responses."""
    id: str = Field(alias="_id", description="Lesion's MongoDB ObjectId")
    lesion_id: str
    patient_id: str
    lesion_location: str
    initial_size_mm: float
    created_at: datetime

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "_id": "507f1f77bcf86cd799439012",
                "lesion_id": "LES-001",
                "patient_id": "PAT-001",
                "lesion_location": "back",
                "initial_size_mm": 12.4,
                "created_at": "2025-01-01T12:00:00Z"
            }
        }
    }
