"""
Pydantic models for Patient collection.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import datetime
from bson import ObjectId


class PatientCreate(BaseModel):
    """Schema for creating a new patient."""
    patient_id: str = Field(..., description="Unique patient identifier (e.g., PAT-001)")
    patient_full_name: str = Field(..., min_length=1, description="Full name of the patient")
    sex: str = Field(..., description="Patient's biological sex")
    date_of_birth: str = Field(..., description="Date of birth in DD/MM/YYYY format")

    @field_validator("sex")
    @classmethod
    def validate_sex(cls, v: str) -> str:
        """Validate sex field."""
        allowed_values = ["male", "female"]
        v_lower = v.lower()
        if v_lower not in allowed_values:
            raise ValueError(f"Sex must be one of {allowed_values}, got '{v}'")
        return v_lower

    @field_validator("date_of_birth")
    @classmethod
    def validate_date_of_birth(cls, v: str) -> str:
        """Validate date of birth format (DD/MM/YYYY)."""
        try:
            datetime.strptime(v, "%d/%m/%Y")
        except ValueError:
            raise ValueError("date_of_birth must be in DD/MM/YYYY format")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "patient_id": "PAT-001",
                "patient_full_name": "John Doe",
                "sex": "male",
                "date_of_birth": "15/03/1980"
            }
        }
    }


class PatientInDB(PatientCreate):
    """Schema for patient stored in database (includes MongoDB _id and created_at)."""
    id: str = Field(alias="_id", description="MongoDB ObjectId as string")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp when patient was created")

    model_config = {
        "populate_by_name": True,
        "arbitrary_types_allowed": True,
        "json_schema_extra": {
            "example": {
                "_id": "507f1f77bcf86cd799439011",
                "patient_id": "PAT-001",
                "patient_full_name": "John Doe",
                "sex": "male",
                "date_of_birth": "15/03/1980",
                "created_at": "2025-01-01T12:00:00Z"
            }
        }
    }


class PatientUpdate(BaseModel):
    """Schema for updating an existing patient (all fields optional)."""
    patient_full_name: Optional[str] = Field(None, min_length=1)
    sex: Optional[str] = None
    date_of_birth: Optional[str] = None

    @field_validator("sex")
    @classmethod
    def validate_sex(cls, v: Optional[str]) -> Optional[str]:
        """Validate sex field if provided - must be 'male' or 'female' (required by ML model)."""
        if v is None:
            return v
        allowed_values = ["male", "female"]
        v_lower = v.lower()
        if v_lower not in allowed_values:
            raise ValueError(f"Sex must be one of {allowed_values}, got '{v}'")
        return v_lower

    @field_validator("date_of_birth")
    @classmethod
    def validate_date_of_birth(cls, v: Optional[str]) -> Optional[str]:
        """Validate date of birth format if provided."""
        if v is None:
            return v
        try:
            datetime.strptime(v, "%d/%m/%Y")
        except ValueError:
            raise ValueError("date_of_birth must be in DD/MM/YYYY format")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "patient_full_name": "John Smith",
                "sex": "male"
            }
        }
    }


class PatientResponse(BaseModel):
    """Schema for patient API responses."""
    id: str = Field(alias="_id", description="Patient's MongoDB ObjectId")
    patient_id: str
    patient_full_name: str
    sex: str
    date_of_birth: str
    created_at: datetime

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "_id": "507f1f77bcf86cd799439011",
                "patient_id": "PAT-001",
                "patient_full_name": "John Doe",
                "sex": "male",
                "date_of_birth": "15/03/1980",
                "created_at": "2025-01-01T12:00:00Z"
            }
        }
    }
