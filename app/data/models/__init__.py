"""
Data models module - Pydantic schemas for MongoDB collections.
"""

from .patient import (
    PatientCreate,
    PatientInDB,
    PatientUpdate,
    PatientResponse,
)
from .lesion import (
    LesionCreate,
    LesionInDB,
    LesionUpdate,
    LesionResponse,
)
from .analysis import (
    ImageData,
    ClinicalData,
    ModelOutput,
    ModelOutputs,
    ShapFeature,
    ShapAnalysis,
    TemporalData,
    AnalysisCaseCreate,
    AnalysisCaseInDB,
    AnalysisCaseUpdate,
    AnalysisCaseResponse,
)

__all__ = [
    # Patient models
    "PatientCreate",
    "PatientInDB",
    "PatientUpdate",
    "PatientResponse",
    # Lesion models
    "LesionCreate",
    "LesionInDB",
    "LesionUpdate",
    "LesionResponse",
    # Analysis models
    "ImageData",
    "ClinicalData",
    "ModelOutput",
    "ModelOutputs",
    "ShapFeature",
    "ShapAnalysis",
    "TemporalData",
    "AnalysisCaseCreate",
    "AnalysisCaseInDB",
    "AnalysisCaseUpdate",
    "AnalysisCaseResponse",
]
