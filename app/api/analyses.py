"""
REST API endpoints for Analysis Cases management.
"""

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import FileResponse
from typing import List
import os

from app.data.managers.analysis_manager import analysis_manager
from app.data.managers.storage_manager import storage_manager
from app.data.models.analysis import (
    AnalysisCaseCreate,
    AnalysisCaseUpdate,
    AnalysisCaseResponse,
)

router = APIRouter(prefix="/api/analyses", tags=["analyses"])


@router.post(
    "",
    response_model=AnalysisCaseResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new analysis case"
)
async def create_analysis(analysis_data: AnalysisCaseCreate):
    """
    Create a new analysis case in the database.

    Args:
        analysis_data: Analysis case information including image, clinical data, model outputs, SHAP analysis, and temporal data

    Returns:
        AnalysisCaseResponse: Created analysis case information

    Raises:
        HTTPException 400: If analysis_id already exists
        HTTPException 500: If database operation fails
    """
    try:
        # Create analysis
        object_id = await analysis_manager.create_analysis(analysis_data)

        # Retrieve and return the created analysis
        analysis = await analysis_manager.get_analysis_by_object_id(object_id)
        if not analysis:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Analysis created but could not be retrieved"
            )

        return analysis

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating analysis: {str(e)}"
        )


@router.get(
    "/{analysis_id}",
    response_model=AnalysisCaseResponse,
    summary="Get an analysis case by ID"
)
async def get_analysis(analysis_id: str):
    """
    Get an analysis case by its unique analysis_id.

    Args:
        analysis_id: The unique analysis identifier (e.g., "AN-001")

    Returns:
        AnalysisCaseResponse: Analysis case information

    Raises:
        HTTPException 404: If analysis not found
    """
    try:
        analysis = await analysis_manager.get_analysis(analysis_id)
        if not analysis:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Analysis '{analysis_id}' not found"
            )

        return analysis

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving analysis: {str(e)}"
        )


@router.get(
    "/{analysis_id}/image",
    response_class=FileResponse,
    summary="Get analysis image"
)
async def get_analysis_image(analysis_id: str):
    """
    Get the lesion image associated with an analysis case.

    This endpoint returns the actual image file that was uploaded
    for this analysis. The image is served directly as a file response.

    Args:
        analysis_id: The unique analysis identifier (e.g., "AN-001")

    Returns:
        FileResponse: The image file (JPEG/PNG)

    Raises:
        HTTPException 404: If analysis not found or image file doesn't exist
        HTTPException 500: If error reading image file
    """
    try:
        # Get analysis to retrieve image path
        analysis = await analysis_manager.get_analysis(analysis_id)
        if not analysis:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Analysis '{analysis_id}' not found"
            )

        # Get image path from analysis
        image_path = analysis.image.path

        # Verify file exists
        if not os.path.exists(image_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Image file not found for analysis '{analysis_id}'"
            )

        # Return image file
        return FileResponse(
            path=image_path,
            media_type="image/jpeg",
            filename=analysis.image.filename
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving image: {str(e)}"
        )


@router.put(
    "/{analysis_id}",
    response_model=AnalysisCaseResponse,
    summary="Update an analysis case"
)
async def update_analysis(analysis_id: str, analysis_data: AnalysisCaseUpdate):
    """
    Update an analysis case's information.

    Args:
        analysis_id: The unique analysis identifier
        analysis_data: Fields to update

    Returns:
        AnalysisCaseResponse: Updated analysis case information

    Raises:
        HTTPException 404: If analysis not found
        HTTPException 400: If no fields to update
    """
    try:
        # Check if analysis exists
        existing_analysis = await analysis_manager.get_analysis(analysis_id)
        if not existing_analysis:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Analysis '{analysis_id}' not found"
            )

        # Update analysis
        updated = await analysis_manager.update_analysis(analysis_id, analysis_data)
        if not updated:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No fields to update"
            )

        # Retrieve and return updated analysis
        analysis = await analysis_manager.get_analysis(analysis_id)
        return analysis

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating analysis: {str(e)}"
        )


@router.delete(
    "/{analysis_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete an analysis case"
)
async def delete_analysis(analysis_id: str):
    """
    Delete an analysis case from the database.

    Args:
        analysis_id: The unique analysis identifier

    Raises:
        HTTPException 404: If analysis not found
    """
    try:
        deleted = await analysis_manager.delete_analysis(analysis_id)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Analysis '{analysis_id}' not found"
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting analysis: {str(e)}"
        )


@router.get(
    "",
    response_model=List[AnalysisCaseResponse],
    summary="List all analysis cases with pagination"
)
async def list_analyses(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=500, description="Maximum number of records to return")
):
    """
    List all analysis cases with pagination.

    Args:
        skip: Number of records to skip (default: 0)
        limit: Maximum number of records to return (default: 100, max: 500)

    Returns:
        List[AnalysisCaseResponse]: List of analysis cases
    """
    try:
        analyses = await analysis_manager.list_analyses(skip=skip, limit=limit)
        return analyses

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing analyses: {str(e)}"
        )


@router.get(
    "/high-risk/list",
    response_model=List[AnalysisCaseResponse],
    summary="Get high-risk analysis cases"
)
async def get_high_risk_analyses(
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="Minimum probability threshold"),
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=500, description="Maximum number of records to return")
):
    """
    Get analysis cases with high malignancy probability.

    Args:
        threshold: Minimum probability threshold (default: 0.5)
        skip: Number of records to skip (default: 0)
        limit: Maximum number of records to return (default: 100, max: 500)

    Returns:
        List[AnalysisCaseResponse]: List of high-risk analysis cases
    """
    try:
        analyses = await analysis_manager.get_high_risk_analyses(
            threshold=threshold,
            skip=skip,
            limit=limit
        )
        return analyses

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving high-risk analyses: {str(e)}"
        )


@router.get(
    "/storage/stats",
    response_model=dict,
    summary="Get storage statistics"
)
async def get_storage_statistics():
    """
    Get statistics about stored images and files.

    Returns:
        dict: Storage statistics including total files, size, and patient count
    """
    try:
        stats = storage_manager.get_storage_stats()
        return stats

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving storage statistics: {str(e)}"
        )


# Patient-specific endpoints
patients_router = APIRouter(prefix="/api/patients", tags=["analyses"])


@patients_router.get(
    "/{patient_id}/analyses",
    response_model=List[AnalysisCaseResponse],
    summary="Get all analyses for a patient"
)
async def get_patient_analyses(patient_id: str):
    """
    Get all analysis cases for a specific patient.

    Args:
        patient_id: The unique patient identifier

    Returns:
        List[AnalysisCaseResponse]: List of analyses for the patient
    """
    try:
        analyses = await analysis_manager.get_analyses_by_patient(patient_id)
        return analyses

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving analyses for patient: {str(e)}"
        )


@patients_router.get(
    "/{patient_id}/analyses/count",
    response_model=dict,
    summary="Count analyses for a patient"
)
async def count_patient_analyses(patient_id: str):
    """
    Count the number of analysis cases for a specific patient.

    Args:
        patient_id: The unique patient identifier

    Returns:
        dict: Count of analyses for the patient
    """
    try:
        count = await analysis_manager.count_analyses_by_patient(patient_id)
        return {
            "patient_id": patient_id,
            "analysis_count": count
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error counting analyses for patient: {str(e)}"
        )


# Lesion-specific endpoints
lesions_router = APIRouter(prefix="/api/lesions", tags=["analyses"])


@lesions_router.get(
    "/{lesion_id}/analyses",
    response_model=List[AnalysisCaseResponse],
    summary="Get all analyses for a lesion"
)
async def get_lesion_analyses(lesion_id: str):
    """
    Get all analysis cases for a specific lesion.

    Args:
        lesion_id: The unique lesion identifier

    Returns:
        List[AnalysisCaseResponse]: List of analyses for the lesion
    """
    try:
        analyses = await analysis_manager.get_analyses_by_lesion(lesion_id)
        return analyses

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving analyses for lesion: {str(e)}"
        )


@lesions_router.get(
    "/{lesion_id}/analyses/count",
    response_model=dict,
    summary="Count analyses for a lesion"
)
async def count_lesion_analyses(lesion_id: str):
    """
    Count the number of analysis cases for a specific lesion.

    Args:
        lesion_id: The unique lesion identifier

    Returns:
        dict: Count of analyses for the lesion
    """
    try:
        count = await analysis_manager.count_analyses_by_lesion(lesion_id)
        return {
            "lesion_id": lesion_id,
            "analysis_count": count
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error counting analyses for lesion: {str(e)}"
        )


@lesions_router.get(
    "/{lesion_id}/progression",
    response_model=List[AnalysisCaseResponse],
    summary="Get temporal progression for a lesion"
)
async def get_lesion_temporal_progression(lesion_id: str):
    """
    Get temporal progression of analysis cases for a lesion, sorted chronologically.

    This endpoint is useful for tracking how a lesion changes over time,
    showing all analyses from oldest to newest.

    Args:
        lesion_id: The unique lesion identifier

    Returns:
        List[AnalysisCaseResponse]: List of analyses sorted by capture date (oldest to newest)
    """
    try:
        analyses = await analysis_manager.get_temporal_progression(lesion_id)
        return analyses

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving temporal progression for lesion: {str(e)}"
        )


@lesions_router.get(
    "/{lesion_id}/analyses/latest",
    response_model=AnalysisCaseResponse,
    summary="Get the latest analysis for a lesion"
)
async def get_latest_lesion_analysis(lesion_id: str):
    """
    Get the most recent analysis for a specific lesion.

    Args:
        lesion_id: The unique lesion identifier

    Returns:
        AnalysisCaseResponse: Most recent analysis for the lesion

    Raises:
        HTTPException 404: If no analyses found for this lesion
    """
    try:
        analysis = await analysis_manager.get_latest_analysis_for_lesion(lesion_id)
        if not analysis:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No analyses found for lesion '{lesion_id}'"
            )

        return analysis

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving latest analysis for lesion: {str(e)}"
        )
