"""
REST API endpoints for Lesion management.
"""

from fastapi import APIRouter, HTTPException, Query, status
from typing import List

from app.data.managers.lesion_manager import lesion_manager
from app.data.models.lesion import (
    LesionCreate,
    LesionUpdate,
    LesionResponse,
)

router = APIRouter(prefix="/api/lesions", tags=["lesions"])


@router.post(
    "",
    response_model=LesionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new lesion"
)
async def create_lesion(lesion_data: LesionCreate):
    """
    Create a new lesion in the database.

    Args:
        lesion_data: Lesion information (lesion_id, patient_id, location, initial_size_mm)

    Returns:
        LesionResponse: Created lesion information

    Raises:
        HTTPException 400: If lesion_id already exists
        HTTPException 500: If database operation fails
    """
    try:
        # Create lesion
        object_id = await lesion_manager.create_lesion(lesion_data)

        # Retrieve and return the created lesion
        lesion = await lesion_manager.get_lesion_by_object_id(object_id)
        if not lesion:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Lesion created but could not be retrieved"
            )

        return lesion

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating lesion: {str(e)}"
        )


@router.get(
    "/{lesion_id}",
    response_model=LesionResponse,
    summary="Get a lesion by ID"
)
async def get_lesion(lesion_id: str):
    """
    Get a lesion by its unique lesion_id.

    Args:
        lesion_id: The unique lesion identifier (e.g., "LES-001")

    Returns:
        LesionResponse: Lesion information

    Raises:
        HTTPException 404: If lesion not found
    """
    try:
        lesion = await lesion_manager.get_lesion(lesion_id)
        if not lesion:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Lesion '{lesion_id}' not found"
            )

        return lesion

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving lesion: {str(e)}"
        )


@router.put(
    "/{lesion_id}",
    response_model=LesionResponse,
    summary="Update a lesion"
)
async def update_lesion(lesion_id: str, lesion_data: LesionUpdate):
    """
    Update a lesion's information.

    Args:
        lesion_id: The unique lesion identifier
        lesion_data: Fields to update

    Returns:
        LesionResponse: Updated lesion information

    Raises:
        HTTPException 404: If lesion not found
        HTTPException 400: If no fields to update
    """
    try:
        # Check if lesion exists
        existing_lesion = await lesion_manager.get_lesion(lesion_id)
        if not existing_lesion:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Lesion '{lesion_id}' not found"
            )

        # Update lesion
        updated = await lesion_manager.update_lesion(lesion_id, lesion_data)
        if not updated:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No fields to update"
            )

        # Retrieve and return updated lesion
        lesion = await lesion_manager.get_lesion(lesion_id)
        return lesion

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating lesion: {str(e)}"
        )


@router.delete(
    "/{lesion_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a lesion"
)
async def delete_lesion(lesion_id: str):
    """
    Delete a lesion from the database.

    Args:
        lesion_id: The unique lesion identifier

    Raises:
        HTTPException 404: If lesion not found
    """
    try:
        deleted = await lesion_manager.delete_lesion(lesion_id)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Lesion '{lesion_id}' not found"
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting lesion: {str(e)}"
        )


@router.get(
    "",
    response_model=List[LesionResponse],
    summary="List all lesions with pagination"
)
async def list_lesions(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=500, description="Maximum number of records to return")
):
    """
    List all lesions with pagination.

    Args:
        skip: Number of records to skip (default: 0)
        limit: Maximum number of records to return (default: 100, max: 500)

    Returns:
        List[LesionResponse]: List of lesions
    """
    try:
        lesions = await lesion_manager.list_lesions(skip=skip, limit=limit)
        return lesions

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing lesions: {str(e)}"
        )


# Patient-specific endpoints
patients_router = APIRouter(prefix="/api/patients", tags=["lesions"])


@patients_router.get(
    "/{patient_id}/lesions",
    response_model=List[LesionResponse],
    summary="Get all lesions for a patient"
)
async def get_patient_lesions(patient_id: str):
    """
    Get all lesions for a specific patient.

    Args:
        patient_id: The unique patient identifier

    Returns:
        List[LesionResponse]: List of lesions for the patient
    """
    try:
        lesions = await lesion_manager.get_lesions_by_patient(patient_id)
        return lesions

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving lesions for patient: {str(e)}"
        )


@patients_router.get(
    "/{patient_id}/lesions/count",
    response_model=dict,
    summary="Count lesions for a patient"
)
async def count_patient_lesions(patient_id: str):
    """
    Count the number of lesions for a specific patient.

    Args:
        patient_id: The unique patient identifier

    Returns:
        dict: Count of lesions for the patient
    """
    try:
        count = await lesion_manager.count_lesions_by_patient(patient_id)
        return {
            "patient_id": patient_id,
            "lesion_count": count
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error counting lesions for patient: {str(e)}"
        )


@router.get(
    "/location/{location}",
    response_model=List[LesionResponse],
    summary="Get lesions by anatomical location"
)
async def get_lesions_by_location(location: str):
    """
    Get all lesions at a specific anatomical location.

    Args:
        location: The anatomical location (e.g., "back", "arm")

    Returns:
        List[LesionResponse]: List of lesions at that location
    """
    try:
        lesions = await lesion_manager.get_lesions_by_location(location)
        return lesions

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving lesions by location: {str(e)}"
        )
