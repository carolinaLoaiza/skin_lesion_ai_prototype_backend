"""
REST API endpoints for Patient management.
"""

from fastapi import APIRouter, HTTPException, Query, status
from typing import List

from app.data.managers.patient_manager import patient_manager
from app.data.models.patient import (
    PatientCreate,
    PatientUpdate,
    PatientResponse,
)

router = APIRouter(prefix="/api/patients", tags=["patients"])


@router.post(
    "",
    response_model=PatientResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new patient"
)
async def create_patient(patient_data: PatientCreate):
    """
    Create a new patient in the database.

    Args:
        patient_data: Patient information (patient_id, full_name, sex, date_of_birth)

    Returns:
        PatientResponse: Created patient information

    Raises:
        HTTPException 400: If patient_id already exists
        HTTPException 500: If database operation fails
    """
    try:
        # Create patient
        object_id = await patient_manager.create_patient(patient_data)

        # Retrieve and return the created patient
        patient = await patient_manager.get_patient_by_object_id(object_id)
        if not patient:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Patient created but could not be retrieved"
            )

        return patient

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating patient: {str(e)}"
        )


@router.get(
    "/{patient_id}",
    response_model=PatientResponse,
    summary="Get a patient by ID"
)
async def get_patient(patient_id: str):
    """
    Get a patient by their unique patient_id.

    Args:
        patient_id: The unique patient identifier (e.g., "PAT-001")

    Returns:
        PatientResponse: Patient information

    Raises:
        HTTPException 404: If patient not found
    """
    try:
        patient = await patient_manager.get_patient(patient_id)
        if not patient:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Patient '{patient_id}' not found"
            )

        return patient

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving patient: {str(e)}"
        )


@router.put(
    "/{patient_id}",
    response_model=PatientResponse,
    summary="Update a patient"
)
async def update_patient(patient_id: str, patient_data: PatientUpdate):
    """
    Update a patient's information.

    Args:
        patient_id: The unique patient identifier
        patient_data: Fields to update

    Returns:
        PatientResponse: Updated patient information

    Raises:
        HTTPException 404: If patient not found
        HTTPException 400: If no fields to update
    """
    try:
        # Check if patient exists
        existing_patient = await patient_manager.get_patient(patient_id)
        if not existing_patient:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Patient '{patient_id}' not found"
            )

        # Update patient
        updated = await patient_manager.update_patient(patient_id, patient_data)
        if not updated:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No fields to update"
            )

        # Retrieve and return updated patient
        patient = await patient_manager.get_patient(patient_id)
        return patient

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating patient: {str(e)}"
        )


@router.delete(
    "/{patient_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a patient"
)
async def delete_patient(patient_id: str):
    """
    Delete a patient from the database.

    Args:
        patient_id: The unique patient identifier

    Raises:
        HTTPException 404: If patient not found
    """
    try:
        deleted = await patient_manager.delete_patient(patient_id)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Patient '{patient_id}' not found"
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting patient: {str(e)}"
        )


@router.get(
    "",
    response_model=List[PatientResponse],
    summary="List all patients with pagination"
)
async def list_patients(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=500, description="Maximum number of records to return")
):
    """
    List all patients with pagination.

    Args:
        skip: Number of records to skip (default: 0)
        limit: Maximum number of records to return (default: 100, max: 500)

    Returns:
        List[PatientResponse]: List of patients
    """
    try:
        patients = await patient_manager.list_patients(skip=skip, limit=limit)
        return patients

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing patients: {str(e)}"
        )


@router.get(
    "/search/by-name",
    response_model=List[PatientResponse],
    summary="Search patients by name"
)
async def search_patients_by_name(
    name: str = Query(..., min_length=1, description="Name or partial name to search for")
):
    """
    Search patients by name (case-insensitive partial match).

    Args:
        name: Name or partial name to search for

    Returns:
        List[PatientResponse]: List of matching patients
    """
    try:
        patients = await patient_manager.search_patients_by_name(name)
        return patients

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error searching patients: {str(e)}"
        )


@router.get(
    "/stats/count",
    response_model=dict,
    summary="Get patient count statistics"
)
async def get_patient_stats():
    """
    Get statistics about patients.

    Returns:
        dict: Statistics including total patient count
    """
    try:
        count = await patient_manager.count_patients()
        return {
            "total_patients": count
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting patient statistics: {str(e)}"
        )
