"""
CRUD Manager for Patient collection.
"""

from typing import List, Optional
from datetime import datetime
from bson import ObjectId
import logging

from app.data.database import db
from app.data.models.patient import PatientCreate, PatientInDB, PatientUpdate

logger = logging.getLogger(__name__)


class PatientManager:
    """Manager for CRUD operations on Patient collection."""

    def __init__(self):
        """Initialize PatientManager."""
        self.collection_name = "patients"

    async def create_patient(self, data: PatientCreate) -> str:
        """
        Create a new patient in the database.

        Args:
            data: PatientCreate schema with patient information

        Returns:
            str: The MongoDB ObjectId of the created patient as string

        Raises:
            ValueError: If patient_id already exists
            Exception: If database operation fails
        """
        try:
            collection = db.get_collection(self.collection_name)

            # Check if patient_id already exists
            existing = await collection.find_one({"patient_id": data.patient_id})
            if existing:
                raise ValueError(f"Patient with patient_id '{data.patient_id}' already exists")

            # Prepare document
            patient_dict = data.model_dump()
            patient_dict["created_at"] = datetime.utcnow()

            # Insert into database
            result = await collection.insert_one(patient_dict)
            logger.info(f"Created patient with ID: {data.patient_id}")

            return str(result.inserted_id)

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Error creating patient: {str(e)}")
            raise

    async def get_patient(self, patient_id: str) -> Optional[PatientInDB]:
        """
        Get a patient by patient_id.

        Args:
            patient_id: The unique patient identifier (e.g., "PAT-001")

        Returns:
            PatientInDB if found, None otherwise
        """
        try:
            collection = db.get_collection(self.collection_name)
            patient_doc = await collection.find_one({"patient_id": patient_id})

            if patient_doc:
                # Convert ObjectId to string for Pydantic
                patient_doc["_id"] = str(patient_doc["_id"])
                return PatientInDB(**patient_doc)

            return None

        except Exception as e:
            logger.error(f"Error getting patient {patient_id}: {str(e)}")
            raise

    async def get_patient_by_object_id(self, object_id: str) -> Optional[PatientInDB]:
        """
        Get a patient by MongoDB ObjectId.

        Args:
            object_id: The MongoDB ObjectId as string

        Returns:
            PatientInDB if found, None otherwise
        """
        try:
            collection = db.get_collection(self.collection_name)
            patient_doc = await collection.find_one({"_id": ObjectId(object_id)})

            if patient_doc:
                patient_doc["_id"] = str(patient_doc["_id"])
                return PatientInDB(**patient_doc)

            return None

        except Exception as e:
            logger.error(f"Error getting patient by ObjectId {object_id}: {str(e)}")
            raise

    async def update_patient(self, patient_id: str, data: PatientUpdate) -> bool:
        """
        Update a patient's information.

        Args:
            patient_id: The unique patient identifier
            data: PatientUpdate schema with fields to update

        Returns:
            bool: True if patient was updated, False if not found

        Raises:
            Exception: If database operation fails
        """
        try:
            collection = db.get_collection(self.collection_name)

            # Only include fields that were actually set
            update_data = data.model_dump(exclude_unset=True)

            if not update_data:
                logger.warning(f"No fields to update for patient {patient_id}")
                return False

            # Update the document
            result = await collection.update_one(
                {"patient_id": patient_id},
                {"$set": update_data}
            )

            if result.modified_count > 0:
                logger.info(f"Updated patient {patient_id}")
                return True

            return False

        except Exception as e:
            logger.error(f"Error updating patient {patient_id}: {str(e)}")
            raise

    async def delete_patient(self, patient_id: str) -> bool:
        """
        Delete a patient from the database.

        Args:
            patient_id: The unique patient identifier

        Returns:
            bool: True if patient was deleted, False if not found

        Raises:
            Exception: If database operation fails
        """
        try:
            collection = db.get_collection(self.collection_name)

            result = await collection.delete_one({"patient_id": patient_id})

            if result.deleted_count > 0:
                logger.info(f"Deleted patient {patient_id}")
                return True

            logger.warning(f"Patient {patient_id} not found for deletion")
            return False

        except Exception as e:
            logger.error(f"Error deleting patient {patient_id}: {str(e)}")
            raise

    async def list_patients(self, skip: int = 0, limit: int = 100) -> List[PatientInDB]:
        """
        List all patients with pagination.

        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            List[PatientInDB]: List of patients
        """
        try:
            collection = db.get_collection(self.collection_name)

            cursor = collection.find().skip(skip).limit(limit).sort("created_at", -1)
            patients = []

            async for doc in cursor:
                doc["_id"] = str(doc["_id"])
                patients.append(PatientInDB(**doc))

            return patients

        except Exception as e:
            logger.error(f"Error listing patients: {str(e)}")
            raise

    async def search_patients_by_name(self, name: str) -> List[PatientInDB]:
        """
        Search patients by name (case-insensitive partial match).

        Args:
            name: Name or partial name to search for

        Returns:
            List[PatientInDB]: List of matching patients
        """
        try:
            collection = db.get_collection(self.collection_name)

            # Case-insensitive regex search
            cursor = collection.find({
                "patient_full_name": {"$regex": name, "$options": "i"}
            }).sort("patient_full_name", 1)

            patients = []
            async for doc in cursor:
                doc["_id"] = str(doc["_id"])
                patients.append(PatientInDB(**doc))

            return patients

        except Exception as e:
            logger.error(f"Error searching patients by name '{name}': {str(e)}")
            raise

    async def count_patients(self) -> int:
        """
        Get total count of patients in the database.

        Returns:
            int: Total number of patients
        """
        try:
            collection = db.get_collection(self.collection_name)
            count = await collection.count_documents({})
            return count

        except Exception as e:
            logger.error(f"Error counting patients: {str(e)}")
            raise

    async def patient_exists(self, patient_id: str) -> bool:
        """
        Check if a patient exists by patient_id.

        Args:
            patient_id: The unique patient identifier

        Returns:
            bool: True if patient exists, False otherwise
        """
        try:
            collection = db.get_collection(self.collection_name)
            count = await collection.count_documents({"patient_id": patient_id})
            return count > 0

        except Exception as e:
            logger.error(f"Error checking if patient exists {patient_id}: {str(e)}")
            raise


# Global instance
patient_manager = PatientManager()
