"""
CRUD Manager for Lesion collection.
"""

from typing import List, Optional
from datetime import datetime
from bson import ObjectId
import logging

from app.data.database import db
from app.data.models.lesion import LesionCreate, LesionInDB, LesionUpdate

logger = logging.getLogger(__name__)


class LesionManager:
    """Manager for CRUD operations on Lesion collection."""

    def __init__(self):
        """Initialize LesionManager."""
        self.collection_name = "lesions"

    async def create_lesion(self, data: LesionCreate) -> str:
        """
        Create a new lesion in the database.

        Args:
            data: LesionCreate schema with lesion information

        Returns:
            str: The MongoDB ObjectId of the created lesion as string

        Raises:
            ValueError: If lesion_id already exists
            Exception: If database operation fails
        """
        try:
            collection = db.get_collection(self.collection_name)

            # Check if lesion_id already exists
            existing = await collection.find_one({"lesion_id": data.lesion_id})
            if existing:
                raise ValueError(f"Lesion with lesion_id '{data.lesion_id}' already exists")

            # Prepare document
            lesion_dict = data.model_dump()
            lesion_dict["created_at"] = datetime.utcnow()

            # Insert into database
            result = await collection.insert_one(lesion_dict)
            logger.info(f"Created lesion with ID: {data.lesion_id} for patient {data.patient_id}")

            return str(result.inserted_id)

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Error creating lesion: {str(e)}")
            raise

    async def get_lesion(self, lesion_id: str) -> Optional[LesionInDB]:
        """
        Get a lesion by lesion_id.

        Args:
            lesion_id: The unique lesion identifier (e.g., "LES-001")

        Returns:
            LesionInDB if found, None otherwise
        """
        try:
            collection = db.get_collection(self.collection_name)
            lesion_doc = await collection.find_one({"lesion_id": lesion_id})

            if lesion_doc:
                # Convert ObjectId to string for Pydantic
                lesion_doc["_id"] = str(lesion_doc["_id"])
                return LesionInDB(**lesion_doc)

            return None

        except Exception as e:
            logger.error(f"Error getting lesion {lesion_id}: {str(e)}")
            raise

    async def get_lesion_by_object_id(self, object_id: str) -> Optional[LesionInDB]:
        """
        Get a lesion by MongoDB ObjectId.

        Args:
            object_id: The MongoDB ObjectId as string

        Returns:
            LesionInDB if found, None otherwise
        """
        try:
            collection = db.get_collection(self.collection_name)
            lesion_doc = await collection.find_one({"_id": ObjectId(object_id)})

            if lesion_doc:
                lesion_doc["_id"] = str(lesion_doc["_id"])
                return LesionInDB(**lesion_doc)

            return None

        except Exception as e:
            logger.error(f"Error getting lesion by ObjectId {object_id}: {str(e)}")
            raise

    async def update_lesion(self, lesion_id: str, data: LesionUpdate) -> bool:
        """
        Update a lesion's information.

        Args:
            lesion_id: The unique lesion identifier
            data: LesionUpdate schema with fields to update

        Returns:
            bool: True if lesion was updated, False if not found

        Raises:
            Exception: If database operation fails
        """
        try:
            collection = db.get_collection(self.collection_name)

            # Only include fields that were actually set
            update_data = data.model_dump(exclude_unset=True)

            if not update_data:
                logger.warning(f"No fields to update for lesion {lesion_id}")
                return False

            # Update the document
            result = await collection.update_one(
                {"lesion_id": lesion_id},
                {"$set": update_data}
            )

            if result.modified_count > 0:
                logger.info(f"Updated lesion {lesion_id}")
                return True

            return False

        except Exception as e:
            logger.error(f"Error updating lesion {lesion_id}: {str(e)}")
            raise

    async def delete_lesion(self, lesion_id: str) -> bool:
        """
        Delete a lesion from the database.

        Args:
            lesion_id: The unique lesion identifier

        Returns:
            bool: True if lesion was deleted, False if not found

        Raises:
            Exception: If database operation fails
        """
        try:
            collection = db.get_collection(self.collection_name)

            result = await collection.delete_one({"lesion_id": lesion_id})

            if result.deleted_count > 0:
                logger.info(f"Deleted lesion {lesion_id}")
                return True

            logger.warning(f"Lesion {lesion_id} not found for deletion")
            return False

        except Exception as e:
            logger.error(f"Error deleting lesion {lesion_id}: {str(e)}")
            raise

    async def get_lesions_by_patient(self, patient_id: str) -> List[LesionInDB]:
        """
        Get all lesions for a specific patient.

        Args:
            patient_id: The patient identifier

        Returns:
            List[LesionInDB]: List of lesions for the patient
        """
        try:
            collection = db.get_collection(self.collection_name)

            cursor = collection.find({"patient_id": patient_id}).sort("created_at", -1)
            lesions = []

            async for doc in cursor:
                doc["_id"] = str(doc["_id"])
                lesions.append(LesionInDB(**doc))

            return lesions

        except Exception as e:
            logger.error(f"Error getting lesions for patient {patient_id}: {str(e)}")
            raise

    async def count_lesions_by_patient(self, patient_id: str) -> int:
        """
        Count the number of lesions for a specific patient.

        Args:
            patient_id: The patient identifier

        Returns:
            int: Number of lesions for the patient
        """
        try:
            collection = db.get_collection(self.collection_name)
            count = await collection.count_documents({"patient_id": patient_id})
            return count

        except Exception as e:
            logger.error(f"Error counting lesions for patient {patient_id}: {str(e)}")
            raise

    async def list_lesions(self, skip: int = 0, limit: int = 100) -> List[LesionInDB]:
        """
        List all lesions with pagination.

        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            List[LesionInDB]: List of lesions
        """
        try:
            collection = db.get_collection(self.collection_name)

            cursor = collection.find().skip(skip).limit(limit).sort("created_at", -1)
            lesions = []

            async for doc in cursor:
                doc["_id"] = str(doc["_id"])
                lesions.append(LesionInDB(**doc))

            return lesions

        except Exception as e:
            logger.error(f"Error listing lesions: {str(e)}")
            raise

    async def count_lesions(self) -> int:
        """
        Get total count of lesions in the database.

        Returns:
            int: Total number of lesions
        """
        try:
            collection = db.get_collection(self.collection_name)
            count = await collection.count_documents({})
            return count

        except Exception as e:
            logger.error(f"Error counting lesions: {str(e)}")
            raise

    async def lesion_exists(self, lesion_id: str) -> bool:
        """
        Check if a lesion exists by lesion_id.

        Args:
            lesion_id: The unique lesion identifier

        Returns:
            bool: True if lesion exists, False otherwise
        """
        try:
            collection = db.get_collection(self.collection_name)
            count = await collection.count_documents({"lesion_id": lesion_id})
            return count > 0

        except Exception as e:
            logger.error(f"Error checking if lesion exists {lesion_id}: {str(e)}")
            raise

    async def get_lesions_by_location(self, location: str) -> List[LesionInDB]:
        """
        Get all lesions at a specific anatomical location.

        Args:
            location: The anatomical location to search for

        Returns:
            List[LesionInDB]: List of lesions at that location
        """
        try:
            collection = db.get_collection(self.collection_name)

            # Case-insensitive search
            cursor = collection.find({
                "lesion_location": {"$regex": f"^{location}$", "$options": "i"}
            }).sort("created_at", -1)

            lesions = []
            async for doc in cursor:
                doc["_id"] = str(doc["_id"])
                lesions.append(LesionInDB(**doc))

            return lesions

        except Exception as e:
            logger.error(f"Error getting lesions by location '{location}': {str(e)}")
            raise


# Global instance
lesion_manager = LesionManager()
