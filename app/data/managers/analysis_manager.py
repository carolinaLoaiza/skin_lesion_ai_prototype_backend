"""
CRUD Manager for Analysis Cases collection.
"""

from typing import List, Optional
from datetime import datetime
from bson import ObjectId
import logging

from app.data.database import db
from app.data.models.analysis import AnalysisCaseCreate, AnalysisCaseInDB, AnalysisCaseUpdate

logger = logging.getLogger(__name__)


class AnalysisManager:
    """Manager for CRUD operations on Analysis Cases collection."""

    def __init__(self):
        """Initialize AnalysisManager."""
        self.collection_name = "analysis_cases"

    async def create_analysis(self, data: AnalysisCaseCreate) -> str:
        """
        Create a new analysis case in the database.

        Args:
            data: AnalysisCaseCreate schema with analysis information

        Returns:
            str: The MongoDB ObjectId of the created analysis case as string

        Raises:
            ValueError: If analysis_id already exists
            Exception: If database operation fails
        """
        try:
            collection = db.get_collection(self.collection_name)

            # Check if analysis_id already exists
            existing = await collection.find_one({"analysis_id": data.analysis_id})
            if existing:
                raise ValueError(f"Analysis with analysis_id '{data.analysis_id}' already exists")

            # Prepare document
            analysis_dict = data.model_dump()
            analysis_dict["created_at"] = datetime.utcnow()

            # Insert into database
            result = await collection.insert_one(analysis_dict)
            logger.info(
                f"Created analysis {data.analysis_id} for patient {data.patient_id}, "
                f"lesion {data.lesion_id}"
            )

            return str(result.inserted_id)

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Error creating analysis: {str(e)}")
            raise

    async def get_analysis(self, analysis_id: str) -> Optional[AnalysisCaseInDB]:
        """
        Get an analysis case by analysis_id.

        Args:
            analysis_id: The unique analysis identifier (e.g., "AN-001")

        Returns:
            AnalysisCaseInDB if found, None otherwise
        """
        try:
            collection = db.get_collection(self.collection_name)
            analysis_doc = await collection.find_one({"analysis_id": analysis_id})

            if analysis_doc:
                # Convert ObjectId to string for Pydantic
                analysis_doc["_id"] = str(analysis_doc["_id"])
                return AnalysisCaseInDB(**analysis_doc)

            return None

        except Exception as e:
            logger.error(f"Error getting analysis {analysis_id}: {str(e)}")
            raise

    async def get_analysis_by_object_id(self, object_id: str) -> Optional[AnalysisCaseInDB]:
        """
        Get an analysis case by MongoDB ObjectId.

        Args:
            object_id: The MongoDB ObjectId as string

        Returns:
            AnalysisCaseInDB if found, None otherwise
        """
        try:
            collection = db.get_collection(self.collection_name)
            analysis_doc = await collection.find_one({"_id": ObjectId(object_id)})

            if analysis_doc:
                analysis_doc["_id"] = str(analysis_doc["_id"])
                return AnalysisCaseInDB(**analysis_doc)

            return None

        except Exception as e:
            logger.error(f"Error getting analysis by ObjectId {object_id}: {str(e)}")
            raise

    async def update_analysis(self, analysis_id: str, data: AnalysisCaseUpdate) -> bool:
        """
        Update an analysis case's information.

        Args:
            analysis_id: The unique analysis identifier
            data: AnalysisCaseUpdate schema with fields to update

        Returns:
            bool: True if analysis was updated, False if not found

        Raises:
            Exception: If database operation fails
        """
        try:
            collection = db.get_collection(self.collection_name)

            # Only include fields that were actually set
            update_data = data.model_dump(exclude_unset=True)

            if not update_data:
                logger.warning(f"No fields to update for analysis {analysis_id}")
                return False

            # Update the document
            result = await collection.update_one(
                {"analysis_id": analysis_id},
                {"$set": update_data}
            )

            if result.modified_count > 0:
                logger.info(f"Updated analysis {analysis_id}")
                return True

            return False

        except Exception as e:
            logger.error(f"Error updating analysis {analysis_id}: {str(e)}")
            raise

    async def delete_analysis(self, analysis_id: str) -> bool:
        """
        Delete an analysis case from the database.

        Args:
            analysis_id: The unique analysis identifier

        Returns:
            bool: True if analysis was deleted, False if not found

        Raises:
            Exception: If database operation fails
        """
        try:
            collection = db.get_collection(self.collection_name)

            result = await collection.delete_one({"analysis_id": analysis_id})

            if result.deleted_count > 0:
                logger.info(f"Deleted analysis {analysis_id}")
                return True

            logger.warning(f"Analysis {analysis_id} not found for deletion")
            return False

        except Exception as e:
            logger.error(f"Error deleting analysis {analysis_id}: {str(e)}")
            raise

    async def get_analyses_by_patient(self, patient_id: str) -> List[AnalysisCaseInDB]:
        """
        Get all analysis cases for a specific patient.

        Args:
            patient_id: The patient identifier

        Returns:
            List[AnalysisCaseInDB]: List of analyses for the patient
        """
        try:
            collection = db.get_collection(self.collection_name)

            cursor = collection.find({"patient_id": patient_id}).sort("temporal_data.capture_date", -1)
            analyses = []

            async for doc in cursor:
                doc["_id"] = str(doc["_id"])
                analyses.append(AnalysisCaseInDB(**doc))

            return analyses

        except Exception as e:
            logger.error(f"Error getting analyses for patient {patient_id}: {str(e)}")
            raise

    async def get_analyses_by_lesion(self, lesion_id: str) -> List[AnalysisCaseInDB]:
        """
        Get all analysis cases for a specific lesion.

        Args:
            lesion_id: The lesion identifier

        Returns:
            List[AnalysisCaseInDB]: List of analyses for the lesion
        """
        try:
            collection = db.get_collection(self.collection_name)

            cursor = collection.find({"lesion_id": lesion_id}).sort("temporal_data.capture_date", -1)
            analyses = []

            async for doc in cursor:
                doc["_id"] = str(doc["_id"])
                analyses.append(AnalysisCaseInDB(**doc))

            return analyses

        except Exception as e:
            logger.error(f"Error getting analyses for lesion {lesion_id}: {str(e)}")
            raise

    async def get_temporal_progression(self, lesion_id: str) -> List[AnalysisCaseInDB]:
        """
        Get temporal progression of analysis cases for a lesion, sorted chronologically.

        This is useful for tracking how a lesion changes over time.

        Args:
            lesion_id: The lesion identifier

        Returns:
            List[AnalysisCaseInDB]: List of analyses sorted by capture date (oldest to newest)
        """
        try:
            collection = db.get_collection(self.collection_name)

            # Sort by capture date in ascending order (oldest first)
            cursor = collection.find({"lesion_id": lesion_id}).sort("temporal_data.capture_date", 1)
            analyses = []

            async for doc in cursor:
                doc["_id"] = str(doc["_id"])
                analyses.append(AnalysisCaseInDB(**doc))

            logger.info(f"Retrieved {len(analyses)} temporal progression records for lesion {lesion_id}")
            return analyses

        except Exception as e:
            logger.error(f"Error getting temporal progression for lesion {lesion_id}: {str(e)}")
            raise

    async def list_analyses(self, skip: int = 0, limit: int = 100) -> List[AnalysisCaseInDB]:
        """
        List all analysis cases with pagination.

        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            List[AnalysisCaseInDB]: List of analyses
        """
        try:
            collection = db.get_collection(self.collection_name)

            cursor = collection.find().skip(skip).limit(limit).sort("created_at", -1)
            analyses = []

            async for doc in cursor:
                doc["_id"] = str(doc["_id"])
                analyses.append(AnalysisCaseInDB(**doc))

            return analyses

        except Exception as e:
            logger.error(f"Error listing analyses: {str(e)}")
            raise

    async def count_analyses(self) -> int:
        """
        Get total count of analysis cases in the database.

        Returns:
            int: Total number of analyses
        """
        try:
            collection = db.get_collection(self.collection_name)
            count = await collection.count_documents({})
            return count

        except Exception as e:
            logger.error(f"Error counting analyses: {str(e)}")
            raise

    async def count_analyses_by_patient(self, patient_id: str) -> int:
        """
        Count the number of analysis cases for a specific patient.

        Args:
            patient_id: The patient identifier

        Returns:
            int: Number of analyses for the patient
        """
        try:
            collection = db.get_collection(self.collection_name)
            count = await collection.count_documents({"patient_id": patient_id})
            return count

        except Exception as e:
            logger.error(f"Error counting analyses for patient {patient_id}: {str(e)}")
            raise

    async def count_analyses_by_lesion(self, lesion_id: str) -> int:
        """
        Count the number of analysis cases for a specific lesion.

        Args:
            lesion_id: The lesion identifier

        Returns:
            int: Number of analyses for the lesion
        """
        try:
            collection = db.get_collection(self.collection_name)
            count = await collection.count_documents({"lesion_id": lesion_id})
            return count

        except Exception as e:
            logger.error(f"Error counting analyses for lesion {lesion_id}: {str(e)}")
            raise

    async def analysis_exists(self, analysis_id: str) -> bool:
        """
        Check if an analysis exists by analysis_id.

        Args:
            analysis_id: The unique analysis identifier

        Returns:
            bool: True if analysis exists, False otherwise
        """
        try:
            collection = db.get_collection(self.collection_name)
            count = await collection.count_documents({"analysis_id": analysis_id})
            return count > 0

        except Exception as e:
            logger.error(f"Error checking if analysis exists {analysis_id}: {str(e)}")
            raise

    async def get_latest_analysis_for_lesion(self, lesion_id: str) -> Optional[AnalysisCaseInDB]:
        """
        Get the most recent analysis for a specific lesion.

        Args:
            lesion_id: The lesion identifier

        Returns:
            AnalysisCaseInDB if found, None otherwise
        """
        try:
            collection = db.get_collection(self.collection_name)

            # Find the most recent analysis by capture date
            analysis_doc = await collection.find_one(
                {"lesion_id": lesion_id},
                sort=[("temporal_data.capture_date", -1)]
            )

            if analysis_doc:
                analysis_doc["_id"] = str(analysis_doc["_id"])
                return AnalysisCaseInDB(**analysis_doc)

            return None

        except Exception as e:
            logger.error(f"Error getting latest analysis for lesion {lesion_id}: {str(e)}")
            raise

    async def get_high_risk_analyses(
        self,
        threshold: float = 0.5,
        skip: int = 0,
        limit: int = 100
    ) -> List[AnalysisCaseInDB]:
        """
        Get analysis cases with high malignancy probability from clinical ML model.

        Args:
            threshold: Minimum probability threshold (default 0.5)
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            List[AnalysisCaseInDB]: List of high-risk analyses
        """
        try:
            collection = db.get_collection(self.collection_name)

            cursor = collection.find({
                "model_outputs.clinical_ml_model.malignant_probability": {"$gte": threshold}
            }).skip(skip).limit(limit).sort("model_outputs.clinical_ml_model.malignant_probability", -1)

            analyses = []
            async for doc in cursor:
                doc["_id"] = str(doc["_id"])
                analyses.append(AnalysisCaseInDB(**doc))

            return analyses

        except Exception as e:
            logger.error(f"Error getting high-risk analyses: {str(e)}")
            raise


# Global instance
analysis_manager = AnalysisManager()
