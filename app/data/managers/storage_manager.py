"""
Storage Manager for handling uploaded images and file operations.
"""

import os
import shutil
from pathlib import Path
from typing import Optional
from datetime import datetime
from fastapi import UploadFile
import logging

logger = logging.getLogger(__name__)


class StorageManager:
    """Manager for file storage operations."""

    def __init__(self, base_upload_dir: str = "uploads"):
        """
        Initialize StorageManager.

        Args:
            base_upload_dir: Base directory for storing uploaded files
        """
        self.base_upload_dir = base_upload_dir
        self._ensure_upload_dir_exists()

    def _ensure_upload_dir_exists(self):
        """Ensure the upload directory exists."""
        Path(self.base_upload_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Upload directory ensured at: {self.base_upload_dir}")

    async def save_uploaded_image(
        self,
        file: UploadFile,
        analysis_id: str,
        patient_id: Optional[str] = None
    ) -> dict:
        """
        Read uploaded image file and return binary data for MongoDB storage.

        NOTE: Images are now stored in MongoDB instead of disk to support
        Render.com deployment (free tier has no persistent disk).

        Args:
            file: The uploaded file from FastAPI
            analysis_id: Unique analysis identifier for naming the file
            patient_id: Optional patient ID (included for compatibility)

        Returns:
            dict: Contains filename, path (virtual), content_type, and binary data

        Raises:
            Exception: If file read operation fails
        """
        try:
            # Determine file extension
            file_extension = ""
            if file.filename:
                file_extension = os.path.splitext(file.filename)[1]
            if not file_extension:
                file_extension = ".jpg"  # Default to .jpg if no extension

            # Create filename
            filename = f"{analysis_id}{file_extension}"

            # Read file content into memory (will be stored in MongoDB)
            image_bytes = await file.read()
            file_size = len(image_bytes)

            # Virtual path indicating MongoDB storage
            virtual_path = f"mongodb://{analysis_id}"

            logger.info(f"Image prepared for MongoDB storage: {filename} ({file_size} bytes)")

            return {
                "filename": filename,
                "path": virtual_path,
                "size": file_size,
                "content_type": file.content_type or "image/jpeg",
                "data": image_bytes  # Binary data for MongoDB
            }

        except Exception as e:
            logger.error(f"Error reading uploaded image: {str(e)}")
            raise

    async def get_image_path(self, filename: str, patient_id: Optional[str] = None) -> str:
        """
        Get the full path to a stored image.

        Args:
            filename: Name of the image file
            patient_id: Optional patient ID if files are organized by patient

        Returns:
            str: Full path to the image file

        Raises:
            FileNotFoundError: If the image file does not exist
        """
        try:
            if patient_id:
                file_path = os.path.join(self.base_upload_dir, patient_id, filename)
            else:
                file_path = os.path.join(self.base_upload_dir, filename)

            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Image file not found: {file_path}")

            return file_path

        except Exception as e:
            logger.error(f"Error getting image path: {str(e)}")
            raise

    async def delete_image(self, filename: str, patient_id: Optional[str] = None) -> bool:
        """
        Delete a stored image file.

        Args:
            filename: Name of the image file to delete
            patient_id: Optional patient ID if files are organized by patient

        Returns:
            bool: True if file was deleted, False if file didn't exist

        Raises:
            Exception: If deletion fails for reasons other than file not existing
        """
        try:
            if patient_id:
                file_path = os.path.join(self.base_upload_dir, patient_id, filename)
            else:
                file_path = os.path.join(self.base_upload_dir, filename)

            if not os.path.exists(file_path):
                logger.warning(f"Image file not found for deletion: {file_path}")
                return False

            os.remove(file_path)
            logger.info(f"Image deleted: {file_path}")

            # Clean up empty patient directories
            if patient_id:
                patient_dir = os.path.join(self.base_upload_dir, patient_id)
                if os.path.exists(patient_dir) and not os.listdir(patient_dir):
                    os.rmdir(patient_dir)
                    logger.info(f"Empty patient directory removed: {patient_dir}")

            return True

        except Exception as e:
            logger.error(f"Error deleting image: {str(e)}")
            raise

    async def delete_patient_images(self, patient_id: str) -> int:
        """
        Delete all images for a specific patient.

        Args:
            patient_id: The patient identifier

        Returns:
            int: Number of images deleted

        Raises:
            Exception: If deletion fails
        """
        try:
            patient_dir = os.path.join(self.base_upload_dir, patient_id)

            if not os.path.exists(patient_dir):
                logger.warning(f"Patient directory not found: {patient_dir}")
                return 0

            # Count and delete all files
            deleted_count = 0
            for filename in os.listdir(patient_dir):
                file_path = os.path.join(patient_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    deleted_count += 1

            # Remove the patient directory
            if os.path.exists(patient_dir):
                os.rmdir(patient_dir)
                logger.info(f"Patient directory removed: {patient_dir}")

            logger.info(f"Deleted {deleted_count} images for patient {patient_id}")
            return deleted_count

        except Exception as e:
            logger.error(f"Error deleting patient images: {str(e)}")
            raise

    def get_storage_stats(self) -> dict:
        """
        Get statistics about stored files.

        Returns:
            dict: Statistics including total files, total size, and patient count
        """
        try:
            total_files = 0
            total_size = 0
            patient_count = 0

            if not os.path.exists(self.base_upload_dir):
                return {
                    "total_files": 0,
                    "total_size_bytes": 0,
                    "total_size_mb": 0.0,
                    "patient_count": 0
                }

            # Walk through all files
            for root, dirs, files in os.walk(self.base_upload_dir):
                # Count patient directories (one level below base)
                if root == self.base_upload_dir:
                    patient_count = len(dirs)

                for file in files:
                    file_path = os.path.join(root, file)
                    if os.path.isfile(file_path):
                        total_files += 1
                        total_size += os.path.getsize(file_path)

            return {
                "total_files": total_files,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "patient_count": patient_count,
                "base_directory": self.base_upload_dir
            }

        except Exception as e:
            logger.error(f"Error getting storage stats: {str(e)}")
            return {
                "total_files": 0,
                "total_size_bytes": 0,
                "total_size_mb": 0.0,
                "patient_count": 0,
                "error": str(e)
            }


# Global instance
storage_manager = StorageManager()
