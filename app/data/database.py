"""
MongoDB database connection manager.
Implements singleton pattern for async database connections using Motor.
"""

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from typing import Optional
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)


class MongoDB:
    """
    Singleton MongoDB connection manager using Motor (async driver).
    """
    _instance: Optional["MongoDB"] = None
    _client: Optional[AsyncIOMotorClient] = None
    _database: Optional[AsyncIOMotorDatabase] = None

    def __new__(cls):
        """Ensure only one instance exists (Singleton pattern)."""
        if cls._instance is None:
            cls._instance = super(MongoDB, cls).__new__(cls)
        return cls._instance

    async def connect(self):
        """
        Establish connection to MongoDB.
        Should be called during application startup.
        """
        if self._client is None:
            try:
                logger.info(f"Connecting to MongoDB at {settings.MONGODB_URI}")
                self._client = AsyncIOMotorClient(settings.MONGODB_URI)
                self._database = self._client[settings.MONGODB_DB_NAME]

                # Test the connection
                await self._client.admin.command('ping')
                logger.info(f"Successfully connected to MongoDB database: {settings.MONGODB_DB_NAME}")
            except Exception as e:
                logger.error(f"Failed to connect to MongoDB: {str(e)}")
                raise

    async def disconnect(self):
        """
        Close MongoDB connection.
        Should be called during application shutdown.
        """
        if self._client is not None:
            try:
                logger.info("Closing MongoDB connection...")
                self._client.close()
                self._client = None
                self._database = None
                logger.info("MongoDB connection closed successfully")
            except Exception as e:
                logger.error(f"Error closing MongoDB connection: {str(e)}")
                raise

    def get_database(self) -> AsyncIOMotorDatabase:
        """
        Get the MongoDB database instance.

        Returns:
            AsyncIOMotorDatabase: The database instance

        Raises:
            RuntimeError: If database is not connected
        """
        if self._database is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._database

    def get_collection(self, collection_name: str):
        """
        Get a specific collection from the database.

        Args:
            collection_name: Name of the collection to retrieve

        Returns:
            AsyncIOMotorCollection: The collection instance

        Raises:
            RuntimeError: If database is not connected
        """
        database = self.get_database()
        return database[collection_name]

    @property
    def is_connected(self) -> bool:
        """Check if database is connected."""
        return self._client is not None and self._database is not None


# Global database instance
db = MongoDB()
