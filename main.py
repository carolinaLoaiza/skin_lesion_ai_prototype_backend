"""
Main entry point for the Skin Lesion AI Prototype Backend API.
This module initializes the FastAPI application and configures middleware, routes, and startup events.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.core.logger import logger
from app.api import prediction_hugging, patients, lesions, analyses
from app.data.database import db


# Initialize FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="API for skin lesion malignancy risk prediction using multiple AI models",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """
    Event handler for application startup.
    Loads ML models and performs initial setup.
    """
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Debug mode: {settings.DEBUG}")

    # Connect to MongoDB
    try:
        await db.connect()
        logger.info("MongoDB connection established")
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {str(e)}")
        logger.warning("Application will continue without database support")

    # Models are hosted on Hugging Face Space - no local model loading needed
    logger.info("Application startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    """
    Event handler for application shutdown.
    Performs cleanup operations.
    """
    logger.info("Shutting down application")

    # Disconnect from MongoDB
    try:
        await db.disconnect()
        logger.info("MongoDB connection closed")
    except Exception as e:
        logger.error(f"Error closing MongoDB connection: {str(e)}")


@app.get("/")
async def root():
    """
    Root endpoint for health check and API information.

    Returns:
        dict: Basic API information
    """
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "database_connected": db.is_connected,
        "endpoints": {
            "predict": "/api/predict",
            "explain": "/api/explain",
            "patients": "/api/patients",
            "lesions": "/api/lesions",
            "analyses": "/api/analyses",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify API availability and MongoDB connection.

    Returns:
        dict: Health status including database connection state
    """
    return {
        "status": "healthy",
        "database": "connected" if db.is_connected else "disconnected"
    }


# Include routers
app.include_router(prediction_hugging.router, prefix="/api", tags=["prediction"])
app.include_router(patients.router, tags=["patients"])
app.include_router(lesions.router, tags=["lesions"])
app.include_router(lesions.patients_router, tags=["lesions"])  # Patient-specific lesion endpoints
app.include_router(analyses.router, tags=["analyses"])
app.include_router(analyses.patients_router, tags=["analyses"])  # Patient-specific analysis endpoints
app.include_router(analyses.lesions_router, tags=["analyses"])  # Lesion-specific analysis endpoints


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
