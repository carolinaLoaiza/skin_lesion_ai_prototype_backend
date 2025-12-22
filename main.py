"""
Main entry point for the Skin Lesion AI Prototype Backend API.
This module initializes the FastAPI application and configures middleware, routes, and startup events.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.core.logger import logger
from app.api import prediction


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
    # Models will be loaded lazily on first request or can be preloaded here
    logger.info("Application startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    """
    Event handler for application shutdown.
    Performs cleanup operations.
    """
    logger.info("Shutting down application")


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
        "endpoints": {
            "predict": "/api/predict",
            "explain": "/api/explain",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify API availability.

    Returns:
        dict: Health status
    """
    return {"status": "healthy"}


# Include routers
app.include_router(prediction.router, prefix="/api", tags=["prediction"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
