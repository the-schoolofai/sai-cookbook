"""
Health Check API Endpoints

Provides health status and service monitoring endpoints.
"""

from datetime import datetime
from fastapi import APIRouter, status
from openai import OpenAI

from app.config.settings import get_settings
from app.models.schemas import HealthResponse
from app.core.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health Check",
    description="Check the health status of the API and its dependencies",
)
async def health_check():
    """
    Comprehensive health check endpoint.

    Returns the status of the API and all its dependencies.
    """
    settings = get_settings()
    services = {}
    overall_status = "healthy"

    # Check OpenAI API connection
    try:
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        # Simple API call to verify connection
        client.models.list()
        services["openai_api"] = "healthy"
        logger.debug("OpenAI API health check passed")
    except Exception as e:
        services["openai_api"] = f"unhealthy: {str(e)}"
        overall_status = "degraded"
        logger.warning(f"OpenAI API health check failed: {e}")

    # Check configuration
    try:
        settings.validate()
        services["configuration"] = "healthy"
    except Exception as e:
        services["configuration"] = f"unhealthy: {str(e)}"
        overall_status = "unhealthy"
        logger.error(f"Configuration validation failed: {e}")

    # Check file system
    try:
        from pathlib import Path

        upload_dir = Path(settings.UPLOAD_DIR)
        if upload_dir.exists() and upload_dir.is_dir():
            services["file_system"] = "healthy"
        else:
            services["file_system"] = "degraded: upload directory not found"
            overall_status = "degraded"
    except Exception as e:
        services["file_system"] = f"unhealthy: {str(e)}"
        overall_status = "degraded"

    return HealthResponse(
        status=overall_status,
        version=settings.API_VERSION,
        timestamp=datetime.now(),
        services=services,
    )


@router.get(
    "/health/ready",
    status_code=status.HTTP_200_OK,
    summary="Readiness Check",
    description="Check if the API is ready to accept requests",
)
async def readiness_check():
    """
    Readiness probe for Kubernetes and container orchestration.

    Returns 200 if the service is ready to accept traffic.
    """
    settings = get_settings()

    try:
        # Verify critical dependencies
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        client.models.list()

        return {
            "status": "ready",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return {
            "status": "not_ready",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


@router.get(
    "/health/live",
    status_code=status.HTTP_200_OK,
    summary="Liveness Check",
    description="Check if the API is alive and running",
)
async def liveness_check():
    """
    Liveness probe for Kubernetes and container orchestration.

    Returns 200 if the service is alive.
    """
    return {
        "status": "alive",
        "timestamp": datetime.now().isoformat(),
    }
