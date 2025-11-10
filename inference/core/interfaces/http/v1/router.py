"""
Main router for v1 API endpoints.

This module aggregates all v1 API endpoints and provides a single router
to be mounted in the main FastAPI application.
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from inference.core import logger
from inference.core.managers.base import ModelManager
from inference.core.version import __version__
from inference.core.env import DISABLE_WORKFLOW_ENDPOINTS

# Import endpoint creators
from inference.core.interfaces.http.v1.endpoints.models import create_model_endpoints
from inference.core.interfaces.http.v1.endpoints.workflows import (
    create_workflow_endpoints,
)


def create_v1_router(model_manager: ModelManager) -> APIRouter:
    """
    Create the main v1 API router with all endpoints.

    Args:
        model_manager: The model manager instance to use for all endpoints

    Returns:
        Configured APIRouter with all v1 endpoints
    """
    # Create main v1 router
    v1_router = APIRouter(prefix="/v1", tags=["v1"])

    # Add version info endpoint
    @v1_router.get(
        "/info",
        summary="V1 API Information",
        description="Get information about the v1 API version and capabilities",
        tags=["v1"],
    )
    def v1_info():
        """Get v1 API version information."""
        return JSONResponse(
            content={
                "api_version": "v1.0",
                "server_version": __version__,
                "description": "Roboflow Inference Server v1 API",
                "features": [
                    "Header-based authentication (no API keys in request bodies)",
                    "Multipart form data for efficient image uploads",
                    "Clean RESTful interfaces with explicit versioning",
                ],
                "authentication": {
                    "methods": [
                        "Authorization: Bearer <api_key>",
                        "X-Roboflow-Api-Key: <api_key>",
                        "Query parameter: ?api_key=<api_key>",
                    ],
                },
                "endpoints": {
                    "models": [
                        "/v1/object-detection/{model_id}",
                        "/v1/instance-segmentation/{model_id}",
                        "/v1/classification/{model_id}",
                        "/v1/keypoint-detection/{model_id}",
                    ],
                    "workflows": [
                        "/v1/workflows/{workspace_id}/{workflow_id}",
                        "/v1/workflows/run",
                    ],
                },
            },
            headers={"X-Inference-Api-Version": "v1.0"},
        )

    # Create and include model endpoints
    logger.info("Registering v1 model inference endpoints")
    model_router = create_model_endpoints(model_manager)
    v1_router.include_router(model_router)

    # Create and include workflow endpoints (if enabled)
    if not DISABLE_WORKFLOW_ENDPOINTS:
        logger.info("Registering v1 workflow endpoints")
        workflow_router = create_workflow_endpoints(model_manager)
        v1_router.include_router(workflow_router)
    else:
        logger.info("Workflow endpoints disabled, skipping v1 workflow routes")

    logger.info("V1 API router created successfully")
    return v1_router


# This will be set by http_api.py when creating the interface
v1_router = None
