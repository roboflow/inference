"""Model registry and server-state HTTP routes (list models, add/clear)."""

from typing import Optional

from fastapi import APIRouter

from inference.core import logger
from inference.core.entities.requests.server_state import (
    AddModelRequest,
    ClearModelRequest,
)
from inference.core.entities.responses.server_state import ModelsDescriptions
from inference.core.env import (
    GET_MODEL_REGISTRY_ENABLED,
    GCP_SERVERLESS,
    LAMBDA,
)
from inference.core.interfaces.http.error_handlers import with_route_exceptions
from inference.core.managers.base import ModelManager
from inference.models.aliases import resolve_roboflow_model_alias

def create_models_router(model_manager: ModelManager) -> APIRouter:
    router = APIRouter()

    if not LAMBDA and GET_MODEL_REGISTRY_ENABLED:

        @router.get(
            "/model/registry",
            response_model=ModelsDescriptions,
            summary="Get model keys",
            description="Get the ID of each loaded model",
        )
        def registry():
            """Get the ID of each loaded model in the registry.

            Returns:
                ModelsDescriptions: The object containing models descriptions
            """
            logger.debug("Reached /model/registry")
            models_descriptions = model_manager.describe_models()
            return ModelsDescriptions.from_models_descriptions(
                models_descriptions=models_descriptions
            )

    if not (LAMBDA or GCP_SERVERLESS):

        @router.post(
            "/model/add",
            response_model=ModelsDescriptions,
            summary="Load a model",
            description="Load the model with the given model ID",
        )
        @with_route_exceptions
        def model_add(
            request: AddModelRequest,
            countinference: Optional[bool] = None,
            service_secret: Optional[str] = None,
        ):
            """Load the model with the given model ID into the model manager.

            Args:
                request (AddModelRequest): The request containing the model ID and optional API key.
                countinference (Optional[bool]): Whether to count inference or not.
                service_secret (Optional[str]): The service secret for the request.

            Returns:
                ModelsDescriptions: The object containing models descriptions
            """
            logger.debug("Reached /model/add")
            de_aliased_model_id = resolve_roboflow_model_alias(
                model_id=request.model_id
            )
            logger.info(f"Loading model: {de_aliased_model_id}")
            model_manager.add_model(
                de_aliased_model_id,
                request.api_key,
                countinference=countinference,
                service_secret=service_secret,
            )
            models_descriptions = model_manager.describe_models()
            return ModelsDescriptions.from_models_descriptions(
                models_descriptions=models_descriptions
            )

        @router.post(
            "/model/remove",
            response_model=ModelsDescriptions,
            summary="Remove a model",
            description="Remove the model with the given model ID",
        )
        @with_route_exceptions
        def model_remove(request: ClearModelRequest):
            """Remove the model with the given model ID from the model manager.

            Args:
                request (ClearModelRequest): The request containing the model ID to be removed.

            Returns:
                ModelsDescriptions: The object containing models descriptions
            """
            logger.debug("Reached /model/remove")
            de_aliased_model_id = resolve_roboflow_model_alias(
                model_id=request.model_id
            )
            model_manager.remove(de_aliased_model_id)
            models_descriptions = model_manager.describe_models()
            return ModelsDescriptions.from_models_descriptions(
                models_descriptions=models_descriptions
            )

        @router.post(
            "/model/clear",
            response_model=ModelsDescriptions,
            summary="Remove all models",
            description="Remove all loaded models",
        )
        @with_route_exceptions
        def model_clear():
            """Remove all loaded models from the model manager.

            Returns:
                ModelsDescriptions: The object containing models descriptions
            """
            logger.debug("Reached /model/clear")
            model_manager.clear()
            models_descriptions = model_manager.describe_models()
            return ModelsDescriptions.from_models_descriptions(
                models_descriptions=models_descriptions
            )

    return router

