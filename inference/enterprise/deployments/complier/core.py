import asyncio
from asyncio import AbstractEventLoop
from typing import Any, Dict, Optional

from inference.core.env import API_KEY, MAX_ACTIVE_MODELS
from inference.core.managers.base import ModelManager
from inference.core.managers.decorators.fixed_size_cache import WithFixedSizeCache
from inference.core.registries.roboflow import RoboflowModelRegistry
from inference.enterprise.deployments.complier.execution_engine import execute_graph
from inference.enterprise.deployments.complier.graph_parser import (
    prepare_execution_graph,
)
from inference.enterprise.deployments.complier.validator import validate_deployment_spec
from inference.enterprise.deployments.entities.deployment_specs import (
    DeploymentSpecification,
)
from inference.enterprise.deployments.errors import InvalidSpecificationVersionError
from inference.models.utils import ROBOFLOW_MODEL_TYPES


def compile_and_execute(
    deployment_spec: dict,
    runtime_parameters: Dict[str, Any],
    api_key: Optional[str] = None,
    loop: Optional[AbstractEventLoop] = None,
) -> dict:
    if loop is None:
        loop = asyncio.get_event_loop()
    return loop.run_until_complete(
        compile_and_execute_async(
            deployment_spec=deployment_spec,
            runtime_parameters=runtime_parameters,
            api_key=api_key,
        )
    )


async def compile_and_execute_async(
    deployment_spec: dict,
    runtime_parameters: Dict[str, Any],
    api_key: Optional[str] = None,
) -> dict:
    if api_key is None:
        api_key = API_KEY
    model_registry = RoboflowModelRegistry(ROBOFLOW_MODEL_TYPES)
    model_manager = ModelManager(model_registry=model_registry)
    model_manager = WithFixedSizeCache(model_manager, max_size=MAX_ACTIVE_MODELS)
    parsed_deployment_spec = DeploymentSpecification.parse_obj(deployment_spec)
    if parsed_deployment_spec.specification.version != "1.0":
        raise InvalidSpecificationVersionError(
            f"Only version 1.0 of deployment specs is supported."
        )
    validate_deployment_spec(deployment_spec=parsed_deployment_spec.specification)
    execution_graph = prepare_execution_graph(
        deployment_spec=parsed_deployment_spec.specification
    )
    return await execute_graph(
        execution_graph=execution_graph,
        runtime_parameters=runtime_parameters,
        model_manager=model_manager,
        api_key=api_key,
    )
