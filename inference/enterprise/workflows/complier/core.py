import asyncio
from asyncio import AbstractEventLoop
from typing import Any, Dict, Optional

from fastapi import BackgroundTasks

from inference.core.cache import cache
from inference.core.env import API_KEY, MAX_ACTIVE_MODELS
from inference.core.managers.base import ModelManager
from inference.core.managers.decorators.fixed_size_cache import WithFixedSizeCache
from inference.core.registries.roboflow import RoboflowModelRegistry
from inference.enterprise.workflows.complier.entities import StepExecutionMode
from inference.enterprise.workflows.complier.execution_engine import execute_graph
from inference.enterprise.workflows.complier.graph_parser import prepare_execution_graph
from inference.enterprise.workflows.complier.steps_executors.active_learning_middlewares import (
    WorkflowsActiveLearningMiddleware,
)
from inference.enterprise.workflows.complier.validator import (
    validate_workflow_specification,
)
from inference.enterprise.workflows.entities.workflows_specification import (
    WorkflowSpecification,
)
from inference.enterprise.workflows.errors import InvalidSpecificationVersionError
from inference.models.utils import ROBOFLOW_MODEL_TYPES


def compile_and_execute(
    workflow_specification: dict,
    runtime_parameters: Dict[str, Any],
    api_key: Optional[str] = None,
    model_manager: Optional[ModelManager] = None,
    loop: Optional[AbstractEventLoop] = None,
    active_learning_middleware: Optional[WorkflowsActiveLearningMiddleware] = None,
    background_tasks: Optional[BackgroundTasks] = None,
    max_concurrent_steps: int = 1,
    step_execution_mode: StepExecutionMode = StepExecutionMode.LOCAL,
) -> dict:
    if loop is None:
        try:
            loop = asyncio.get_event_loop()
        except:
            loop = asyncio.new_event_loop()
    return loop.run_until_complete(
        compile_and_execute_async(
            workflow_specification=workflow_specification,
            runtime_parameters=runtime_parameters,
            model_manager=model_manager,
            api_key=api_key,
            active_learning_middleware=active_learning_middleware,
            background_tasks=background_tasks,
            max_concurrent_steps=max_concurrent_steps,
            step_execution_mode=step_execution_mode,
        )
    )


async def compile_and_execute_async(
    workflow_specification: dict,
    runtime_parameters: Dict[str, Any],
    model_manager: Optional[ModelManager] = None,
    api_key: Optional[str] = None,
    active_learning_middleware: Optional[WorkflowsActiveLearningMiddleware] = None,
    background_tasks: Optional[BackgroundTasks] = None,
    max_concurrent_steps: int = 1,
    step_execution_mode: StepExecutionMode = StepExecutionMode.LOCAL,
) -> dict:
    if api_key is None:
        api_key = API_KEY
    if model_manager is None:
        model_registry = RoboflowModelRegistry(ROBOFLOW_MODEL_TYPES)
        model_manager = ModelManager(model_registry=model_registry)
        model_manager = WithFixedSizeCache(model_manager, max_size=MAX_ACTIVE_MODELS)
    if active_learning_middleware is None:
        active_learning_middleware = WorkflowsActiveLearningMiddleware(cache=cache)
    parsed_workflow_specification = WorkflowSpecification.parse_obj(
        workflow_specification
    )
    if parsed_workflow_specification.specification.version != "1.0":
        raise InvalidSpecificationVersionError(
            f"Only version 1.0 of workflow specification is supported."
        )
    validate_workflow_specification(
        workflow_specification=parsed_workflow_specification.specification
    )
    execution_graph = prepare_execution_graph(
        workflow_specification=parsed_workflow_specification.specification
    )
    return await execute_graph(
        execution_graph=execution_graph,
        runtime_parameters=runtime_parameters,
        model_manager=model_manager,
        active_learning_middleware=active_learning_middleware,
        background_tasks=background_tasks,
        api_key=api_key,
        max_concurrent_steps=max_concurrent_steps,
        step_execution_mode=step_execution_mode,
    )
