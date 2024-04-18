import asyncio
from asyncio import AbstractEventLoop
from typing import List, Optional

from fastapi import BackgroundTasks

from inference.core import logger
from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.interfaces.camera.entities import VideoFrame
from inference.core.managers.base import ModelManager
from inference.enterprise.workflows.complier.core import compile_and_execute
from inference.enterprise.workflows.complier.entities import StepExecutionMode
from inference.enterprise.workflows.complier.steps_executors.active_learning_middlewares import (
    WorkflowsActiveLearningMiddleware,
)
from inference.enterprise.workflows.execution_engine.core import ExecutionEngine


def run_video_frame_through_workflow(
    video_frames: List[VideoFrame],
    workflow_specification: dict,
    model_manager: ModelManager,
    image_input_name: str,
    workflows_parameters: Optional[dict],
    api_key: Optional[str],
    workflows_active_learning_middleware: WorkflowsActiveLearningMiddleware,
    background_tasks: Optional[BackgroundTasks],
) -> List[dict]:
    if len(video_frames) > 1:
        logger.warning(
            f"Workflows in InferencePipeline do not support multiple video sources. Using the first video frame."
        )
    step_execution_mode = StepExecutionMode.LOCAL
    if workflows_parameters is None:
        workflows_parameters = {}
    workflows_parameters[image_input_name] = video_frames[0].image
    return [
        compile_and_execute(
            workflow_specification=workflow_specification,
            runtime_parameters=workflows_parameters,
            model_manager=model_manager,
            api_key=api_key,
            max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
            step_execution_mode=step_execution_mode,
            active_learning_middleware=workflows_active_learning_middleware,
            background_tasks=background_tasks,
        )
    ]


class WorkflowRunner:

    def __init__(self):
        self._event_loop: Optional[AbstractEventLoop] = None

    def run_workflow(
        self,
        video_frames: List[VideoFrame],
        workflows_parameters: Optional[dict],
        execution_engine: ExecutionEngine,
        image_input_name: str,
    ) -> List[dict]:
        if self._event_loop is None:
            try:
                event_loop = asyncio.get_event_loop()
            except:
                event_loop = asyncio.new_event_loop()
            self._event_loop = event_loop
        if len(video_frames) > 1:
            logger.warning(
                f"Workflows in InferencePipeline do not support multiple video sources. Using the first video frame."
            )
        if workflows_parameters is None:
            workflows_parameters = {}
        workflows_parameters[image_input_name] = video_frames[0].image
        return [
            execution_engine.run(
                runtime_parameters=workflows_parameters, event_loop=self._event_loop
            )
        ]
