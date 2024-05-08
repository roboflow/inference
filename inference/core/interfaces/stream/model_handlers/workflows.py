import asyncio
from asyncio import AbstractEventLoop
from typing import List, Optional

from inference.core import logger
from inference.core.interfaces.camera.entities import VideoFrame
from inference.enterprise.workflows.execution_engine.core import ExecutionEngine


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
