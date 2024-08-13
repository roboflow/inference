import asyncio
from typing import List, Optional

from inference.core.interfaces.camera.entities import VideoFrame
from inference.core.workflows.execution_engine.core import ExecutionEngine


class WorkflowRunner:

    def run_workflow(
        self,
        video_frames: List[VideoFrame],
        workflows_parameters: Optional[dict],
        execution_engine: ExecutionEngine,
        image_input_name: str,
    ) -> List[dict]:
        if workflows_parameters is None:
            workflows_parameters = {}
        # TODO: pass fps reflecting each stream to workflows_parameters
        fps = video_frames[0].fps
        workflows_parameters[image_input_name] = [
            video_frame.image for video_frame in video_frames
        ]
        return execution_engine.run(
            runtime_parameters=workflows_parameters,
            fps=fps,
        )
