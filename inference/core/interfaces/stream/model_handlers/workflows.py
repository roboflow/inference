from typing import List, Optional

from inference.core.interfaces.camera.entities import VideoFrame
from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference.core.workflows.execution_engine.entities.base import VideoMetadata


class WorkflowRunner:

    def run_workflow(
        self,
        video_frames: List[VideoFrame],
        workflows_parameters: Optional[dict],
        execution_engine: ExecutionEngine,
        image_input_name: str,
        video_metadata_input_name: str,
    ) -> List[dict]:
        if workflows_parameters is None:
            workflows_parameters = {}
        # TODO: pass fps reflecting each stream to workflows_parameters
        fps = video_frames[0].fps
        if fps is None:
            # for FPS reporting we expect 0 when FPS cannot be determined
            fps = 0
        workflows_parameters[image_input_name] = [
            video_frame.image for video_frame in video_frames
        ]
        workflows_parameters[video_metadata_input_name] = [
            VideoMetadata(
                video_identifier=(
                    str(video_frame.source_id)
                    if video_frame.source_id
                    else "default_source"
                ),
                frame_number=video_frame.frame_id,
                frame_timestamp=video_frame.frame_timestamp,
                fps=video_frame.fps,
                comes_from_video_file=video_frame.comes_from_video_file,
            )
            for video_frame in video_frames
        ]
        return execution_engine.run(
            runtime_parameters=workflows_parameters,
            fps=fps,
        )
