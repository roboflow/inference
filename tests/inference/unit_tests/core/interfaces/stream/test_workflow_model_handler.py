from datetime import datetime

import numpy as np

from inference.core.interfaces.camera.entities import VideoFrame
from inference.core.interfaces.stream.model_handlers.workflows import WorkflowRunner


class CapturingExecutionEngine:
    def __init__(self):
        self.runtime_parameters = None

    def run(self, runtime_parameters, **kwargs):
        self.runtime_parameters = runtime_parameters
        return []


def test_workflow_runner_preserves_zero_source_id_as_video_identifier():
    engine = CapturingExecutionEngine()
    frames = [
        VideoFrame(
            image=np.zeros((1, 1, 3), dtype=np.uint8),
            frame_id=1,
            frame_timestamp=datetime.now(),
            source_id=0,
        ),
        VideoFrame(
            image=np.zeros((1, 1, 3), dtype=np.uint8),
            frame_id=1,
            frame_timestamp=datetime.now(),
            source_id=1,
        ),
    ]

    WorkflowRunner().run_workflow(
        video_frames=frames,
        workflows_parameters={},
        execution_engine=engine,
        image_input_name="image",
        video_metadata_input_name="video_metadata",
    )

    assert [
        metadata.video_identifier
        for metadata in engine.runtime_parameters["video_metadata"]
    ] == ["0", "1"]
