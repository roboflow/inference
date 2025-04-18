import os
from typing import Union, List, Optional

import supervision as sv
from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame
from inference.core.interfaces.stream.entities import AnyPrediction

VIDEO_REFERENCE = os.environ["VIDEO_REFERENCE"]
WORKSPACE = os.environ["WORKSPACE"]
WORKFLOW_ID = os.environ["WORKFLOW_ID"]


def main() -> None:
    monitor = sv.FPSMonitor(sample_size=128)

    def sink(
        predictions: Union[List[Optional[AnyPrediction]], AnyPrediction],
        frames: Union[List[Optional[VideoFrame]], VideoFrame],
    ) -> None:
        if not isinstance(frames, list):
            frames = [frames]
            predictions = [predictions]
        for _ in predictions:
            monitor.tick()
        print(f"FPS: {monitor.fps}, CROPS: {sum(len(p) for p in predictions)} from {len(frames)} frames")

    pipeline = InferencePipeline.init_with_workflow(
        video_reference=[VIDEO_REFERENCE] * 6,
        workspace_name=WORKSPACE,
        workflow_id=WORKFLOW_ID,
        on_prediction=sink,
        batch_collection_timeout=0.02,
    )
    pipeline.start()
    pipeline.join()


if __name__ == '__main__':
    main()
