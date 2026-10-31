import datetime

import numpy as np
import supervision as sv

from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    VideoMetadata,
    WorkflowImageData,
)


def manifest_accepted_kind_names(manifest_cls, field_name="predictions"):
    """Extract accepted kind names from a manifest's Selector field via JSON schema."""
    schema = manifest_cls.model_json_schema()
    kind_entries = schema["properties"][field_name].get("kind", [])
    return {entry["name"] for entry in kind_entries}


FRAME1_XYXY = np.array(
    [[10, 10, 20, 20], [21, 10, 31, 20], [31, 10, 41, 20], [100, 100, 110, 110]]
)
FRAME2_XYXY = np.array(
    [[12, 10, 22, 20], [23, 10, 33, 20], [33, 10, 43, 20], [110, 100, 120, 110]]
)
FRAME3_XYXY = np.array([[14, 10, 24, 20], [25, 10, 35, 20], [35, 10, 45, 20]])


def make_detections(xyxy: np.ndarray, confidence: float = 0.9) -> sv.Detections:
    return sv.Detections(
        xyxy=xyxy,
        confidence=np.full(len(xyxy), confidence),
        class_id=np.ones(len(xyxy), dtype=int),
    )


def make_metadata(
    frame_number: int,
    fps: float = 1,
    comes_from_video_file: bool = True,
    timestamp_offset: int = 0,
) -> VideoMetadata:
    return VideoMetadata(
        video_identifier="vid_1",
        frame_number=frame_number,
        fps=fps,
        frame_timestamp=datetime.datetime.fromtimestamp(
            1726570875 + timestamp_offset
        ).astimezone(tz=datetime.timezone.utc),
        comes_from_video_file=comes_from_video_file,
    )


def wrap_with_workflow_image(metadata: VideoMetadata) -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some"),
        numpy_image=np.zeros((192, 168, 3), dtype=np.uint8),
        video_metadata=metadata,
    )
