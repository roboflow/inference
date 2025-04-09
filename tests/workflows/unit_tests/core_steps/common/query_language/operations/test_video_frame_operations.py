from datetime import datetime
from typing import Optional

import numpy as np
import pytest

from inference.core.workflows.core_steps.common.query_language.errors import (
    InvalidInputTypeError,
)
from inference.core.workflows.core_steps.common.query_language.operations.core import (
    execute_operations,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    VideoMetadata,
    WorkflowImageData,
)


def test_extract_video_frame_when_invalid_input_provided() -> None:
    # given
    operations = [
        {
            "type": "ExtractFrameMetadata",
            "property_name": "frame_timestamp",
        }
    ]

    # when
    with pytest.raises(InvalidInputTypeError):
        _ = execute_operations(
            value="invalid",
            operations=operations,
        )


def test_extract_video_frame_timestamp() -> None:
    # given
    operations = [
        {
            "type": "ExtractFrameMetadata",
            "property_name": "frame_timestamp",
        }
    ]
    timestamp = datetime.now()
    image = _create_video_frame(timestamp=timestamp)

    # when
    result = execute_operations(
        value=image,
        operations=operations,
    )

    # then
    assert result is timestamp


def test_extract_and_serialize_video_frame_timestamp() -> None:
    # given
    operations = [
        {
            "type": "ExtractFrameMetadata",
            "property_name": "frame_timestamp",
        },
        {
            "type": "TimestampToISOFormat",
        },
    ]
    timestamp = datetime.now()
    image = _create_video_frame(timestamp=timestamp)

    # when
    result = execute_operations(
        value=image,
        operations=operations,
    )

    # then
    assert result == timestamp.isoformat()


def test_extract_video_frame_number() -> None:
    # given
    operations = [
        {
            "type": "ExtractFrameMetadata",
            "property_name": "frame_number",
        }
    ]
    image = _create_video_frame(frame_number=39)

    # when
    result = execute_operations(
        value=image,
        operations=operations,
    )

    # then
    assert result == 39


def test_extract_video_frame_seconds_from_start() -> None:
    # given
    operations = [
        {
            "type": "ExtractFrameMetadata",
            "property_name": "seconds_since_start",
        }
    ]
    image = _create_video_frame(frame_number=61, fps=30.0)

    # when
    result = execute_operations(
        value=image,
        operations=operations,
    )

    # then
    assert abs(result - 2.0) < 1e-5


def _create_video_frame(
    timestamp: Optional[datetime] = None,
    fps: float = 30.0,
    frame_number: int = 60,
) -> WorkflowImageData:
    if timestamp is None:
        timestamp = datetime.now()
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some_image"),
        numpy_image=np.zeros((192, 168, 3), dtype=np.uint8),
        video_metadata=VideoMetadata(
            video_identifier="rtsp://some/stream",
            frame_number=frame_number,
            frame_timestamp=timestamp,
            fps=fps,
        ),
    )
