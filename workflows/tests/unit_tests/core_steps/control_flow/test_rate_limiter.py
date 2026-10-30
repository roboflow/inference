from datetime import datetime, timezone
from typing import List, Optional
from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import pytest
from pydantic import ValidationError
from roboflow_workflows.core_steps.flow_control.rate_limiter import (
    v1 as rate_limiter_v1,
)
from roboflow_workflows.core_steps.flow_control.rate_limiter.v1 import (
    RateLimiterBlockV1,
    RateLimiterManifest,
)
from roboflow_workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    VideoMetadata,
    WorkflowImageData,
)


@pytest.mark.parametrize(
    "depends_on_selector", ["$inputs.image", "$inputs.param", "$steps.some.data"]
)
def test_rate_limiter_manifest_parsing_when_input_is_valid(
    depends_on_selector: str,
) -> None:
    # given
    raw_manifest = {
        "type": "roboflow_core/rate_limiter@v1",
        "name": "rate_limiter",
        "cooldown_seconds": 10,
        "depends_on": depends_on_selector,
        "next_steps": ["$steps.some"],
    }

    # when
    result = RateLimiterManifest.model_validate(raw_manifest)

    # then
    assert result == RateLimiterManifest(
        type="roboflow_core/rate_limiter@v1",
        name="rate_limiter",
        cooldown_seconds=10,
        depends_on=depends_on_selector,
        next_steps=["$steps.some"],
    )


def test_rate_limiter_manifest_parsing_when_cooldown_is() -> None:
    # given
    raw_manifest = {
        "type": "roboflow_core/rate_limiter@v1",
        "name": "rate_limiter",
        "cooldown_seconds": -0.1,
        "depends_on": "$steps.some.data",
        "next_steps": ["$steps.some"],
    }

    # when
    with pytest.raises(ValidationError):
        _ = RateLimiterManifest.model_validate(raw_manifest)


def _image(video_metadata: Optional[VideoMetadata] = None) -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some"),
        numpy_image=np.zeros((32, 32, 3), dtype=np.uint8),
        video_metadata=video_metadata,
    )


def _video_frame(
    frame_number: int,
    *,
    fps: Optional[float] = 1,
    comes_from_video_file: Optional[bool] = True,
    frame_timestamp: Optional[datetime] = None,
) -> WorkflowImageData:
    if frame_timestamp is None:
        frame_timestamp = datetime.fromtimestamp(1726570875, tz=timezone.utc)
    video_metadata = VideoMetadata(
        video_identifier="video",
        frame_number=frame_number,
        frame_timestamp=frame_timestamp,
        fps=fps,
        comes_from_video_file=comes_from_video_file,
    )
    image = _image(video_metadata=video_metadata)

    return image


def _run_frames(
    images: List[Optional[WorkflowImageData]],
    *,
    cooldown_seconds: float,
) -> List[bool]:
    block = RateLimiterBlockV1()
    passed = []
    for image in images:
        result = block.run(
            cooldown_seconds=cooldown_seconds,
            depends_on="$steps.model",
            next_steps=["$steps.upload"],
            video_reference_image=image,
        )
        passed.append(result.context == ["$steps.upload"])

    return passed


@mock.patch.object(rate_limiter_v1, "datetime")
def test_rate_limiter_uses_wall_clock_for_image_without_video_metadata(
    datetime_mock: MagicMock,
) -> None:
    # given
    wall_clock = [
        datetime.fromtimestamp(1726570875 + offset) for offset in (0, 0.5, 1.0, 2.5)
    ]
    datetime_mock.now.side_effect = wall_clock
    images = [_image() for _ in wall_clock]

    # when
    result = _run_frames(images, cooldown_seconds=1.0)

    # then
    assert result == [
        True,
        False,
        True,
        True,
    ], "Expected still images to be throttled by wall-clock time, not by frame 0 of placeholder metadata"


@mock.patch.object(rate_limiter_v1, "datetime")
def test_rate_limiter_uses_wall_clock_when_reference_image_is_not_provided(
    datetime_mock: MagicMock,
) -> None:
    # given
    wall_clock = [
        datetime.fromtimestamp(1726570875 + offset) for offset in (0, 0.5, 1.0, 2.5)
    ]
    datetime_mock.now.side_effect = wall_clock

    # when
    result = _run_frames([None] * len(wall_clock), cooldown_seconds=1.0)

    # then
    assert result == [True, False, True, True]


@mock.patch.object(rate_limiter_v1, "datetime")
def test_rate_limiter_uses_video_time_for_video_file_frames(
    datetime_mock: MagicMock,
) -> None:
    # given
    datetime_mock.now.return_value = datetime.fromtimestamp(1726570875)
    images = [_video_frame(frame_number) for frame_number in (1, 3, 5)]

    # when
    result = _run_frames(images, cooldown_seconds=3.0)

    # then
    assert result == [
        True,
        False,
        True,
    ], "Expected frames 1 and 5 at 1 FPS to be 4 seconds of video time apart"


@mock.patch.object(rate_limiter_v1, "datetime")
def test_rate_limiter_lets_video_file_frames_through_once_cooldown_elapses(
    datetime_mock: MagicMock,
) -> None:
    # given
    datetime_mock.now.return_value = datetime.fromtimestamp(1726570875)
    images = [_video_frame(frame_number) for frame_number in (1, 3, 5)]

    # when
    result = _run_frames(images, cooldown_seconds=2.0)

    # then
    assert result == [True, True, True]


@pytest.mark.parametrize(
    "comes_from_video_file, fps",
    [(False, 30), (None, 30), (True, 0), (True, None)],
    ids=["stream", "unknown_source", "video_file_with_zero_fps", "video_file_no_fps"],
)
@mock.patch.object(rate_limiter_v1, "datetime")
def test_rate_limiter_uses_frame_timestamp_when_video_time_is_not_available(
    datetime_mock: MagicMock,
    comes_from_video_file: Optional[bool],
    fps: Optional[float],
) -> None:
    # given
    datetime_mock.now.return_value = datetime.fromtimestamp(1726570875)
    images = [
        _video_frame(
            frame_number,
            fps=fps,
            comes_from_video_file=comes_from_video_file,
            frame_timestamp=datetime.fromtimestamp(
                1726570875 + offset, tz=timezone.utc
            ),
        )
        for frame_number, offset in ((1, 0), (2, 0.5), (3, 1.0), (4, 2.5))
    ]

    # when
    result = _run_frames(images, cooldown_seconds=1.0)

    # then
    assert result == [True, False, True, True]
