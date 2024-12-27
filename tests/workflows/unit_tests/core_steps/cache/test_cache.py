import datetime

import numpy as np

from inference.core.workflows.core_steps.cache.cache_get.v1 import CacheGetBlockV1
from inference.core.workflows.core_steps.cache.cache_set.v1 import CacheSetBlockV1
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    VideoMetadata,
    WorkflowImageData,
)


def test_cache_on_video() -> None:
    # given
    metadata = VideoMetadata(
        video_identifier="vid",
        frame_number=10,
        frame_timestamp=datetime.datetime.fromtimestamp(1726570875).astimezone(
            tz=datetime.timezone.utc
        ),
    )
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some"),
        numpy_image=np.zeros((192, 168, 3), dtype=np.uint8),
        video_metadata=metadata,
    )
    cache_get_block = CacheGetBlockV1(step_execution_mode=StepExecutionMode.LOCAL)
    cache_set_block = CacheSetBlockV1(step_execution_mode=StepExecutionMode.LOCAL)

    # empty result
    get_empty = cache_get_block.run(image=image, key="foo")
    assert get_empty == {
        "output": False,
    }

    # set then get
    cache_set_block.run(image=image, key="foo", value="bar")
    get_full = cache_get_block.run(image=image, key="foo")
    assert get_full == {
        "output": "bar",
    }


def test_cache_with_no_metadata() -> None:
    # given
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some"),
        numpy_image=np.zeros((192, 168, 3), dtype=np.uint8),
    )
    cache_get_block = CacheGetBlockV1(step_execution_mode=StepExecutionMode.LOCAL)
    cache_set_block = CacheSetBlockV1(step_execution_mode=StepExecutionMode.LOCAL)

    # empty result
    get_empty = cache_get_block.run(image=image, key="foo")
    assert get_empty == {
        "output": False,
    }

    # set then get
    cache_set_block.run(image=image, key="foo", value="bar")
    get_full = cache_get_block.run(image=image, key="foo")
    assert get_full == {
        "output": "bar",
    }


def test_cache_on_multiple_videos() -> None:
    # given
    metadata_1 = VideoMetadata(
        video_identifier="vid_1",
        frame_number=10,
        frame_timestamp=datetime.datetime.fromtimestamp(1726570875).astimezone(
            tz=datetime.timezone.utc
        ),
    )
    image_1 = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some"),
        numpy_image=np.zeros((192, 168, 3), dtype=np.uint8),
        video_metadata=metadata_1,
    )

    metadata_2 = VideoMetadata(
        video_identifier="vid_2",
        frame_number=10,
        frame_timestamp=datetime.datetime.fromtimestamp(1726570875).astimezone(
            tz=datetime.timezone.utc
        ),
    )
    image_2 = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some"),
        numpy_image=np.zeros((192, 168, 3), dtype=np.uint8),
        video_metadata=metadata_2,
    )

    cache_get_block = CacheGetBlockV1(step_execution_mode=StepExecutionMode.LOCAL)
    cache_set_block = CacheSetBlockV1(step_execution_mode=StepExecutionMode.LOCAL)

    # empty result
    get_empty = cache_get_block.run(image=image_1, key="foo")
    assert get_empty == {
        "output": False,
    }

    # set then get
    cache_set_block.run(image=image_1, key="foo", value="bar")
    get_full = cache_get_block.run(image=image_1, key="foo")
    assert get_full == {
        "output": "bar",
    }

    # make sure it doesn't bleed over
    get_empty = cache_get_block.run(image=image_2, key="foo")
    assert get_empty == {
        "output": False,
    }
