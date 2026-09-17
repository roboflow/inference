from datetime import datetime

import numpy as np
import pytest

from inference.core.workflows.core_steps.transformations.absolute_static_crop.v1 import (
    take_static_crop,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    OriginCoordinatesSystem,
    VideoMetadata,
    WorkflowImageData,
)


def test_take_absolute_static_crop() -> None:
    # given
    np_image = np.zeros((100, 100, 3), dtype=np.uint8)
    np_image[50:70, 45:55] = 30  # painted the crop into (30, 30, 30)
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="origin_image"),
        numpy_image=np_image,
        video_metadata=VideoMetadata(
            video_identifier="some",
            frame_number=0,
            frame_timestamp=datetime.now(),
            fps=100,
        ),
    )

    # when
    result = take_static_crop(
        image=image,
        x_center=50,
        y_center=60,
        width=10,
        height=20,
    )

    # then
    assert (
        result.numpy_image == (np.ones((20, 10, 3), dtype=np.uint8) * 30)
    ).all(), "Crop must have the exact size and color"
    assert result.parent_metadata.parent_id.startswith(
        "absolute_static_crop."
    ), "Parent must be set at crop step identifier"
    assert result.parent_metadata.origin_coordinates == OriginCoordinatesSystem(
        left_top_x=45,
        left_top_y=50,
        origin_width=100,
        origin_height=100,
    ), "Origin coordinates of crop and image size metadata must be maintained through the operation"
    assert (
        result.workflow_root_ancestor_metadata.origin_coordinates
        == OriginCoordinatesSystem(
            left_top_x=45,
            left_top_y=50,
            origin_width=100,
            origin_height=100,
        )
    ), "Root Origin coordinates of crop and image size metadata must be maintained through the operation"
    assert (
        result.video_metadata.video_identifier.startswith("some")
        and result.video_metadata.video_identifier != "some"
    ), "Expected to generate new video identifier"
    assert result.video_metadata.fps == 100, "Expected to preserve video metadata"


def test_take_absolute_static_crop_when_output_crop_is_empty() -> None:
    # given
    np_image = np.zeros((100, 100, 3), dtype=np.uint8)
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="origin_image"),
        numpy_image=np_image,
    )

    # when
    result = take_static_crop(
        image=image,
        x_center=50,
        y_center=60,
        width=0,
        height=0,
    )

    # then
    assert result is None, "Expected no crop as result"


def test_take_absolute_static_crop_clamps_out_of_bounds_crop() -> None:
    # given
    np_image = np.zeros((100, 100, 3), dtype=np.uint8)
    np_image[0:20, 0:10] = 30
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="origin_image"),
        numpy_image=np_image,
    )

    # when - crop centred at the image corner extends past the top-left edge
    result = take_static_crop(
        image=image,
        x_center=0,
        y_center=0,
        width=20,
        height=40,
    )

    # then - clamped to the in-bounds region instead of wrapping via negative indices
    assert result is not None, "Expected clamped crop as result"
    assert result.numpy_image.shape == (20, 10, 3)
    assert (
        result.numpy_image == (np.ones((20, 10, 3), dtype=np.uint8) * 30)
    ).all(), "Expected the top-left in-bounds region to be returned"
    offset = result.parent_metadata.origin_coordinates
    assert (offset.left_top_x, offset.left_top_y) == (0, 0)


def test_take_absolute_static_crop_tensor_sibling_clamps_out_of_bounds_crop() -> None:
    # given
    pytest.importorskip("torch")
    pytest.importorskip("inference_models")
    from inference.core.workflows.core_steps.transformations.absolute_static_crop.v1_tensor import (
        take_static_crop as take_static_crop_tensor,
    )

    np_image = np.zeros((100, 100, 3), dtype=np.uint8)
    np_image[0:20, 0:10] = 30
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="origin_image"),
        numpy_image=np_image,
    )

    # when
    result = take_static_crop_tensor(
        image=image,
        x_center=0,
        y_center=0,
        width=20,
        height=40,
    )

    # then
    assert result is not None, "Expected clamped crop as result"
    assert result.numpy_image.shape == (20, 10, 3)
    assert (
        result.numpy_image == (np.ones((20, 10, 3), dtype=np.uint8) * 30)
    ).all(), "Expected the top-left in-bounds region to be returned"
    offset = result.parent_metadata.origin_coordinates
    assert (offset.left_top_x, offset.left_top_y) == (0, 0)


def test_take_absolute_static_crop_tensor_sibling_clamps_on_tensor_path() -> None:
    # given
    torch = pytest.importorskip("torch")
    pytest.importorskip("inference_models")
    from inference.core.workflows.core_steps.transformations.absolute_static_crop.v1_tensor import (
        take_static_crop as take_static_crop_tensor,
    )

    tensor_image = torch.zeros((3, 100, 100), dtype=torch.uint8)
    tensor_image[:, 0:20, 0:10] = 30
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="origin_image"),
        tensor_image=tensor_image,
    )

    # when
    result = take_static_crop_tensor(
        image=image,
        x_center=0,
        y_center=0,
        width=20,
        height=40,
    )

    # then
    assert result is not None, "Expected clamped crop as result"
    assert tuple(result.tensor_image.shape) == (3, 20, 10)
    assert bool((result.tensor_image == 30).all())


def test_take_absolute_static_crop_tensor_sibling_returns_none_when_crop_fully_outside() -> (
    None
):
    # given
    pytest.importorskip("torch")
    pytest.importorskip("inference_models")
    from inference.core.workflows.core_steps.transformations.absolute_static_crop.v1_tensor import (
        take_static_crop as take_static_crop_tensor,
    )

    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="origin_image"),
        numpy_image=np.zeros((100, 100, 3), dtype=np.uint8),
    )

    # when - the whole crop lies left of / above the image
    result = take_static_crop_tensor(
        image=image,
        x_center=-50,
        y_center=-50,
        width=20,
        height=20,
    )

    # then
    assert result is None, "Expected no crop as result"
