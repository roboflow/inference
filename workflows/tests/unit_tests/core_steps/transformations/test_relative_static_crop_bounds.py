from datetime import datetime, timezone

import numpy as np
import pytest
import torch
from roboflow_workflows.core_steps.transformations.relative_static_crop.v1 import (
    BlockManifest,
    take_static_crop,
)
from roboflow_workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    VideoMetadata,
    WorkflowImageData,
)


@pytest.mark.parametrize("backing", ["numpy", "tensor"])
@pytest.mark.parametrize(
    "shape,parameters,expected_box",
    [
        ((100, 100), (0.5, 0.5, 0.4, 0.4), (30, 30, 70, 70)),
        ((100, 100), (0.1, 0.5, 0.4, 0.4), (0, 30, 30, 70)),
        ((100, 100), (0.5, 0.1, 0.4, 0.4), (30, 0, 70, 30)),
        ((100, 100), (0.1, 0.1, 0.4, 0.4), (0, 0, 30, 30)),
        ((100, 100), (0.0, 0.5, 0.4, 0.4), (0, 30, 20, 70)),
        ((100, 100), (0.5, 0.0, 0.4, 0.4), (30, 0, 70, 20)),
        ((100, 100), (0.9, 0.5, 0.4, 0.4), (70, 30, 100, 70)),
        ((100, 100), (0.5, 0.9, 0.4, 0.4), (30, 70, 70, 100)),
        ((100, 100), (0.9, 0.9, 0.4, 0.4), (70, 70, 100, 100)),
        ((100, 100), (0.5, 0.5, 1.0, 1.0), (0, 0, 100, 100)),
        ((100, 100), (0.5, 0.5, 0.0, 0.4), None),
        ((100, 100), (0.5, 0.5, 0.4, 0.0), None),
        ((99, 101), (0.5, 0.5, 0.2, 0.2), (40, 40, 60, 60)),
        ((100, 100), (0.5, 0.5, 0.001, 0.4), None),
    ],
)
def test_relative_crop_intersects_canvas_without_changing_rounding(
    shape: tuple,
    parameters: tuple,
    expected_box: tuple | None,
    backing: str,
) -> None:
    pixels = np.arange(shape[0] * shape[1] * 3, dtype=np.uint32).astype(np.uint8)
    pixels = pixels.reshape(*shape, 3)
    if backing == "numpy":
        image = WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="image"),
            numpy_image=pixels.copy(),
        )
    else:
        image = WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="image"),
            tensor_image=torch.from_numpy(pixels[:, :, ::-1].copy()).permute(2, 0, 1),
        )
    x_center, y_center, width, height = parameters
    # All cases are valid public block inputs, including zero and edge centers.
    BlockManifest(
        type="RelativeStaticCrop",
        name="crop",
        images="$inputs.image",
        x_center=x_center,
        y_center=y_center,
        width=width,
        height=height,
    )

    result = take_static_crop(
        image=image,
        x_center=x_center,
        y_center=y_center,
        width=width,
        height=height,
    )

    np.testing.assert_array_equal(image.numpy_image, pixels)
    if expected_box is None:
        assert result is None
        return

    assert result is not None
    x1, y1, x2, y2 = expected_box
    np.testing.assert_array_equal(result.numpy_image, pixels[y1:y2, x1:x2])
    assert result.parent_metadata.origin_coordinates.left_top_x == x1
    assert result.parent_metadata.origin_coordinates.left_top_y == y1
    assert result.parent_metadata.origin_coordinates.origin_width == shape[1]
    assert result.parent_metadata.origin_coordinates.origin_height == shape[0]
    assert result.workflow_root_ancestor_metadata.parent_id == "image"
    assert result.workflow_root_ancestor_metadata.origin_coordinates.left_top_x == x1
    assert result.workflow_root_ancestor_metadata.origin_coordinates.left_top_y == y1


@pytest.mark.parametrize(
    "parameters,equivalent,box",
    [
        ((0.1, 0.1, 0.4, 0.4), (0.0, 0.1, 0.6, 0.4), (0, 0, 30, 30)),
        ((0.9, 0.9, 0.4, 0.4), (1.0, 0.9, 0.6, 0.4), (70, 70, 100, 100)),
        ((0.5, 0.1, 0.4, 0.4), (0.5, 0.0, 0.4, 0.6), (30, 0, 70, 30)),
        ((0.5, 0.9, 0.4, 0.4), (0.5, 1.0, 0.4, 0.6), (30, 70, 70, 100)),
    ],
)
def test_relative_crop_preserves_nested_origin_and_video_identity(
    parameters: tuple, equivalent: tuple, box: tuple
) -> None:
    pixels = (
        np.arange(200 * 200 * 3, dtype=np.uint32).astype(np.uint8).reshape(200, 200, 3)
    )
    frame = VideoMetadata(
        video_identifier="camera",
        frame_number=7,
        frame_timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        fps=30.0,
        measured_fps=29.5,
        comes_from_video_file=True,
        total_frames=100,
    )
    root = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="root"),
        numpy_image=pixels.copy(),
        video_metadata=frame,
    )
    image = WorkflowImageData.create_crop(
        origin_image_data=root,
        crop_identifier="outer",
        cropped_image=root.numpy_image[40:140, 50:150],
        offset_x=50,
        offset_y=40,
        preserve_video_metadata=True,
        video_crop_identifier="outer-region",
    )
    original_frame = image.video_metadata.model_dump()

    first = take_static_crop(image, *parameters)
    assert first is not None
    # A different requested box with the same visible intersection identifies
    # the same video region. Frame metadata changes must not reset its identity.
    image.video_metadata.frame_number = 8
    second = take_static_crop(image, *equivalent)
    different = take_static_crop(
        image=image, x_center=0.5, y_center=0.5, width=0.2, height=0.2
    )

    assert second is not None and different is not None
    x1, y1, x2, y2 = box
    np.testing.assert_array_equal(
        first.numpy_image, pixels[40 + y1 : 40 + y2, 50 + x1 : 50 + x2]
    )
    np.testing.assert_array_equal(second.numpy_image, first.numpy_image)
    assert first.parent_metadata.origin_coordinates.left_top_x == x1
    assert first.parent_metadata.origin_coordinates.left_top_y == y1
    assert first.workflow_root_ancestor_metadata.parent_id == "root"
    assert (
        first.workflow_root_ancestor_metadata.origin_coordinates.left_top_x == 50 + x1
    )
    assert (
        first.workflow_root_ancestor_metadata.origin_coordinates.left_top_y == 40 + y1
    )
    assert first.workflow_root_ancestor_metadata.origin_coordinates.origin_width == 200
    assert first.workflow_root_ancestor_metadata.origin_coordinates.origin_height == 200
    assert first.parent_metadata.parent_id != second.parent_metadata.parent_id
    assert (
        first.video_metadata.video_identifier == second.video_metadata.video_identifier
    )
    assert (
        different.video_metadata.video_identifier
        != first.video_metadata.video_identifier
    )
    assert first.video_metadata.frame_number == 7
    assert second.video_metadata.frame_number == 8
    assert first.video_metadata.model_dump(exclude={"video_identifier"}) == {
        key: value for key, value in original_frame.items() if key != "video_identifier"
    }
    assert image.video_metadata.video_identifier == original_frame["video_identifier"]
    np.testing.assert_array_equal(root.numpy_image, pixels)
