import base64
import os
from datetime import datetime
from unittest import mock
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from inference.core.workflows.execution_engine.entities import base
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    ImageParentMetadata,
    OriginCoordinatesSystem,
    ParentOrigin,
    VideoMetadata,
    WorkflowImageData,
)


def test_initialising_batch_with_misaligned_indices() -> None:
    # when
    with pytest.raises(ValueError):
        _ = Batch.init(
            content=[1, "2", None, 3.0],
            indices=[(0,), (3,)],
        )


def test_standard_iteration_through_batch() -> None:
    # given
    batch = Batch.init(
        content=[1, "2", None, 3.0],
        indices=[(0,), (1,), (2,), (3,)],
    )

    # when
    result = list(batch)

    # then
    assert result == [1, "2", None, 3.0]


def test_standard_iteration_through_batch_with_indices() -> None:
    # given
    batch = Batch.init(
        content=[1, "2", None, 3.0],
        indices=[(0,), (1,), (2,), (3,)],
    )

    # when
    result = list(batch.iter_with_indices())

    # then
    assert result == [((0,), 1), ((1,), "2"), ((2,), None), ((3,), 3.0)]


def test_getting_batch_length() -> None:
    # given
    batch = Batch.init(
        content=[1, "2", None, 3.0],
        indices=[(0,), (1,), (2,), (3,)],
    )

    # when
    result = len(batch)

    # then
    assert result == 4


def test_getting_batch_element_when_valid_element_is_chosen() -> None:
    # given
    batch = Batch.init(
        content=[1, "2", None, 3.0],
        indices=[(0,), (1,), (2,), (3,)],
    )

    # when
    result = batch[1]

    # then
    assert result == "2"


def test_getting_batch_element_when_valid_invalid_element_is_chosen() -> None:
    # given
    batch = Batch.init(
        content=[1, "2", None, 3.0],
        indices=[(0,), (1,), (2,), (3,)],
    )

    # when
    with pytest.raises(IndexError):
        _ = batch[5]


def test_filtering_out_batch_elements() -> None:
    # given
    batch = Batch.init(
        content=[1, "2", None, 3.0],
        indices=[(0,), (1,), (2,), (3,)],
    )

    # when
    result = batch.remove_by_indices(indices_to_remove={(1,), (2,), (5,)})

    # then
    assert result.indices == [
        (0,),
        (3,),
    ], "Expected to see only first and last original index"
    assert list(result) == [
        1,
        3.0,
    ], "Expected to see only first and last elements of original content"


def test_remove_by_indices_with_nested_batch_removes_indices_inside_nested() -> None:
    # given: top-level batch with one nested batch (indices (0,0), (0,1)) and one scalar
    nested = Batch.init(
        content=["a", "b"],
        indices=[(0, 0), (0, 1)],
    )
    batch = Batch.init(
        content=[nested, "scalar"],
        indices=[(0,), (1,)],
    )

    # when: remove index that exists only inside the nested batch
    result = batch.remove_by_indices(indices_to_remove={(0, 1)})

    # then: top-level unchanged; nested batch has (0,1) removed
    assert result.indices == [(0,), (1,)]
    assert len(result) == 2
    first_element = result[0]
    assert isinstance(first_element, Batch)
    assert first_element.indices == [(0, 0)]
    assert list(first_element) == ["a"]
    assert result[1] == "scalar"


def test_remove_by_indices_with_nested_batch_removes_top_level_index() -> None:
    # given: top-level batch with one nested batch and one scalar
    nested = Batch.init(
        content=["a", "b"],
        indices=[(0, 0), (0, 1)],
    )
    batch = Batch.init(
        content=[nested, "scalar"],
        indices=[(0,), (1,)],
    )

    # when: remove top-level index (0,) - the whole nested batch is dropped
    result = batch.remove_by_indices(indices_to_remove={(0,)})

    # then: only the scalar remains
    assert result.indices == [(1,)]
    assert list(result) == ["scalar"]


def test_remove_by_indices_with_nested_batch_removes_mixed_indices() -> None:
    # given: two nested batches at top level
    nested_a = Batch.init(
        content=["a1", "a2", "a3"],
        indices=[(0, 0), (0, 1), (0, 2)],
    )
    nested_b = Batch.init(
        content=["b1", "b2"],
        indices=[(1, 0), (1, 1)],
    )
    batch = Batch.init(
        content=[nested_a, nested_b],
        indices=[(0,), (1,)],
    )

    # when: remove one index from first nested and one from second nested
    result = batch.remove_by_indices(indices_to_remove={(0, 1), (1, 0)})

    # then: both nested batches kept but filtered
    assert result.indices == [(0,), (1,)]
    first = result[0]
    second = result[1]
    assert isinstance(first, Batch)
    assert isinstance(second, Batch)
    assert first.indices == [(0, 0), (0, 2)]
    assert list(first) == ["a1", "a3"]
    assert second.indices == [(1, 1)]
    assert list(second) == ["b2"]


def test_broadcast_batch_when_requested_size_is_equal_to_batch_size() -> None:
    # given
    batch = Batch.init(
        content=[1, "2", None, 3.0],
        indices=[(0,), (1,), (2,), (3,)],
    )

    # when
    result = batch.broadcast(n=4)

    # then
    assert list(result) == [1, "2", None, 3.0]


def test_broadcast_batch_when_requested_size_is_valid_and_batch_size_is_one() -> None:
    # given
    batch = Batch.init(content=[1], indices=[(0,)])

    # when
    result = batch.broadcast(n=4)

    # then
    assert list(result) == [1, 1, 1, 1]


def test_broadcast_batch_when_requested_size_is_valid_and_batch_size_is_not_matching() -> (
    None
):
    # given
    batch = Batch.init(content=[1, 2], indices=[(0,), (1,)])

    # when
    with pytest.raises(ValueError):
        _ = batch.broadcast(n=4)


def test_broadcast_batch_when_requested_size_is_invalid() -> None:
    # given
    batch = Batch.init(content=[1, 2], indices=[(0,), (1,)])

    # when
    with pytest.raises(ValueError):
        _ = batch.broadcast(n=0)


def test_init_workflow_image_data_when_image_representation_not_provided() -> None:
    # when
    with pytest.raises(ValueError):
        # we requre some form of image to be provided, not only metadata
        _ = WorkflowImageData(parent_metadata=ImageParentMetadata(parent_id="parent"))


def test_getting_parent_metadata_when_np_representation_is_provided() -> None:
    # given
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        numpy_image=np.zeros((192, 168, 3), dtype=np.uint8),
    )

    # when
    result = image.parent_metadata

    # then
    assert result == ImageParentMetadata(
        parent_id="parent",
        origin_coordinates=OriginCoordinatesSystem(
            left_top_x=0,
            left_top_y=0,
            origin_width=168,
            origin_height=192,
        ),
    ), "Expected origin coordinates to be provided with default coordinates system"


def test_getting_parent_metadata_when_base64_representation_is_provided() -> None:
    # given
    base64_image = base64.b64encode(
        cv2.imencode(".jpg", np.zeros((192, 168, 3), dtype=np.uint8))[1]
    ).decode("ascii")
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        base64_image=base64_image,
    )

    # when
    result = image.parent_metadata

    # then
    assert result == ImageParentMetadata(
        parent_id="parent",
        origin_coordinates=OriginCoordinatesSystem(
            left_top_x=0,
            left_top_y=0,
            origin_width=168,
            origin_height=192,
        ),
    ), "Expected origin coordinates to be provided with default coordinates system"


def test_getting_workflow_root_ancestor_metadata_when_np_representation_is_provided_and_no_explicit_root_pointed() -> (
    None
):
    # given
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        numpy_image=np.zeros((192, 168, 3), dtype=np.uint8),
    )

    # when
    result = image.workflow_root_ancestor_metadata

    # then
    assert result == ImageParentMetadata(
        parent_id="parent",
        origin_coordinates=OriginCoordinatesSystem(
            left_top_x=0,
            left_top_y=0,
            origin_width=168,
            origin_height=192,
        ),
    ), "Expected origin coordinates to be provided with default coordinates system"


def test_getting_workflow_root_ancestor_metadata_when_base64_representation_is_provided_and_no_explicit_root_pointed() -> (
    None
):
    # given
    base64_image = base64.b64encode(
        cv2.imencode(".jpg", np.zeros((192, 168, 3), dtype=np.uint8))[1]
    ).decode("ascii")
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        base64_image=base64_image,
    )

    # when
    result = image.workflow_root_ancestor_metadata

    # then
    assert result == ImageParentMetadata(
        parent_id="parent",
        origin_coordinates=OriginCoordinatesSystem(
            left_top_x=0,
            left_top_y=0,
            origin_width=168,
            origin_height=192,
        ),
    ), "Expected origin coordinates to be provided with default coordinates system"


def test_getting_workflow_root_ancestor_metadata_when_np_representation_is_provided_and_explicit_root_pointed() -> (
    None
):
    # given
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        workflow_root_ancestor_metadata=ImageParentMetadata(parent_id="root"),
        numpy_image=np.zeros((192, 168, 3), dtype=np.uint8),
    )

    # when
    result = image.workflow_root_ancestor_metadata

    # then
    assert result == ImageParentMetadata(
        parent_id="root",
        origin_coordinates=OriginCoordinatesSystem(
            left_top_x=0,
            left_top_y=0,
            origin_width=168,
            origin_height=192,
        ),
    ), "Expected origin coordinates to be provided with default coordinates system"


def test_getting_workflow_root_ancestor_metadata_when_base64_representation_is_provided_and_explicit_root_pointed() -> (
    None
):
    # given
    base64_image = base64.b64encode(
        cv2.imencode(".jpg", np.zeros((192, 168, 3), dtype=np.uint8))[1]
    ).decode("ascii")
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        workflow_root_ancestor_metadata=ImageParentMetadata(parent_id="root"),
        base64_image=base64_image,
    )

    # when
    result = image.workflow_root_ancestor_metadata

    # then
    assert result == ImageParentMetadata(
        parent_id="root",
        origin_coordinates=OriginCoordinatesSystem(
            left_top_x=0,
            left_top_y=0,
            origin_width=168,
            origin_height=192,
        ),
    ), "Expected origin coordinates to be provided with default coordinates system"


def test_getting_workflow_root_ancestor_metadata_when_coordinates_provided_explicitly() -> (
    None
):
    # given
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        workflow_root_ancestor_metadata=ImageParentMetadata(
            parent_id="root",
            origin_coordinates=OriginCoordinatesSystem(
                left_top_x=100,
                left_top_y=200,
                origin_width=1000,
                origin_height=2000,
            ),
        ),
        numpy_image=np.zeros((192, 168, 3), dtype=np.uint8),
    )

    # when
    result = image.workflow_root_ancestor_metadata

    # then
    assert result == ImageParentMetadata(
        parent_id="root",
        origin_coordinates=OriginCoordinatesSystem(
            left_top_x=100,
            left_top_y=200,
            origin_width=1000,
            origin_height=2000,
        ),
    ), "Expected origin coordinates to be provided with explicit coordinates system"


def test_getting_np_image_when_image_provided_in_np_representation() -> None:
    # given
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        numpy_image=np.zeros((192, 168, 3), dtype=np.uint8),
    )

    # when
    result = image.numpy_image

    # then
    assert np.allclose(result, np.zeros((192, 168, 3), dtype=np.uint8))


def test_getting_np_image_when_image_provided_in_base64_representation() -> None:
    # given
    base64_image = base64.b64encode(
        cv2.imencode(".jpg", np.zeros((192, 168, 3), dtype=np.uint8))[1]
    ).decode("ascii")
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        base64_image=base64_image,
    )

    # when
    result = image.numpy_image

    # then
    assert np.allclose(result, np.zeros((192, 168, 3), dtype=np.uint8))


def test_getting_np_image_when_image_provided_as_file(empty_directory: str) -> None:
    # given
    np_image = np.zeros((192, 168, 3), dtype=np.uint8)
    image_path = os.path.join(empty_directory, "file.jpg")
    cv2.imwrite(image_path, np_image)
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        image_reference=image_path,
    )

    # when
    result = image.numpy_image

    # then
    assert np.allclose(result, np.zeros((192, 168, 3), dtype=np.uint8))


@mock.patch.object(base, "load_image_from_url")
def test_getting_np_image_when_image_provided_as_url(
    load_image_from_url_mock: MagicMock,
) -> None:
    # given
    np_image = np.zeros((192, 168, 3), dtype=np.uint8)
    load_image_from_url_mock.return_value = np_image
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        image_reference="http://some.com/image.jpg",
    )

    # when
    result = image.numpy_image

    # then
    assert np.allclose(result, np.zeros((192, 168, 3), dtype=np.uint8))


def test_getting_base64_image_when_image_provided_in_np_representation() -> None:
    # given
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        numpy_image=np.zeros((192, 168, 3), dtype=np.uint8),
    )

    # when
    result = image.base64_image

    # then
    try:
        result_image = cv2.imdecode(
            np.frombuffer(base64.b64decode(result), np.uint8), cv2.IMREAD_ANYCOLOR
        )
    except Exception:
        result_image = cv2.imdecode(
            np.fromstring(base64.b64decode(result), np.uint8), cv2.IMREAD_ANYCOLOR
        )
    assert np.allclose(result_image, np.zeros((192, 168, 3), dtype=np.uint8))


def test_getting_base64_image_when_image_provided_in_base64_representation() -> None:
    # given
    base64_image = base64.b64encode(
        cv2.imencode(".jpg", np.zeros((192, 168, 3), dtype=np.uint8))[1]
    ).decode("ascii")
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        base64_image=base64_image,
    )

    # when
    result = image.base64_image

    # then
    assert result == base64_image


def test_getting_base64_image_when_image_provided_as_file(empty_directory: str) -> None:
    # given
    np_image = np.zeros((192, 168, 3), dtype=np.uint8)
    image_path = os.path.join(empty_directory, "file.jpg")
    cv2.imwrite(image_path, np_image)
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        image_reference=image_path,
    )

    # when
    result = image.base64_image

    # then
    try:
        result_image = cv2.imdecode(
            np.frombuffer(base64.b64decode(result), np.uint8), cv2.IMREAD_ANYCOLOR
        )
    except Exception:
        result_image = cv2.imdecode(
            np.fromstring(base64.b64decode(result), np.uint8), cv2.IMREAD_ANYCOLOR
        )
    assert np.allclose(result_image, np.zeros((192, 168, 3), dtype=np.uint8))


@mock.patch.object(base, "load_image_from_url")
def test_getting_base64_image_when_image_provided_as_url(
    load_image_from_url_mock: MagicMock,
) -> None:
    # given
    np_image = np.zeros((192, 168, 3), dtype=np.uint8)
    load_image_from_url_mock.return_value = np_image
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        image_reference="http://some.com/image.jpg",
    )

    # when
    result = image.base64_image

    # then
    try:
        result_image = cv2.imdecode(
            np.frombuffer(base64.b64decode(result), np.uint8), cv2.IMREAD_ANYCOLOR
        )
    except Exception:
        result_image = cv2.imdecode(
            np.fromstring(base64.b64decode(result), np.uint8), cv2.IMREAD_ANYCOLOR
        )
    assert np.allclose(result_image, np.zeros((192, 168, 3), dtype=np.uint8))


def test_workflow_image_data_to_inference_format_when_numpy_preferred() -> None:
    # given
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        numpy_image=np.zeros((192, 168, 3), dtype=np.uint8),
    )

    # when
    result = image.to_inference_format(numpy_preferred=True)

    # then
    assert result["type"] == "numpy_object"
    assert np.allclose(result["value"], np.zeros((192, 168, 3), dtype=np.uint8))


def test_workflow_image_data_to_inference_format_when_numpy_not_preferred_but_available_on_the_spot() -> (
    None
):
    # given
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        numpy_image=np.zeros((192, 168, 3), dtype=np.uint8),
    )

    # when
    result = image.to_inference_format(numpy_preferred=True)

    # then
    assert result["type"] == "numpy_object"
    assert np.allclose(result["value"], np.zeros((192, 168, 3), dtype=np.uint8))


def test_workflow_image_data_to_inference_format_when_numpy_not_preferred_but_url_available_on_the_spot() -> (
    None
):
    # given
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        image_reference="http://some.com/image.jpg",
    )

    # when
    result = image.to_inference_format(numpy_preferred=False)

    # then
    assert result["type"] == "url"
    assert result["value"] == "http://some.com/image.jpg"


def test_workflow_image_data_to_inference_format_when_numpy_not_preferred_but_file_path_available_on_the_spot() -> (
    None
):
    # given
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        image_reference="./some/file.jpeg",
    )

    # when
    result = image.to_inference_format(numpy_preferred=False)

    # then
    assert result["type"] == "file"
    assert result["value"] == "./some/file.jpeg"


def test_workflow_image_data_to_inference_format_when_numpy_not_preferred_but_base64_available_on_the_spot() -> (
    None
):
    # given
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        base64_image="base64_value",
    )

    # when
    result = image.to_inference_format(numpy_preferred=False)

    # then
    assert result["type"] == "base64"
    assert result["value"] == "base64_value"


@mock.patch.object(base, "datetime")
def test_workflow_image_default_video_metadata_generation(
    datetime_mock: MagicMock,
) -> None:
    # given
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        numpy_image=np.zeros((192, 168, 3), dtype=np.uint8),
    )
    timestamp = datetime.now()
    datetime_mock.now.return_value = timestamp

    # when
    result = image.video_metadata

    # then
    assert result == VideoMetadata(
        video_identifier="parent",
        frame_number=0,
        frame_timestamp=timestamp,
        fps=30,
        comes_from_video_file=None,
    )


def test_workflow_image_video_metadata_preservation() -> None:
    # given
    metadata = VideoMetadata(
        video_identifier="parent",
        frame_number=0,
        frame_timestamp=datetime.now(),
        fps=40,
        comes_from_video_file=None,
    )
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        numpy_image=np.zeros((192, 168, 3), dtype=np.uint8),
        video_metadata=metadata,
    )

    # when
    result = image.video_metadata

    # then
    assert result is metadata


def test_workflow_image_replace_image_operation() -> None:
    # given
    metadata = VideoMetadata(
        video_identifier="parent",
        frame_number=0,
        frame_timestamp=datetime.now(),
        fps=40,
        comes_from_video_file=None,
    )
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        workflow_root_ancestor_metadata=ImageParentMetadata(parent_id="root"),
        numpy_image=np.zeros((192, 168, 3), dtype=np.uint8),
        base64_image="some",
        image_reference="ref",
        video_metadata=metadata,
    )
    updated_image = np.ones((192, 168, 3), dtype=np.uint8)

    # when
    result = WorkflowImageData.copy_and_replace(
        origin_image_data=image,
        numpy_image=updated_image,
    )

    # then
    assert result is not image
    assert result.parent_metadata == image.parent_metadata
    assert (
        result.workflow_root_ancestor_metadata == image.workflow_root_ancestor_metadata
    )
    assert result.video_metadata == metadata
    assert np.allclose(result.numpy_image, updated_image)
    assert result.base64_image != "some"


def test_workflow_image_replace_base64_image_operation() -> None:
    # given
    metadata = VideoMetadata(
        video_identifier="parent",
        frame_number=0,
        frame_timestamp=datetime.now(),
        fps=40,
        comes_from_video_file=None,
    )
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        workflow_root_ancestor_metadata=ImageParentMetadata(parent_id="root"),
        numpy_image=np.zeros((192, 168, 3), dtype=np.uint8),
        base64_image="some",
        image_reference="ref",
        video_metadata=metadata,
    )
    updated_image = np.ones((192, 168, 3), dtype=np.uint8)
    is_success, buffer = cv2.imencode(".jpg", updated_image)
    base64_image = base64.b64encode(buffer.tobytes())

    # when
    result = WorkflowImageData.copy_and_replace(
        origin_image_data=image,
        base64_image=base64_image,
    )

    # then
    assert result is not image
    assert result.parent_metadata == image.parent_metadata
    assert (
        result.workflow_root_ancestor_metadata == image.workflow_root_ancestor_metadata
    )
    assert result.video_metadata == metadata
    assert np.allclose(result.numpy_image, updated_image)
    assert result.base64_image == base64_image


def test_workflow_image_replace_all_image_representations() -> None:
    # given
    metadata = VideoMetadata(
        video_identifier="parent",
        frame_number=0,
        frame_timestamp=datetime.now(),
        fps=40,
        comes_from_video_file=None,
    )
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        workflow_root_ancestor_metadata=ImageParentMetadata(parent_id="root"),
        numpy_image=np.zeros((192, 168, 3), dtype=np.uint8),
        base64_image="some",
        image_reference="ref",
        video_metadata=metadata,
    )
    updated_image = np.ones((192, 168, 3), dtype=np.uint8)

    # when
    result = WorkflowImageData.copy_and_replace(
        origin_image_data=image,
        base64_image="dummy",
        numpy_image=updated_image,
    )

    # then
    assert result is not image
    assert result.parent_metadata == image.parent_metadata
    assert (
        result.workflow_root_ancestor_metadata == image.workflow_root_ancestor_metadata
    )
    assert result.video_metadata == metadata
    assert np.allclose(result.numpy_image, updated_image)
    assert result.base64_image == "dummy"


def test_workflow_image_replace_parent_metadata() -> None:
    # given
    metadata = VideoMetadata(
        video_identifier="parent",
        frame_number=0,
        frame_timestamp=datetime.now(),
        fps=40,
        comes_from_video_file=None,
    )
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        workflow_root_ancestor_metadata=ImageParentMetadata(parent_id="root"),
        numpy_image=np.zeros((192, 168, 3), dtype=np.uint8),
        base64_image="some",
        image_reference="ref",
        video_metadata=metadata,
    )
    updated_parent = ImageParentMetadata(parent_id="updated_parent")

    # when
    result = WorkflowImageData.copy_and_replace(
        origin_image_data=image,
        parent_metadata=updated_parent,
    )

    # then
    assert result is not image
    assert result.numpy_image.sum() == 0, "Image not changed"
    assert result.parent_metadata.parent_id == updated_parent.parent_id
    assert (
        result.workflow_root_ancestor_metadata == image.workflow_root_ancestor_metadata
    )
    assert result.video_metadata == metadata


def test_workflow_image_replace_parent_metadata_when_root_metadata_is_the_same_as_parent_one() -> (
    None
):
    # given
    metadata = VideoMetadata(
        video_identifier="parent",
        frame_number=0,
        frame_timestamp=datetime.now(),
        fps=40,
        comes_from_video_file=None,
    )
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        workflow_root_ancestor_metadata=None,
        numpy_image=np.zeros((192, 168, 3), dtype=np.uint8),
        base64_image="some",
        image_reference="ref",
        video_metadata=metadata,
    )
    updated_parent = ImageParentMetadata(parent_id="updated_parent")

    # when
    result = WorkflowImageData.copy_and_replace(
        origin_image_data=image,
        parent_metadata=updated_parent,
    )

    # then
    assert result is not image
    assert result.numpy_image.sum() == 0, "Image not changed"
    assert result.parent_metadata.parent_id == updated_parent.parent_id
    assert result.workflow_root_ancestor_metadata.parent_id == updated_parent.parent_id
    assert result.video_metadata == metadata


def test_workflow_image_replace_video_metadata() -> None:
    # given
    metadata = VideoMetadata(
        video_identifier="parent",
        frame_number=0,
        frame_timestamp=datetime.now(),
        fps=40,
        comes_from_video_file=None,
    )
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        workflow_root_ancestor_metadata=ImageParentMetadata(parent_id="root"),
        numpy_image=np.zeros((192, 168, 3), dtype=np.uint8),
        base64_image="some",
        image_reference="ref",
        video_metadata=metadata,
    )
    new_metadata = VideoMetadata(
        video_identifier="new_metadata",
        frame_number=0,
        frame_timestamp=datetime.now(),
        fps=40,
        comes_from_video_file=None,
    )

    # when
    result = WorkflowImageData.copy_and_replace(
        origin_image_data=image,
        video_metadata=new_metadata,
    )

    # then
    assert result is not image
    assert result.parent_metadata == image.parent_metadata
    assert (
        result.workflow_root_ancestor_metadata == image.workflow_root_ancestor_metadata
    )
    assert result.video_metadata == new_metadata
    assert result.numpy_image.sum() == 0, "Image not changed"
    assert result.base64_image == "some"


def test_workflow_image_create_crop_operation() -> None:
    # given
    metadata = VideoMetadata(
        video_identifier="video_id",
        frame_number=0,
        frame_timestamp=datetime.now(),
        fps=40,
        comes_from_video_file=None,
    )
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        workflow_root_ancestor_metadata=ImageParentMetadata(parent_id="root"),
        numpy_image=np.zeros((192, 168, 3), dtype=np.uint8),
        base64_image="some",
        image_reference="ref",
        video_metadata=metadata,
    )

    # when
    result = WorkflowImageData.create_crop(
        origin_image_data=image,
        crop_identifier="my_crop",
        cropped_image=np.zeros((64, 60, 3)),
        offset_x=100,
        offset_y=20,
    )

    # then
    assert np.allclose(result.numpy_image, np.zeros((64, 60, 3)))
    assert result.parent_metadata.parent_id == "my_crop"
    assert result.parent_metadata.origin_coordinates == OriginCoordinatesSystem(
        left_top_x=100,
        left_top_y=20,
        origin_width=168,
        origin_height=192,
    )
    assert result.workflow_root_ancestor_metadata.parent_id == "root"
    assert (
        result.workflow_root_ancestor_metadata.origin_coordinates
        == OriginCoordinatesSystem(
            left_top_x=100,
            left_top_y=20,
            origin_width=168,
            origin_height=192,
        )
    )
    assert (
        result.video_metadata.video_identifier == "my_crop"
    ), "Expected default metadata"
    assert result.video_metadata.fps == 30, "Expected default metadata"


def test_workflow_image_build_create_crop_with_video_metadata_preservation() -> None:
    # given
    metadata = VideoMetadata(
        video_identifier="video_id",
        frame_number=0,
        frame_timestamp=datetime.now(),
        fps=40,
        comes_from_video_file=None,
    )
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        workflow_root_ancestor_metadata=ImageParentMetadata(parent_id="root"),
        numpy_image=np.zeros((192, 168, 3), dtype=np.uint8),
        base64_image="some",
        image_reference="ref",
        video_metadata=metadata,
    )

    # when
    result = WorkflowImageData.create_crop(
        origin_image_data=image,
        crop_identifier="my_crop",
        cropped_image=np.zeros((64, 60, 3)),
        offset_x=100,
        offset_y=20,
        preserve_video_metadata=True,
    )

    # then
    assert np.allclose(result.numpy_image, np.zeros((64, 60, 3)))
    assert result.parent_metadata.parent_id == "my_crop"
    assert result.parent_metadata.origin_coordinates == OriginCoordinatesSystem(
        left_top_x=100,
        left_top_y=20,
        origin_width=168,
        origin_height=192,
    )
    assert result.workflow_root_ancestor_metadata.parent_id == "root"
    assert (
        result.workflow_root_ancestor_metadata.origin_coordinates
        == OriginCoordinatesSystem(
            left_top_x=100,
            left_top_y=20,
            origin_width=168,
            origin_height=192,
        )
    )
    assert (
        result.video_metadata.video_identifier == "video_id | crop: my_crop"
    ), "Expected preserved metadata with updated id"
    assert result.video_metadata.fps == 40, "Expected preserved metadata"


def test_parent_origin_from_origin_coordinates_system() -> None:
    # given
    origin_coords = OriginCoordinatesSystem(
        left_top_x=100,
        left_top_y=200,
        origin_width=800,
        origin_height=600,
    )

    # when
    result = ParentOrigin.from_origin_coordinates_system(origin_coords)

    # then
    assert result.offset_x == 100
    assert result.offset_y == 200
    assert result.width == 800
    assert result.height == 600


def test_parent_origin_to_origin_coordinates_system() -> None:
    # given
    parent_origin = ParentOrigin(
        offset_x=100,
        offset_y=200,
        width=800,
        height=600,
    )

    # when
    result = parent_origin.to_origin_coordinates_system()

    # then
    assert result.left_top_x == 100
    assert result.left_top_y == 200
    assert result.origin_width == 800
    assert result.origin_height == 600


def test_parent_origin_validation_rejects_zero_width() -> None:
    # when
    with pytest.raises(Exception):  # pydantic ValidationError
        _ = ParentOrigin(
            offset_x=0,
            offset_y=0,
            width=0,
            height=100,
        )


def test_parent_origin_validation_rejects_zero_height() -> None:
    # when
    with pytest.raises(Exception):  # pydantic ValidationError
        _ = ParentOrigin(
            offset_x=0,
            offset_y=0,
            width=100,
            height=0,
        )


def test_parent_origin_validation_rejects_negative_width() -> None:
    # when
    with pytest.raises(Exception):  # pydantic ValidationError
        _ = ParentOrigin(
            offset_x=0,
            offset_y=0,
            width=-100,
            height=100,
        )


def test_parent_origin_validation_rejects_negative_height() -> None:
    # when
    with pytest.raises(Exception):  # pydantic ValidationError
        _ = ParentOrigin(
            offset_x=0,
            offset_y=0,
            width=100,
            height=-100,
        )


# ---------------------------------------------------------------------------
# Tensor-native representation tests
# ---------------------------------------------------------------------------


import torch  # noqa: E402  (kept low to avoid touching the existing import block)

from inference.core.env import (  # noqa: E402
    ENABLE_TENSOR_DATA_REPRESENTATION,
    WORKFLOWS_IMAGE_TENSOR_DEVICE,
)


def test_init_workflow_image_data_from_tensor_only() -> None:
    # given
    # Allocated on the configured device so the no-copy identity below holds even
    # when WORKFLOWS_IMAGE_TENSOR_DEVICE is cuda.
    tensor = torch.zeros(
        (3, 10, 20), dtype=torch.uint8, device=WORKFLOWS_IMAGE_TENSOR_DEVICE
    )

    # when
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        tensor_image=tensor,
    )

    # then
    assert image.tensor_image is tensor
    assert image.tensor_image.device == WORKFLOWS_IMAGE_TENSOR_DEVICE


def test_workflow_image_data_numpy_fallback_does_rgb_to_bgr() -> None:
    # given
    # Tensor is CHW uint8 RGB by convention. Bake distinct R/G/B values so
    # the channel order is asserted, not just the shape.
    tensor = torch.zeros((3, 2, 2), dtype=torch.uint8)
    tensor[0] = 10  # R
    tensor[1] = 20  # G
    tensor[2] = 30  # B
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        tensor_image=tensor,
    )

    # when
    numpy_image = image.numpy_image

    # then
    # numpy is HWC uint8 BGR, so [..., 0]=B=30, [..., 1]=G=20, [..., 2]=R=10.
    assert numpy_image.shape == (2, 2, 3)
    assert numpy_image.dtype == np.uint8
    assert np.all(numpy_image[..., 0] == 30)
    assert np.all(numpy_image[..., 1] == 20)
    assert np.all(numpy_image[..., 2] == 10)


def test_workflow_image_data_tensor_fallback_does_bgr_to_rgb() -> None:
    # given
    # numpy is HWC uint8 BGR.
    numpy_image = np.zeros((2, 2, 3), dtype=np.uint8)
    numpy_image[..., 0] = 30  # B
    numpy_image[..., 1] = 20  # G
    numpy_image[..., 2] = 10  # R
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        numpy_image=numpy_image,
    )

    # when
    tensor = image.tensor_image

    # then
    # tensor is CHW uint8 RGB, contiguous.
    assert tuple(tensor.shape) == (3, 2, 2)
    assert tensor.dtype == torch.uint8
    assert tensor.is_contiguous()
    assert torch.all(tensor[0] == 10)
    assert torch.all(tensor[1] == 20)
    assert torch.all(tensor[2] == 30)


def test_workflow_image_data_chw_round_trip_is_pixel_exact_and_non_square() -> None:
    # given a non-square BGR image so an H<->W transpose would be detectable
    numpy_image = np.zeros((4, 6, 3), dtype=np.uint8)
    numpy_image[..., 0] = 30  # B
    numpy_image[..., 1] = 20  # G
    numpy_image[..., 2] = 10  # R
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        numpy_image=numpy_image,
    )

    # when
    tensor = image.tensor_image  # CHW RGB
    round_tripped = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        tensor_image=tensor,
    ).numpy_image  # back to HWC BGR

    # then
    assert tuple(tensor.shape) == (3, 4, 6)  # C, H, W
    assert tensor.is_contiguous()
    assert image._read_shape_without_materialization() == (4, 6)  # H, W
    assert round_tripped.shape == (4, 6, 3)
    assert np.array_equal(round_tripped, numpy_image)


def test_workflow_image_data_numpy_fallback_caches() -> None:
    # given
    tensor = torch.zeros((3, 4, 3), dtype=torch.uint8)
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        tensor_image=tensor,
    )

    # when
    first = image.numpy_image
    second = image.numpy_image

    # then
    assert first is second, "lazy materialization should cache the numpy array"


def test_workflow_image_data_tensor_fallback_caches() -> None:
    # given
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        numpy_image=np.zeros((3, 4, 3), dtype=np.uint8),
    )

    # when
    first = image.tensor_image
    second = image.tensor_image

    # then
    assert first is second, "lazy materialization should cache the tensor"


def test_workflow_image_data_parent_metadata_without_forcing_materialization() -> None:
    # given
    # Tensor-only construction. Reading parent_metadata must populate
    # origin coordinates without going through numpy materialization.
    tensor = torch.zeros((3, 7, 11), dtype=torch.uint8)
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        tensor_image=tensor,
    )

    # when
    parent_metadata = image.parent_metadata

    # then
    assert parent_metadata.origin_coordinates == OriginCoordinatesSystem(
        left_top_x=0,
        left_top_y=0,
        origin_width=11,
        origin_height=7,
    )
    # Sanity: shape was read from the tensor; the numpy cache is still empty.
    assert image._numpy_image is None


def test_workflow_image_create_crop_from_tensor() -> None:
    # given
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        workflow_root_ancestor_metadata=ImageParentMetadata(parent_id="root"),
        tensor_image=torch.zeros(
            (3, 192, 168), dtype=torch.uint8, device=WORKFLOWS_IMAGE_TENSOR_DEVICE
        ),
    )
    cropped = torch.zeros(
        (3, 64, 60), dtype=torch.uint8, device=WORKFLOWS_IMAGE_TENSOR_DEVICE
    )

    # when
    result = WorkflowImageData.create_crop_from_tensor(
        origin_image_data=image,
        crop_identifier="my_crop",
        cropped_tensor_image=cropped,
        offset_x=100,
        offset_y=20,
    )

    # then
    assert result.tensor_image is cropped
    assert result.parent_metadata.parent_id == "my_crop"
    assert result.parent_metadata.origin_coordinates == OriginCoordinatesSystem(
        left_top_x=100,
        left_top_y=20,
        origin_width=168,
        origin_height=192,
    )
    assert result.workflow_root_ancestor_metadata.parent_id == "root"
    assert (
        result.workflow_root_ancestor_metadata.origin_coordinates
        == OriginCoordinatesSystem(
            left_top_x=100,
            left_top_y=20,
            origin_width=168,
            origin_height=192,
        )
    )


def test_workflow_image_copy_and_replace_preserves_tensor() -> None:
    # given
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        tensor_image=torch.zeros(
            (3, 4, 3), dtype=torch.uint8, device=WORKFLOWS_IMAGE_TENSOR_DEVICE
        ),
    )

    # when
    # copy_and_replace with no image-related kwargs must carry the tensor over.
    result = WorkflowImageData.copy_and_replace(
        origin_image_data=image,
        parent_metadata=ImageParentMetadata(parent_id="new_parent"),
    )

    # then
    assert result.tensor_image is image.tensor_image
    assert result.parent_metadata.parent_id == "new_parent"


def test_workflow_image_copy_and_replace_swaps_tensor_for_numpy() -> None:
    # given
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        tensor_image=torch.zeros((3, 4, 3), dtype=torch.uint8),
    )
    replacement = np.zeros((5, 6, 3), dtype=np.uint8)

    # when
    result = WorkflowImageData.copy_and_replace(
        origin_image_data=image,
        numpy_image=replacement,
    )

    # then
    # When any image-representation kwarg is provided, the others reset to
    # whatever was passed (None by default). Tensor should be gone.
    assert result._tensor_image is None
    assert result.numpy_image is replacement


def test_workflow_image_data_single_channel_numpy_to_tensor() -> None:
    # given - Convert Grayscale / Threshold blocks produce 2-D (H, W) arrays
    gray = np.arange(24, dtype=np.uint8).reshape(4, 6)
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        numpy_image=gray,
    )

    # when
    tensor = image.tensor_image

    # then - (1, H, W), values untouched (no channel reversal for single-channel)
    assert tuple(tensor.shape) == (1, 4, 6)
    assert tensor.dtype == torch.uint8
    assert tensor.is_contiguous()
    assert np.array_equal(tensor.squeeze(0).cpu().numpy(), gray)


def test_workflow_image_data_single_channel_tensor_to_numpy_round_trip() -> None:
    # given
    gray = np.arange(24, dtype=np.uint8).reshape(4, 6)
    tensor_born = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        tensor_image=torch.from_numpy(gray.copy()).unsqueeze(0),
    )

    # when
    round_tripped = tensor_born.numpy_image

    # then - back to the 2-D (H, W) shape numpy-land blocks produce
    assert round_tripped.shape == (4, 6)
    assert np.array_equal(round_tripped, gray)
    assert tensor_born._read_shape_without_materialization() == (4, 6)


def test_workflow_image_data_declared_numpy_mutation_refreshes_tensor() -> None:
    # given - numpy-born image with the tensor sibling already derived; both
    # representations stay cached for readers (fan-out reads are free)
    numpy_image = np.zeros((4, 6, 3), dtype=np.uint8)
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        numpy_image=numpy_image,
    )
    first_tensor = image.tensor_image
    assert image.tensor_image is first_tensor, "readers keep the cached tensor"

    # when - the copy_image=False visualization pattern: mutate numpy in place,
    # then DECLARE the mutation per the class contract
    image.numpy_image[0, 0] = (1, 2, 3)  # BGR
    undeclared_tensor = image.tensor_image
    image.declare_numpy_image_mutated()
    refreshed_tensor = image.tensor_image

    # then - before the declaration the sibling cache is stale (the documented
    # limitation of undeclared in-place mutation); after it, the tensor is
    # re-derived from the mutated pixels (RGB order at [:, 0, 0])
    assert undeclared_tensor is first_tensor and torch.all(undeclared_tensor == 0)
    assert tuple(int(c) for c in refreshed_tensor[:, 0, 0]) == (3, 2, 1)


def test_workflow_image_data_declared_tensor_mutation_refreshes_numpy() -> None:
    # given - tensor-born image (the tensor-mode video/crop case) with the numpy
    # sibling already derived; tensor residency survives numpy reads
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        tensor_image=torch.zeros((3, 4, 6), dtype=torch.uint8),
    )
    _ = image.numpy_image
    assert image.is_tensor_materialised(), "numpy reads must not evict the tensor"

    # when - mutate the tensor in place and declare it
    image.tensor_image[:, 0, 0] = torch.tensor([3, 2, 1], dtype=torch.uint8)  # RGB
    image.declare_tensor_image_mutated()

    # then - numpy is re-derived from the mutated tensor (BGR order at [0, 0])
    assert np.array_equal(image.numpy_image[0, 0], np.array([1, 2, 3], dtype=np.uint8))


def test_workflow_image_data_tensor_from_base64_does_not_cache_numpy() -> None:
    # given - PNG so the decode is lossless and pixel assertions are exact
    numpy_image = np.zeros((4, 6, 3), dtype=np.uint8)
    numpy_image[..., 0] = 30  # B
    numpy_image[..., 1] = 20  # G
    numpy_image[..., 2] = 10  # R
    encoded = base64.b64encode(cv2.imencode(".png", numpy_image)[1]).decode("ascii")
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        base64_image=encoded,
    )

    # when - the tensor is requested FIRST on a base64-born image
    tensor = image.tensor_image

    # then - the source decodes straight into the tensor representation; the
    # transient decode buffer must NOT be left behind as a cached numpy
    assert image._numpy_image is None, "decode must not leave a cached numpy behind"
    assert tuple(tensor.shape) == (3, 4, 6)
    assert torch.all(tensor[0] == 10)  # R
    assert torch.all(tensor[1] == 20)  # G
    assert torch.all(tensor[2] == 30)  # B
    # numpy, when later needed, re-derives from the tensor - pixel-exact
    assert np.array_equal(image.numpy_image, numpy_image)


def test_workflow_image_data_tensor_from_base64_jpeg_uses_configured_device() -> None:
    numpy_image = np.zeros((4, 6, 3), dtype=np.uint8)
    encoded_bytes = cv2.imencode(".jpg", numpy_image)[1].tobytes()
    encoded = base64.b64encode(encoded_bytes).decode("ascii")
    expected = torch.zeros(
        (3, 4, 6), dtype=torch.uint8, device=WORKFLOWS_IMAGE_TENSOR_DEVICE
    )
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        base64_image=encoded,
    )

    with mock.patch.object(base, "decode_jpeg", return_value=expected) as decode:
        result = image.tensor_image

    assert result is expected
    payload = decode.call_args.args[0]
    assert bytes(payload.tolist()) == encoded_bytes
    assert decode.call_args.kwargs == {
        "mode": base.ImageReadMode.RGB,
        "device": WORKFLOWS_IMAGE_TENSOR_DEVICE,
    }


def test_workflow_image_data_shape_read_fallback_materializes_per_flag() -> None:
    # given - a base64-born image with no in-memory representation yet
    encoded = base64.b64encode(
        cv2.imencode(".png", np.zeros((7, 11, 3), dtype=np.uint8))[1]
    ).decode("ascii")
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        base64_image=encoded,
    )

    # when
    shape = image._read_shape_without_materialization()

    # then - the flag-appropriate representation (and only it) materialised
    assert shape == (7, 11)
    if ENABLE_TENSOR_DATA_REPRESENTATION:
        assert image._tensor_image is not None, "tensor mode materialises the tensor"
        assert image._numpy_image is None, "tensor mode must not cache numpy"
    else:
        assert image._numpy_image is not None, "numpy mode materialises numpy"
        assert image._tensor_image is None, "numpy mode must not build the tensor"


def test_workflow_image_data_tensor_from_file_reference_does_not_cache_numpy(
    tmp_path,
) -> None:
    # given - a lossless PNG on disk referenced by path
    numpy_image = np.zeros((4, 6, 3), dtype=np.uint8)
    numpy_image[..., 0] = 30  # B
    numpy_image[..., 1] = 20  # G
    numpy_image[..., 2] = 10  # R
    path = str(tmp_path / "reference.png")
    assert cv2.imwrite(path, numpy_image)
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        image_reference=path,
    )

    # when - the tensor is requested FIRST on a reference-born image
    tensor = image.tensor_image

    # then - decoded straight to CHW RGB, no cached numpy left behind
    assert image._numpy_image is None, "decode must not leave a cached numpy behind"
    assert tuple(tensor.shape) == (3, 4, 6)
    assert torch.all(tensor[0] == 10)  # R
    assert torch.all(tensor[1] == 20)  # G
    assert torch.all(tensor[2] == 30)  # B
    assert np.array_equal(image.numpy_image, numpy_image)
