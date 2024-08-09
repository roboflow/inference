import base64
import os
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
