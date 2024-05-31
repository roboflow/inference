import base64
import os
from unittest import mock
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from inference.core.workflows.entities import base
from inference.core.workflows.entities.base import (
    Batch,
    ImageParentMetadata,
    OriginCoordinatesSystem,
    WorkflowImageData,
)


def test_standard_iteration_through_batch() -> None:
    # given
    batch = Batch(content=[1, "2", None, 3.0])

    # when
    result = list(batch)

    # then
    assert result == [1, "2", None, 3.0]


def test_getting_batch_length() -> None:
    # given
    batch = Batch(content=[1, "2", None, 3.0])

    # when
    result = len(batch)

    # then
    assert result == 4


def test_getting_batch_element_when_single_element_chosen() -> None:
    # given
    batch = Batch(content=[1, "2", None, 3.0])

    # when
    result = batch[1]

    # then
    assert result == "2"


def test_getting_batch_element_when_boolean_mask_as_list_declared() -> None:
    # given
    batch = Batch(content=[1, "2", None, 3.0])

    # when
    result = batch[[True, False, True, False]]

    # then
    assert list(result) == [1, None]


def test_getting_batch_element_when_boolean_mask_as_np_array_declared() -> None:
    # given
    batch = Batch(content=[1, "2", None, 3.0])

    # when
    result = batch[np.array([True, False, True, False])]

    # then
    assert list(result) == [1, None]


def test_getting_batch_element_when_mask_of_invalid_size_declared() -> None:
    # given
    batch = Batch(content=[1, "2", None, 3.0])

    # when
    with pytest.raises(ValueError):
        _ = batch[[True, False, False]]


def test_iterating_over_non_empty_elements_of_batch() -> None:
    # given
    batch = Batch(content=[1, "2", None, 3.0])

    # when
    result = list(batch.iter_nonempty())

    # then
    assert result == [1, "2", 3.0]


def test_iterating_over_non_empty_elements_of_batch_returning_indices() -> None:
    # given
    batch = Batch(content=[1, "2", None, 3.0])

    # when
    result = list(batch.iter_nonempty(return_index=True))

    # then
    assert result == [(0, 1), (1, "2"), (3, 3.0)]


def test_iter_selected_batch_elements() -> None:
    # given
    batch = Batch(content=[1, "2", None, 3.0])

    # when
    result = list(batch.iter_selected(mask=[True, False, False, False]))

    # then
    assert result == [1]


def test_iter_selected_batch_elements_returning_indices() -> None:
    # given
    batch = Batch(content=[1, "2", None, 3.0])

    # when
    result = list(
        batch.iter_selected(mask=[False, True, False, False], return_index=True)
    )

    # then
    assert result == [(1, "2")]


def test_align_batch_results_when_mask_provided() -> None:
    # given
    batch = Batch(content=[1, "2", None, 3.0])
    partial_result = ["A", "C", "D"]

    # when
    result = batch.align_batch_results(
        results=partial_result,
        null_element="null",
        mask=[True, False, True, True],
    )

    # then
    assert result == ["A", "null", "C", "D"]


def test_align_batch_results_when_mask_assumed() -> None:
    # given
    batch = Batch(content=[1, "2", None, 3.0])
    partial_result = ["A", "B", "D"]

    # when
    result = batch.align_batch_results(
        results=partial_result,
        null_element="null",
    )

    # then
    assert result == ["A", "B", "null", "D"]


def test_align_batch_results_when_mask_provided_partial_result_shape_missmatch() -> (
    None
):
    # given
    batch = Batch(content=[1, "2", None, 3.0])
    partial_result = ["A", "B", "C", "D"]

    # when
    with pytest.raises(ValueError):
        _ = batch.align_batch_results(
            results=partial_result,
            null_element="null",
            mask=[True, False, True, True],
        )


def test_mask_empty_elements_in_batch() -> None:
    # given
    batch = Batch(content=[1, "2", None, 3.0])

    # when
    result = batch.mask_empty_elements()

    # then
    assert result == [True, True, False, True]


def test_broadcast_batch_when_requested_size_is_equal_to_batch_size() -> None:
    # given
    batch = Batch(content=[1, "2", None, 3.0])

    # when
    result = batch.broadcast(n=4)

    # then
    assert list(result) == [1, "2", None, 3.0]


def test_broadcast_batch_when_requested_size_is_valid_and_batch_size_is_one() -> None:
    # given
    batch = Batch(content=[1])

    # when
    result = batch.broadcast(n=4)

    # then
    assert list(result) == [1, 1, 1, 1]


def test_broadcast_batch_when_requested_size_is_valid_and_batch_size_is_not_matching() -> (
    None
):
    # given
    batch = Batch(content=[1, 2])

    # when
    with pytest.raises(ValueError):
        _ = batch.broadcast(n=4)


def test_broadcast_batch_when_requested_size_is_invalid() -> None:
    # given
    batch = Batch(content=[1, 2])

    # when
    with pytest.raises(ValueError):
        _ = batch.broadcast(n=0)


def test_mask_common_empty_elements_in_batches_when_no_batches_provided() -> None:
    # when
    result = Batch.mask_common_empty_elements(batches=[])

    # then
    assert result == []


def test_mask_common_empty_elements_in_batches_when_not_equally_long_batches_provided() -> (
    None
):
    # when
    with pytest.raises(ValueError):
        _ = Batch.mask_common_empty_elements(
            batches=[Batch(content=[1, "2", None, 3.0]), Batch(content=[1, "2", None])]
        )


def test_mask_common_empty_elements_in_batches_when_equally_long_batches_provided() -> (
    None
):
    # when
    result = Batch.mask_common_empty_elements(
        batches=[
            Batch(content=[1, "2", None, 3.0]),
            Batch(content=[1, "2", None, 2.0]),
            Batch(content=[None, "2", 3, 2.0]),
        ]
    )

    # then
    assert result == [False, True, False, True]


def test_zip_nonempty_batches_when_empty_batch_list_provided() -> None:
    # when
    result = list(Batch.zip_nonempty(batches=[]))

    # then
    assert result == []


def test_zip_nonempty_when_empty_batches_provided() -> None:
    # when
    result = list(Batch.zip_nonempty(batches=[Batch(content=[]), Batch(content=[])]))

    # then
    assert result == []


def test_zip_nonempty_when_batches_with_empty_values_only_provided() -> None:
    # when
    result = list(
        Batch.zip_nonempty(batches=[Batch(content=[None]), Batch(content=[None])])
    )

    # then
    assert result == []


def test_zip_nonempty_when_batches_with_empty_and_non_empty_provided() -> None:
    # when
    result = list(
        Batch.zip_nonempty(
            batches=[
                Batch(content=[None, 1, None, 4, 6]),
                Batch(content=[None, 2, 3, 5, None]),
            ]
        )
    )

    # then
    assert result == [(1, 2), (4, 5)]


def test_align_batches_results_when_malformed_results_provided() -> None:
    # given
    batches = [
        Batch(content=[None, 1, None, 4, 6]),
        Batch(content=[None, 2, 3, 5, None]),
    ]
    partial_results = [1, 2, 3]  # to many items

    # when
    with pytest.raises(ValueError):
        _ = Batch.align_batches_results(
            batches=batches,
            results=partial_results,
            null_element="null",
        )


def test_align_batches_results_when_valid_results_provided() -> None:
    # given
    batches = [
        Batch(content=[None, 1, None, 4, 6]),
        Batch(content=[None, 2, 3, 5, None]),
    ]
    partial_results = [1, 2]  # to many items

    # when
    result = Batch.align_batches_results(
        batches=batches,
        results=partial_results,
        null_element="null",
    )

    # then
    assert result == ["null", 1, "null", 2, "null"]


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
