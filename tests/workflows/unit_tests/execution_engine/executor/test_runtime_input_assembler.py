import time
from datetime import datetime
from typing import Any
from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import pytest

from inference.core.workflows.errors import RuntimeInputError
from inference.core.workflows.execution_engine.entities.base import (
    VideoMetadata,
    WorkflowImage,
    WorkflowParameter,
    WorkflowVideoMetadata,
)
from inference.core.workflows.execution_engine.v1.executor import (
    runtime_input_assembler,
)
from inference.core.workflows.execution_engine.v1.executor.runtime_input_assembler import (
    assemble_runtime_parameters,
)


def test_assemble_runtime_parameters_when_image_is_not_provided() -> None:
    # given
    runtime_parameters = {}
    defined_inputs = [WorkflowImage(type="WorkflowImage", name="image")]

    # when
    with pytest.raises(RuntimeInputError):
        _ = assemble_runtime_parameters(
            runtime_parameters=runtime_parameters,
            defined_inputs=defined_inputs,
        )


@mock.patch.object(runtime_input_assembler, "load_image_from_url")
def test_assemble_runtime_parameters_when_image_is_provided_as_single_element_dict(
    load_image_from_url_mock: MagicMock,
) -> None:
    # given
    load_image_from_url_mock.return_value = np.zeros((192, 168, 3), dtype=np.uint8)
    runtime_parameters = {
        "image1": {
            "type": "url",
            "value": "https://some.com/image.jpg",
        }
    }
    defined_inputs = [WorkflowImage(type="WorkflowImage", name="image1")]

    # when
    result = assemble_runtime_parameters(
        runtime_parameters=runtime_parameters,
        defined_inputs=defined_inputs,
    )

    # then
    assert (
        len(result["image1"]) == 1
    ), "Single image to be transformed into 1-element batch"
    assert np.allclose(
        result["image1"][0].numpy_image, np.zeros((192, 168, 3), dtype=np.uint8)
    ), "Expected image to be placed correctly"
    assert (
        result["image1"][0].parent_metadata.parent_id == "image1"
    ), "Expected parent id to be given after input param name"


def test_assemble_runtime_parameters_when_image_is_provided_as_single_element_dict_pointing_local_file_when_load_of_local_files_allowed(
    example_image_file: str,
) -> None:
    # given
    runtime_parameters = {
        "image1": {
            "type": "file",
            "value": example_image_file,
        }
    }
    defined_inputs = [WorkflowImage(type="WorkflowImage", name="image1")]

    # when
    result = assemble_runtime_parameters(
        runtime_parameters=runtime_parameters,
        defined_inputs=defined_inputs,
    )

    # then
    assert (
        len(result["image1"]) == 1
    ), "Single image to be transformed into 1-element batch"
    assert np.allclose(
        result["image1"][0].numpy_image, np.zeros((192, 168, 3), dtype=np.uint8)
    ), "Expected image to be placed correctly"
    assert (
        result["image1"][0].parent_metadata.parent_id == "image1"
    ), "Expected parent id to be given after input param name"


def test_assemble_runtime_parameters_when_image_is_provided_as_single_element_dict_pointing_local_file_when_load_of_local_files_not_allowed(
    example_image_file: str,
) -> None:
    # given
    runtime_parameters = {
        "image1": {
            "type": "file",
            "value": example_image_file,
        }
    }
    defined_inputs = [WorkflowImage(type="WorkflowImage", name="image1")]

    # when
    with pytest.raises(RuntimeInputError):
        _ = assemble_runtime_parameters(
            runtime_parameters=runtime_parameters,
            defined_inputs=defined_inputs,
            prevent_local_images_loading=True,
        )


def test_assemble_runtime_parameters_when_image_is_provided_as_single_element_np_array() -> (
    None
):
    # given
    runtime_parameters = {"image1": np.zeros((192, 168, 3), dtype=np.uint8)}
    defined_inputs = [WorkflowImage(type="WorkflowImage", name="image1")]

    # when
    result = assemble_runtime_parameters(
        runtime_parameters=runtime_parameters,
        defined_inputs=defined_inputs,
    )

    # then
    assert (
        len(result["image1"]) == 1
    ), "Image is to be transformed into single element list"
    assert result["image1"][0].parent_metadata.parent_id == "image1"
    assert np.allclose(
        result["image1"][0].numpy_image, np.zeros((192, 168, 3), dtype=np.uint8)
    )


def test_assemble_runtime_parameters_when_image_is_provided_as_unknown_element() -> (
    None
):
    # given
    runtime_parameters = {"image1": "some"}
    defined_inputs = [WorkflowImage(type="WorkflowImage", name="image1")]

    # when
    with pytest.raises(RuntimeInputError):
        _ = assemble_runtime_parameters(
            runtime_parameters=runtime_parameters,
            defined_inputs=defined_inputs,
        )


def test_assemble_runtime_parameters_when_image_is_provided_in_batch() -> None:
    # given
    runtime_parameters = {
        "image1": [
            np.zeros((192, 168, 3), dtype=np.uint8),
            {
                "type": "numpy_object",
                "value": np.ones((256, 256, 3), dtype=np.uint8),
            },
        ]
    }
    defined_inputs = [WorkflowImage(type="WorkflowImage", name="image1")]

    # when
    result = assemble_runtime_parameters(
        runtime_parameters=runtime_parameters,
        defined_inputs=defined_inputs,
    )

    # then
    assert (
        len(result["image1"]) == 2
    ), "All batche elements should be included in result"
    assert result["image1"][0].parent_metadata.parent_id == "image1.[0]"
    assert np.allclose(
        result["image1"][0].numpy_image, np.zeros((192, 168, 3), dtype=np.uint8)
    )
    assert result["image1"][1].parent_metadata.parent_id == "image1.[1]"
    assert np.allclose(
        result["image1"][1].numpy_image,
        np.ones((256, 256, 3), dtype=np.uint8),
    )


def test_assemble_runtime_parameters_when_image_is_provided_with_video_metadata() -> (
    None
):
    # given
    runtime_parameters = {
        "image1": [
            {
                "type": "numpy_object",
                "value": np.zeros((192, 168, 3), dtype=np.uint8),
                "video_metadata": {
                    "video_identifier": "some_id",
                    "frame_number": 37,
                    "frame_timestamp": datetime.now().isoformat(),
                    "fps": 35,
                },
            },
            {
                "type": "numpy_object",
                "value": np.ones((256, 256, 3), dtype=np.uint8),
                "video_metadata": VideoMetadata(
                    video_identifier="video_id",
                    frame_number=127,
                    frame_timestamp=datetime.now(),
                    fps=40,
                    comes_from_video_file=None,
                ),
            },
        ]
    }
    defined_inputs = [WorkflowImage(type="WorkflowImage", name="image1")]

    # when
    result = assemble_runtime_parameters(
        runtime_parameters=runtime_parameters,
        defined_inputs=defined_inputs,
    )

    # then
    assert (
        len(result["image1"]) == 2
    ), "All batche elements should be included in result"
    assert result["image1"][0].parent_metadata.parent_id == "image1.[0]"
    assert result["image1"][0].video_metadata.video_identifier == "some_id"
    assert np.allclose(
        result["image1"][0].numpy_image, np.zeros((192, 168, 3), dtype=np.uint8)
    )
    assert result["image1"][1].parent_metadata.parent_id == "image1.[1]"
    assert np.allclose(
        result["image1"][1].numpy_image,
        np.ones((256, 256, 3), dtype=np.uint8),
    )
    assert result["image1"][1].video_metadata.video_identifier == "video_id"


def test_assemble_runtime_parameters_when_parameter_not_provided() -> None:
    # given
    runtime_parameters = {}
    defined_inputs = [WorkflowParameter(type="WorkflowParameter", name="parameter")]

    # when
    result = assemble_runtime_parameters(
        runtime_parameters=runtime_parameters,
        defined_inputs=defined_inputs,
    )

    # then
    assert result["parameter"] is None


def test_assemble_runtime_parameters_when_parameter_provided() -> None:
    # given
    runtime_parameters = {"parameter": 37}
    defined_inputs = [WorkflowParameter(type="WorkflowParameter", name="parameter")]

    # when
    result = assemble_runtime_parameters(
        runtime_parameters=runtime_parameters,
        defined_inputs=defined_inputs,
    )

    # then
    assert result["parameter"] == 37


def test_assemble_runtime_parameters_when_images_with_different_matching_batch_sizes_provided() -> (
    None
):
    # given
    runtime_parameters = {
        "image1": [
            np.zeros((192, 168, 3), dtype=np.uint8),
            {
                "type": "numpy_object",
                "value": np.ones((256, 256, 3), dtype=np.uint8),
            },
        ],
        "image2": np.zeros((192, 168, 3), dtype=np.uint8),
    }
    defined_inputs = [
        WorkflowImage(type="WorkflowImage", name="image1"),
        WorkflowImage(type="WorkflowImage", name="image2"),
    ]

    # when
    result = assemble_runtime_parameters(
        runtime_parameters=runtime_parameters,
        defined_inputs=defined_inputs,
    )

    # then
    assert (
        len(result["image1"]) == 2
    ), "Expected 2 elements in batch of `image1` parameter"
    assert (
        len(result["image2"]) == 2
    ), "Expected 2 elements in batch of `image2` parameter - broadcasting should happen"
    assert np.allclose(
        result["image2"][0].numpy_image, np.zeros((192, 168, 3), dtype=np.uint8)
    ), "Empty image expected"
    assert np.allclose(
        result["image2"][1].numpy_image, np.zeros((192, 168, 3), dtype=np.uint8)
    ), "Empty image expected"


def test_assemble_runtime_parameters_when_images_with_different_and_not_matching_batch_sizes_provided() -> (
    None
):
    # given
    runtime_parameters = {
        "image1": [
            np.zeros((192, 168, 3), dtype=np.uint8),
            {
                "type": "numpy_object",
                "value": np.ones((256, 256, 3), dtype=np.uint8),
            },
        ],
        "image2": [np.zeros((192, 168, 3), dtype=np.uint8)] * 3,
    }
    defined_inputs = [
        WorkflowImage(type="WorkflowImage", name="image1"),
        WorkflowImage(type="WorkflowImage", name="image2"),
    ]

    # when
    with pytest.raises(RuntimeInputError):
        _ = assemble_runtime_parameters(
            runtime_parameters=runtime_parameters,
            defined_inputs=defined_inputs,
        )


def test_assemble_runtime_parameters_when_video_metadata_with_different_matching_batch_sizes_provided() -> (
    None
):
    # given
    runtime_parameters = {
        "meta1": [
            {
                "video_identifier": "a",
                "frame_number": 1,
                "frame_timestamp": datetime.now().isoformat(),
                "fps": 50,
                "comes_from_video_file": True,
            },
            {
                "video_identifier": "b",
                "frame_number": 1,
                "frame_timestamp": datetime.now().isoformat(),
                "fps": 50,
                "comes_from_video_file": True,
            },
        ],
        "meta2": {
            "video_identifier": "c",
            "frame_number": 1,
            "frame_timestamp": datetime.now().isoformat(),
            "fps": 50,
            "comes_from_video_file": True,
        },
    }
    defined_inputs = [
        WorkflowVideoMetadata(type="WorkflowVideoMetadata", name="meta1"),
        WorkflowVideoMetadata(type="WorkflowVideoMetadata", name="meta2"),
    ]

    # when
    result = assemble_runtime_parameters(
        runtime_parameters=runtime_parameters,
        defined_inputs=defined_inputs,
    )

    # then
    assert (
        len(result["meta1"]) == 2
    ), "Expected 2 elements in batch of `image1` parameter"
    assert (
        len(result["meta2"]) == 2
    ), "Expected 2 elements in batch of `image2` parameter - broadcasting should happen"
    assert [
        result["meta2"][0].video_identifier,
        result["meta2"][1].video_identifier,
    ] == ["c", "c"], "Expected broadcasting of meta2 value"


def test_assemble_runtime_parameters_when_video_metadata_declared_but_not_provided() -> (
    None
):
    # given
    defined_inputs = [WorkflowVideoMetadata(type="WorkflowVideoMetadata", name="meta1")]

    # when
    with pytest.raises(RuntimeInputError):
        _ = assemble_runtime_parameters(
            runtime_parameters={},
            defined_inputs=defined_inputs,
        )


@pytest.mark.parametrize(
    "timestamp", [1, time.time(), datetime.now(), datetime.now().isoformat()]
)
def test_assemble_runtime_parameters_when_video_metadata_declared_and_provided_as_dict(
    timestamp: Any,
) -> None:
    # given
    defined_inputs = [WorkflowVideoMetadata(type="WorkflowVideoMetadata", name="meta1")]
    runtime_parameters = {
        "meta1": {
            "video_identifier": "a",
            "frame_number": 1,
            "frame_timestamp": timestamp,
            "fps": 50,
            "comes_from_video_file": True,
        },
    }

    # when
    result = assemble_runtime_parameters(
        runtime_parameters=runtime_parameters,
        defined_inputs=defined_inputs,
    )

    # then
    assert result["meta1"][0].video_identifier == "a"
    assert isinstance(result["meta1"][0].frame_timestamp, datetime)


def test_assemble_runtime_parameters_when_video_metadata_declared_and_provided_as_object() -> (
    None
):
    # given
    defined_inputs = [WorkflowVideoMetadata(type="WorkflowVideoMetadata", name="meta1")]
    runtime_parameters = {
        "meta1": VideoMetadata(
            video_identifier="a",
            frame_number=1,
            frame_timestamp=datetime.now(),
            fps=50,
            comes_from_video_file=False,
        ),
    }

    # when
    result = assemble_runtime_parameters(
        runtime_parameters=runtime_parameters,
        defined_inputs=defined_inputs,
    )

    # then
    assert result["meta1"][0].video_identifier == "a"
    assert isinstance(result["meta1"][0].frame_timestamp, datetime)


def test_assemble_runtime_parameters_when_video_metadata_with_different_and_not_matching_batch_sizes_provided() -> (
    None
):
    # given
    runtime_parameters = {
        "meta1": [
            {
                "video_identifier": "a",
                "frame_number": 1,
                "frame_timestamp": datetime.now().isoformat(),
                "fps": 50,
                "comes_from_video_file": True,
            },
            {
                "video_identifier": "b",
                "frame_number": 1,
                "frame_timestamp": datetime.now().isoformat(),
                "fps": 50,
                "comes_from_video_file": True,
            },
        ],
        "meta2": [
            {
                "video_identifier": "c",
                "frame_number": 1,
                "frame_timestamp": datetime.now().isoformat(),
                "fps": 50,
                "comes_from_video_file": True,
            }
        ]
        * 3,
    }
    defined_inputs = [
        WorkflowVideoMetadata(type="WorkflowVideoMetadata", name="meta1"),
        WorkflowVideoMetadata(type="WorkflowVideoMetadata", name="meta2"),
    ]

    # when
    with pytest.raises(RuntimeInputError):
        _ = assemble_runtime_parameters(
            runtime_parameters=runtime_parameters,
            defined_inputs=defined_inputs,
        )
