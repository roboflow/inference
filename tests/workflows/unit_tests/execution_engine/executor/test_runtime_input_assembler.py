import time
from datetime import datetime
from typing import Any
from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import pytest

from inference.core.workflows.core_steps.common import deserializers
from inference.core.workflows.core_steps.loader import KINDS_DESERIALIZERS
from inference.core.workflows.errors import RuntimeInputError
from inference.core.workflows.execution_engine.entities.base import (
    VideoMetadata,
    WorkflowBatchInput,
    WorkflowImage,
    WorkflowImageData,
    WorkflowParameter,
    WorkflowVideoMetadata,
)
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    DICTIONARY_KIND,
    FLOAT_KIND,
    IMAGE_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
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
            kinds_deserializers=KINDS_DESERIALIZERS,
        )


@mock.patch.object(deserializers, "load_image_from_url")
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
        kinds_deserializers=KINDS_DESERIALIZERS,
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
        kinds_deserializers=KINDS_DESERIALIZERS,
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
            kinds_deserializers=KINDS_DESERIALIZERS,
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
        kinds_deserializers=KINDS_DESERIALIZERS,
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
            kinds_deserializers=KINDS_DESERIALIZERS,
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
        kinds_deserializers=KINDS_DESERIALIZERS,
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
        kinds_deserializers=KINDS_DESERIALIZERS,
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
        kinds_deserializers=KINDS_DESERIALIZERS,
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
        kinds_deserializers=KINDS_DESERIALIZERS,
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
        "image3": [np.zeros((192, 168, 3), dtype=np.uint8)],
    }
    defined_inputs = [
        WorkflowImage(type="WorkflowImage", name="image1"),
        WorkflowImage(type="WorkflowImage", name="image2"),
        WorkflowImage(type="WorkflowImage", name="image3"),
    ]

    # when
    result = assemble_runtime_parameters(
        runtime_parameters=runtime_parameters,
        defined_inputs=defined_inputs,
        kinds_deserializers=KINDS_DESERIALIZERS,
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
    assert np.allclose(
        result["image3"][0].numpy_image, np.zeros((192, 168, 3), dtype=np.uint8)
    ), "Empty image expected"
    assert np.allclose(
        result["image3"][1].numpy_image, np.zeros((192, 168, 3), dtype=np.uint8)
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
            kinds_deserializers=KINDS_DESERIALIZERS,
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
        kinds_deserializers=KINDS_DESERIALIZERS,
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
            kinds_deserializers=KINDS_DESERIALIZERS,
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
        kinds_deserializers=KINDS_DESERIALIZERS,
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
        kinds_deserializers=KINDS_DESERIALIZERS,
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
            kinds_deserializers=KINDS_DESERIALIZERS,
        )


def test_assemble_runtime_parameters_when_parameters_at_different_dimensionality_depth_emerge() -> (
    None
):
    # given
    runtime_parameters = {
        "image1": [
            np.zeros((192, 168, 3), dtype=np.uint8),
            np.zeros((192, 168, 3), dtype=np.uint8),
        ],
        "image2": [
            [
                np.zeros((192, 168, 3), dtype=np.uint8),
                np.zeros((192, 168, 3), dtype=np.uint8),
            ],
            [
                np.zeros((192, 168, 3), dtype=np.uint8),
            ],
        ],
        "image3": [
            [
                [np.zeros((192, 168, 3), dtype=np.uint8)],
                [
                    np.zeros((192, 168, 3), dtype=np.uint8),
                    np.zeros((192, 168, 3), dtype=np.uint8),
                ],
            ],
            [
                [np.zeros((192, 168, 3), dtype=np.uint8)],
                [
                    np.zeros((192, 168, 3), dtype=np.uint8),
                    np.zeros((192, 168, 3), dtype=np.uint8),
                ],
                [np.zeros((192, 168, 3), dtype=np.uint8)],
            ],
        ],
    }
    defined_inputs = [
        WorkflowBatchInput(type="WorkflowBatchInput", name="image1", kind=["image"]),
        WorkflowBatchInput(
            type="WorkflowBatchInput",
            name="image2",
            kind=[IMAGE_KIND],
            dimensionality=2,
        ),
        WorkflowBatchInput(
            type="WorkflowBatchInput", name="image3", kind=["image"], dimensionality=3
        ),
    ]

    # when
    result = assemble_runtime_parameters(
        runtime_parameters=runtime_parameters,
        defined_inputs=defined_inputs,
        kinds_deserializers=KINDS_DESERIALIZERS,
    )

    # then
    assert len(result["image1"]) == 2, "image1 is 1D batch of size (2, )"
    assert all(
        isinstance(e, WorkflowImageData) for e in result["image1"]
    ), "Expected deserialized image data at the bottom level of batch"
    # then
    sizes_of_image2 = [len(e) for e in result["image2"]]
    assert sizes_of_image2 == [2, 1], "image1 is 2D batch of size [(2, ), (1, )]"
    assert all(
        isinstance(e, WorkflowImageData)
        for nested_batch in result["image2"]
        for e in nested_batch
    ), "Expected deserialized image data at the bottom level of batch"
    sizes_of_image3 = [
        [len(e) for e in inner_batch] for inner_batch in result["image3"]
    ]
    assert sizes_of_image3 == [
        [1, 2],
        [1, 2, 1],
    ], "image1 is 3D batch of size [[(1, ), (2, )], [(1, ), (2, ), (1, )]]"
    assert all(
        isinstance(e, WorkflowImageData)
        for nested_batch in result["image3"]
        for inner_batch in nested_batch
        for e in inner_batch
    ), "Expected deserialized image data at the bottom level of batch"


def test_assemble_runtime_parameters_when_basic_types_are_passed_as_batch_oriented_inputs() -> (
    None
):
    # given
    runtime_parameters = {
        "string_param": ["a", "b"],
        "float_param": [1.0, 2.0],
        "int_param": [3, 4],
        "list_param": [["some", "list"], ["other", "list"]],
        "boolean_param": [False, True],
        "dict_param": [{"some": "dict"}, {"other": "dict"}],
    }
    defined_inputs = [
        WorkflowBatchInput(
            type="WorkflowBatchInput", name="string_param", kind=[STRING_KIND.name]
        ),
        WorkflowBatchInput(
            type="WorkflowBatchInput", name="float_param", kind=[FLOAT_KIND.name]
        ),
        WorkflowBatchInput(
            type="WorkflowBatchInput", name="int_param", kind=[INTEGER_KIND]
        ),
        WorkflowBatchInput(
            type="WorkflowBatchInput", name="list_param", kind=[LIST_OF_VALUES_KIND]
        ),
        WorkflowBatchInput(
            type="WorkflowBatchInput", name="boolean_param", kind=[BOOLEAN_KIND]
        ),
        WorkflowBatchInput(
            type="WorkflowBatchInput", name="dict_param", kind=[DICTIONARY_KIND]
        ),
    ]

    # when
    result = assemble_runtime_parameters(
        runtime_parameters=runtime_parameters,
        defined_inputs=defined_inputs,
        kinds_deserializers=KINDS_DESERIALIZERS,
    )

    # then
    assert result == {
        "string_param": ["a", "b"],
        "float_param": [1.0, 2.0],
        "int_param": [3, 4],
        "list_param": [["some", "list"], ["other", "list"]],
        "boolean_param": [False, True],
        "dict_param": [{"some": "dict"}, {"other": "dict"}],
    }, "Expected values not to be changed"


def test_assemble_runtime_parameters_when_input_batch_shallower_than_declared() -> None:
    # given
    runtime_parameters = {
        "string_param": ["a", "b"],
        "float_param": [1.0, 2.0],
    }
    defined_inputs = [
        WorkflowBatchInput(
            type="WorkflowBatchInput", name="string_param", kind=[STRING_KIND.name]
        ),
        WorkflowBatchInput(
            type="WorkflowBatchInput",
            name="float_param",
            kind=[FLOAT_KIND.name],
            dimensionality=2,
        ),
    ]

    # when
    with pytest.raises(RuntimeInputError):
        _ = assemble_runtime_parameters(
            runtime_parameters=runtime_parameters,
            defined_inputs=defined_inputs,
            kinds_deserializers=KINDS_DESERIALIZERS,
        )


def test_assemble_runtime_parameters_when_input_batch_deeper_than_declared() -> None:
    # given
    runtime_parameters = {
        "string_param": ["a", "b"],
        "float_param": [[1.0], [2.0]],
    }
    defined_inputs = [
        WorkflowBatchInput(
            type="WorkflowBatchInput", name="string_param", kind=[STRING_KIND.name]
        ),
        WorkflowBatchInput(
            type="WorkflowBatchInput", name="float_param", kind=[FLOAT_KIND.name]
        ),
    ]

    # when
    with pytest.raises(RuntimeInputError):
        _ = assemble_runtime_parameters(
            runtime_parameters=runtime_parameters,
            defined_inputs=defined_inputs,
            kinds_deserializers=KINDS_DESERIALIZERS,
        )
