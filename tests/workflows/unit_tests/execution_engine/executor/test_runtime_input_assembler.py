from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import pytest

from inference.core.workflows.errors import RuntimeInputError
from inference.core.workflows.execution_engine.entities.base import (
    WorkflowImage,
    WorkflowParameter,
)
from inference.core.workflows.execution_engine.v1.executor import (
    runtime_input_assembler,
)
from inference.core.workflows.execution_engine.v1.executor.runtime_input_assembler import (
    assembly_runtime_parameters,
)


def test_assembly_runtime_parameters_when_image_is_not_provided() -> None:
    # given
    runtime_parameters = {}
    defined_inputs = [WorkflowImage(type="WorkflowImage", name="image")]

    # when
    with pytest.raises(RuntimeInputError):
        _ = assembly_runtime_parameters(
            runtime_parameters=runtime_parameters,
            defined_inputs=defined_inputs,
        )


@mock.patch.object(runtime_input_assembler, "load_image_from_url")
def test_assembly_runtime_parameters_when_image_is_provided_as_single_element_dict(
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
    result = assembly_runtime_parameters(
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


def test_assembly_runtime_parameters_when_image_is_provided_as_single_element_dict_pointing_local_file_when_load_of_local_files_allowed(
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
    result = assembly_runtime_parameters(
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


def test_assembly_runtime_parameters_when_image_is_provided_as_single_element_dict_pointing_local_file_when_load_of_local_files_not_allowed(
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
        _ = assembly_runtime_parameters(
            runtime_parameters=runtime_parameters,
            defined_inputs=defined_inputs,
            prevent_local_images_loading=True,
        )


def test_assembly_runtime_parameters_when_image_is_provided_as_single_element_np_array() -> (
    None
):
    # given
    runtime_parameters = {"image1": np.zeros((192, 168, 3), dtype=np.uint8)}
    defined_inputs = [WorkflowImage(type="WorkflowImage", name="image1")]

    # when
    result = assembly_runtime_parameters(
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


def test_assembly_runtime_parameters_when_image_is_provided_as_unknown_element() -> (
    None
):
    # given
    runtime_parameters = {"image1": "some"}
    defined_inputs = [WorkflowImage(type="WorkflowImage", name="image1")]

    # when
    with pytest.raises(RuntimeInputError):
        _ = assembly_runtime_parameters(
            runtime_parameters=runtime_parameters,
            defined_inputs=defined_inputs,
        )


def test_assembly_runtime_parameters_when_image_is_provided_in_batch() -> None:
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
    result = assembly_runtime_parameters(
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


def test_assembly_runtime_parameters_when_parameter_not_provided() -> None:
    # given
    runtime_parameters = {}
    defined_inputs = [WorkflowParameter(type="WorkflowParameter", name="parameter")]

    # when
    result = assembly_runtime_parameters(
        runtime_parameters=runtime_parameters,
        defined_inputs=defined_inputs,
    )

    # then
    assert result["parameter"] is None


def test_assembly_runtime_parameters_when_parameter_provided() -> None:
    # given
    runtime_parameters = {"parameter": 37}
    defined_inputs = [WorkflowParameter(type="WorkflowParameter", name="parameter")]

    # when
    result = assembly_runtime_parameters(
        runtime_parameters=runtime_parameters,
        defined_inputs=defined_inputs,
    )

    # then
    assert result["parameter"] == 37
