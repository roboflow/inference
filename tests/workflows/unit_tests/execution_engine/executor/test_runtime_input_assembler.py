import numpy as np
import pytest

from inference.core.workflows.entities.base import WorkflowImage, WorkflowParameter
from inference.core.workflows.errors import RuntimeInputError
from inference.core.workflows.execution_engine.executor.runtime_input_assembler import (
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


def test_assembly_runtime_parameters_when_image_is_provided_as_single_element_dict() -> (
    None
):
    # given
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
    assert result["image1"] == [
        {"type": "url", "value": "https://some.com/image.jpg", "parent_id": "image1"}
    ]


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
    assert result["image1"][0]["type"] == "numpy_object"
    assert result["image1"][0]["parent_id"] == "image1"
    assert np.allclose(
        result["image1"][0]["value"], np.zeros((192, 168, 3), dtype=np.uint8)
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
                "type": "url",
                "value": "https://some.com/image.jpg",
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
    assert result["image1"][0]["type"] == "numpy_object"
    assert result["image1"][0]["parent_id"] == "image1.[0]"
    assert np.allclose(
        result["image1"][0]["value"], np.zeros((192, 168, 3), dtype=np.uint8)
    )
    assert result["image1"][1] == {
        "type": "url",
        "value": "https://some.com/image.jpg",
        "parent_id": "image1.[1]",
    }


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
