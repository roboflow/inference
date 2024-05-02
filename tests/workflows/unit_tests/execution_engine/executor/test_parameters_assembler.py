import pytest

from inference.enterprise.workflows.core_steps.fusion.detections_consensus import (
    AggregationMode,
    BlockManifest,
)
from inference.enterprise.workflows.entities.base import OutputDefinition
from inference.enterprise.workflows.entities.types import (
    BATCH_OF_OBJECT_DETECTION_PREDICTION_KIND,
    BATCH_OF_STRING_KIND,
    IMAGE_METADATA_KIND,
    STRING_KIND,
)
from inference.enterprise.workflows.errors import (
    ExecutionEngineNotImplementedError,
    ExecutionEngineRuntimeError,
)
from inference.enterprise.workflows.execution_engine.executor.execution_cache import (
    ExecutionCache,
)
from inference.enterprise.workflows.execution_engine.executor.parameters_assembler import (
    assembly_step_parameters,
    retrieve_step_output,
    retrieve_value_from_runtime_input,
)


def test_retrieve_value_from_runtime_input_when_miss_detected() -> None:
    # when
    with pytest.raises(ExecutionEngineRuntimeError):
        _ = retrieve_value_from_runtime_input(
            selector="$inputs.param",
            runtime_parameters={},
            accepts_batch_input=False,
            step_name="my_step",
        )


def test_retrieve_value_from_runtime_input_when_value_can_be_provided() -> None:
    # when
    result = retrieve_value_from_runtime_input(
        selector="$inputs.param",
        runtime_parameters={
            "param": [3, 4],
        },
        accepts_batch_input=False,
        step_name="my_step",
    )

    # then
    assert result == [3, 4]


def test_retrieve_value_from_runtime_input_when_value_is_batch_of_images_which_is_not_supported() -> (
    None
):
    # when
    with pytest.raises(ExecutionEngineNotImplementedError):
        _ = retrieve_value_from_runtime_input(
            selector="$inputs.param",
            runtime_parameters={
                "param": [
                    {
                        "type": "url",
                        "value": "https://some.com/image1.jpg",
                    },
                    {
                        "type": "url",
                        "value": "https://some.com/image2.jpg",
                    },
                ],
            },
            accepts_batch_input=False,
            step_name="my_step",
        )


def test_retrieve_value_from_runtime_input_when_value_is_image_and_batch_input_is_not_supported() -> (
    None
):
    # when
    result = retrieve_value_from_runtime_input(
        selector="$inputs.param",
        runtime_parameters={
            "param": [
                {
                    "type": "url",
                    "value": "https://some.com/image1.jpg",
                }
            ],
        },
        accepts_batch_input=False,
        step_name="my_step",
    )

    # then
    assert result == {
        "type": "url",
        "value": "https://some.com/image1.jpg",
    }


def test_retrieve_value_from_runtime_input_when_value_is_image_and_batch_input_is_supported() -> (
    None
):
    # when
    result = retrieve_value_from_runtime_input(
        selector="$inputs.param",
        runtime_parameters={
            "param": [
                {
                    "type": "url",
                    "value": "https://some.com/image1.jpg",
                }
            ],
        },
        accepts_batch_input=True,
        step_name="my_step",
    )

    # then
    assert result == [
        {
            "type": "url",
            "value": "https://some.com/image1.jpg",
        }
    ]


def test_retrieve_value_from_runtime_input_when_value_is_batch_of_images_and_batch_input_is_supported() -> (
    None
):
    # when
    result = retrieve_value_from_runtime_input(
        selector="$inputs.param",
        runtime_parameters={
            "param": [
                {
                    "type": "url",
                    "value": "https://some.com/image1.jpg",
                },
                {
                    "type": "url",
                    "value": "https://some.com/image2.jpg",
                },
            ],
        },
        accepts_batch_input=True,
        step_name="my_step",
    )

    # then
    assert result == [
        {
            "type": "url",
            "value": "https://some.com/image1.jpg",
        },
        {
            "type": "url",
            "value": "https://some.com/image2.jpg",
        },
    ]


def test_retrieve_step_output_when_output_not_registered() -> None:
    # given
    execution_cache = ExecutionCache.init()

    # when
    with pytest.raises(ExecutionEngineRuntimeError):
        _ = retrieve_step_output(
            selector="$steps.some.output",
            execution_cache=execution_cache,
            accepts_batch_input=True,
            step_name="my_step",
        )


def test_retrieve_step_output_when_non_batch_output_registered_and_input_compatible_with_batches() -> (
    None
):
    # given
    execution_cache = ExecutionCache.init()
    execution_cache.register_step(
        step_name="some",
        output_definitions=[OutputDefinition(name="output", kind=[STRING_KIND])],
        compatible_with_batches=False,
    )
    execution_cache.register_step_outputs(
        step_name="some", outputs=[{"output": "value"}]
    )

    # when
    result = retrieve_step_output(
        selector="$steps.some.output",
        execution_cache=execution_cache,
        accepts_batch_input=True,
        step_name="my_step",
    )

    # then
    assert result == "value"


def test_retrieve_step_output_when_non_batch_output_registered_and_input_not_compatible_with_batches() -> (
    None
):
    # given
    execution_cache = ExecutionCache.init()
    execution_cache.register_step(
        step_name="some",
        output_definitions=[OutputDefinition(name="output", kind=[STRING_KIND])],
        compatible_with_batches=False,
    )
    execution_cache.register_step_outputs(
        step_name="some", outputs=[{"output": "value"}]
    )

    # when
    result = retrieve_step_output(
        selector="$steps.some.output",
        execution_cache=execution_cache,
        accepts_batch_input=False,
        step_name="my_step",
    )

    # then
    assert result == "value"


def test_retrieve_step_output_when_batch_output_registered_of_size_one_and_input_not_compatible_with_batches() -> (
    None
):
    # given
    execution_cache = ExecutionCache.init()
    execution_cache.register_step(
        step_name="some",
        output_definitions=[
            OutputDefinition(name="output", kind=[BATCH_OF_STRING_KIND])
        ],
        compatible_with_batches=True,
    )
    execution_cache.register_step_outputs(
        step_name="some", outputs=[{"output": "value"}]
    )

    # when
    result = retrieve_step_output(
        selector="$steps.some.output",
        execution_cache=execution_cache,
        accepts_batch_input=False,
        step_name="my_step",
    )

    # then
    assert result == "value"


def test_retrieve_step_output_when_batch_output_registered_input_not_compatible_with_batches() -> (
    None
):
    # given
    execution_cache = ExecutionCache.init()
    execution_cache.register_step(
        step_name="some",
        output_definitions=[
            OutputDefinition(name="output", kind=[BATCH_OF_STRING_KIND])
        ],
        compatible_with_batches=True,
    )
    execution_cache.register_step_outputs(
        step_name="some", outputs=[{"output": "value_1"}, {"output": "value_2"}]
    )

    # when
    with pytest.raises(ExecutionEngineNotImplementedError):
        _ = retrieve_step_output(
            selector="$steps.some.output",
            execution_cache=execution_cache,
            accepts_batch_input=False,
            step_name="my_step",
        )


def test_retrieve_step_output_when_batch_output_registered_input_compatible_with_batches() -> (
    None
):
    # given
    execution_cache = ExecutionCache.init()
    execution_cache.register_step(
        step_name="some",
        output_definitions=[
            OutputDefinition(name="output", kind=[BATCH_OF_STRING_KIND])
        ],
        compatible_with_batches=True,
    )
    execution_cache.register_step_outputs(
        step_name="some", outputs=[{"output": "value_1"}, {"output": "value_2"}]
    )

    # when
    result = retrieve_step_output(
        selector="$steps.some.output",
        execution_cache=execution_cache,
        accepts_batch_input=True,
        step_name="my_step",
    )

    # then
    assert result == ["value_1", "value_2"]


def test_assembly_step_parameters() -> None:
    # given
    execution_cache = ExecutionCache.init()
    execution_cache.register_step(
        step_name="some",
        output_definitions=[
            OutputDefinition(
                name="predictions", kind=[BATCH_OF_OBJECT_DETECTION_PREDICTION_KIND]
            ),
            OutputDefinition(name="image_metadata", kind=[IMAGE_METADATA_KIND]),
        ],
        compatible_with_batches=True,
    )
    execution_cache.register_step_outputs(
        step_name="some",
        outputs=[
            {"predictions": "a", "image_metadata": 1},
            {"predictions": "b", "image_metadata": 2},
        ],
    )
    manifest = BlockManifest(
        type="DetectionsConsensus",
        name="my_step",
        predictions=[
            "$steps.some.predictions",
            "$steps.some.predictions",
        ],
        image_metadata="$steps.some.image_metadata",
        required_votes=1,
        iou_threshold="$inputs.iou_threshold",
    )

    # when
    result = assembly_step_parameters(
        step_manifest=manifest,
        runtime_parameters={"iou_threshold": 0.9},
        execution_cache=execution_cache,
        accepts_batch_input=True,
    )

    # then
    assert result["predictions"] == [
        ["a", "b"],
        ["a", "b"],
    ], "Expected to see 2x dummy predictions"
    assert result["image_metadata"] == [1, 2], "Expected to see dummy image metadata"
    assert result["required_votes"] == 1, "Expected to see default value for block"
    assert result["class_aware"] is True, "Expected to see default value for block"
    assert (
        abs(result["iou_threshold"] - 0.9) < 1e-5
    ), "Expected to see value provided in runtime parameters"
    assert abs(result["confidence"]) < 1e-5, "Expected to see default value for block"
    assert (
        result["classes_to_consider"] is None
    ), "Expected to see default value for block"
    assert result["required_objects"] is None, "Expected to see default value for block"
    assert (
        result["presence_confidence_aggregation"] is AggregationMode.MAX
    ), "Expected to see default value for block"
    assert (
        result["detections_merge_confidence_aggregation"] is AggregationMode.AVERAGE
    ), "Expected to see default value for block"
    assert (
        result["detections_merge_coordinates_aggregation"] is AggregationMode.AVERAGE
    ), "Expected to see default value for block"
