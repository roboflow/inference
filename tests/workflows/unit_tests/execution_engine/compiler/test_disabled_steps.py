from inference.core.workflows.execution_engine.introspection.blocks_loader import (
    load_workflow_blocks,
)
from inference.core.workflows.execution_engine.v1.compiler.disabled_steps import (
    strip_disabled_steps,
)


def _definition(disabled_nodes):
    return {
        "version": "1.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "steps": [
            {
                "type": "roboflow_core/roboflow_object_detection_model@v2",
                "name": "crop_model",
                "image": "$inputs.image",
                "model_id": "some/1",
            },
            {
                "type": "roboflow_core/dynamic_crop@v1",
                "name": "crops",
                "image": "$inputs.image",
                "predictions": "$steps.crop_model.predictions",
            },
            {
                "type": "roboflow_core/roboflow_object_detection_model@v2",
                "name": "main_model",
                "image": "$inputs.image",
                "model_id": "other/1",
            },
        ],
        "outputs": [
            {"type": "JsonField", "name": "crops", "selector": "$steps.crops.crops"},
            {
                "type": "JsonField",
                "name": "predictions",
                "selector": "$steps.main_model.predictions",
            },
        ],
        "metadata": {"ui": {"nodes": disabled_nodes}},
    }


def test_strip_disabled_steps_is_noop_when_nothing_disabled() -> None:
    # given
    definition = _definition(disabled_nodes={})

    # when
    result = strip_disabled_steps(definition, available_blocks=load_workflow_blocks())

    # then
    assert result is definition


def test_strip_disabled_steps_removes_step_and_cascades_to_dependants() -> None:
    # given
    definition = _definition(disabled_nodes={"$steps.crop_model": {"disabled": True}})

    # when
    result = strip_disabled_steps(definition, available_blocks=load_workflow_blocks())

    # then
    assert [step["name"] for step in result["steps"]] == ["main_model"]
    assert [output["name"] for output in result["outputs"]] == ["predictions"]
    # input definition must not be mutated
    assert len(definition["steps"]) == 3


def test_strip_disabled_steps_ignores_non_step_nodes_and_false_flags() -> None:
    # given
    definition = _definition(
        disabled_nodes={
            "$inputs.image": {"disabled": True},
            "$steps.crop_model": {"disabled": False},
        }
    )

    # when
    result = strip_disabled_steps(definition, available_blocks=load_workflow_blocks())

    # then
    assert result is definition


def test_strip_disabled_steps_removes_reference_from_optional_list_without_cascade() -> (
    None
):
    # given
    definition = {
        "version": "1.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "steps": [
            {
                "type": "roboflow_core/roboflow_object_detection_model@v2",
                "name": "a",
                "image": "$inputs.image",
                "model_id": "some/1",
            },
            {
                "type": "roboflow_core/roboflow_object_detection_model@v2",
                "name": "b",
                "image": "$inputs.image",
                "model_id": "some/2",
            },
            {
                "type": "roboflow_core/detections_consensus@v1",
                "name": "consensus",
                "predictions_batches": ["$steps.a.predictions", "$steps.b.predictions"],
                "required_votes": 1,
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "result",
                "selector": "$steps.consensus.predictions",
            }
        ],
        "metadata": {"ui": {"nodes": {"$steps.a": {"disabled": True}}}},
    }

    # when
    result = strip_disabled_steps(definition, available_blocks=load_workflow_blocks())

    # then
    assert [step["name"] for step in result["steps"]] == ["b", "consensus"]
    assert result["steps"][1]["predictions_batches"] == ["$steps.b.predictions"]
    assert len(result["outputs"]) == 1


def test_strip_disabled_steps_cascades_to_steps_gated_only_by_disabled_conditional() -> (
    None
):
    # given
    definition = {
        "version": "1.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "steps": [
            {
                "type": "roboflow_core/roboflow_object_detection_model@v2",
                "name": "model",
                "image": "$inputs.image",
                "model_id": "some/1",
            },
            {
                "type": "roboflow_core/continue_if@v1",
                "name": "gate",
                "condition_statement": {
                    "type": "StatementGroup",
                    "statements": [
                        {
                            "type": "BinaryStatement",
                            "left_operand": {
                                "type": "DynamicOperand",
                                "operand_name": "predictions",
                            },
                            "comparator": {"type": "(Detections) not empty"},
                            "right_operand": {"type": "StaticOperand", "value": None},
                        }
                    ],
                },
                "evaluation_parameters": {"predictions": "$steps.model.predictions"},
                "next_steps": ["$steps.second_model"],
            },
            {
                "type": "roboflow_core/roboflow_object_detection_model@v2",
                "name": "second_model",
                "image": "$inputs.image",
                "model_id": "some/2",
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "second",
                "selector": "$steps.second_model.predictions",
            }
        ],
        "metadata": {"ui": {"nodes": {"$steps.gate": {"disabled": True}}}},
    }

    # when
    result = strip_disabled_steps(definition, available_blocks=load_workflow_blocks())

    # then
    assert [step["name"] for step in result["steps"]] == ["model"]
    assert result["outputs"] == []
