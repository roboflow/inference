"""
Post-inlining JSON shape for the nested workflow used in integration tests under
``inner_workflow_inlining/test_inner_workflow_continue_if_inside_inner_with_crop_batch_lineage.py``.

Workflow dicts here are aligned with that module's ``_inner_continue_if_then_pick`` and
``_nested_workflow`` helpers so the integration scenario and this structural check stay in sync.
"""

from __future__ import annotations

import copy
from unittest import mock

from inference.core.workflows.execution_engine.introspection import blocks_loader
from inference.core.workflows.execution_engine.introspection.blocks_loader import (
    load_workflow_blocks,
)
from inference.core.workflows.execution_engine.v1.inner_workflow.compiler_bridge import (
    validate_inner_workflow_composition_from_raw_workflow_definition,
)
from inference.core.workflows.execution_engine.v1.inner_workflow.inline import (
    inline_inner_workflow_steps,
)
from inference.core.workflows.execution_engine.v1.inner_workflow.reference_resolution import (
    normalize_inner_workflow_references_in_definition,
)

_SCALAR_ONLY_ECHO_PLUGIN = (
    "tests.workflows.integration_tests.execution.stub_plugins.scalar_only_block_plugin"
)

_CLASSIFICATION_REQUEST_CONFIDENCE_THRESHOLD = 0.2
_CONTINUE_IF_CONFIDENCE_THRESHOLD = 0.5


def _inner_continue_if_then_pick() -> dict:
    return {
        "version": "1.0",
        "inputs": [
            {
                "type": "WorkflowBatchInput",
                "name": "crops",
                "kind": ["image"],
                "dimensionality": 2,
            },
            {
                "type": "WorkflowParameter",
                "name": "crop_label",
                "default_value": "",
            },
        ],
        "steps": [
            {
                "type": "roboflow_core/roboflow_classification_model@v2",
                "name": "breds_classification",
                "image": "$inputs.crops",
                "model_id": "dog-breed/1",
                "confidence": _CLASSIFICATION_REQUEST_CONFIDENCE_THRESHOLD,
            },
            {
                "type": "roboflow_core/continue_if@v1",
                "name": "continue_if",
                "condition_statement": {
                    "type": "StatementGroup",
                    "statements": [
                        {
                            "type": "BinaryStatement",
                            "left_operand": {
                                "type": "DynamicOperand",
                                "operand_name": "predictions",
                                "operations": [
                                    {
                                        "type": "ClassificationPropertyExtract",
                                        "property_name": "top_class_confidence",
                                    }
                                ],
                            },
                            "comparator": {"type": "(Number) >="},
                            "right_operand": {
                                "type": "StaticOperand",
                                "value": _CONTINUE_IF_CONFIDENCE_THRESHOLD,
                            },
                        }
                    ],
                },
                "evaluation_parameters": {
                    "predictions": "$steps.breds_classification.predictions",
                },
                "next_steps": ["$steps.echo"],
            },
            {
                "type": "scalar_only_echo",
                "name": "echo",
                "value": "$inputs.crop_label",
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "echo",
                "selector": "$steps.echo.output",
            },
        ],
    }


def _nested_workflow(inner: dict) -> dict:
    return {
        "version": "1.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
            {"type": "WorkflowParameter", "name": "crop_label"},
        ],
        "steps": [
            {
                "type": "roboflow_core/roboflow_object_detection_model@v2",
                "name": "general_detection",
                "image": "$inputs.image",
                "model_id": "yolov8n-640",
                "class_filter": ["dog"],
            },
            {
                "type": "roboflow_core/dynamic_crop@v1",
                "name": "cropping",
                "image": "$inputs.image",
                "predictions": "$steps.general_detection.predictions",
            },
            {
                "type": "roboflow_core/inner_workflow@v1",
                "name": "nested_inner_workflow",
                "workflow_definition": inner,
                "parameter_bindings": {
                    "crops": "$steps.cropping.crops",
                    "crop_label": "$inputs.crop_label",
                },
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "from_child",
                "selector": "$steps.nested_inner_workflow.echo",
            },
        ],
    }


def test_nested_continue_if_inner_workflow_inlined_raw_definition_shape() -> None:
    """Pins the raw dict after ``inline_inner_workflow_steps`` (same order as ``compile_workflow_graph``)."""
    init_parameters = {"workflows_core.api_key": None}
    inner = _inner_continue_if_then_pick()
    raw = _nested_workflow(inner)

    with mock.patch.object(
        blocks_loader,
        "get_plugin_modules",
        return_value=[_SCALAR_ONLY_ECHO_PLUGIN],
    ):
        blocks_loader.clear_caches()
        try:
            normalized = normalize_inner_workflow_references_in_definition(
                workflow_definition=copy.deepcopy(raw),
                init_parameters=init_parameters,
            )
            validate_inner_workflow_composition_from_raw_workflow_definition(normalized)
            available_blocks = load_workflow_blocks(
                execution_engine_version=None,
                profiler=None,
            )
            inlined = inline_inner_workflow_steps(
                copy.deepcopy(normalized),
                available_blocks=available_blocks,
                profiler=None,
            )
        finally:
            blocks_loader.clear_caches()

    assert inlined["version"] == raw["version"]
    assert inlined["inputs"] == raw["inputs"]
    assert inlined["outputs"] == [
        {
            "type": "JsonField",
            "name": "from_child",
            "selector": "$steps.nested_inner_workflow__echo.output",
        }
    ]

    by_name = {s["name"]: s for s in inlined["steps"]}
    assert set(by_name) == {
        "general_detection",
        "cropping",
        "nested_inner_workflow__breds_classification",
        "nested_inner_workflow__continue_if",
        "nested_inner_workflow__echo",
    }
    assert not any(
        s.get("type") == "roboflow_core/inner_workflow@v1" for s in inlined["steps"]
    )

    assert by_name["general_detection"] == raw["steps"][0]
    assert by_name["cropping"] == raw["steps"][1]

    inner_breds = inner["steps"][0]
    inner_continue_if = inner["steps"][1]
    inner_echo = inner["steps"][2]
    inlined_breds = by_name["nested_inner_workflow__breds_classification"]
    inlined_continue_if = by_name["nested_inner_workflow__continue_if"]
    inlined_echo = by_name["nested_inner_workflow__echo"]

    assert inlined_breds["type"] == inner_breds["type"]
    assert inlined_breds["model_id"] == inner_breds["model_id"]
    assert inlined_breds["confidence"] == inner_breds["confidence"]
    assert inlined_breds["image"] == "$steps.cropping.crops"

    assert inlined_continue_if["type"] == inner_continue_if["type"]
    assert (
        inlined_continue_if["condition_statement"]
        == inner_continue_if["condition_statement"]
    )
    assert inlined_continue_if["evaluation_parameters"] == {
        "predictions": "$steps.nested_inner_workflow__breds_classification.predictions",
    }
    assert inlined_continue_if["next_steps"] == ["$steps.nested_inner_workflow__echo"]

    assert inlined_echo["type"] == inner_echo["type"]
    assert inlined_echo["value"] == "$inputs.crop_label"
