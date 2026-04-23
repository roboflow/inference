"""
Equivalence like ``test_inner_workflow_after_continue_if_with_crop_batch_lineage``,
but ``roboflow_core/continue_if@v1`` lives in the inner workflow while the parent still
runs detection, crop, and classification.

One test keeps ``scalar_only_echo`` in the child after ``continue_if``; another ends the
child on ``continue_if`` only and runs ``scalar_only_echo`` in the parent, wiring
``next_steps`` to ``$steps.echo`` so control flow reaches the outer step after inlining.
"""

from unittest import mock

import numpy as np

from inference.core.entities.requests.inference import (
    ClassificationInferenceRequest,
    ObjectDetectionInferenceRequest,
)
from inference.core.entities.responses.inference import (
    ClassificationInferenceResponse,
    ClassificationPrediction,
    InferenceResponseImage,
    ObjectDetectionInferenceResponse,
    ObjectDetectionPrediction,
)
from inference.core.managers.base import ModelManager
from tests.workflows.integration_tests.execution.inner_workflow_inlining._common import (
    execution_engine,
)

_CLASSIFICATION_REQUEST_CONFIDENCE_THRESHOLD = 0.2
_CONTINUE_IF_CONFIDENCE_THRESHOLD = 0.5


def _infer_from_request_sync_factory(h: int, w: int):
    def infer_from_request_sync(model_id: str, request):
        bbox_w, bbox_h = 100, 100

        if isinstance(request, ObjectDetectionInferenceRequest):
            assert model_id == "yolov8n-640"
            imgs = request.image if isinstance(request.image, list) else [request.image]
            assert len(imgs) == 1
            assert request.class_filter == ["dog"]

            return ObjectDetectionInferenceResponse(
                image=InferenceResponseImage(width=w, height=h),
                predictions=[
                    ObjectDetectionPrediction(
                        x=80,
                        y=80,
                        width=bbox_w,
                        height=bbox_h,
                        confidence=0.99,
                        class_id=0,
                        detection_id="mock-d0",
                        parent_id="root",
                        **{"class": "dog"},
                    ),
                    ObjectDetectionPrediction(
                        x=300,
                        y=80,
                        width=bbox_w,
                        height=bbox_h,
                        confidence=0.99,
                        class_id=0,
                        detection_id="mock-d1",
                        parent_id="root",
                        **{"class": "dog"},
                    ),
                ],
            )

        if isinstance(request, ClassificationInferenceRequest):
            assert model_id == "dog-breed/1"
            assert request.confidence == _CLASSIFICATION_REQUEST_CONFIDENCE_THRESHOLD
            imgs = request.image if isinstance(request.image, list) else [request.image]
            assert len(imgs) == 2

            return [
                ClassificationInferenceResponse(
                    image=InferenceResponseImage(width=bbox_w, height=bbox_h),
                    predictions=[
                        ClassificationPrediction(
                            class_id=0,
                            confidence=0.9,
                            **{"class": "dog"},
                        ),
                    ],
                    top="dog",
                    confidence=0.9,
                ),
                ClassificationInferenceResponse(
                    image=InferenceResponseImage(width=bbox_w, height=bbox_h),
                    predictions=[
                        ClassificationPrediction(
                            class_id=1,
                            confidence=0.4,
                            **{"class": "dog"},
                        ),
                    ],
                    top="dog",
                    confidence=0.4,
                ),
            ]

        raise AssertionError(f"Unexpected request type: {type(request)!r}")

    return infer_from_request_sync


def _inner_continue_if_then_pick() -> dict:
    return {
        "version": "1.0",
        "inputs": [
            {
                "type": "WorkflowBatchInput",
                "name": "classification_predictions",
                "kind": ["classification_prediction"],
                "dimensionality": 1,
            },
            {
                "type": "WorkflowParameter",
                "name": "crop_label",
                "default_value": "",
            },
        ],
        "steps": [
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
                    "predictions": "$inputs.classification_predictions",
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


def _inner_continue_if_only_outer_echo_name_matches_parent() -> dict:
    """
    Child ends with ``continue_if`` only; ``next_steps`` names the parent's ``echo`` step
    (present only after inlining next to the expanded inner graph).
    """
    return {
        "version": "1.0",
        "inputs": [
            {
                "type": "WorkflowBatchInput",
                "name": "classification_predictions",
                "kind": ["classification_prediction"],
                "dimensionality": 1,
            },
        ],
        "steps": [
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
                    "predictions": "$inputs.classification_predictions",
                },
                "next_steps": ["$steps.echo"],
            },
        ],
        "outputs": [],
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
                "type": "roboflow_core/roboflow_object_detection_model@v3",
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
                "type": "roboflow_core/roboflow_classification_model@v3",
                "name": "breds_classification",
                "image": "$steps.cropping.crops",
                "model_id": "dog-breed/1",
                "confidence_mode": "custom",
                "custom_confidence": _CLASSIFICATION_REQUEST_CONFIDENCE_THRESHOLD,
            },
            {
                "type": "roboflow_core/inner_workflow@v1",
                "name": "nested_inner_workflow",
                "workflow_definition": inner,
                "parameter_bindings": {
                    "classification_predictions": "$steps.breds_classification.predictions",
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


def _nested_workflow_continue_if_inner_echo_outer(inner: dict) -> dict:
    """Like ``_nested_workflow`` but ``scalar_only_echo`` is a sibling step named ``echo``."""
    return {
        "version": "1.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
            {"type": "WorkflowParameter", "name": "crop_label"},
        ],
        "steps": [
            {
                "type": "roboflow_core/roboflow_object_detection_model@v3",
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
                "type": "roboflow_core/roboflow_classification_model@v3",
                "name": "breds_classification",
                "image": "$steps.cropping.crops",
                "model_id": "dog-breed/1",
                "confidence_mode": "custom",
                "custom_confidence": _CLASSIFICATION_REQUEST_CONFIDENCE_THRESHOLD,
            },
            {
                "type": "roboflow_core/inner_workflow@v1",
                "name": "nested_inner_workflow",
                "workflow_definition": inner,
                "parameter_bindings": {
                    "classification_predictions": "$steps.breds_classification.predictions",
                },
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
                "name": "from_child",
                "selector": "$steps.echo.output",
            },
        ],
    }


def _flat_workflow() -> dict:
    return {
        "version": "1.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
            {"type": "WorkflowParameter", "name": "crop_label"},
        ],
        "steps": [
            {
                "type": "roboflow_core/roboflow_object_detection_model@v3",
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
                "type": "roboflow_core/roboflow_classification_model@v3",
                "name": "breds_classification",
                "image": "$steps.cropping.crops",
                "model_id": "dog-breed/1",
                "confidence_mode": "custom",
                "custom_confidence": _CLASSIFICATION_REQUEST_CONFIDENCE_THRESHOLD,
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
                "name": "from_child",
                "selector": "$steps.echo.output",
            },
        ],
    }


def test_inlined_continue_if_inside_inner_matches_flat_workflow(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
) -> None:
    h, w = dogs_image.shape[:2]
    infer_mock = mock.MagicMock(
        side_effect=_infer_from_request_sync_factory(h, w),
    )
    inner = _inner_continue_if_then_pick()

    with mock.patch.object(ModelManager, "add_model"), mock.patch.object(
        ModelManager,
        "infer_from_request_sync",
        new=infer_mock,
    ):
        nested_engine = execution_engine(model_manager, _nested_workflow(inner))
        flat_engine = execution_engine(model_manager, _flat_workflow())
        nested_result = nested_engine.run(
            runtime_parameters={
                "image": dogs_image,
                "crop_label": "passed",
            },
        )
        flat_result = flat_engine.run(
            runtime_parameters={
                "image": dogs_image,
                "crop_label": "passed",
            },
        )

    assert infer_mock.call_count == 4
    for idx in (0, 2):
        od_mid, od_req = infer_mock.call_args_list[idx].args
        assert od_mid == "yolov8n-640"
        assert isinstance(od_req, ObjectDetectionInferenceRequest)
        od_imgs = od_req.image if isinstance(od_req.image, list) else [od_req.image]
        assert len(od_imgs) == 1

    for idx in (1, 3):
        cls_mid, cls_req = infer_mock.call_args_list[idx].args
        assert cls_mid == "dog-breed/1"
        assert isinstance(cls_req, ClassificationInferenceRequest)
        assert cls_req.confidence == _CLASSIFICATION_REQUEST_CONFIDENCE_THRESHOLD
        cls_imgs = cls_req.image if isinstance(cls_req.image, list) else [cls_req.image]
        assert len(cls_imgs) == 2

    assert nested_result == flat_result
    assert len(nested_result) == 1
    assert nested_result[0]["from_child"] == ["passed", None]


def test_inlined_continue_if_last_in_inner_echo_on_outer_matches_flat_workflow(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
) -> None:
    h, w = dogs_image.shape[:2]
    infer_mock = mock.MagicMock(
        side_effect=_infer_from_request_sync_factory(h, w),
    )
    inner = _inner_continue_if_only_outer_echo_name_matches_parent()

    with mock.patch.object(ModelManager, "add_model"), mock.patch.object(
        ModelManager,
        "infer_from_request_sync",
        new=infer_mock,
    ):
        nested_engine = execution_engine(
            model_manager,
            _nested_workflow_continue_if_inner_echo_outer(inner),
        )
        flat_engine = execution_engine(model_manager, _flat_workflow())
        nested_result = nested_engine.run(
            runtime_parameters={
                "image": dogs_image,
                "crop_label": "passed",
            },
        )
        flat_result = flat_engine.run(
            runtime_parameters={
                "image": dogs_image,
                "crop_label": "passed",
            },
        )

    assert infer_mock.call_count == 4
    for idx in (0, 2):
        od_mid, od_req = infer_mock.call_args_list[idx].args
        assert od_mid == "yolov8n-640"
        assert isinstance(od_req, ObjectDetectionInferenceRequest)
        od_imgs = od_req.image if isinstance(od_req.image, list) else [od_req.image]
        assert len(od_imgs) == 1

    for idx in (1, 3):
        cls_mid, cls_req = infer_mock.call_args_list[idx].args
        assert cls_mid == "dog-breed/1"
        assert isinstance(cls_req, ClassificationInferenceRequest)
        assert cls_req.confidence == _CLASSIFICATION_REQUEST_CONFIDENCE_THRESHOLD
        cls_imgs = cls_req.image if isinstance(cls_req.image, list) else [cls_req.image]
        assert len(cls_imgs) == 2

    assert nested_result == flat_result
    assert len(nested_result) == 1
    assert nested_result[0]["from_child"] == ["passed", None]
