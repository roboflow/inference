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
import pytest
import torch

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
from inference.core.env import ENABLE_TENSOR_DATA_REPRESENTATION
from inference.core.managers.base import ModelManager
from inference_models.models.base.classification import (
    ClassificationPrediction as NativeClassificationPrediction,
)
from inference_models.models.base.object_detection import Detections as NativeDetections
from tests.workflows.integration_tests.execution.inner_workflow_inlining._common import (
    execution_engine,
)

# Under ENABLE_TENSOR_DATA_REPRESENTATION both the OD block AND the classification
# block run inference through `run_tensor_native_inference` (+ `get_class_names`),
# not `infer_from_request_sync`. The numpy tests skip when the flag is on; the
# `*_tensor_native` parity tests (skipped when off) drive the same scenario with a
# single `run_tensor_native_inference` mock that dispatches on `model_id` and a
# `get_class_names` that dispatches likewise. The workflow output (`from_child`) is
# a scalar echo list (`["passed", None]`) gated by the `continue_if` on the
# classification top-class confidence, so the value-level expectations are identical
# to numpy — only the mock seams change.
_NUMPY_ONLY = pytest.mark.skipif(
    ENABLE_TENSOR_DATA_REPRESENTATION,
    reason="numpy ModelManager mock; OD/classification blocks are native under the "
    "flag — see *_tensor_native",
)
_TENSOR_ONLY = pytest.mark.skipif(
    not ENABLE_TENSOR_DATA_REPRESENTATION,
    reason="tensor-native variant; runs only with ENABLE_TENSOR_DATA_REPRESENTATION=True",
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


@_NUMPY_ONLY
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


def _run_tensor_native_inference_factory(h: int, w: int):
    # Native equivalent of _infer_from_request_sync_factory. Under the flag BOTH the
    # OD block and the classification block dispatch through `run_tensor_native_inference`,
    # keyed by `model_id` (the numpy mock keyed on request type instead):
    #  * "yolov8n-640" -> List[Detections] (one per input image). Two boxes, both
    #    class_id=0 -> "dog" via get_class_names, so both survive the native
    #    class_filter=["dog"]. xyxy from the numpy centre/size boxes
    #    (80,80,100,100) and (300,80,100,100). detection_id rides in bboxes_metadata.
    #  * "dog-breed/1" -> ONE batched ClassificationPrediction over the 2 crops:
    #    class_id (bs,) and confidence (bs, num_classes) full softmax. top_class_confidence
    #    reads confidence[0, class_id] per single-row slice, so row 0 -> 0.9 (>=0.5, passes)
    #    and row 1 -> 0.4 (<0.5, blocked), reproducing the numpy ["passed", None] gating.
    def run_tensor_native_inference(model_id: str, images, **kwargs):
        imgs = images if isinstance(images, list) else [images]
        if model_id == "yolov8n-640":
            assert len(imgs) == 1, f"Mock Expected 1 OD image, got {len(imgs)}"
            return [
                NativeDetections(
                    xyxy=torch.tensor(
                        [[30, 30, 130, 130], [250, 30, 350, 130]],
                        dtype=torch.float32,
                    ),
                    class_id=torch.tensor([0, 0], dtype=torch.long),
                    confidence=torch.tensor([0.99, 0.99], dtype=torch.float32),
                    bboxes_metadata=[
                        {"detection_id": "mock-d0"},
                        {"detection_id": "mock-d1"},
                    ],
                )
            ]
        if model_id == "dog-breed/1":
            assert len(imgs) == 2, f"Mock Expected 2 crop images, got {len(imgs)}"
            return NativeClassificationPrediction(
                class_id=torch.tensor([0, 1], dtype=torch.long),
                confidence=torch.tensor([[0.9, 0.1], [0.6, 0.4]], dtype=torch.float32),
            )
        raise AssertionError(f"Unexpected model_id: {model_id!r}")

    return run_tensor_native_inference


def _get_class_names_native(model_id: str):
    # Single get_class_names mock dispatching on model_id (mirrors the request-type
    # dispatch in the numpy infer mock). OD class 0 -> "dog" so both boxes survive the
    # native class_filter=["dog"]. The classification names don't affect the continue_if
    # gate (it reads confidence[0, class_id]), but the list must cover the class ids.
    if model_id == "yolov8n-640":
        return ["dog"]
    if model_id == "dog-breed/1":
        return ["dog", "other"]
    raise AssertionError(f"Unexpected model_id: {model_id!r}")


def _assert_native_call_order(run_mock: mock.MagicMock) -> None:
    # Both engines run OD then classification, so the 4 calls alternate
    # yolov8n-640, dog-breed/1. model_id/images may arrive positional OR kwarg.
    assert run_mock.call_count == 4
    for idx in (0, 2):
        call = run_mock.call_args_list[idx]
        model_id = call.kwargs.get("model_id", call.args[0] if call.args else None)
        assert model_id == "yolov8n-640"
        od_imgs = call.kwargs.get(
            "images", call.args[1] if len(call.args) > 1 else None
        )
        assert od_imgs is not None and len(od_imgs) == 1
    for idx in (1, 3):
        call = run_mock.call_args_list[idx]
        model_id = call.kwargs.get("model_id", call.args[0] if call.args else None)
        assert model_id == "dog-breed/1"
        cls_imgs = call.kwargs.get(
            "images", call.args[1] if len(call.args) > 1 else None
        )
        assert cls_imgs is not None and len(cls_imgs) == 2


@_TENSOR_ONLY
def test_inlined_continue_if_inside_inner_matches_flat_workflow_tensor_native(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
) -> None:
    h, w = dogs_image.shape[:2]
    run_mock = mock.MagicMock(
        side_effect=_run_tensor_native_inference_factory(h, w),
    )
    inner = _inner_continue_if_then_pick()

    with mock.patch.object(ModelManager, "add_model"), mock.patch.object(
        ModelManager, "get_class_names", side_effect=_get_class_names_native
    ), mock.patch.object(
        ModelManager,
        "run_tensor_native_inference",
        new=run_mock,
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

    _assert_native_call_order(run_mock)

    # `from_child` is a plain scalar echo list (no native Detections), so the value-level
    # expectation is identical to numpy and safe to assert directly on both runs.
    assert nested_result == flat_result
    assert len(nested_result) == 1
    assert nested_result[0]["from_child"] == ["passed", None]


@_NUMPY_ONLY
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


@_TENSOR_ONLY
def test_inlined_continue_if_last_in_inner_echo_on_outer_matches_flat_workflow_tensor_native(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
) -> None:
    h, w = dogs_image.shape[:2]
    run_mock = mock.MagicMock(
        side_effect=_run_tensor_native_inference_factory(h, w),
    )
    inner = _inner_continue_if_only_outer_echo_name_matches_parent()

    with mock.patch.object(ModelManager, "add_model"), mock.patch.object(
        ModelManager, "get_class_names", side_effect=_get_class_names_native
    ), mock.patch.object(
        ModelManager,
        "run_tensor_native_inference",
        new=run_mock,
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

    _assert_native_call_order(run_mock)

    # `from_child` is a plain scalar echo list (no native Detections), so the value-level
    # expectation is identical to numpy and safe to assert directly on both runs.
    assert nested_result == flat_result
    assert len(nested_result) == 1
    assert nested_result[0]["from_child"] == ["passed", None]
