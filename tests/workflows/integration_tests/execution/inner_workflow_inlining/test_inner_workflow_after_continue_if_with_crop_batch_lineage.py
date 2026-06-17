"""
Equivalence for ``test_inner_workflow_after_continue_if_with_crop_batch_lineage``.
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
    echo_child_workflow,
    execution_engine,
)

# Under ENABLE_TENSOR_DATA_REPRESENTATION both the OD block and the classification
# block run inference through `run_tensor_native_inference` (+ `get_class_names`),
# not `infer_from_request_sync`. The OD block emits native `inference_models.Detections`
# and the classification block emits native `inference_models.ClassificationPrediction`.
# The numpy test skips when the flag is on; the `*_tensor_native` parity test (skipped
# when off) drives the same scenario natively. The continue_if gate, the crop-batch
# lineage and the dimension-collapse echo output are representation-independent, so the
# final `["passed", None]` is asserted identically in both variants.
_NUMPY_ONLY = pytest.mark.skipif(
    ENABLE_TENSOR_DATA_REPRESENTATION,
    reason="numpy ModelManager mock; OD + classification blocks are native under the "
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
                "next_steps": ["$steps.nested_inner_workflow"],
            },
            {
                "type": "roboflow_core/inner_workflow@v1",
                "name": "nested_inner_workflow",
                "workflow_definition": inner,
                "parameter_bindings": {
                    "child_msg": "$inputs.crop_label",
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
def test_inlined_continue_if_echo_matches_inner_workflow(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
) -> None:
    h, w = dogs_image.shape[:2]
    infer_mock = mock.MagicMock(
        side_effect=_infer_from_request_sync_factory(h, w),
    )
    inner = echo_child_workflow()

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


# Class names for the native path. The OD `class_filter=["dog"]` is applied natively by
# the block against get_class_names("yolov8n-640"), so "dog" must sit at the OD
# class_id (0); the classification class_id-to-name map needs an entry at index 1 too.
_OD_CLASS_NAMES = ["dog"]
_CLS_CLASS_NAMES = ["dog", "cat"]


def _run_tensor_native_inference_factory(h: int, w: int):
    # Native equivalent of _infer_from_request_sync_factory, dispatching on model_id
    # because under the flag BOTH the OD block and the classification block call
    # `run_tensor_native_inference`.
    #
    #   OD (yolov8n-640): list of ONE raw `Detections` for the single input image, two
    #     "dog" boxes (xyxy from the numpy x/y/w/h centres: (80,80,100,100) ->
    #     [30,30,130,130]; (300,80,100,100) -> [250,30,350,130]). detection_id rides in
    #     bboxes_metadata; the OD block attaches the image lineage + class names itself.
    #   classification (dog-breed/1): ONE batched `ClassificationPrediction` over the two
    #     crops. class_id is (bs,), confidence is the FULL (bs, num_classes) softmax. The
    #     continue_if gate reads top_class_confidence = confidence[i, class_id[i]]:
    #       crop 0 -> class_id 0, confidence[0,0]=0.9  >= 0.5  -> PASS  -> echo "passed"
    #       crop 1 -> class_id 1, confidence[1,1]=0.4  <  0.5  -> GATED -> None
    #     (mirrors the numpy responses top=dog @0.9 and @0.4). The block fans this batched
    #     object out per-crop and attaches images_metadata itself.
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
            assert (
                kwargs.get("confidence")
                == _CLASSIFICATION_REQUEST_CONFIDENCE_THRESHOLD
            )
            return NativeClassificationPrediction(
                class_id=torch.tensor([0, 1], dtype=torch.long),
                confidence=torch.tensor(
                    [[0.9, 0.1], [0.6, 0.4]], dtype=torch.float32
                ),
            )
        raise AssertionError(f"Unexpected model_id: {model_id!r}")

    return run_tensor_native_inference


def _get_class_names_factory():
    def get_class_names(model_id: str):
        if model_id == "yolov8n-640":
            return list(_OD_CLASS_NAMES)
        if model_id == "dog-breed/1":
            return list(_CLS_CLASS_NAMES)
        raise AssertionError(f"Unexpected model_id: {model_id!r}")

    return get_class_names


@_TENSOR_ONLY
def test_inlined_continue_if_echo_matches_inner_workflow_tensor_native(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
) -> None:
    h, w = dogs_image.shape[:2]
    run_mock = mock.MagicMock(
        side_effect=_run_tensor_native_inference_factory(h, w),
    )
    inner = echo_child_workflow()

    with mock.patch.object(ModelManager, "add_model"), mock.patch.object(
        ModelManager,
        "get_class_names",
        new=mock.MagicMock(side_effect=_get_class_names_factory()),
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

    # Same dispatch count as the numpy mock (4): OD + classification per engine run.
    # model_id/images may be positional OR kwarg — read tolerantly.
    assert run_mock.call_count == 4
    od_calls = []
    cls_calls = []
    for call in run_mock.call_args_list:
        model_id = call.kwargs.get("model_id", call.args[0] if call.args else None)
        images = call.kwargs.get(
            "images", call.args[1] if len(call.args) > 1 else None
        )
        assert images is not None
        if model_id == "yolov8n-640":
            assert len(images) == 1
            od_calls.append(call)
        elif model_id == "dog-breed/1":
            assert len(images) == 2
            cls_calls.append(call)
        else:
            raise AssertionError(f"Unexpected model_id: {model_id!r}")
    assert len(od_calls) == 2
    assert len(cls_calls) == 2

    # The continue_if gate, crop-batch lineage and dimension-collapse echo are
    # representation-independent: crop 0 passes (top conf 0.9 >= 0.5) -> "passed",
    # crop 1 is gated (top conf 0.4 < 0.5) -> None. Native Detections /
    # ClassificationPrediction never surface in `from_child` (the echo emits scalars),
    # so the output is identical to the numpy variant. Assert it on BOTH runs rather
    # than comparing nested_result == flat_result (native carriers have no value __eq__).
    assert len(nested_result) == 1
    assert len(flat_result) == 1
    assert nested_result[0]["from_child"] == ["passed", None]
    assert flat_result[0]["from_child"] == ["passed", None]
