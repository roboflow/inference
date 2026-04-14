"""
End-to-end tests for roboflow_core/inner_workflow@v1 (nested workflow execution).
"""

from typing import Any, Dict, Optional
from unittest import mock

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference.core.workflows.execution_engine.v1.inner_workflow.reference_resolution import (
    WORKFLOWS_CORE_INNER_WORKFLOW_SPEC_RESOLVER,
)


def _execution_engine(
    model_manager: ModelManager, workflow_definition: Dict[str, Any]
) -> ExecutionEngine:
    return ExecutionEngine.init(
        workflow_definition=workflow_definition,
        init_parameters={
            "workflows_core.model_manager": model_manager,
            "workflows_core.api_key": None,
            "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
        },
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )


def _echo_child_workflow() -> dict:
    """Minimal child: one WorkflowParameter, one formatter step, one output."""
    return {
        "version": "1.0",
        "inputs": [
            {
                "type": "WorkflowParameter",
                "name": "child_msg",
                "default_value": "default-child",
            },
        ],
        "steps": [
            {
                "type": "roboflow_core/first_non_empty_or_default@v1",
                "name": "pick",
                "data": ["$inputs.child_msg"],
                "default": "fallback-inner",
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "echo",
                "selector": "$steps.pick.output",
            },
        ],
    }


def _child_dynamic_crop_from_parent_detections() -> dict:
    """Child: crop using parent image + OD predictions; expose per-crop translated detections."""
    return {
        "version": "1.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
            {
                "type": "WorkflowBatchInput",
                "name": "predictions",
                "kind": ["object_detection_prediction"],
            },
        ],
        "steps": [
            {
                "type": "roboflow_core/dynamic_crop@v1",
                "name": "cropping",
                "image": "$inputs.image",
                "predictions": "$inputs.predictions",
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "crop_predictions",
                "selector": "$steps.cropping.predictions",
            },
        ],
    }


def test_inner_workflow_maps_parent_input_to_child_output(
    model_manager: ModelManager,
) -> None:
    inner = _echo_child_workflow()
    workflow_definition = {
        "version": "1.0",
        "inputs": [
            {
                "type": "WorkflowParameter",
                "name": "parent_msg",
                "default_value": "unused-default",
            },
        ],
        "steps": [
            {
                "type": "roboflow_core/inner_workflow@v1",
                "name": "nested",
                "workflow_definition": inner,
                "parameter_bindings": {
                    "child_msg": "$inputs.parent_msg",
                },
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "from_child",
                "selector": "$steps.nested.echo",
            },
        ],
    }
    engine = _execution_engine(model_manager, workflow_definition)

    result = engine.run(
        runtime_parameters={"parent_msg": "hello-from-parent"},
    )

    assert len(result) == 1
    assert result[0] == {"from_child": "hello-from-parent"}


def test_inner_workflow_resolves_saved_workflow_by_id_via_custom_resolver(
    model_manager: ModelManager,
) -> None:
    """Reference fields are expanded at compile time using an injected resolver."""

    def resolver(
        workspace_id: str,
        workflow_id: str,
        workflow_version_id: Optional[str],
        init_parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        assert workspace_id == "stub-ws"
        assert workflow_id == "stub-id"
        assert workflow_version_id is None
        return _echo_child_workflow()

    workflow_definition = {
        "version": "1.0",
        "inputs": [
            {
                "type": "WorkflowParameter",
                "name": "parent_msg",
                "default_value": "unused-default",
            },
        ],
        "steps": [
            {
                "type": "roboflow_core/inner_workflow@v1",
                "name": "nested_by_ref",
                "workflow_workspace_id": "stub-ws",
                "workflow_id": "stub-id",
                "parameter_bindings": {
                    "child_msg": "$inputs.parent_msg",
                },
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "from_child",
                "selector": "$steps.nested_by_ref.echo",
            },
        ],
    }
    engine = ExecutionEngine.init(
        workflow_definition=workflow_definition,
        init_parameters={
            "workflows_core.model_manager": model_manager,
            "workflows_core.api_key": None,
            "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
            WORKFLOWS_CORE_INNER_WORKFLOW_SPEC_RESOLVER: resolver,
        },
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    result = engine.run(
        runtime_parameters={"parent_msg": "hello-from-ref"},
    )

    assert len(result) == 1
    assert result[0] == {"from_child": "hello-from-ref"}


def test_stacked_inner_workflow_runs_at_depth_two(
    model_manager: ModelManager,
) -> None:
    """Parent inner_workflow wraps a child workflow that itself contains inner_workflow."""
    inner = _echo_child_workflow()
    middle = {
        "version": "1.0",
        "inputs": [
            {
                "type": "WorkflowParameter",
                "name": "wrapper_msg",
                "default_value": "unused-middle",
            },
        ],
        "steps": [
            {
                "type": "roboflow_core/inner_workflow@v1",
                "name": "inner_nested",
                "workflow_definition": inner,
                "parameter_bindings": {
                    "child_msg": "$inputs.wrapper_msg",
                },
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "wrapped",
                "selector": "$steps.inner_nested.echo",
            },
        ],
    }
    workflow_definition = {
        "version": "1.0",
        "inputs": [
            {
                "type": "WorkflowParameter",
                "name": "root_msg",
                "default_value": "unused-root",
            },
        ],
        "steps": [
            {
                "type": "roboflow_core/inner_workflow@v1",
                "name": "outer_nested",
                "workflow_definition": middle,
                "parameter_bindings": {
                    "wrapper_msg": "$inputs.root_msg",
                },
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "final",
                "selector": "$steps.outer_nested.wrapped",
            },
        ],
    }
    engine = _execution_engine(model_manager, workflow_definition)

    result = engine.run(runtime_parameters={"root_msg": "depth-two-value"})

    assert len(result) == 1
    assert result[0] == {"final": "depth-two-value"}


def test_inner_workflow_receives_parameter_from_upstream_parent_step(
    model_manager: ModelManager,
) -> None:
    """Bindings may reference ``$steps.<name>.<output>`` from a step that ran before the nest."""
    inner = _echo_child_workflow()
    workflow_definition = {
        "version": "1.0",
        "inputs": [
            {
                "type": "WorkflowParameter",
                "name": "base",
                "default_value": "",
            },
        ],
        "steps": [
            {
                "type": "roboflow_core/first_non_empty_or_default@v1",
                "name": "prepare",
                "data": ["$inputs.base"],
                "default": "unset",
            },
            {
                "type": "roboflow_core/inner_workflow@v1",
                "name": "nested",
                "workflow_definition": inner,
                "parameter_bindings": {
                    "child_msg": "$steps.prepare.output",
                },
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "from_child",
                "selector": "$steps.nested.echo",
            },
        ],
    }
    engine = _execution_engine(model_manager, workflow_definition)

    result = engine.run(runtime_parameters={"base": "value-from-prepare-step"})

    assert len(result) == 1
    assert result[0] == {"from_child": "value-from-prepare-step"}


def test_inner_workflow_with_list_valued_workflow_parameter(
    model_manager: ModelManager,
) -> None:
    """
    In Roboflow Inference, ``WorkflowParameter`` is a scalar input (not batch dimensionality
    like ``WorkflowBatchInput``). The runtime value may still be a JSON list: the nested
    workflow receives that list as a single ``child_msg``, and the parent returns one result
    dict whose ``from_child`` is that list.
    """
    inner = _echo_child_workflow()
    workflow_definition = {
        "version": "1.0",
        "inputs": [
            {
                "type": "WorkflowParameter",
                "name": "parent_msg",
                "default_value": "",
            },
        ],
        "steps": [
            {
                "type": "roboflow_core/inner_workflow@v1",
                "name": "nested",
                "workflow_definition": inner,
                "parameter_bindings": {
                    "child_msg": "$inputs.parent_msg",
                },
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "from_child",
                "selector": "$steps.nested.echo",
            },
        ],
    }
    engine = _execution_engine(model_manager, workflow_definition)

    result = engine.run(
        runtime_parameters={"parent_msg": ["alpha", "beta", "gamma"]},
    )

    assert len(result) == 1
    assert result[0]["from_child"] == ["alpha", "beta", "gamma"]


def test_inner_workflow_with_batch_workflow_batch_input(
    model_manager: ModelManager,
) -> None:
    """
    Same runtime payload as ``test_inner_workflow_with_list_valued_workflow_parameter`` (a flat list
    of strings under ``parent_msg``), but the input is declared ``WorkflowBatchInput`` with
    ``kind`` ``string`` and ``dimensionality`` 1.

    Unlike a list-valued ``WorkflowParameter``, a batch-oriented root input drives **one
    workflow result per batch element**: each run carries a scalar slice into the nested
    workflow, so you get three top-level dicts with string ``from_child`` values, not one dict
    whose ``from_child`` is a list of three strings.
    """
    inner = _echo_child_workflow()
    workflow_definition = {
        "version": "1.0",
        "inputs": [
            {
                "type": "WorkflowBatchInput",
                "name": "parent_msg",
                "kind": ["string"],
                "dimensionality": 1,
            },
        ],
        "steps": [
            {
                "type": "roboflow_core/inner_workflow@v1",
                "name": "nested",
                "workflow_definition": inner,
                "parameter_bindings": {
                    "child_msg": "$inputs.parent_msg",
                },
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "from_child",
                "selector": "$steps.nested.echo",
            },
        ],
    }
    engine = _execution_engine(model_manager, workflow_definition)

    result = engine.run(
        runtime_parameters={"parent_msg": ["alpha", "beta", "gamma"]},
    )

    assert len(result) == 3
    assert [row["from_child"] for row in result] == ["alpha", "beta", "gamma"]


def test_inner_workflow_child_runs_dynamic_crop_on_parent_detections(
    model_manager: ModelManager,
    dogs_image,
) -> None:
    """
    Parent runs mocked OD (three boxes, classes ``x``, ``y``, ``z``) and passes the image plus
    ``general_detection.predictions`` into a child workflow. The child runs ``dynamic_crop`` and
    exposes per-crop translated ``sv.Detections`` (class, confidence, ``detection_id``, valid ``xyxy``).
    """
    import supervision as sv

    from inference.core.entities.requests.inference import (
        ObjectDetectionInferenceRequest,
    )
    from inference.core.entities.responses.inference import (
        InferenceResponseImage,
        ObjectDetectionInferenceResponse,
        ObjectDetectionPrediction,
    )

    h, w = dogs_image.shape[:2]

    def infer_from_request_sync(
        model_id: str, request: ObjectDetectionInferenceRequest
    ):
        imgs = request.image if isinstance(request.image, list) else [request.image]
        assert len(imgs) == 1, f"Mock Expected 1 image, got {len(imgs)}"

        return ObjectDetectionInferenceResponse(
            image=InferenceResponseImage(width=w, height=h),
            predictions=[
                ObjectDetectionPrediction(
                    x=60,
                    y=60,
                    width=80,
                    height=80,
                    confidence=0.99,
                    class_id=0,
                    detection_id="mock-d0",
                    **{"class": "x"},
                ),
                ObjectDetectionPrediction(
                    x=250,
                    y=60,
                    width=80,
                    height=80,
                    confidence=0.99,
                    class_id=1,
                    detection_id="mock-d1",
                    **{"class": "y"},
                ),
                ObjectDetectionPrediction(
                    x=450,
                    y=60,
                    width=80,
                    height=80,
                    confidence=0.99,
                    class_id=2,
                    detection_id="mock-d2",
                    **{"class": "z"},
                ),
            ],
        )

    infer_mock = mock.MagicMock(side_effect=infer_from_request_sync)

    inner = _child_dynamic_crop_from_parent_detections()
    workflow_definition = {
        "version": "1.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "steps": [
            {
                "type": "roboflow_core/roboflow_object_detection_model@v2",
                "name": "general_detection",
                "image": "$inputs.image",
                "model_id": "yolov8n-640",
            },
            {
                "type": "roboflow_core/inner_workflow@v1",
                "name": "nested",
                "workflow_definition": inner,
                "parameter_bindings": {
                    "image": "$inputs.image",
                    "predictions": "$steps.general_detection.predictions",
                },
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "from_child",
                "selector": "$steps.nested.crop_predictions",
            },
        ],
    }
    with mock.patch.object(ModelManager, "add_model"), mock.patch.object(
        ModelManager,
        "infer_from_request_sync",
        new=infer_mock,
    ):
        engine = _execution_engine(model_manager, workflow_definition)
        result = engine.run(runtime_parameters={"image": dogs_image})

    infer_mock.assert_called_once()
    call_model_id, call_request = infer_mock.call_args.args

    assert call_model_id == "yolov8n-640"
    assert isinstance(call_request, ObjectDetectionInferenceRequest)
    imgs = (
        call_request.image
        if isinstance(call_request.image, list)
        else [call_request.image]
    )
    assert len(imgs) == 1

    assert len(result) == 1
    crop_preds = result[0]["from_child"]
    assert isinstance(crop_preds, list)
    assert len(crop_preds) == 3
    expected = [
        ("mock-d0", "x", 0, 0.99),
        ("mock-d1", "y", 1, 0.99),
        ("mock-d2", "z", 2, 0.99),
    ]
    for det, (det_id, class_name, class_id, conf) in zip(crop_preds, expected):
        assert isinstance(det, sv.Detections)
        assert len(det) == 1
        assert det["detection_id"][0] == det_id
        assert det.class_id[0] == class_id
        assert det.confidence[0] == conf
        assert det["class_name"][0] == class_name
        xyxy = det.xyxy[0]
        assert xyxy[0] >= 0 and xyxy[1] >= 0
        assert xyxy[2] > xyxy[0] and xyxy[3] > xyxy[1]


def test_inner_workflow_parent_detection_offset_after_nested_crop(
    model_manager: ModelManager,
    dogs_image,
) -> None:
    """
    Same mocked three-box OD and nested ``dynamic_crop`` child as
    ``test_inner_workflow_child_runs_dynamic_crop_on_parent_detections``. The parent then runs
    ``roboflow_core/detection_offset@v1`` on ``$steps.nested.crop_predictions`` (10 px width/height, pixels).
    We assert boxes match the pixel-offset formula vs crop-only baselines, class metadata is unchanged, and new
    ``detection_id`` values are issued (see ``detection_offset`` TODO on parent coordinates).
    """
    import numpy as np
    import supervision as sv

    from inference.core.entities.requests.inference import (
        ObjectDetectionInferenceRequest,
    )
    from inference.core.entities.responses.inference import (
        InferenceResponseImage,
        ObjectDetectionInferenceResponse,
        ObjectDetectionPrediction,
    )

    h, w = dogs_image.shape[:2]

    def infer_from_request_sync(
        model_id: str, request: ObjectDetectionInferenceRequest
    ):
        imgs = request.image if isinstance(request.image, list) else [request.image]
        assert len(imgs) == 1

        return ObjectDetectionInferenceResponse(
            image=InferenceResponseImage(width=w, height=h),
            predictions=[
                ObjectDetectionPrediction(
                    x=60,
                    y=60,
                    width=80,
                    height=80,
                    confidence=0.99,
                    class_id=0,
                    detection_id="mock-d0",
                    **{"class": "x"},
                ),
                ObjectDetectionPrediction(
                    x=250,
                    y=60,
                    width=80,
                    height=80,
                    confidence=0.99,
                    class_id=1,
                    detection_id="mock-d1",
                    **{"class": "y"},
                ),
                ObjectDetectionPrediction(
                    x=450,
                    y=60,
                    width=80,
                    height=80,
                    confidence=0.99,
                    class_id=2,
                    detection_id="mock-d2",
                    **{"class": "z"},
                ),
            ],
        )

    infer_mock = mock.MagicMock(side_effect=infer_from_request_sync)

    inner = _child_dynamic_crop_from_parent_detections()

    offset_x = 10
    offset_y = 10

    def _parent_workflow(*, apply_detection_offset: bool) -> dict:
        steps: list = [
            {
                "type": "roboflow_core/roboflow_object_detection_model@v2",
                "name": "general_detection",
                "image": "$inputs.image",
                "model_id": "yolov8n-640",
            },
            {
                "type": "roboflow_core/inner_workflow@v1",
                "name": "nested",
                "workflow_definition": inner,
                "parameter_bindings": {
                    "image": "$inputs.image",
                    "predictions": "$steps.general_detection.predictions",
                },
            },
        ]
        if apply_detection_offset:
            steps.append(
                {
                    "type": "roboflow_core/detection_offset@v1",
                    "name": "padded",
                    "predictions": "$steps.nested.crop_predictions",
                    "offset_width": offset_x,
                    "offset_height": offset_y,
                    "units": "Pixels",
                }
            )
        return {
            "version": "1.0",
            "inputs": [{"type": "WorkflowImage", "name": "image"}],
            "steps": steps,
            "outputs": [
                {
                    "type": "JsonField",
                    "name": (
                        "padded_preds" if apply_detection_offset else "crop_preds"
                    ),
                    "selector": (
                        "$steps.padded.predictions"
                        if apply_detection_offset
                        else "$steps.nested.crop_predictions"
                    ),
                },
            ],
        }

    with mock.patch.object(ModelManager, "add_model"), mock.patch.object(
        ModelManager,
        "infer_from_request_sync",
        new=infer_mock,
    ):
        eng_crop = _execution_engine(
            model_manager,
            _parent_workflow(apply_detection_offset=False),
        )
        res_crop = eng_crop.run(runtime_parameters={"image": dogs_image})

        eng_off = _execution_engine(
            model_manager,
            _parent_workflow(apply_detection_offset=True),
        )
        res_off = eng_off.run(runtime_parameters={"image": dogs_image})

    infer_mock.assert_called()
    assert infer_mock.call_count == 2

    before = res_crop[0]["crop_preds"]
    after = res_off[0]["padded_preds"]
    assert len(before) == len(after) == 3

    dx, dy = offset_x // 2, offset_y // 2
    for det_b, det_a in zip(before, after):
        assert isinstance(det_b, sv.Detections) and isinstance(det_a, sv.Detections)
        assert len(det_b) == len(det_a) == 1

        hid, wid = det_b["image_dimensions"][0]
        x1, y1, x2, y2 = det_b.xyxy[0]
        expected = np.array(
            [
                max(0.0, x1 - dx),
                max(0.0, y1 - dy),
                min(float(wid), x2 + dx),
                min(float(hid), y2 + dy),
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(
            det_a.xyxy[0].astype(np.float32),
            expected,
            rtol=0,
            atol=1e-3,
        )

        assert str(det_a.data["detection_id"][0]) != str(det_b.data["detection_id"][0])
        assert len(str(det_a.data["detection_id"][0])) == 36
        assert det_a.data["class_name"][0] == det_b.data["class_name"][0]
        assert det_a.class_id[0] == det_b.class_id[0]
        np.testing.assert_allclose(
            det_a.confidence[0], det_b.confidence[0], rtol=0, atol=1e-6
        )


def test_inner_workflow_after_continue_if_with_crop_batch_lineage(
    model_manager: ModelManager,
    dogs_image,
) -> None:
    """
    Same crop-level batch and confidence gate as ``test_workflow_with_two_stage_models_and_flow_control``,
    but the gated step is ``inner_workflow`` echoing a value derived from ``crop_label``.

    Object detection and classification are mocked so the test does not call Roboflow inference.

    With a list-valued ``WorkflowParameter`` for ``crop_label``, SIMD ``inner_workflow`` still passes the
    full list into each nested run; only the first crop passes ``continue_if``, so the nested echo runs
    once with that list while the second crop slot is ``None`` in the aggregated output.
    """
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

    h, w = dogs_image.shape[:2]

    def infer_from_request_sync(model_id: str, request):
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
                        width=100,
                        height=100,
                        confidence=0.99,
                        class_id=0,
                        detection_id="mock-d0",
                        parent_id="root",
                        **{"class": "dog"},
                    ),
                    ObjectDetectionPrediction(
                        x=300,
                        y=80,
                        width=100,
                        height=100,
                        confidence=0.99,
                        class_id=0,
                        detection_id="mock-d1",
                        parent_id="root",
                        **{"class": "dog"},
                    ),
                ],
            )
        if isinstance(request, ClassificationInferenceRequest):
            assert model_id == "dog-breed-xpaq6/1"
            assert request.confidence == 0.09
            imgs = request.image if isinstance(request.image, list) else [request.image]
            assert len(imgs) == 2
            crop_w, crop_h = 100, 100
            return [
                ClassificationInferenceResponse(
                    image=InferenceResponseImage(width=crop_w, height=crop_h),
                    predictions=[
                        ClassificationPrediction(
                            class_id=0,
                            confidence=0.9,
                            **{"class": "passed-crop"},
                        ),
                    ],
                    top="passed-crop",
                    confidence=0.9,
                ),
                ClassificationInferenceResponse(
                    image=InferenceResponseImage(width=crop_w, height=crop_h),
                    predictions=[
                        ClassificationPrediction(
                            class_id=1,
                            confidence=0.1,
                            **{"class": "skipped-crop"},
                        ),
                    ],
                    top="skipped-crop",
                    confidence=0.1,
                ),
            ]
        raise AssertionError(f"Unexpected request type: {type(request)!r}")

    infer_mock = mock.MagicMock(side_effect=infer_from_request_sync)

    inner = _echo_child_workflow()
    workflow_definition = {
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
                "type": "roboflow_core/roboflow_classification_model@v2",
                "name": "breds_classification",
                "image": "$steps.cropping.crops",
                "model_id": "dog-breed-xpaq6/1",
                "confidence": 0.09,
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
                            "right_operand": {"type": "StaticOperand", "value": 0.35},
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
    with mock.patch.object(ModelManager, "add_model"), mock.patch.object(
        ModelManager,
        "infer_from_request_sync",
        new=infer_mock,
    ):
        engine = ExecutionEngine.init(
            workflow_definition=workflow_definition,
            init_parameters={
                "workflows_core.model_manager": model_manager,
                "workflows_core.api_key": None,
                "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
            },
            max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
        )

        result = engine.run(
            runtime_parameters={
                "image": dogs_image,
                "crop_label": ["passed-crop", "skipped-crop"],
            },
        )

    assert infer_mock.call_count == 2
    od_mid, od_req = infer_mock.call_args_list[0].args
    assert od_mid == "yolov8n-640"
    assert isinstance(od_req, ObjectDetectionInferenceRequest)
    od_imgs = od_req.image if isinstance(od_req.image, list) else [od_req.image]
    assert len(od_imgs) == 1

    cls_mid, cls_req = infer_mock.call_args_list[1].args
    assert cls_mid == "dog-breed-xpaq6/1"
    assert isinstance(cls_req, ClassificationInferenceRequest)
    assert cls_req.confidence == 0.09
    cls_imgs = cls_req.image if isinstance(cls_req.image, list) else [cls_req.image]
    assert len(cls_imgs) == 2

    assert len(result) == 1
    assert result[0]["from_child"] == [["passed-crop", "skipped-crop"], None]


def test_parent_combines_outputs_from_two_parallel_inner_workflows(
    model_manager: ModelManager,
) -> None:
    """Two sibling ``inner_workflow`` steps each map a parent input into the same child shape."""
    inner = _echo_child_workflow()
    workflow_definition = {
        "version": "1.0",
        "inputs": [
            {
                "type": "WorkflowParameter",
                "name": "msg_a",
                "default_value": "",
            },
            {
                "type": "WorkflowParameter",
                "name": "msg_b",
                "default_value": "",
            },
        ],
        "steps": [
            {
                "type": "roboflow_core/inner_workflow@v1",
                "name": "branch_a",
                "workflow_definition": inner,
                "parameter_bindings": {
                    "child_msg": "$inputs.msg_a",
                },
            },
            {
                "type": "roboflow_core/inner_workflow@v1",
                "name": "branch_b",
                "workflow_definition": inner,
                "parameter_bindings": {
                    "child_msg": "$inputs.msg_b",
                },
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "first",
                "selector": "$steps.branch_a.echo",
            },
            {
                "type": "JsonField",
                "name": "second",
                "selector": "$steps.branch_b.echo",
            },
        ],
    }
    engine = _execution_engine(model_manager, workflow_definition)

    result = engine.run(
        runtime_parameters={"msg_a": "left-branch", "msg_b": "right-branch"},
    )

    assert len(result) == 1
    assert result[0] == {"first": "left-branch", "second": "right-branch"}
