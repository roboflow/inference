from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import supervision as sv

from inference.core.workflows.core_steps.sinks.roboflow.vision_events.v1 import (
    BlockManifest,
    RoboflowVisionEventsBlockV1,
    _build_event_data,
    _build_event_payload,
    _convert_classification_to_vision_events_format,
    _convert_sv_detections_to_vision_events_format,
    _detect_prediction_type,
    _upload_image,
)
from inference.core.workflows.execution_engine.constants import (
    KEYPOINTS_CLASS_ID_KEY_IN_SV_DETECTIONS,
    KEYPOINTS_CLASS_NAME_KEY_IN_SV_DETECTIONS,
    KEYPOINTS_CONFIDENCE_KEY_IN_SV_DETECTIONS,
    KEYPOINTS_XY_KEY_IN_SV_DETECTIONS,
    POLYGON_KEY_IN_SV_DETECTIONS,
    PREDICTION_TYPE_KEY,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


def _make_workflow_image(width: int = 100, height: int = 100) -> WorkflowImageData:
    """Create a simple WorkflowImageData for testing."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="test"),
        numpy_image=image,
    )


# === Manifest Validation ===


def test_manifest_parsing_valid() -> None:
    raw_manifest = {
        "type": "roboflow_core/roboflow_vision_events@v1",
        "name": "test_step",
        "event_type": "quality_check",
        "solution": "my-solution",
    }
    manifest = BlockManifest.model_validate(raw_manifest)
    assert manifest.event_type == "quality_check"
    assert manifest.solution == "my-solution"
    assert manifest.fire_and_forget is True
    assert manifest.disable_sink is False
    assert manifest.input_image is None
    assert manifest.output_image is None


def test_manifest_parsing_missing_event_type() -> None:
    raw_manifest = {
        "type": "roboflow_core/roboflow_vision_events@v1",
        "name": "test_step",
        "solution": "my-solution",
    }
    with pytest.raises(Exception):
        BlockManifest.model_validate(raw_manifest)


def test_manifest_parsing_missing_solution() -> None:
    raw_manifest = {
        "type": "roboflow_core/roboflow_vision_events@v1",
        "name": "test_step",
        "event_type": "quality_check",
    }
    with pytest.raises(Exception):
        BlockManifest.model_validate(raw_manifest)


# === Detection Type Auto-Detection ===


def _make_detections(
    n: int = 2,
    with_mask: bool = False,
    with_polygon: bool = False,
    with_keypoints: bool = False,
    prediction_type: str = "object-detection",
) -> sv.Detections:
    """Helper to build sv.Detections with various data."""
    xyxy = np.array([[10, 20, 50, 60], [100, 200, 150, 260]], dtype=float)[:n]
    confidence = np.array([0.9, 0.8])[:n]
    class_id = np.array([0, 1])[:n]

    data = {
        "class_name": np.array(["cat", "dog"])[:n],
        "detection_id": np.array(["id1", "id2"])[:n],
        PREDICTION_TYPE_KEY: np.array([prediction_type] * n),
    }

    mask = None
    if with_mask:
        mask = np.zeros((n, 100, 100), dtype=bool)
        # Create a simple square mask
        mask[:, 20:60, 10:50] = True

    if with_polygon:
        data[POLYGON_KEY_IN_SV_DETECTIONS] = np.array(
            [
                np.array([[10, 20], [50, 20], [50, 60], [10, 60]], dtype=float),
                np.array([[100, 200], [150, 200], [150, 260], [100, 260]], dtype=float),
            ],
            dtype=object,
        )[:n]

    if with_keypoints:
        data[KEYPOINTS_XY_KEY_IN_SV_DETECTIONS] = np.array(
            [
                np.array([[15.0, 25.0], [30.0, 40.0], [45.0, 55.0]]),
                np.array([[110.0, 210.0], [125.0, 230.0], [140.0, 250.0]]),
            ],
            dtype=object,
        )[:n]
        data[KEYPOINTS_CLASS_ID_KEY_IN_SV_DETECTIONS] = np.array(
            [
                np.array([0, 1, 2]),
                np.array([0, 1, 2]),
            ],
            dtype=object,
        )[:n]
        data[KEYPOINTS_CLASS_NAME_KEY_IN_SV_DETECTIONS] = np.array(
            [
                np.array(["nose", "left_eye", "right_eye"]),
                np.array(["nose", "left_eye", "right_eye"]),
            ],
            dtype=object,
        )[:n]
        data[KEYPOINTS_CONFIDENCE_KEY_IN_SV_DETECTIONS] = np.array(
            [
                np.array([0.95, 0.90, 0.85]),
                np.array([0.92, 0.88, 0.80]),
            ],
            dtype=object,
        )[:n]

    return sv.Detections(
        xyxy=xyxy,
        confidence=confidence,
        class_id=class_id,
        mask=mask,
        data=data,
    )


def test_detect_prediction_type_object_detection() -> None:
    detections = _make_detections()
    assert _detect_prediction_type(detections) == "object_detection"


def test_detect_prediction_type_instance_segmentation_with_mask() -> None:
    detections = _make_detections(with_mask=True)
    assert _detect_prediction_type(detections) == "instance_segmentation"


def test_detect_prediction_type_instance_segmentation_with_polygon() -> None:
    detections = _make_detections(with_polygon=True)
    assert _detect_prediction_type(detections) == "instance_segmentation"


def test_detect_prediction_type_keypoint_detection() -> None:
    detections = _make_detections(with_keypoints=True)
    assert _detect_prediction_type(detections) == "keypoint_detection"


def test_detect_prediction_type_empty_detections() -> None:
    detections = sv.Detections.empty()
    assert _detect_prediction_type(detections) == "object_detection"


def test_detect_prediction_type_from_prediction_type_key() -> None:
    detections = _make_detections(prediction_type="instance-segmentation")
    # No mask or polygon, so it falls through to prediction_type key
    assert _detect_prediction_type(detections) == "instance_segmentation"


# === Object Detection Conversion ===


def test_convert_object_detections_center_based_coordinates() -> None:
    """Verify xyxy → center-based conversion is correct."""
    detections = _make_detections(n=1)
    obj_dets, seg, kp = _convert_sv_detections_to_vision_events_format(detections)

    assert len(obj_dets) == 1
    assert len(seg) == 0
    assert len(kp) == 0

    det = obj_dets[0]
    # xyxy = [10, 20, 50, 60] → w=40, h=40, cx=30, cy=40
    assert det["class"] == "cat"
    assert det["x"] == pytest.approx(30.0)
    assert det["y"] == pytest.approx(40.0)
    assert det["width"] == pytest.approx(40.0)
    assert det["height"] == pytest.approx(40.0)
    assert det["confidence"] == pytest.approx(0.9)


def test_convert_multiple_object_detections() -> None:
    detections = _make_detections(n=2)
    obj_dets, _, _ = _convert_sv_detections_to_vision_events_format(detections)

    assert len(obj_dets) == 2
    # Second detection: xyxy = [100, 200, 150, 260] → w=50, h=60, cx=125, cy=230
    assert obj_dets[1]["class"] == "dog"
    assert obj_dets[1]["x"] == pytest.approx(125.0)
    assert obj_dets[1]["y"] == pytest.approx(230.0)
    assert obj_dets[1]["width"] == pytest.approx(50.0)
    assert obj_dets[1]["height"] == pytest.approx(60.0)


# === Instance Segmentation Conversion ===


def test_convert_instance_segmentation_with_polygon() -> None:
    detections = _make_detections(n=1, with_polygon=True)
    obj_dets, seg, _ = _convert_sv_detections_to_vision_events_format(detections)

    assert len(obj_dets) == 0
    assert len(seg) == 1

    s = seg[0]
    assert s["class"] == "cat"
    assert s["x"] == pytest.approx(30.0)
    assert s["y"] == pytest.approx(40.0)
    assert "points" in s
    assert len(s["points"]) == 4
    # Points should be [[x,y], ...] format
    assert s["points"][0] == [10.0, 20.0]
    assert s["points"][1] == [50.0, 20.0]


def test_convert_instance_segmentation_with_mask() -> None:
    detections = _make_detections(n=1, with_mask=True)
    obj_dets, seg, _ = _convert_sv_detections_to_vision_events_format(detections)

    # Should have found polygon from mask
    assert len(seg) + len(obj_dets) == 1
    if len(seg) == 1:
        assert "points" in seg[0]
        assert len(seg[0]["points"]) >= 3


# === Keypoint Detection Conversion ===


def test_convert_keypoint_detection() -> None:
    detections = _make_detections(n=1, with_keypoints=True)
    _, _, kp = _convert_sv_detections_to_vision_events_format(detections)

    assert len(kp) == 1
    k = kp[0]
    assert k["class"] == "cat"
    assert k["x"] == pytest.approx(30.0)
    assert k["y"] == pytest.approx(40.0)
    assert "keypoints" in k
    assert len(k["keypoints"]) == 3
    assert k["keypoints"][0] == {"id": 0, "x": 15.0, "y": 25.0}
    assert k["keypoints"][1] == {"id": 1, "x": 30.0, "y": 40.0}
    assert k["keypoints"][2] == {"id": 2, "x": 45.0, "y": 55.0}


# === Classification Conversion ===


def test_convert_classification_multiclass() -> None:
    prediction = {
        "predictions": [
            {"class_name": "cat", "class_id": 0, "confidence": 0.85},
            {"class_name": "dog", "class_id": 1, "confidence": 0.15},
        ]
    }
    classifications = _convert_classification_to_vision_events_format(prediction)
    assert len(classifications) == 2
    assert classifications[0] == {"class": "cat", "confidence": 0.85}
    assert classifications[1] == {"class": "dog", "confidence": 0.15}


def test_convert_classification_multilabel() -> None:
    prediction = {
        "predictions": {
            "cat": {"confidence": 0.9, "class_id": 0},
            "dog": {"confidence": 0.3, "class_id": 1},
        }
    }
    classifications = _convert_classification_to_vision_events_format(prediction)
    assert len(classifications) == 2
    classes = {c["class"] for c in classifications}
    assert classes == {"cat", "dog"}
    cat = next(c for c in classifications if c["class"] == "cat")
    assert cat["confidence"] == pytest.approx(0.9)


def test_convert_classification_with_top_field() -> None:
    prediction = {"top": "cat", "confidence": 0.95}
    classifications = _convert_classification_to_vision_events_format(prediction)
    assert len(classifications) == 1
    assert classifications[0] == {"class": "cat", "confidence": 0.95}


def test_convert_classification_predicted_classes() -> None:
    prediction = {
        "predicted_classes": ["cat", "dog"],
        "predictions": {
            "cat": {"confidence": 0.9, "class_id": 0},
            "dog": {"confidence": 0.7, "class_id": 1},
        },
    }
    classifications = _convert_classification_to_vision_events_format(prediction)
    assert len(classifications) == 2
    cat = next(c for c in classifications if c["class"] == "cat")
    assert cat["confidence"] == pytest.approx(0.9)


def test_convert_empty_detections() -> None:
    detections = sv.Detections.empty()
    obj_dets, seg, kp = _convert_sv_detections_to_vision_events_format(detections)
    assert obj_dets == []
    assert seg == []
    assert kp == []


# === Event Payload Building ===


def test_build_event_payload_all_fields() -> None:
    payload = _build_event_payload(
        event_type="quality_check",
        solution="my-solution",
        images=[{"label": "input", "sourceId": "src-1"}],
        event_data={"result": "pass"},
        custom_metadata={"camera_id": "cam-01"},
    )

    assert payload["eventType"] == "quality_check"
    assert payload["useCaseId"] == "my-solution"
    assert payload["eventData"] == {"result": "pass"}
    assert payload["customMetadata"] == {"camera_id": "cam-01"}
    assert payload["displayImagePosition"] == 0
    assert "eventId" in payload
    assert "timestamp" in payload
    assert len(payload["images"]) == 1


def test_build_event_payload_minimal() -> None:
    payload = _build_event_payload(
        event_type="custom",
        solution="test",
        images=[],
        event_data={},
        custom_metadata={},
    )

    assert payload["eventType"] == "custom"
    assert payload["useCaseId"] == "test"
    assert payload["images"] == []
    assert "eventData" not in payload
    assert "customMetadata" not in payload
    assert "deviceId" not in payload  # device_id not supported
    assert "displayImagePosition" not in payload
    assert "eventId" in payload
    assert "timestamp" in payload


# === Image Upload (mocked) ===


@patch(
    "inference.core.workflows.core_steps.sinks.roboflow.vision_events.v1.requests.post"
)
def test_upload_image_success(mock_post: MagicMock) -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "sourceId": "src-123",
        "url": "https://example.com/img.jpg",
    }
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    image = _make_workflow_image()
    source_id, url = _upload_image("https://api.roboflow.com", "test-key", image)

    assert source_id == "src-123"
    assert url == "https://example.com/img.jpg"

    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args
    assert "vision-events/upload" in call_kwargs[0][0]
    assert call_kwargs[1]["headers"]["Authorization"] == "Bearer test-key"


@patch(
    "inference.core.workflows.core_steps.sinks.roboflow.vision_events.v1.requests.post"
)
def test_upload_image_failure(mock_post: MagicMock) -> None:
    import requests

    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
        response=mock_response
    )
    mock_post.return_value = mock_response

    image = _make_workflow_image()
    with pytest.raises(requests.exceptions.HTTPError):
        _upload_image("https://api.roboflow.com", "test-key", image)


# === Block run() Tests ===


def test_run_missing_api_key() -> None:
    block = RoboflowVisionEventsBlockV1(
        api_key=None,
        background_tasks=None,
        thread_pool_executor=None,
    )
    with pytest.raises(ValueError, match="API key"):
        block.run(
            input_image=None,
            output_image=None,
            predictions=None,
            event_type="custom",
            solution="test",
            custom_metadata={},
            fire_and_forget=False,
            disable_sink=False,
        )


def test_run_disabled() -> None:
    block = RoboflowVisionEventsBlockV1(
        api_key="test-key",
        background_tasks=None,
        thread_pool_executor=None,
    )
    result = block.run(
        input_image=None,
        output_image=None,
        predictions=None,
        event_type="custom",
        solution="test",
        custom_metadata={},
        fire_and_forget=False,
        disable_sink=True,
    )
    assert isinstance(result, dict)
    assert result["error_status"] is False
    assert "disabled" in result["message"].lower()


@patch(
    "inference.core.workflows.core_steps.sinks.roboflow.vision_events.v1._execute_vision_event"
)
def test_run_fire_and_forget_background_tasks(mock_execute: MagicMock) -> None:
    background_tasks = MagicMock()
    block = RoboflowVisionEventsBlockV1(
        api_key="test-key",
        background_tasks=background_tasks,
        thread_pool_executor=None,
    )
    result = block.run(
        input_image=None,
        output_image=None,
        predictions=None,
        event_type="custom",
        solution="test",
        custom_metadata={},
        fire_and_forget=True,
        disable_sink=False,
    )

    background_tasks.add_task.assert_called_once()
    assert result["error_status"] is False
    assert "background" in result["message"].lower()


@patch(
    "inference.core.workflows.core_steps.sinks.roboflow.vision_events.v1._execute_vision_event"
)
def test_run_fire_and_forget_thread_pool(mock_execute: MagicMock) -> None:
    thread_pool = MagicMock()
    block = RoboflowVisionEventsBlockV1(
        api_key="test-key",
        background_tasks=None,
        thread_pool_executor=thread_pool,
    )
    result = block.run(
        input_image=None,
        output_image=None,
        predictions=None,
        event_type="custom",
        solution="test",
        custom_metadata={},
        fire_and_forget=True,
        disable_sink=False,
    )

    thread_pool.submit.assert_called_once()
    assert result["error_status"] is False
    assert "background" in result["message"].lower()


@patch(
    "inference.core.workflows.core_steps.sinks.roboflow.vision_events.v1._execute_vision_event"
)
def test_run_synchronous(mock_execute: MagicMock) -> None:
    mock_execute.return_value = (False, "Vision event sent successfully")
    block = RoboflowVisionEventsBlockV1(
        api_key="test-key",
        background_tasks=None,
        thread_pool_executor=None,
    )
    result = block.run(
        input_image=None,
        output_image=None,
        predictions=None,
        event_type="custom",
        solution="test",
        custom_metadata={},
        fire_and_forget=False,
        disable_sink=False,
    )

    mock_execute.assert_called_once()
    assert result["error_status"] is False
    assert result["message"] == "Vision event sent successfully"


# === Non-SIMD / Compilation Regression Tests (ENT-1126) ===


def test_manifest_is_not_simd() -> None:
    """Block must be non-SIMD so the engine broadcasts scalar params per image."""
    assert BlockManifest.accepts_batch_input() is False


def test_batch_selector_on_scalar_field_passes_compile_check() -> None:
    """Regression test for ENT-1126.

    Exercises ``verify_declared_batch_compatibility_against_actual_inputs``
    (graph_constructor.py:1659), the exact compile-time check that rejected
    batch-oriented selectors on ``item_count`` when the block was SIMD.

    A StepNode backed by our manifest with a batch-oriented input on
    ``item_count`` must NOT raise ``ExecutionGraphStructureError``.
    """
    from inference.core.workflows.errors import ExecutionGraphStructureError
    from inference.core.workflows.execution_engine.v1.compiler.entities import (
        DynamicStepInputDefinition,
        NodeInputCategory,
        ParameterSpecification,
        StepNode,
    )
    from inference.core.workflows.execution_engine.v1.compiler.graph_constructor import (
        verify_declared_batch_compatibility_against_actual_inputs,
    )

    manifest = BlockManifest(
        type="roboflow_core/roboflow_vision_events@v1",
        name="vision_events",
        event_type="inventory_count",
        solution="test-solution",
        item_count="$steps.counter.count",
    )

    step_node = StepNode(
        node_category="step_node",
        name="vision_events",
        selector="$steps.vision_events",
        data_lineage=[],
        step_manifest=manifest,
        input_data={
            "item_count": DynamicStepInputDefinition(
                parameter_specification=ParameterSpecification(
                    parameter_name="item_count",
                    nested_element_key=None,
                    nested_element_index=None,
                ),
                category=NodeInputCategory.BATCH_STEP_OUTPUT,
                data_lineage=["<workflow_input>"],
                selector="$steps.counter.count",
            ),
        },
        batch_oriented_parameters=set(),
    )

    # batch_compatibility_of_properties says item_count is NOT batch-compatible
    batch_compat = {"item_count": {False}}

    # Must not raise. Before the fix (when accepts_batch_input() was True),
    # this exact call raised ExecutionGraphStructureError because a
    # batch-oriented selector was plugged into a non-batch parameter.
    result = verify_declared_batch_compatibility_against_actual_inputs(
        node="$steps.vision_events",
        step_node_data=step_node,
        input_data=step_node.input_data,
        batch_compatibility_of_properties=batch_compat,
    )
    assert isinstance(result, set)
