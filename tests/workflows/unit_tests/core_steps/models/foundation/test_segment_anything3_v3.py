"""Unit tests for SAM3 v3 block class_mapping feature."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import supervision as sv

from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.models.foundation.segment_anything3.v3 import (
    BlockManifest,
    SegmentAnything3BlockV3,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


def _make_detections(class_names: list[str]) -> sv.Detections:
    n = len(class_names)
    return sv.Detections(
        xyxy=np.array([[0, 0, 10, 10]] * n, dtype=np.float32),
        confidence=np.array([0.9] * n, dtype=np.float32),
        data={"class_name": np.array(class_names)},
    )


def _make_result(class_names: list[str]) -> list[dict]:
    return [{"predictions": _make_detections(class_names)}]


@pytest.fixture
def mock_workflow_image_data():
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="test"),
        numpy_image=img,
    )


# --- Manifest tests ---


def test_manifest_parsing_with_class_mapping():
    """Test that BlockManifest accepts the class_mapping field."""
    data = {
        "type": "roboflow_core/sam3@v3",
        "name": "my_sam3_step",
        "images": "$inputs.image",
        "class_names": ["cat", "dog"],
        "class_mapping": {"cat": "gato", "dog": "perro"},
    }
    result = BlockManifest.model_validate(data)
    assert result.class_mapping == {"cat": "gato", "dog": "perro"}


def test_manifest_parsing_without_class_mapping():
    """Test that class_mapping is optional and defaults to None."""
    data = {
        "type": "roboflow_core/sam3@v3",
        "name": "my_sam3_step",
        "images": "$inputs.image",
        "class_names": ["cat", "dog"],
    }
    result = BlockManifest.model_validate(data)
    assert result.class_mapping is None


# --- _apply_class_mapping unit tests ---


def test_apply_class_mapping_full():
    """Test remapping all class names."""
    result = _make_result(["cat", "dog"])
    mapped = SegmentAnything3BlockV3._apply_class_mapping(
        result, {"cat": "gato", "dog": "perro"}
    )
    assert list(mapped[0]["predictions"].data["class_name"]) == ["gato", "perro"]


def test_apply_class_mapping_partial():
    """Test remapping only some class names, leaving others unchanged."""
    result = _make_result(["cat", "dog", "bird"])
    mapped = SegmentAnything3BlockV3._apply_class_mapping(result, {"cat": "gato"})
    assert list(mapped[0]["predictions"].data["class_name"]) == [
        "gato",
        "dog",
        "bird",
    ]


def test_apply_class_mapping_no_matching_keys():
    """Test that unmatched mapping keys leave predictions unchanged."""
    result = _make_result(["cat", "dog"])
    mapped = SegmentAnything3BlockV3._apply_class_mapping(result, {"fish": "pez"})
    assert list(mapped[0]["predictions"].data["class_name"]) == ["cat", "dog"]


def test_apply_class_mapping_multiple_images():
    """Test remapping across multiple images in a batch."""
    result = [
        {"predictions": _make_detections(["cat"])},
        {"predictions": _make_detections(["dog"])},
    ]
    mapped = SegmentAnything3BlockV3._apply_class_mapping(
        result, {"cat": "gato", "dog": "perro"}
    )
    assert list(mapped[0]["predictions"].data["class_name"]) == ["gato"]
    assert list(mapped[1]["predictions"].data["class_name"]) == ["perro"]


def test_apply_class_mapping_empty_result():
    """Test that an empty result list is handled gracefully."""
    result = []
    mapped = SegmentAnything3BlockV3._apply_class_mapping(result, {"cat": "gato"})
    assert mapped == []


def test_apply_class_mapping_empty_mapping():
    """Test that an empty mapping leaves predictions unchanged."""
    result = _make_result(["cat", "dog"])
    mapped = SegmentAnything3BlockV3._apply_class_mapping(result, {})
    assert list(mapped[0]["predictions"].data["class_name"]) == ["cat", "dog"]


# --- Block-level run() tests ---


@patch.object(SegmentAnything3BlockV3, "run_locally")
def test_run_with_class_mapping_remaps_predictions(
    mock_run_locally, mock_workflow_image_data
):
    """Test that block.run() applies class_mapping to predictions from run_locally."""
    mock_run_locally.return_value = _make_result(["cat", "dog"])
    block = SegmentAnything3BlockV3(
        model_manager=MagicMock(),
        api_key="test_key",
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    result = block.run(
        images=[mock_workflow_image_data],
        model_id="sam3/sam3_final",
        class_names=["cat", "dog"],
        confidence=0.5,
        class_mapping={"cat": "gato", "dog": "perro"},
    )

    assert list(result[0]["predictions"].data["class_name"]) == ["gato", "perro"]


@patch.object(SegmentAnything3BlockV3, "run_locally")
def test_run_without_class_mapping_leaves_predictions_unchanged(
    mock_run_locally, mock_workflow_image_data
):
    """Test that block.run() without class_mapping does not alter predictions."""
    mock_run_locally.return_value = _make_result(["cat", "dog"])
    block = SegmentAnything3BlockV3(
        model_manager=MagicMock(),
        api_key="test_key",
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    result = block.run(
        images=[mock_workflow_image_data],
        model_id="sam3/sam3_final",
        class_names=["cat", "dog"],
        confidence=0.5,
    )

    assert list(result[0]["predictions"].data["class_name"]) == ["cat", "dog"]


@patch.object(SegmentAnything3BlockV3, "run_locally")
def test_run_with_partial_class_mapping(mock_run_locally, mock_workflow_image_data):
    """Test that block.run() with partial class_mapping only remaps matched classes."""
    mock_run_locally.return_value = _make_result(["cat", "dog", "bird"])
    block = SegmentAnything3BlockV3(
        model_manager=MagicMock(),
        api_key="test_key",
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    result = block.run(
        images=[mock_workflow_image_data],
        model_id="sam3/sam3_final",
        class_names=["cat", "dog", "bird"],
        confidence=0.5,
        class_mapping={"cat": "gato"},
    )

    assert list(result[0]["predictions"].data["class_name"]) == [
        "gato",
        "dog",
        "bird",
    ]


def _sam3_polygon_response():
    """One text prompt, one polygon; the shapes
    `_convert_polygon_response_to_inference_format` reads off the response."""
    prediction = MagicMock()
    prediction.confidence = 0.9
    prediction.masks = [[[0, 0], [8, 0], [8, 6], [0, 6]]]
    prompt_result = MagicMock()
    prompt_result.prompt_index = 0
    prompt_result.predictions = [prediction]
    response = MagicMock()
    response.prompt_results = [prompt_result]
    response.predictions = [prediction]
    return response


_SAM3_POLYGON_JSON = {
    "prompt_results": [
        {
            "prompt_index": 0,
            "predictions": [
                {"confidence": 0.9, "masks": [[[0, 0], [8, 0], [8, 6], [0, 6]]]}
            ],
        }
    ]
}


def _one_polygon_image_batch():
    from inference.core.workflows.execution_engine.entities.base import Batch

    return Batch(
        content=[
            WorkflowImageData(
                parent_metadata=ImageParentMetadata(parent_id="p"),
                numpy_image=np.zeros((10, 20, 3), dtype=np.uint8),
            )
        ],
        indices=[(0,)],
    )


_POLYGON_RUN_KWARGS = dict(
    class_names=["cat"],
    confidence=0.5,
    per_class_confidence=None,
    apply_nms=False,
    nms_iou_threshold=0.9,
    output_format="polygons",
)


def test_v3_local_polygon_path_converts_through_supervision() -> None:
    """Drives `run_locally` with a stubbed provider so the changed
    `_convert_polygon_response_to_inference_format(...).to_dict()` ->
    `sv.Detections.from_inference` path actually executes. Before Task 11.4
    Step 8 this raised `TypeError: … object is not subscriptable`."""
    model_manager = MagicMock()
    model_manager.infer_from_request_sync.return_value = _sam3_polygon_response()
    block = SegmentAnything3BlockV3(
        model_manager=model_manager,
        api_key="k",
        step_execution_mode=StepExecutionMode.LOCAL,
    )
    result = block.run_locally(
        images=_one_polygon_image_batch(),
        model_id="sam3/sam3_final",
        **_POLYGON_RUN_KWARGS
    )
    detections = result[0]["predictions"]
    assert len(detections) == 1
    assert detections.xyxy.tolist() == [[0.0, 0.0, 8.0, 6.0]]
    model_manager.infer_from_request_sync.assert_called_once()


def test_v3_remote_polygon_path_converts_through_supervision() -> None:
    """The REMOTE branch converts `_convert_polygon_json_response_to_inference_format(...)`
    through the same `to_dict()` seam (`v3.py:517`)."""
    import inference.core.workflows.core_steps.models.foundation.segment_anything3.v3 as v3_module

    with patch.object(v3_module, "InferenceHTTPClient") as client_cls:
        client_cls.return_value.sam3_concept_segment.return_value = _SAM3_POLYGON_JSON
        block = SegmentAnything3BlockV3(
            model_manager=MagicMock(),
            api_key="k",
            step_execution_mode=StepExecutionMode.REMOTE,
        )
        result = block.run_remotely(
            images=_one_polygon_image_batch(),
            model_id="sam3/sam3_final",
            **_POLYGON_RUN_KWARGS
        )
    assert len(result[0]["predictions"]) == 1
    assert result[0]["predictions"].xyxy.tolist() == [[0.0, 0.0, 8.0, 6.0]]


def test_v3_proxy_polygon_path_converts_through_supervision() -> None:
    """The inference-proxy branch (`run_via_request`, `v3.py:622`) - the third
    changed call site."""
    import inference.core.workflows.core_steps.models.foundation.segment_anything3.v3 as v3_module

    response = MagicMock()
    response.json.return_value = _SAM3_POLYGON_JSON
    with patch.object(v3_module.requests, "post", return_value=response):
        block = SegmentAnything3BlockV3(
            model_manager=MagicMock(),
            api_key="k",
            step_execution_mode=StepExecutionMode.LOCAL,
        )
        result = block.run_via_request(
            images=_one_polygon_image_batch(), **_POLYGON_RUN_KWARGS
        )
    assert len(result[0]["predictions"]) == 1
    assert result[0]["predictions"].xyxy.tolist() == [[0.0, 0.0, 8.0, 6.0]]


def test_v3_polygon_dataclass_matches_the_pydantic_form_through_supervision() -> None:
    """Explains the contract the block tests rely on: the response dataclass,
    passed as a dict, produces the same Detections the pydantic response did."""
    from inference.core.entities.responses.inference import (
        InferenceResponseImage,
        InstanceSegmentationInferenceResponse,
        InstanceSegmentationPrediction,
        Point,
    )
    from inference.core.workflows.core_steps.common.inference_response_dc import (
        InferenceResponseImageDC,
        InstanceSegmentationInferenceResponseDC,
    )

    polygon = [(0.0, 0.0), (8.0, 0.0), (8.0, 6.0), (0.0, 6.0)]
    prediction = InstanceSegmentationPrediction(
        **{
            "x": 4.0,
            "y": 3.0,
            "width": 8.0,
            "height": 6.0,
            "confidence": 0.75,
            "class": "cat",
            "class_id": 2,
            "detection_id": "fixed",
            "points": [Point(x=px, y=py) for px, py in polygon],
        }
    )
    local = InstanceSegmentationInferenceResponseDC(
        image=InferenceResponseImageDC(width=20, height=10), predictions=[prediction]
    )
    pydantic = InstanceSegmentationInferenceResponse(
        image=InferenceResponseImage(width=20, height=10), predictions=[prediction]
    )
    from_local = sv.Detections.from_inference(local.to_dict())
    from_pydantic = sv.Detections.from_inference(pydantic)
    assert np.array_equal(from_local.xyxy, from_pydantic.xyxy)
    assert np.array_equal(from_local.class_id, from_pydantic.class_id)
    assert np.array_equal(from_local.confidence, from_pydantic.confidence)
    assert (from_local.mask is None) == (from_pydantic.mask is None)
    if from_local.mask is not None:
        assert np.array_equal(from_local.mask, from_pydantic.mask)
