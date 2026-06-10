"""Unit tests for SAM 3 Interactive block (point / box prompting via SAM3 PVS)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import supervision as sv

from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.models.foundation.segment_anything3_interactive.v1 import (
    BlockManifest,
    SegmentAnything3InteractiveBlockV1,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


@pytest.fixture
def mock_model_manager():
    mock = MagicMock()
    mock_prediction = MagicMock()
    mock_prediction.masks = [[[0, 0], [100, 0], [100, 100], [0, 100]]]
    mock_prediction.confidence = 0.95
    mock.infer_from_request_sync.return_value = MagicMock(predictions=[mock_prediction])
    return mock


@pytest.fixture
def mock_workflow_image_data():
    start_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some"),
        numpy_image=start_image,
    )


def _example_detections() -> sv.Detections:
    detections = sv.Detections(
        xyxy=np.array([[10, 10, 50, 50]]),
        confidence=np.array([0.9]),
        class_id=np.array([0]),
    )
    detections["class_name"] = np.array(["object"])
    detections["detection_id"] = np.array(["det_1"])
    return detections


def test_manifest_parsing_with_literal_points() -> None:
    data = {
        "type": "roboflow_core/sam3_interactive@v1",
        "name": "sam3_interactive",
        "images": "$inputs.image",
        "points": [
            {"x": 320, "y": 240, "positive": True},
            {"x": 100, "y": 100, "positive": False},
        ],
    }
    result = BlockManifest.model_validate(data)
    assert result.type == "roboflow_core/sam3_interactive@v1"
    assert len(result.points) == 2


def test_manifest_parsing_with_points_selector() -> None:
    data = {
        "type": "roboflow_core/sam3_interactive@v1",
        "name": "sam3_interactive",
        "images": "$inputs.image",
        "points": "$inputs.points",
        "boxes": "$steps.detection.predictions",
    }
    result = BlockManifest.model_validate(data)
    assert result.points == "$inputs.points"
    assert result.boxes == "$steps.detection.predictions"


def test_manifest_parsing_with_sequence_points() -> None:
    data = {
        "type": "roboflow_core/sam3_interactive@v1",
        "name": "sam3_interactive",
        "images": "$inputs.image",
        "points": [[320, 240], [100, 100, False]],
    }
    result = BlockManifest.model_validate(data)
    assert len(result.points) == 2


def test_manifest_parsing_rejects_invalid_points() -> None:
    data = {
        "type": "roboflow_core/sam3_interactive@v1",
        "name": "sam3_interactive",
        "images": "$inputs.image",
        "points": [{"x": 320}],
    }
    with pytest.raises(Exception):
        _ = BlockManifest.model_validate(data)


def test_run_raises_when_no_prompts_given(
    mock_model_manager, mock_workflow_image_data
) -> None:
    block = SegmentAnything3InteractiveBlockV1(
        model_manager=mock_model_manager,
        api_key="test_api_key",
        step_execution_mode=StepExecutionMode.LOCAL,
    )
    with pytest.raises(ValueError):
        _ = block.run(
            images=[mock_workflow_image_data],
            points=None,
            boxes=None,
            threshold=0.0,
            multimask_output=True,
        )


def test_run_locally_with_point_prompts(
    mock_model_manager, mock_workflow_image_data
) -> None:
    block = SegmentAnything3InteractiveBlockV1(
        model_manager=mock_model_manager,
        api_key="test_api_key",
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    result = block.run(
        images=[mock_workflow_image_data],
        points=[
            {"x": 320, "y": 240, "positive": True},
            {"x": 100, "y": 100, "positive": False},
        ],
        boxes=None,
        threshold=0.0,
        multimask_output=True,
    )

    assert len(result) == 1
    assert "predictions" in result[0]
    mock_model_manager.add_model.assert_called_once()
    inference_request = mock_model_manager.infer_from_request_sync.call_args[0][1]
    prompts = inference_request.prompts.prompts
    assert len(prompts) == 1
    assert len(prompts[0].points) == 2
    assert prompts[0].points[0].x == 320
    assert prompts[0].points[0].positive is True
    assert prompts[0].points[1].positive is False


def test_run_locally_with_boxes_and_points(
    mock_model_manager, mock_workflow_image_data
) -> None:
    block = SegmentAnything3InteractiveBlockV1(
        model_manager=mock_model_manager,
        api_key="test_api_key",
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    result = block.run(
        images=[mock_workflow_image_data],
        points=[[320, 240]],
        boxes=[_example_detections()],
        threshold=0.0,
        multimask_output=True,
    )

    assert len(result) == 1
    inference_request = mock_model_manager.infer_from_request_sync.call_args[0][1]
    prompts = inference_request.prompts.prompts
    # one prompt per box + one prompt for the points
    assert len(prompts) == 2
    assert prompts[0].box is not None
    assert prompts[0].box.x == 30  # box centre of [10, 10, 50, 50]
    assert prompts[1].points[0].x == 320


@patch(
    "inference.core.workflows.core_steps.models.foundation.segment_anything3_interactive.v1.InferenceHTTPClient"
)
def test_run_remotely_calls_sam3_visual_segment(
    mock_client_cls, mock_model_manager, mock_workflow_image_data
) -> None:
    mock_client = MagicMock()
    mock_client.sam3_visual_segment.return_value = {
        "predictions": [
            {
                "confidence": 0.95,
                "masks": [[[0, 0], [100, 0], [100, 100], [0, 100]]],
            }
        ],
        "time": 0.1,
    }
    mock_client_cls.return_value = mock_client

    block = SegmentAnything3InteractiveBlockV1(
        model_manager=mock_model_manager,
        api_key="test_api_key",
        step_execution_mode=StepExecutionMode.REMOTE,
    )

    result = block.run(
        images=[mock_workflow_image_data],
        points=[{"x": 320, "y": 240, "positive": True}],
        boxes=None,
        threshold=0.0,
        multimask_output=True,
    )

    assert len(result) == 1
    assert "predictions" in result[0]
    mock_client.sam3_visual_segment.assert_called_once()
    call_kwargs = mock_client.sam3_visual_segment.call_args.kwargs
    assert call_kwargs["prompts"] == [
        {"points": [{"x": 320.0, "y": 240.0, "positive": True}]}
    ]


@patch(
    "inference.core.workflows.core_steps.models.foundation.segment_anything3_interactive.v1.InferenceHTTPClient"
)
def test_run_remotely_with_boxes(
    mock_client_cls, mock_model_manager, mock_workflow_image_data
) -> None:
    mock_client = MagicMock()
    mock_client.sam3_visual_segment.return_value = {
        "predictions": [
            {
                "confidence": 0.95,
                "masks": [[[0, 0], [100, 0], [100, 100], [0, 100]]],
            }
        ],
        "time": 0.1,
    }
    mock_client_cls.return_value = mock_client

    block = SegmentAnything3InteractiveBlockV1(
        model_manager=mock_model_manager,
        api_key="test_api_key",
        step_execution_mode=StepExecutionMode.REMOTE,
    )

    result = block.run(
        images=[mock_workflow_image_data],
        points=None,
        boxes=[_example_detections()],
        threshold=0.0,
        multimask_output=True,
    )

    assert len(result) == 1
    call_kwargs = mock_client.sam3_visual_segment.call_args.kwargs
    assert call_kwargs["prompts"] == [
        {"box": {"x": 30.0, "y": 30.0, "width": 40.0, "height": 40.0}}
    ]
    # class name of the box prompt should be forwarded to predicted masks
    predictions = result[0]["predictions"]
    assert list(predictions["class_name"]) == ["object"]
