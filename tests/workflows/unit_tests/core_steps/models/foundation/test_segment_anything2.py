"""Unit tests for Segment Anything 2 block including remote execution."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.models.foundation.segment_anything2.v1 import (
    BlockManifest,
    SegmentAnything2BlockV1,
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
    mock.run_sam2_segmentation.return_value = [MagicMock(predictions=[mock_prediction])]
    return mock


@pytest.fixture
def mock_workflow_image_data():
    start_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some"),
        numpy_image=start_image,
    )


def test_manifest_parsing_valid():
    data = {
        "type": "roboflow_core/segment_anything@v1",
        "name": "my_sam2_step",
        "images": "$inputs.image",
        "version": "hiera_tiny",
    }
    result = BlockManifest.model_validate(data)
    assert result.type == "roboflow_core/segment_anything@v1"
    assert result.version == "hiera_tiny"


@patch(
    "inference.core.workflows.core_steps.models.foundation.segment_anything2.v1.InferenceHTTPClient"
)
def test_run_remotely_calls_sam2_segment_image(
    mock_client_cls, mock_model_manager, mock_workflow_image_data
):
    """Test that remote execution uses the sam2_segment_image client method."""
    mock_client = MagicMock()
    mock_client.sam2_segment_image.return_value = {
        "predictions": [
            {
                "confidence": 0.95,
                "masks": [[[0, 0], [100, 0], [100, 100], [0, 100]]],
            }
        ]
    }
    mock_client_cls.return_value = mock_client

    block = SegmentAnything2BlockV1(
        model_manager=mock_model_manager,
        api_key="test_api_key",
        step_execution_mode=StepExecutionMode.REMOTE,
    )

    result = block.run(
        images=[mock_workflow_image_data],
        boxes=None,
        version="hiera_tiny",
        threshold=0.0,
        multimask_output=True,
    )

    assert len(result) == 1
    assert "predictions" in result[0]
    mock_client.sam2_segment_image.assert_called_once()


@patch(
    "inference.core.workflows.core_steps.models.foundation.segment_anything2.v1.InferenceHTTPClient"
)
def test_run_remotely_with_prompts(
    mock_client_cls, mock_model_manager, mock_workflow_image_data
):
    """Test that remote execution passes prompts correctly."""
    import supervision as sv

    mock_client = MagicMock()
    mock_client.sam2_segment_image.return_value = {
        "predictions": [
            {
                "confidence": 0.95,
                "masks": [[[0, 0], [100, 0], [100, 100], [0, 100]]],
            }
        ]
    }
    mock_client_cls.return_value = mock_client

    # Create mock detections with boxes
    detections = sv.Detections(
        xyxy=np.array([[10, 10, 50, 50]]),
        confidence=np.array([0.9]),
        class_id=np.array([0]),
    )
    detections["class_name"] = np.array(["object"])
    detections["detection_id"] = np.array(["det_1"])

    block = SegmentAnything2BlockV1(
        model_manager=mock_model_manager,
        api_key="test_api_key",
        step_execution_mode=StepExecutionMode.REMOTE,
    )

    result = block.run(
        images=[mock_workflow_image_data],
        boxes=[detections],
        version="hiera_tiny",
        threshold=0.0,
        multimask_output=True,
    )

    assert len(result) == 1
    mock_client.sam2_segment_image.assert_called_once()
    # Verify prompts were passed
    call_args = mock_client.sam2_segment_image.call_args
    assert call_args.kwargs.get("prompts") is not None


def test_convert_sam2_response_produces_the_same_dict_as_the_pydantic_form() -> None:
    import numpy as np

    from inference.core.entities.responses.inference import (
        InferenceResponseImage,
        InstanceSegmentationInferenceResponse,
        InstanceSegmentationPrediction,
        Point,
    )
    from inference.core.workflows.core_steps.common.segmentation_entities import (
        Sam2SegmentationPrediction,
    )
    from inference.core.workflows.core_steps.models.foundation.segment_anything2.v1 import (
        convert_sam2_segmentation_response_to_inference_instances_seg_response,
    )
    from inference.core.workflows.execution_engine.entities.base import (
        ImageParentMetadata,
        WorkflowImageData,
    )

    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="p"),
        numpy_image=np.zeros((20, 10, 3), dtype=np.uint8),
    )
    result = convert_sam2_segmentation_response_to_inference_instances_seg_response(
        sam2_segmentation_predictions=[
            # the parser's own input shape: a str confidence the pydantic class coerces
            Sam2SegmentationPrediction(
                masks=[[[0, 0], [4, 0], [4, 4]]], confidence="0.9"
            )
        ],
        image=image,
        prompt_class_ids=[1],
        prompt_class_names=["cat"],
        prompt_detection_ids=["d1"],
        threshold=0.1,
    )
    produced = result.to_dict()
    expected = InstanceSegmentationInferenceResponse(
        image=InferenceResponseImage(width=10, height=20),
        predictions=[
            InstanceSegmentationPrediction(
                **{
                    "x": 2.0,
                    "y": 2.0,
                    "width": 4.0,
                    "height": 4.0,
                    "confidence": 0.9,
                    "class": "cat",
                    "class_id": 1,
                    "parent_id": "d1",
                    "detection_id": produced["predictions"][0]["detection_id"],
                    "points": [Point(x=0, y=0), Point(x=4, y=0), Point(x=4, y=4)],
                }
            )
        ],
    ).model_dump(by_alias=True, exclude_none=True)
    assert produced == expected


def test_sam2_v1_local_sends_the_box_centre_prompt_it_computed() -> None:
    """Drives `run_locally` with one detection so the box-centre arithmetic and
    the prompt encoding both execute. A replacement range that swallowed the
    `cx`/`cy` assignments raises NameError here."""
    import supervision as sv

    from inference.core.roboflow_api import ModelEndpointType
    from inference.core.workflows.core_steps.models.foundation.segment_anything2.v1 import (
        DETECTION_ID_FIELD,
        DETECTIONS_CLASS_NAME_FIELD,
    )
    from inference.core.workflows.execution_engine.entities.base import Batch

    detections = sv.Detections(
        xyxy=np.array([[10.0, 10.0, 50.0, 50.0]], dtype=np.float32),
        confidence=np.array([0.9], dtype=np.float32),
        class_id=np.array([0]),
        data={
            DETECTIONS_CLASS_NAME_FIELD: np.array(["object"]),
            DETECTION_ID_FIELD: np.array(["d1"]),
        },
    )
    model_manager = MagicMock()
    model_manager.run_sam2_segmentation.return_value = [MagicMock(predictions=[])]
    block = SegmentAnything2BlockV1(
        model_manager=model_manager,
        api_key="k",
        step_execution_mode=StepExecutionMode.LOCAL,
    )
    images = Batch(
        content=[
            WorkflowImageData(
                parent_metadata=ImageParentMetadata(parent_id="p"),
                numpy_image=np.zeros((100, 100, 3), dtype=np.uint8),
            )
        ],
        indices=[(0,)],
    )
    block.run_locally(
        images=images,
        boxes=Batch(content=[detections], indices=[(0,)]),
        version="hiera_large",
        threshold=0.0,
        multimask_output=True,
    )
    prompts = model_manager.run_sam2_segmentation.call_args.kwargs["prompts"]
    # centre of [10, 10, 50, 50] is (30, 30) with width/height 40
    assert prompts == [{"box": {"x": 30.0, "y": 30.0, "width": 40.0, "height": 40.0}}]
    # Registration stayed in the block, in its original position. Asserted by
    # enum coercion: `load_core_model` passes the enum before Phase 9 and the
    # string "core_model" after it (Task 9.2) - `ModelEndpointType(...)` maps
    # both to CORE_MODEL, and `ModelEndpointType.CORE_MODEL == "core_model"`
    # is False (plain Enum, roboflow_api.py:582).
    args, kwargs = model_manager.add_model.call_args
    assert args == ("sam2/hiera_large", "k")
    assert set(kwargs) == {"endpoint_type"}
    assert ModelEndpointType(kwargs["endpoint_type"]) is ModelEndpointType.CORE_MODEL
