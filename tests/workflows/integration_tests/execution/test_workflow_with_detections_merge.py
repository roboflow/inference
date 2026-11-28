import numpy as np
import pytest
import supervision as sv
import torch

from inference.core.env import (
    ENABLE_TENSOR_DATA_REPRESENTATION,
    WORKFLOWS_MAX_CONCURRENT_STEPS,
)
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.constants import (
    CLASS_NAMES_KEY,
    DETECTION_ID_KEY,
)
from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference_models.models.base.object_detection import Detections as NativeDetections
from tests.workflows.integration_tests.execution.tensor_input_utils import (
    numpy_image_as_tensor,
)
from tests.workflows.integration_tests.execution.workflows_gallery_collector.decorators import (
    add_to_workflows_gallery,
)

# Under ENABLE_TENSOR_DATA_REPRESENTATION the detections_merge block emits a native
# inference_models.Detections instead of sv.Detections. The numpy-shaped assertions
# below run only with the flag off; an equivalent *_tensor_native test asserts the same
# facts against the native carrier with the flag on.
_NUMPY_ONLY = pytest.mark.skipif(
    ENABLE_TENSOR_DATA_REPRESENTATION,
    reason="sv.Detections output; native under ENABLE_TENSOR_DATA_REPRESENTATION "
    "— see the *_tensor_native parity test",
)
_TENSOR_ONLY = pytest.mark.skipif(
    not ENABLE_TENSOR_DATA_REPRESENTATION,
    reason="tensor-native variant; runs only with ENABLE_TENSOR_DATA_REPRESENTATION=True",
)

DETECTIONS_MERGE_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
    ],
    "steps": [
        {
            "type": "ObjectDetectionModel",
            "name": "detection",
            "image": "$inputs.image",
            "model_id": "yolov8n-640",
        },
        {
            "type": "roboflow_core/detections_merge@v1",
            "name": "detections_merge",
            "predictions": "$steps.detection.predictions",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "result",
            "selector": "$steps.detections_merge.predictions",
        }
    ],
}


@_NUMPY_ONLY
@add_to_workflows_gallery(
    category="Basic Workflows",
    use_case_title="Workflow with detections merge",
    use_case_description="""
This workflow demonstrates how to merge multiple object detections into a single bounding box.
This is useful when you want to:
- Combine overlapping detections of the same object
- Create a single region that contains multiple detected objects
- Simplify multiple detections into one larger detection
    """,
    workflow_definition=DETECTIONS_MERGE_WORKFLOW,
    workflow_name_in_app="merge-detections",
)
def test_detections_merge_workflow(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=DETECTIONS_MERGE_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [dogs_image],
        }
    )

    # then
    assert len(result) == 1, "One set of outputs expected"
    assert "result" in result[0], "Output must contain key 'result'"
    assert isinstance(
        result[0]["result"], sv.Detections
    ), "Output must be instance of sv.Detections"

    # Check that we have exactly one merged detection
    assert len(result[0]["result"]) == 1, "Should have exactly one merged detection"

    # Check that the merged detection has all required fields
    assert "class_name" in result[0]["result"].data, "Should have class_name in data"
    assert (
        "detection_id" in result[0]["result"].data
    ), "Should have detection_id in data"

    # Check that the bounding box has reasonable dimensions
    merged_bbox = result[0]["result"].xyxy[0]
    image_height, image_width = dogs_image.shape[:2]

    # Check that coordinates are within image bounds
    assert 0 <= merged_bbox[0] <= image_width, "x1 should be within image bounds"
    assert 0 <= merged_bbox[1] <= image_height, "y1 should be within image bounds"
    assert 0 <= merged_bbox[2] <= image_width, "x2 should be within image bounds"
    assert 0 <= merged_bbox[3] <= image_height, "y2 should be within image bounds"

    # Check that the box has reasonable dimensions
    assert merged_bbox[2] > merged_bbox[0], "x2 should be greater than x1"
    assert merged_bbox[3] > merged_bbox[1], "y2 should be greater than y1"

    # Check that the box is large enough to likely contain the dogs
    box_width = merged_bbox[2] - merged_bbox[0]
    box_height = merged_bbox[3] - merged_bbox[1]
    assert box_width > 100, "Merged box should be reasonably wide"
    assert box_height > 100, "Merged box should be reasonably tall"


@_TENSOR_ONLY
def test_detections_merge_workflow_tensor_native(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
    image_as_workflow_input,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=DETECTIONS_MERGE_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [image_as_workflow_input(dogs_image)],
        }
    )

    # then
    assert len(result) == 1, "One set of outputs expected"
    assert "result" in result[0], "Output must contain key 'result'"
    merged = result[0]["result"]
    assert isinstance(
        merged, NativeDetections
    ), "Output must be instance of native inference_models.Detections"

    # Check that we have exactly one merged detection
    assert merged.xyxy.shape[0] == 1, "Should have exactly one merged detection"

    # Check that the merged detection carries the required fields on the native
    # carrier: class name lives in image_metadata[CLASS_NAMES_KEY]; detection_id is a
    # per-box entry in bboxes_metadata (the native equivalents of sv.Detections.data).
    assert (
        CLASS_NAMES_KEY in merged.image_metadata
    ), "Should have class_names in metadata"
    assert (
        "merged_detection" in merged.image_metadata[CLASS_NAMES_KEY].values()
    ), "Merged detection should carry the merged class name"
    assert merged.bboxes_metadata is not None, "Should have per-box metadata"
    assert (
        DETECTION_ID_KEY in merged.bboxes_metadata[0]
    ), "Should have detection_id in per-box metadata"

    # Check that the bounding box has reasonable dimensions
    merged_bbox = merged.xyxy[0].tolist()
    image_height, image_width = dogs_image.shape[:2]

    # Check that coordinates are within image bounds
    assert 0 <= merged_bbox[0] <= image_width, "x1 should be within image bounds"
    assert 0 <= merged_bbox[1] <= image_height, "y1 should be within image bounds"
    assert 0 <= merged_bbox[2] <= image_width, "x2 should be within image bounds"
    assert 0 <= merged_bbox[3] <= image_height, "y2 should be within image bounds"

    # Check that the box has reasonable dimensions
    assert merged_bbox[2] > merged_bbox[0], "x2 should be greater than x1"
    assert merged_bbox[3] > merged_bbox[1], "y2 should be greater than y1"

    # Check that the box is large enough to likely contain the dogs
    box_width = merged_bbox[2] - merged_bbox[0]
    box_height = merged_bbox[3] - merged_bbox[1]
    assert box_width > 100, "Merged box should be reasonably wide"
    assert box_height > 100, "Merged box should be reasonably tall"


@_TENSOR_ONLY
def test_detections_merge_workflow_with_tensor_input(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
) -> None:
    # Same as test_detections_merge_workflow_tensor_native, but the image arrives ALREADY
    # materialised as a CHW RGB device tensor (is_tensor_materialised() == True), so the OD
    # block runs its on-device tensor path. Results must match the numpy-input variant.
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=DETECTIONS_MERGE_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when — feed the fixture as a pre-materialised tensor
    result = execution_engine.run(
        runtime_parameters={
            "image": [numpy_image_as_tensor(dogs_image)],
        }
    )

    # then
    assert len(result) == 1, "One set of outputs expected"
    assert "result" in result[0], "Output must contain key 'result'"
    merged = result[0]["result"]
    assert isinstance(
        merged, NativeDetections
    ), "Output must be instance of native inference_models.Detections"
    assert merged.xyxy.shape[0] == 1, "Should have exactly one merged detection"
    assert (
        CLASS_NAMES_KEY in merged.image_metadata
    ), "Should have class_names in metadata"
    assert (
        "merged_detection" in merged.image_metadata[CLASS_NAMES_KEY].values()
    ), "Merged detection should carry the merged class name"
    assert merged.bboxes_metadata is not None, "Should have per-box metadata"
    assert (
        DETECTION_ID_KEY in merged.bboxes_metadata[0]
    ), "Should have detection_id in per-box metadata"

    merged_bbox = merged.xyxy[0].tolist()
    image_height, image_width = dogs_image.shape[:2]
    assert 0 <= merged_bbox[0] <= image_width, "x1 should be within image bounds"
    assert 0 <= merged_bbox[1] <= image_height, "y1 should be within image bounds"
    assert 0 <= merged_bbox[2] <= image_width, "x2 should be within image bounds"
    assert 0 <= merged_bbox[3] <= image_height, "y2 should be within image bounds"
    assert merged_bbox[2] > merged_bbox[0], "x2 should be greater than x1"
    assert merged_bbox[3] > merged_bbox[1], "y2 should be greater than y1"
    assert merged_bbox[2] - merged_bbox[0] > 100, "Merged box should be reasonably wide"
    assert merged_bbox[3] - merged_bbox[1] > 100, "Merged box should be reasonably tall"


@pytest.mark.parametrize("origin", [(40, 30), (0, 0)])
@pytest.mark.parametrize(
    ("box_dtype", "empty"),
    [
        pytest.param(np.float32, False, id="float"),
        pytest.param(np.uint8, False, id="uint8"),
        pytest.param(np.float32, True, id="empty"),
    ],
)
def test_detections_merge_preserves_image_frame(
    origin: tuple[int, int], box_dtype: type, empty: bool
) -> None:
    definition = {
        "version": "1.0",
        "inputs": [
            {
                "type": "WorkflowBatchInput",
                "name": "detections",
                "kind": ["object_detection_prediction"],
                "dimensionality": 1,
            }
        ],
        "steps": [
            {
                "type": "roboflow_core/detections_merge@v1",
                "name": "merge",
                "predictions": "$inputs.detections",
                "class_name": "union",
            }
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": frame,
                "selector": "$steps.merge.predictions",
                "coordinates_system": frame,
            }
            for frame in ("own", "parent")
        ],
    }
    frame_metadata = {
        "image_dimensions": [60, 80],
        "parent_id": "crop",
        "parent_coordinates": list(origin),
        "parent_dimensions": [120, 160],
        "root_parent_id": "image",
        "root_parent_coordinates": list(origin),
        "root_parent_dimensions": [120, 160],
    }
    boxes = [] if empty else [[10, 12, 20, 22], [16, 18, 30, 28]]
    confidences = [] if empty else [0.85, 0.95]
    classes = [0] * len(boxes)
    if ENABLE_TENSOR_DATA_REPRESENTATION:
        detections = NativeDetections(
            xyxy=torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            confidence=torch.tensor(confidences, dtype=torch.float32),
            class_id=torch.tensor(classes, dtype=torch.long),
            image_metadata={**frame_metadata, CLASS_NAMES_KEY: {0: "object"}},
        )
    else:
        detections = sv.Detections(
            xyxy=np.array(boxes, dtype=box_dtype).reshape(-1, 4),
            confidence=np.array(confidences, dtype=np.float32),
            class_id=np.array(classes, dtype=int),
            data={
                key: np.array([value] * len(boxes))
                for key, value in frame_metadata.items()
            },
            metadata=frame_metadata if empty else {},
        )

    engine = ExecutionEngine.init(workflow_definition=definition, init_parameters={})
    try:
        (result,) = engine.run(
            runtime_parameters={"detections": [detections]}, serialize_results=True
        )
    finally:
        engine.close()

    assert result["own"]["image"] == {"width": 80, "height": 60}
    assert result["parent"]["image"] == {"width": 160, "height": 120}
    if empty:
        assert result["own"]["predictions"] == []
        assert result["parent"]["predictions"] == []
    else:
        (own,) = result["own"]["predictions"]
        (parent,) = result["parent"]["predictions"]
        assert (own["x"], own["y"]) == (20.0, 20.0)
        assert (parent["x"], parent["y"]) == (20.0 + origin[0], 20.0 + origin[1])
        for row in (own, parent):
            assert (row["width"], row["height"]) == (20.0, 16.0)
            assert row["class"] == "union"
            assert row["confidence"] == pytest.approx(0.85)
        assert own["parent_id"] == "crop"
        assert parent["parent_id"] == "image"
