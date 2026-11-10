import numpy as np
import pytest
import supervision as sv
import torch

from inference.core.env import ENABLE_TENSOR_DATA_REPRESENTATION
from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference_models.models.base.object_detection import Detections


@pytest.mark.parametrize(
    ("scale", "origin", "expected_parent"),
    [
        (0.5, (40, 30), [60, 54, 100, 86]),
        (0.5, (0, 0), [20, 24, 60, 56]),
        (2.0, (40, 30), [45, 36, 55, 44]),
    ],
)
def test_workflow_preserves_own_and_restores_parent_coordinates(
    scale: float, origin: tuple[int, int], expected_parent: list[int]
) -> None:
    """Both output selectors keep their frame when a prediction has a root scale."""
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
                "type": "roboflow_core/detections_filter@v1",
                "name": "identity",
                "predictions": "$inputs.detections",
                "operations": [],
            }
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": frame,
                "selector": "$steps.identity.predictions",
                "coordinates_system": frame,
            }
            for frame in ("own", "parent")
        ],
    }
    root_dimensions = [144, 220]
    metadata = {
        "image_dimensions": root_dimensions if origin == (0, 0) else [64, 80],
        "parent_dimensions": root_dimensions,
        "root_parent_dimensions": root_dimensions,
        "parent_coordinates": list(origin),
        "root_parent_coordinates": list(origin),
        "scaling_relative_to_root_parent": scale,
        "root_parent_id": "root",
        "parent_id": "root" if origin == (0, 0) else "crop",
        "prediction_type": "object-detection",
    }
    box = [[10.0, 12.0, 30.0, 28.0]]
    if ENABLE_TENSOR_DATA_REPRESENTATION:
        prediction = Detections(
            xyxy=torch.tensor(box),
            class_id=torch.tensor([0]),
            confidence=torch.tensor([0.95]),
            bboxes_metadata=[{"detection_id": "box"}],
            image_metadata={**metadata, "class_names": {0: "object"}},
        )
    else:
        prediction = sv.Detections(
            xyxy=np.array(box),
            class_id=np.array([0]),
            confidence=np.array([0.95]),
            data={
                **{key: np.array([value]) for key, value in metadata.items()},
                "detection_id": np.array(["box"]),
                "class_name": np.array(["object"]),
            },
        )

    engine = ExecutionEngine.init(workflow_definition=definition, init_parameters={})
    try:
        result = engine.run(
            runtime_parameters={"detections": [prediction]}, serialize_results=False
        )[0]
    finally:
        engine.close()

    assert isinstance(result["own"], type(prediction))
    assert result["own"].xyxy.tolist() == box
    assert result["parent"].xyxy.tolist() == [expected_parent]
