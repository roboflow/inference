import numpy as np
from roboflow_workflows.execution_engine.core import ExecutionEngine


def test_dynamic_crop_bounds_through_execution_engine() -> None:
    definition = {
        "version": "1.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
            {
                "type": "WorkflowBatchInput",
                "name": "detections",
                "kind": ["object_detection_prediction"],
                "dimensionality": 1,
            },
        ],
        "steps": [
            {
                "type": "roboflow_core/dynamic_crop@v1",
                "name": "crop",
                "images": "$inputs.image",
                "predictions": "$inputs.detections",
            }
        ],
        "outputs": [
            {"type": "JsonField", "name": "crops", "selector": "$steps.crop.crops"}
        ],
    }
    pixels = np.arange(20 * 30 * 3, dtype=np.uint16).astype(np.uint8).reshape(20, 30, 3)
    # The serialized boundary is accepted by both NumPy and tensor-native modes.
    detections = {
        "image": {"height": 20, "width": 30},
        "predictions": [
            {
                "x": 3,
                "y": 10,
                "width": 18,
                "height": 12,
                "confidence": 0.9,
                "class_id": 0,
                "class": "thing",
                "detection_id": "partial",
            },
            {
                "x": -12.5,
                "y": 10,
                "width": 15,
                "height": 12,
                "confidence": 0.9,
                "class_id": 0,
                "class": "thing",
                "detection_id": "outside",
            },
            {
                "x": 8.5,
                "y": 11,
                "width": 11,
                "height": 12,
                "confidence": 0.9,
                "class_id": 0,
                "class": "thing",
                "detection_id": "inside",
            },
        ],
    }
    engine = ExecutionEngine.init(workflow_definition=definition, init_parameters={})
    try:
        result = engine.run(
            runtime_parameters={"image": pixels, "detections": [detections]}
        )
    finally:
        engine.close()

    assert len(result) == 1
    crops = result[0]["crops"]
    assert len(crops) == 3
    assert crops[1] is None
    np.testing.assert_array_equal(crops[0].numpy_image, pixels[4:16, :12])
    np.testing.assert_array_equal(crops[2].numpy_image, pixels[5:17, 3:14])
    assert crops[0].parent_metadata.parent_id == "partial"
    assert crops[2].parent_metadata.parent_id == "inside"
