# How to run `workflow` with new `execution engine`?

This is example `workflow` run through `execution engine`:

```python
from inference.core.managers.base import ModelManager
from inference.core.registries.roboflow import (
    RoboflowModelRegistry,
)
from inference.models.utils import ROBOFLOW_MODEL_TYPES
from inference.enterprise.workflows.execution_engine.core import ExecutionEngine

model_registry = RoboflowModelRegistry(ROBOFLOW_MODEL_TYPES)
model_manager = ModelManager(model_registry=model_registry)


SIMPLE_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "InferenceImage", "name": "image" },
        {"type": "InferenceParameter", "name": "confidence"},
    ],
    "steps": [
        {
            "type": "ObjectDetectionModel",
            "name": "step_1",
            "image": "$inputs.image",
            "model_id": "yolov8n-640",
            "confidence": "$inputs.confidence",
            "iou_threshold": 0.3,
        },
        {
            "type": "DetectionOffset",
            "name": "step_2",
            "predictions": "$steps.step_1.predictions",
            "offset_x": 20,
            "offset_y": 20,
            "image_metadata": "$steps.step_1.image",
            "prediction_type": "$steps.step_1.prediction_type",
        },
        {
            "type": "Crop",
            "name": "step_3",
            "image": "$inputs.image",
            "predictions": "$steps.step_2.predictions",
        },
        {
            "type": "ObjectDetectionModel",
            "name": "step_4",
            "image": "$steps.step_3.crops",
            "model_id": "yolov8n-640",
            "confidence": 0.4,
            "iou_threshold": 0.3,
        },
    ],
    "outputs": [
        { "type": "JsonField", "name": "predictions_1", "selector": "$steps.step_1.predictions" },
        { "type": "JsonField", "name": "crops", "selector": "$steps.step_3.crops" },
        { "type": "JsonField", "name": "predictions_2", "selector": "$steps.step_4.predictions" },
    ]   
}


execution_engine = ExecutionEngine.init(
    workflow_definition=SIMPLE_WORKFLOW,
    init_parameters={
        "workflows_core.model_manager": model_manager,  # this init parameter is required by Roboflow `blocks` we use
        "workflows_core.api_key": "XXX",  # this init parameter is required by Roboflow `blocks` we use
    },
)

result = execution_engine.run(
    runtime_parameters={
        "image": {"type": "url", "value": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7d/Titans_Texans.jpg/1200px-Titans_Texans.jpg"},
        "confidence": 0.8,
    }
)
```
