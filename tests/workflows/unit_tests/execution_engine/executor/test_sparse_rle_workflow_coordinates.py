import numpy as np
import supervision as sv

from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.constants import (
    POLYGON_KEY_IN_SV_DETECTIONS,
)
from inference.core.workflows.execution_engine.core import ExecutionEngine

WORKFLOW_WITH_CROP_SEGMENTATION_AND_FILTER = {
    "version": "1.0",
    "inputs": [{"type": "WorkflowImage", "name": "image"}],
    "steps": [
        {
            "type": "RelativeStaticCrop",
            "name": "crop",
            "image": "$inputs.image",
            "x_center": 0.4,
            "y_center": 0.375,
            "width": 0.4,
            "height": 0.5,
        },
        {
            "type": "roboflow_core/roboflow_instance_segmentation_model@v3",
            "name": "segmentation",
            "images": "$steps.crop.crops",
            "model_id": "rfdetr-seg-nano/1",
            "confidence_mode": "custom",
            "custom_confidence": 0.4,
        },
        {
            "type": "roboflow_core/detections_filter@v1",
            "name": "only_cars",
            "predictions": "$steps.segmentation.predictions",
            "operations": [
                {
                    "type": "DetectionsFilter",
                    "filter_operation": {
                        "type": "StatementGroup",
                        "operator": "and",
                        "statements": [
                            {
                                "type": "BinaryStatement",
                                "left_operand": {
                                    "type": "DynamicOperand",
                                    "operand_name": "_",
                                    "operations": [
                                        {
                                            "type": "ExtractDetectionProperty",
                                            "property_name": "class_name",
                                        }
                                    ],
                                },
                                "comparator": {"type": "in (Sequence)"},
                                "right_operand": {
                                    "type": "StaticOperand",
                                    "value": ["car"],
                                },
                            }
                        ],
                    },
                }
            ],
            "operations_parameters": {},
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "model_predictions",
            "selector": "$steps.segmentation.predictions",
        },
        {
            "type": "JsonField",
            "name": "filtered_predictions",
            "selector": "$steps.only_cars.predictions",
        },
    ],
}


class _FakeSparseRLEWorkflowResponse:
    _rfdetr_sparse_rle_postprocess = True

    def model_dump(self, by_alias: bool, exclude_none: bool) -> dict:
        assert by_alias is True
        assert exclude_none is True
        return {
            "image": {"width": 40, "height": 40},
            "predictions": [
                {
                    "x": 10,
                    "y": 10,
                    "width": 10,
                    "height": 10,
                    "confidence": 0.9,
                    "class": "car",
                    "class_id": 0,
                    # Forces the workflow converter through sv.Detections.from_inference,
                    # which is where stale polygon metadata can enter this path.
                    "rle": {"size": [40, 40], "counts": "unused-by-test"},
                }
            ],
        }


class _FakeModelManager:
    def __init__(self) -> None:
        self.add_model_calls = []
        self.infer_calls = []

    def add_model(self, model_id: str, api_key: str) -> None:
        self.add_model_calls.append((model_id, api_key))

    def infer_from_request_sync(self, model_id: str, request):
        self.infer_calls.append((model_id, request))
        return [_FakeSparseRLEWorkflowResponse()]

    def __contains__(self, model_id: str) -> bool:
        return False


def test_sparse_rle_segmentation_after_relative_crop_serializes_direct_output_in_root_coordinates(
    monkeypatch,
) -> None:
    # given
    model_manager = _FakeModelManager()
    image = np.zeros((80, 100, 3), dtype=np.uint8)

    def fake_from_inference(payload: dict) -> sv.Detections:
        assert payload["image"] == {"width": 40, "height": 40}
        mask = np.zeros((1, 40, 40), dtype=np.bool_)
        mask[0, 5:16, 5:16] = True
        return sv.Detections(
            xyxy=np.array([[5, 5, 15, 15]]),
            mask=mask,
            confidence=np.array([0.9]),
            class_id=np.array([0]),
            data={
                "class_name": np.array(["car"]),
                # Crop-local polygon metadata must not win over the shifted mask
                # when the sparse RLE workflow output is serialized.
                POLYGON_KEY_IN_SV_DETECTIONS: np.array(
                    [[[5, 5], [5, 15], [15, 15], [15, 5]]]
                ),
            },
        )

    monkeypatch.setattr(sv.Detections, "from_inference", fake_from_inference)
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_CROP_SEGMENTATION_AND_FILTER,
        init_parameters={
            "workflows_core.model_manager": model_manager,
            "workflows_core.api_key": None,
            "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
        },
        max_concurrent_steps=1,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={"image": image},
        serialize_results=True,
    )

    # then
    assert len(result) == 1
    model_prediction = result[0]["model_predictions"]["predictions"][0]
    filtered_prediction = result[0]["filtered_predictions"]["predictions"][0]
    assert model_prediction["x"] == 30
    assert model_prediction["y"] == 20
    assert model_prediction["width"] == 10
    assert model_prediction["height"] == 10
    assert _points_bounds(model_prediction["points"]) == (25, 15, 35, 25)
    assert "polygon" not in model_prediction
    assert _points_bounds(filtered_prediction["points"]) == (25, 15, 35, 25)
    assert "polygon" not in filtered_prediction


def _points_bounds(points: list[dict]) -> tuple[float, float, float, float]:
    x_values = [point["x"] for point in points]
    y_values = [point["y"] for point in points]
    return min(x_values), min(y_values), max(x_values), max(y_values)
