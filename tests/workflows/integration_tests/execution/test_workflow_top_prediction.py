import numpy as np
import supervision as sv

from inference.core.env import USE_INFERENCE_MODELS, WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine

TOP_PREDICTION_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "model_id"},
        {"type": "WorkflowParameter", "name": "confidence", "default_value": 0.3},
        {"type": "WorkflowParameter", "name": "classes"},
    ],
    "steps": [
        {
            "type": "ObjectDetectionModel",
            "name": "model",
            "image": "$inputs.image",
            "model_id": "$inputs.model_id",
            "confidence": "$inputs.confidence",
        },
        {
            "type": "DetectionsTransformation",
            "name": "take_top_prediction",
            "predictions": "$steps.model.predictions",
            "operations": [{"type": "DetectionsSelection", "mode": "top_confidence"}],
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "all_predictions",
            "selector": "$steps.model.predictions",
        },
        {
            "type": "JsonField",
            "name": "top_prediction",
            "selector": "$steps.take_top_prediction.predictions",
        },
    ],
}

EXPECTED_OBJECT_DETECTION_BBOXES_OLD_INFERENCE = np.array(
    [
        [180, 273, 244, 383],
        [271, 266, 328, 383],
        [552, 259, 598, 365],
        [113, 269, 145, 347],
        [416, 258, 457, 365],
        [521, 257, 555, 360],
        [387, 264, 414, 342],
        [158, 267, 183, 349],
        [324, 256, 345, 320],
        [341, 261, 362, 338],
        [247, 251, 262, 284],
        [239, 251, 249, 282],
    ]
)
EXPECTED_OBJECT_DETECTION_CONFIDENCES_OLD_INFERENCE = np.array(
    [
        0.84284,
        0.83957,
        0.81555,
        0.80455,
        0.75804,
        0.75794,
        0.71715,
        0.71408,
        0.71003,
        0.56938,
        0.54092,
        0.43511,
    ]
)


EXPECTED_OBJECT_DETECTION_BBOXES_NEW_INFERENCE = np.array(
    [
        [180, 273, 244, 384],
        [271, 267, 328, 384],
        [552, 260, 598, 365],
        [113, 270, 145, 348],
        [416, 259, 457, 365],
        [521, 257, 555, 360],
        [387, 264, 414, 342],
        [158, 268, 183, 350],
        [324, 257, 345, 321],
        [341, 262, 362, 338],
        [247, 251, 262, 285],
        [240, 251, 250, 282],
        [412, 265, 432, 337],
        [145, 265, 165, 329],
        [300, 264, 319, 296],
    ]
)
EXPECTED_OBJECT_DETECTION_CONFIDENCES_NEW_INFERENCE = np.array(
    [
        0.84284,
        0.83957,
        0.81555,
        0.80455,
        0.75804,
        0.75794,
        0.71715,
        0.71408,
        0.71003,
        0.56938,
        0.54092,
        0.51359,
        0.36387,
        0.35772,
        0.34012,
    ]
)


def test_filtering_workflow_to_include_only_top_prediction(
    model_manager: ModelManager,
    crowd_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=TOP_PREDICTION_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": crowd_image,
            "model_id": "yolov8n-640",
            "classes": {"person"},
        }
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 1, "Single image provided - single output expected"
    all_detections: sv.Detections = result[0]["all_predictions"]
    top_detections: sv.Detections = result[0]["top_prediction"]

    if not USE_INFERENCE_MODELS:
        assert len(all_detections) == 12, "Expected 12 total predictions"
        assert np.allclose(
            all_detections.xyxy,
            EXPECTED_OBJECT_DETECTION_BBOXES_OLD_INFERENCE,
            atol=1,
        ), "Expected bboxes to match what was validated manually as workflow outcome"
        assert np.allclose(
            all_detections.confidence,
            EXPECTED_OBJECT_DETECTION_CONFIDENCES_OLD_INFERENCE,
            atol=0.01,
        ), "Expected confidences to match what was validated manually as workflow outcome"
    else:
        assert len(all_detections) == 15, "Expected 15 total predictions"
        assert np.allclose(
            all_detections.xyxy,
            EXPECTED_OBJECT_DETECTION_BBOXES_NEW_INFERENCE,
            atol=1,
        ), "Expected bboxes to match what was validated manually as workflow outcome"
        assert np.allclose(
            all_detections.confidence,
            EXPECTED_OBJECT_DETECTION_CONFIDENCES_NEW_INFERENCE,
            atol=0.01,
        ), "Expected confidences to match what was validated manually as workflow outcome"
    assert len(top_detections) == 1, "Expected only one top prediction"
    assert np.allclose(
        top_detections.xyxy,
        [EXPECTED_OBJECT_DETECTION_BBOXES_OLD_INFERENCE[0]],
        atol=1,
    ), "Expected top bbox to match what was validated manually as workflow outcome"
    assert np.allclose(
        top_detections.confidence,
        [EXPECTED_OBJECT_DETECTION_CONFIDENCES_OLD_INFERENCE[0]],
        atol=0.01,
    ), "Expected top confidence to match what was validated manually as workflow outcome"


def test_filtering_workflow_by_top_prediction_with_no_detections(
    model_manager: ModelManager,
    red_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=TOP_PREDICTION_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": red_image,
            "model_id": "yolov8n-640",
            "classes": {"not_present"},
        }
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 1, "Single image provided - single output expected"
    all_detections: sv.Detections = result[0]["all_predictions"]
    top_detections: sv.Detections = result[0]["top_prediction"]

    assert len(all_detections) == 0, "Expected 0 total predictions"
    assert len(top_detections) == 0, "Expected top prediction to be an empty array"


SORT_DETECTIONS_WORKFLOW_LAST = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "model_id"},
        {"type": "WorkflowParameter", "name": "confidence", "default_value": 0.3},
        {"type": "WorkflowParameter", "name": "classes"},
    ],
    "steps": [
        {
            "type": "ObjectDetectionModel",
            "name": "model",
            "image": "$inputs.image",
            "model_id": "$inputs.model_id",
            "confidence": "$inputs.confidence",
        },
        {
            "type": "roboflow_core/property_definition@v1",
            "name": "property_definition",
            "data": "$steps.model.predictions",
            "operations": [
                {"type": "SortDetections", "mode": "confidence", "ascending": True},
                {"type": "DetectionsSelection", "mode": "last"},
            ],
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "all_predictions",
            "selector": "$steps.model.predictions",
        },
        {
            "type": "JsonField",
            "name": "selected_box",
            "selector": "$steps.property_definition.output",
        },
    ],
}


def test_extracting_largest_bbox_from_detections(
    model_manager: ModelManager,
    crowd_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=SORT_DETECTIONS_WORKFLOW_LAST,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": crowd_image,
            "model_id": "yolov8n-640",
            "classes": {"person"},
        }
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 1, "Single image provided - single output expected"
    all_detections: sv.Detections = result[0]["all_predictions"]
    selected_box: sv.Detections = result[0]["selected_box"]

    if not USE_INFERENCE_MODELS:
        assert len(all_detections) == 12, "Expected 12 total predictions"
        assert np.allclose(
            all_detections.xyxy,
            EXPECTED_OBJECT_DETECTION_BBOXES_OLD_INFERENCE,
            atol=1,
        ), "Expected bboxes to match what was validated manually as workflow outcome"
        assert np.allclose(
            all_detections.confidence,
            EXPECTED_OBJECT_DETECTION_CONFIDENCES_OLD_INFERENCE,
            atol=0.01,
        ), "Expected confidences to match what was validated manually as workflow outcome"
    else:
        assert len(all_detections) == 15, "Expected 15 total predictions"
        assert np.allclose(
            all_detections.xyxy,
            EXPECTED_OBJECT_DETECTION_BBOXES_NEW_INFERENCE,
            atol=1,
        ), "Expected bboxes to match what was validated manually as workflow outcome"
        assert np.allclose(
            all_detections.confidence,
            EXPECTED_OBJECT_DETECTION_CONFIDENCES_NEW_INFERENCE,
            atol=0.01,
        ), "Expected confidences to match what was validated manually as workflow outcome"
    assert len(selected_box) == 1, "Expected only one top prediction"
    assert np.allclose(
        selected_box.xyxy,
        [EXPECTED_OBJECT_DETECTION_BBOXES_OLD_INFERENCE[0]],
        atol=1,
    ), "Expected top bbox to match what was validated manually as workflow outcome"
    assert np.allclose(
        selected_box.confidence,
        [EXPECTED_OBJECT_DETECTION_CONFIDENCES_OLD_INFERENCE[0]],
        atol=0.01,
    ), "Expected top confidence to match what was validated manually as workflow outcome"


SORT_DETECTIONS_WORKFLOW_FIRST = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "model_id"},
        {"type": "WorkflowParameter", "name": "confidence", "default_value": 0.3},
        {"type": "WorkflowParameter", "name": "classes"},
    ],
    "steps": [
        {
            "type": "ObjectDetectionModel",
            "name": "model",
            "image": "$inputs.image",
            "model_id": "$inputs.model_id",
            "confidence": "$inputs.confidence",
        },
        {
            "type": "roboflow_core/property_definition@v1",
            "name": "property_definition",
            "data": "$steps.model.predictions",
            "operations": [
                {"type": "SortDetections", "mode": "confidence", "ascending": True},
                {"type": "DetectionsSelection", "mode": "first"},
            ],
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "all_predictions",
            "selector": "$steps.model.predictions",
        },
        {
            "type": "JsonField",
            "name": "first_box",
            "selector": "$steps.property_definition.output",
        },
    ],
}


def test_extracting_smallest_bbox_from_detections(
    model_manager: ModelManager,
    crowd_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=SORT_DETECTIONS_WORKFLOW_FIRST,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": crowd_image,
            "model_id": "yolov8n-640",
            "classes": {"person"},
        }
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 1, "Single image provided - single output expected"
    all_detections: sv.Detections = result[0]["all_predictions"]
    first_box: sv.Detections = result[0]["first_box"]

    if not USE_INFERENCE_MODELS:
        assert len(all_detections) == 12, "Expected 12 total predictions"
        assert np.allclose(
            all_detections.xyxy,
            EXPECTED_OBJECT_DETECTION_BBOXES_OLD_INFERENCE,
            atol=1,
        ), "Expected bboxes to match what was validated manually as workflow outcome"
        assert np.allclose(
            all_detections.confidence,
            EXPECTED_OBJECT_DETECTION_CONFIDENCES_OLD_INFERENCE,
            atol=0.01,
        ), "Expected confidences to match what was validated manually as workflow outcome"
        assert len(first_box) == 1, "Expected only one prediction"
        assert np.allclose(
            first_box.xyxy,
            [EXPECTED_OBJECT_DETECTION_BBOXES_OLD_INFERENCE[-1]],
            atol=1,
        ), "Expected top bbox to match what was validated manually as workflow outcome"
        assert np.allclose(
            first_box.confidence,
            [EXPECTED_OBJECT_DETECTION_CONFIDENCES_OLD_INFERENCE[-1]],
            atol=0.01,
        ), "Expected top confidence to match what was validated manually as workflow outcome"

    else:
        assert len(all_detections) == 15, "Expected 15 total predictions"
        assert np.allclose(
            all_detections.xyxy,
            EXPECTED_OBJECT_DETECTION_BBOXES_NEW_INFERENCE,
            atol=1,
        ), "Expected bboxes to match what was validated manually as workflow outcome"
        assert np.allclose(
            all_detections.confidence,
            EXPECTED_OBJECT_DETECTION_CONFIDENCES_NEW_INFERENCE,
            atol=0.01,
        ), "Expected confidences to match what was validated manually as workflow outcome"
        assert len(first_box) == 1, "Expected only one prediction"
        assert np.allclose(
            first_box.xyxy,
            [EXPECTED_OBJECT_DETECTION_BBOXES_NEW_INFERENCE[-1]],
            atol=1,
        ), "Expected top bbox to match what was validated manually as workflow outcome"
        assert np.allclose(
            first_box.confidence,
            [EXPECTED_OBJECT_DETECTION_CONFIDENCES_NEW_INFERENCE[-1]],
            atol=0.01,
        ), "Expected top confidence to match what was validated manually as workflow outcome"
