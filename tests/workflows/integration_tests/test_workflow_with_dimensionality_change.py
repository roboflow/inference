from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import pytest

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference.core.workflows.execution_engine.introspection import blocks_loader

DETECTIONS_TO_PARENT_COORDINATES_BATCH_VARIANT_WORKFLOW = {
    "version": "1.0",
    "inputs": [{"type": "WorkflowImage", "name": "image"}],
    "steps": [
        {
            "type": "ObjectDetectionModel",
            "name": "first_detection",
            "image": "$inputs.image",
            "model_id": "yolov8n-640",
        },
        {
            "type": "DetectionsTransformation",
            "name": "enlarging_boxes",
            "predictions": "$steps.first_detection.predictions",
            "operations": [
                {"type": "DetectionsOffset", "offset_x": 50, "offset_y": 50}
            ],
        },
        {
            "type": "Crop",
            "name": "cropping",
            "image": "$inputs.image",
            "predictions": "$steps.enlarging_boxes.predictions",
        },
        {
            "type": "ObjectDetectionModel",
            "name": "specialised_detection",
            "image": "$steps.cropping.crops",
            "model_id": "yolov8n-640",
        },
        {
            "type": "DetectionsToParentsCoordinatesBatch",
            "name": "coordinates_transform",
            "images": "$inputs.image",
            "images_predictions": "$steps.specialised_detection.predictions",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "predictions_in_own_coordinates",
            "selector": "$steps.specialised_detection.predictions",
            "coordinates_system": "own",
        },
        {
            "type": "JsonField",
            "name": "predictions_in_original_coordinates",
            "selector": "$steps.coordinates_transform.predictions",
            "coordinates_system": "own",
        },
    ],
}


@pytest.mark.asyncio
@mock.patch.object(blocks_loader, "get_plugin_modules")
async def test_workflow_with_detections_coordinates_transformation_in_batch_variant(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
    crowd_image: np.ndarray,
) -> None:
    # given
    get_plugin_modules_mock.return_value = [
        "tests.workflows.integration_tests.dimensionality_manipulation_plugin"
    ]
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=DETECTIONS_TO_PARENT_COORDINATES_BATCH_VARIANT_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = await execution_engine.run_async(
        runtime_parameters={"image": [crowd_image, crowd_image]}
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 2, "Two images provided, so two output elements expected"
    assert set(result[0].keys()) == {
        "predictions_in_own_coordinates",
        "predictions_in_original_coordinates",
    }, "Expected all declared outputs to be delivered"
    assert set(result[1].keys()) == {
        "predictions_in_own_coordinates",
        "predictions_in_original_coordinates",
    }, "Expected all declared outputs to be delivered"
    assert len(result[0]["predictions_in_own_coordinates"]) == len(
        result[0]["predictions_in_original_coordinates"]
    ), "Expected the same number of nested detections in both outputs for input image 0"
    assert len(result[1]["predictions_in_own_coordinates"]) == len(
        result[1]["predictions_in_original_coordinates"]
    ), "Expected the same number of nested detections in both outputs for input image 1"
    for own_coords_detection, original_coords_detection in zip(
        result[0]["predictions_in_own_coordinates"],
        result[0]["predictions_in_original_coordinates"],
    ):
        assert len(own_coords_detection) == len(
            original_coords_detection
        ), "Expected number of bounding boxes in nested sv.Detections not to change"
        assert np.all(
            own_coords_detection["parent_id"] != original_coords_detection["parent_id"]
        ), "Expected parent_id to be modified"
        assert original_coords_detection["parent_id"].tolist() == ["image.[0]"] * len(
            original_coords_detection
        ), "Expected parent of each bounding box to point into original input image instead of crop ID"
    for own_coords_detection, original_coords_detection in zip(
        result[1]["predictions_in_own_coordinates"],
        result[1]["predictions_in_original_coordinates"],
    ):
        assert len(own_coords_detection) == len(
            original_coords_detection
        ), "Expected number of bounding boxes in nested sv.Detections not to change"
        assert np.all(
            own_coords_detection["parent_id"] != original_coords_detection["parent_id"]
        ), "Expected parent_id to be modified"
        assert original_coords_detection["parent_id"].tolist() == ["image.[1]"] * len(
            original_coords_detection
        ), "Expected parent of each bounding box to point into original input image instead of crop ID"


DETECTIONS_TO_PARENT_COORDINATES_NON_BATCH_VARIANT_WORKFLOW = {
    "version": "1.0",
    "inputs": [{"type": "WorkflowImage", "name": "image"}],
    "steps": [
        {
            "type": "ObjectDetectionModel",
            "name": "first_detection",
            "image": "$inputs.image",
            "model_id": "yolov8n-640",
        },
        {
            "type": "DetectionsTransformation",
            "name": "enlarging_boxes",
            "predictions": "$steps.first_detection.predictions",
            "operations": [
                {"type": "DetectionsOffset", "offset_x": 50, "offset_y": 50}
            ],
        },
        {
            "type": "Crop",
            "name": "cropping",
            "image": "$inputs.image",
            "predictions": "$steps.enlarging_boxes.predictions",
        },
        {
            "type": "ObjectDetectionModel",
            "name": "specialised_detection",
            "image": "$steps.cropping.crops",
            "model_id": "yolov8n-640",
        },
        {
            "type": "DetectionsToParentsCoordinatesNonBatch",
            "name": "coordinates_transform",
            "image": "$inputs.image",
            "image_predictions": "$steps.specialised_detection.predictions",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "predictions_in_own_coordinates",
            "selector": "$steps.specialised_detection.predictions",
            "coordinates_system": "own",
        },
        {
            "type": "JsonField",
            "name": "predictions_in_original_coordinates",
            "selector": "$steps.coordinates_transform.predictions",
            "coordinates_system": "own",
        },
    ],
}


@pytest.mark.asyncio
@mock.patch.object(blocks_loader, "get_plugin_modules")
async def test_workflow_with_detections_coordinates_transformation_in_non_batch_variant(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
    crowd_image: np.ndarray,
) -> None:
    # given
    get_plugin_modules_mock.return_value = [
        "tests.workflows.integration_tests.dimensionality_manipulation_plugin"
    ]
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=DETECTIONS_TO_PARENT_COORDINATES_NON_BATCH_VARIANT_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = await execution_engine.run_async(
        runtime_parameters={"image": [crowd_image, crowd_image]}
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 2, "Two images provided, so two output elements expected"
    assert set(result[0].keys()) == {
        "predictions_in_own_coordinates",
        "predictions_in_original_coordinates",
    }, "Expected all declared outputs to be delivered"
    assert set(result[1].keys()) == {
        "predictions_in_own_coordinates",
        "predictions_in_original_coordinates",
    }, "Expected all declared outputs to be delivered"
    assert len(result[0]["predictions_in_own_coordinates"]) == len(
        result[0]["predictions_in_original_coordinates"]
    ), "Expected the same number of nested detections in both outputs for input image 0"
    assert len(result[1]["predictions_in_own_coordinates"]) == len(
        result[1]["predictions_in_original_coordinates"]
    ), "Expected the same number of nested detections in both outputs for input image 1"
    for own_coords_detection, original_coords_detection in zip(
        result[0]["predictions_in_own_coordinates"],
        result[0]["predictions_in_original_coordinates"],
    ):
        assert len(own_coords_detection) == len(
            original_coords_detection
        ), "Expected number of bounding boxes in nested sv.Detections not to change"
        assert np.all(
            own_coords_detection["parent_id"] != original_coords_detection["parent_id"]
        ), "Expected parent_id to be modified"
        assert original_coords_detection["parent_id"].tolist() == ["image.[0]"] * len(
            original_coords_detection
        ), "Expected parent of each bounding box to point into original input image instead of crop ID"
    for own_coords_detection, original_coords_detection in zip(
        result[1]["predictions_in_own_coordinates"],
        result[1]["predictions_in_original_coordinates"],
    ):
        assert len(own_coords_detection) == len(
            original_coords_detection
        ), "Expected number of bounding boxes in nested sv.Detections not to change"
        assert np.all(
            own_coords_detection["parent_id"] != original_coords_detection["parent_id"]
        ), "Expected parent_id to be modified"
        assert original_coords_detection["parent_id"].tolist() == ["image.[1]"] * len(
            original_coords_detection
        ), "Expected parent of each bounding box to point into original input image instead of crop ID"
