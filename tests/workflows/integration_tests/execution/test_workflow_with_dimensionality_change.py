from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import pytest

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
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


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_workflow_with_detections_coordinates_transformation_in_batch_variant(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
    crowd_image: np.ndarray,
) -> None:
    """
    In this test case scenario we rely on custom block `DetectionsToParentsCoordinatesBatch`
    that is supposed to take:
    * original image (before crop)
    * detections (performed on crops made from original image)
    and aligns coordinates system of predictions (in very naive way, not fully functional) to
    be represented in original image coordinates.
    Which means that block operates on input occupying in two different dimensionality levels.
    Block produces outputs at "detections" dimensionality level.
    Block takes BATCHED as input

    What this workflow do is making detection, then cropping images according to
    detections, then performing detections on crop and then making detections on crops.
    Detections from crops are submitted along with original image into
    `DetectionsToParentsCoordinatesBatch` block instance.

    What is verified from EE standpoint:
    * ability to operate with steps that take inputs sitting at different dimensionality level
    * ability to select output dimensionality when inputs are ay different dimensionality levels
    * ability for steps under that conditions to operate in batch mode
    """
    # given
    get_plugin_modules_mock.return_value = [
        "tests.workflows.integration_tests.execution.stub_plugins.dimensionality_manipulation_plugin"
    ]
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=DETECTIONS_TO_PARENT_COORDINATES_BATCH_VARIANT_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
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
        print(own_coords_detection["parent_id"])
        print(original_coords_detection["parent_id"])
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


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_workflow_with_detections_coordinates_transformation_in_non_batch_variant(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
    crowd_image: np.ndarray,
) -> None:
    """
    In this test case scenario we rely on custom block `DetectionsToParentsCoordinatesNonBatch`
    that is supposed to take:
    * original image (before crop)
    * detections (performed on crops made from original image)
    and aligns coordinates system of predictions (in very naive way, not fully functional) to
    be represented in original image coordinates.
    Which means that block operates on input occupying in two different dimensionality levels.
    Block produces outputs at "detections" dimensionality level.
    Block takes NON-BATCHED as input

    What this workflow do is making detection, then cropping images according to
    detections, then performing detections on crop and then making detections on crops.
    Detections from crops are submitted along with original image into
    `DetectionsToParentsCoordinatesNonBatch` block instance.

    What is verified from EE standpoint:
    * ability to operate with steps that take inputs sitting at different dimensionality level
    * ability to select output dimensionality when inputs are ay different dimensionality levels
    * ability for steps under that conditions to operate in non-batch mode
    """
    # given
    get_plugin_modules_mock.return_value = [
        "tests.workflows.integration_tests.execution.stub_plugins.dimensionality_manipulation_plugin"
    ]
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=DETECTIONS_TO_PARENT_COORDINATES_NON_BATCH_VARIANT_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
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


DETECTIONS_STITCHING_BATCH_VARIANT_WORKFLOW = {
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
            "type": "StitchDetectionsBatch",
            "name": "stitch",
            "images": "$inputs.image",
            "images_predictions": "$steps.specialised_detection.predictions",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "predictions_for_crops",
            "selector": "$steps.specialised_detection.predictions",
            "coordinates_system": "own",
        },
        {
            "type": "JsonField",
            "name": "aggregated_predictions",
            "selector": "$steps.stitch.predictions",
            "coordinates_system": "own",
        },
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_workflow_with_detections_stitching_in_batch_variant(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
    crowd_image: np.ndarray,
    dogs_image: np.ndarray,
) -> None:
    """
    In this test case scenario we rely on custom block `StitchDetectionsBatch`
    that is supposed to take:
    * original image (before crop)
    * detections (performed on crops made from original image)
    and merge together detections for all crops coming from the same image.
    Which means that block operates on input occupying in two different dimensionality levels.
    Block produces outputs at "original image" dimensionality level.
    Block takes BATCHED as input

    What this workflow do is making detection, then cropping images according to
    detections, then performing detections on crop and then making detections on crops.
    Detections from crops are submitted along with original image into
    `StitchDetectionsBatch` block instance to merge and get results at image
    dimensionality level

    What is verified from EE standpoint:
    * ability to operate with steps that take inputs sitting at different dimensionality level
    * ability to select output dimensionality when inputs are ay different dimensionality levels
    * ability for steps under that conditions to operate in batch mode
    """
    # given
    get_plugin_modules_mock.return_value = [
        "tests.workflows.integration_tests.execution.stub_plugins.dimensionality_manipulation_plugin"
    ]
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=DETECTIONS_STITCHING_BATCH_VARIANT_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={"image": [crowd_image, dogs_image]}
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 2, "Two images provided, so two output elements expected"
    assert set(result[0].keys()) == {
        "predictions_for_crops",
        "aggregated_predictions",
    }, "Expected all declared outputs to be delivered"
    assert set(result[1].keys()) == {
        "predictions_for_crops",
        "aggregated_predictions",
    }, "Expected all declared outputs to be delivered"
    assert sum(len(e) for e in result[0]["predictions_for_crops"]) == len(
        result[0]["aggregated_predictions"]
    ), "Number of bounding boxes in aggregated prediction must match sum of number of boxes for all crops"
    assert sum(len(e) for e in result[1]["predictions_for_crops"]) == len(
        result[1]["aggregated_predictions"]
    ), "Number of bounding boxes in aggregated prediction must match sum of number of boxes for all crops"


DETECTIONS_STITCHING_NON_BATCH_VARIANT_WORKFLOW = {
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
            "type": "StitchDetectionsNonBatch",
            "name": "stitch",
            "image": "$inputs.image",
            "image_predictions": "$steps.specialised_detection.predictions",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "predictions_for_crops",
            "selector": "$steps.specialised_detection.predictions",
            "coordinates_system": "own",
        },
        {
            "type": "JsonField",
            "name": "aggregated_predictions",
            "selector": "$steps.stitch.predictions",
            "coordinates_system": "own",
        },
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_workflow_with_detections_stitching_in_batch_variant(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
    crowd_image: np.ndarray,
    dogs_image: np.ndarray,
) -> None:
    """
    In this test case scenario we rely on custom block `StitchDetectionsNonBatch`
    that is supposed to take:
    * original image (before crop)
    * detections (performed on crops made from original image)
    and merge together detections for all crops coming from the same image.
    Which means that block operates on input occupying in two different dimensionality levels.
    Block produces outputs at "original image" dimensionality level.
    Block takes NON-BATCHED as input

    What this workflow do is making detection, then cropping images according to
    detections, then performing detections on crop and then making detections on crops.
    Detections from crops are submitted along with original image into
    `StitchDetectionsNonBatch` block instance to merge and get results at image
    dimensionality level

    What is verified from EE standpoint:
    * ability to operate with steps that take inputs sitting at different dimensionality level
    * ability to select output dimensionality when inputs are ay different dimensionality levels
    * ability for steps under that conditions to operate in non-batch mode
    """
    # given
    get_plugin_modules_mock.return_value = [
        "tests.workflows.integration_tests.execution.stub_plugins.dimensionality_manipulation_plugin"
    ]
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=DETECTIONS_STITCHING_NON_BATCH_VARIANT_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={"image": [crowd_image, dogs_image]}
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 2, "Two images provided, so two output elements expected"
    assert set(result[0].keys()) == {
        "predictions_for_crops",
        "aggregated_predictions",
    }, "Expected all declared outputs to be delivered"
    assert set(result[1].keys()) == {
        "predictions_for_crops",
        "aggregated_predictions",
    }, "Expected all declared outputs to be delivered"
    assert sum(len(e) for e in result[0]["predictions_for_crops"]) == len(
        result[0]["aggregated_predictions"]
    ), "Number of bounding boxes in aggregated prediction must match sum of number of boxes for all crops"
    assert sum(len(e) for e in result[1]["predictions_for_crops"]) == len(
        result[1]["aggregated_predictions"]
    ), "Number of bounding boxes in aggregated prediction must match sum of number of boxes for all crops"


DETECTIONS_TILING_BATCH_VARIANT_WORKFLOW = {
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
            "type": "TileDetectionsBatch",
            "name": "tiles",
            "images_crops": "$steps.cropping.crops",
            "crops_predictions": "$steps.specialised_detection.predictions",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "visualisation",
            "selector": "$steps.tiles.visualisations",
            "coordinates_system": "own",
        }
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_workflow_with_detections_tiling_in_batch_variant(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
    crowd_image: np.ndarray,
    dogs_image: np.ndarray,
) -> None:
    """
    In this test case scenario we rely on custom block `TileDetectionsBatch`
    that is supposed to take:
    * cropped images
    * detections for cropped images
    and overlays predictions on crops then making tiles out of visualisation - decreasing
    data dimensionality by one.
    Block takes BATCHED as input

    What this workflow do is making detection, then cropping images according to
    detections, then performing detections on crop and then making detections on crops.
    Detections from crops are submitted along with cropped images into
    `TileDetectionsBatch` block instance to create tiles and send back into the
    dimensionality level of original image

    What is verified from EE standpoint:
    * ability to decrease dimensionality
    * ability to unwrap one dimensionality level of input data before submitting into block
    execution
    * properly handling the scenario when block operates in batch mode
    """
    # given
    get_plugin_modules_mock.return_value = [
        "tests.workflows.integration_tests.execution.stub_plugins.dimensionality_manipulation_plugin"
    ]
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=DETECTIONS_TILING_BATCH_VARIANT_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={"image": [crowd_image, dogs_image]}
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 2, "Two images provided, so two output elements expected"
    assert set(result[0].keys()) == {
        "visualisation",
    }, "Expected all declared outputs to be delivered"
    assert set(result[1].keys()) == {
        "visualisation",
    }, "Expected all declared outputs to be delivered"
    assert result[0]["visualisation"].shape == (
        416,
        362,
        3,
    ), "Expected visualisation to be image of shape (416, 362, 3)"
    assert result[1]["visualisation"].shape == (
        296,
        522,
        3,
    ), "Expected visualisation to be image of shape (296, 522, 3)"


DETECTIONS_TILING_NON_BATCH_VARIANT_WORKFLOW = {
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
            "type": "TileDetectionsNonBatch",
            "name": "tiles",
            "crops": "$steps.cropping.crops",
            "crops_predictions": "$steps.specialised_detection.predictions",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "visualisation",
            "selector": "$steps.tiles.visualisations",
            "coordinates_system": "own",
        }
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_workflow_with_detections_tiling_in_non_batch_variant(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
    crowd_image: np.ndarray,
    dogs_image: np.ndarray,
) -> None:
    """
    In this test case scenario we rely on custom block `TileDetectionsNonBatch`
    that is supposed to take:
    * cropped images
    * detections for cropped images
    and overlays predictions on crops then making tiles out of visualisation - decreasing
    data dimensionality by one.
    Block takes NON-BATCHED as input

    What this workflow do is making detection, then cropping images according to
    detections, then performing detections on crop and then making detections on crops.
    Detections from crops are submitted along with cropped images into
    `TileDetectionsNonBatch` block instance to create tiles and send back into the
    dimensionality level of original image

    What is verified from EE standpoint:
    * ability to decrease dimensionality
    * ability to unwrap one dimensionality level of input data before submitting into block
    execution
    * properly handling the scenario when block operates in non-batch mode
    """
    # given
    get_plugin_modules_mock.return_value = [
        "tests.workflows.integration_tests.execution.stub_plugins.dimensionality_manipulation_plugin"
    ]
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=DETECTIONS_TILING_NON_BATCH_VARIANT_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={"image": [crowd_image, dogs_image]}
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 2, "Two images provided, so two output elements expected"
    assert set(result[0].keys()) == {
        "visualisation",
    }, "Expected all declared outputs to be delivered"
    assert set(result[1].keys()) == {
        "visualisation",
    }, "Expected all declared outputs to be delivered"
    assert result[0]["visualisation"].shape == (
        416,
        362,
        3,
    ), "Expected visualisation to be image of shape (416, 362, 3)"
    assert result[1]["visualisation"].shape == (
        296,
        522,
        3,
    ), "Expected visualisation to be image of shape (296, 522, 3)"
