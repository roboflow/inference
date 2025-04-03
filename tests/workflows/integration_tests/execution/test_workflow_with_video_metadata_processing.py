import time
from datetime import datetime
from unittest import mock
from unittest.mock import MagicMock

import numpy as np

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference.core.workflows.execution_engine.entities.base import VideoMetadata
from inference.core.workflows.execution_engine.introspection import blocks_loader

WORKFLOW_PROCESSING_VIDEO_METADATA = {
    "version": "1.1",
    "inputs": [{"type": "WorkflowVideoMetadata", "name": "video_metadata"}],
    "steps": [
        {
            "type": "ExampleVideoMetadataProcessing",
            "name": "metadata_handler",
            "metadata": "$inputs.video_metadata",
        }
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "frame_number",
            "selector": "$steps.metadata_handler.frame_number",
        },
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_workflow_processing_video_metadata(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
) -> None:
    """
    In this test scenario, we verify compatibility of new input type (WorkflowVideoMetadata)
    with Workflows compiler and execution engine.
    """
    # given
    get_plugin_modules_mock.return_value = [
        "tests.workflows.integration_tests.execution.stub_plugins.plugin_handling_video_metadata"
    ]
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_PROCESSING_VIDEO_METADATA,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "video_metadata": [
                {
                    "video_identifier": "a",
                    "frame_number": 21,
                    "frame_timestamp": datetime.now().isoformat(),
                    "fps": 50,
                    "comes_from_video_file": True,
                },
                VideoMetadata(
                    video_identifier="b",
                    frame_number=37,
                    frame_timestamp=time.time(),
                    fps=None,
                    comes_from_video_file=None,
                ),
            ]
        }
    )

    # then
    assert isinstance(result, list), "Expected list to be delivered"
    assert (
        len(result) == 2
    ), "Expected 2 element in the output for two input batch elements"
    assert set(result[0].keys()) == {
        "frame_number",
    }, "Expected all declared outputs to be delivered"
    assert set(result[1].keys()) == {
        "frame_number",
    }, "Expected all declared outputs to be delivered"
    assert (
        result[0]["frame_number"] == 21
    ), "Expected step to correctly extract frame number"
    assert (
        result[1]["frame_number"] == 37
    ), "Expected step to correctly extract frame number"


WORKFLOW_WITH_TRACKER = {
    "version": "1.1",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowVideoMetadata", "name": "video_metadata"},
    ],
    "steps": [
        {
            "type": "RoboflowObjectDetectionModel",
            "name": "detection",
            "image": "$inputs.image",
            "model_id": "yolov8n-640",
        },
        {
            "type": "Tracker",
            "name": "tracker",
            "metadata": "$inputs.video_metadata",
            "predictions": "$steps.detection.predictions",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "predictions",
            "selector": "$steps.detection.predictions",
        },
        {
            "type": "JsonField",
            "name": "tracker_id",
            "selector": "$steps.tracker.tracker_id",
        },
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_workflow_with_tracker(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
    dogs_image: np.ndarray,
    crowd_image: np.ndarray,
    license_plate_image: np.ndarray,
) -> None:
    """
    In this test scenario, we verify compatibility of new input type (WorkflowVideoMetadata)
    with Workflows compiler and execution engine.
    """
    # given
    get_plugin_modules_mock.return_value = [
        "tests.workflows.integration_tests.execution.stub_plugins.plugin_handling_video_metadata"
    ]
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_TRACKER,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )
    metadata_dogs_image = {
        "video_identifier": "a",
        "frame_number": 1,
        "frame_timestamp": datetime.now().isoformat(),
        "fps": 50,
        "comes_from_video_file": True,
    }
    metadata_crowd_image = {
        "video_identifier": "b",
        "frame_number": 1,
        "frame_timestamp": datetime.now().isoformat(),
        "fps": 50,
        "comes_from_video_file": True,
    }
    metadata_license_plate_image = {
        "video_identifier": "c",
        "frame_number": 1,
        "frame_timestamp": datetime.now().isoformat(),
        "fps": 50,
        "comes_from_video_file": True,
    }

    # when
    # simulating sequence of video frames to be processed from different sources
    result_1 = execution_engine.run(
        runtime_parameters={
            "image": [dogs_image, crowd_image],
            "video_metadata": [metadata_dogs_image, metadata_crowd_image],
        }
    )
    result_2 = execution_engine.run(
        runtime_parameters={
            "image": [crowd_image],
            "video_metadata": [metadata_crowd_image],
        }
    )
    result_3 = execution_engine.run(
        runtime_parameters={
            "image": [dogs_image, license_plate_image],
            "video_metadata": [metadata_dogs_image, metadata_license_plate_image],
        }
    )
    first_dogs_frame_tracker_ids = result_1[0]["tracker_id"]
    second_dogs_frame_tracker_ids = result_3[0]["tracker_id"]
    first_crowd_frame_tracker_ids = result_1[1]["tracker_id"]
    second_crowd_frame_tracker_ids = result_2[0]["tracker_id"]
    first_license_plate_frame_tracker_ids = result_3[1]["tracker_id"]

    # then
    assert (
        first_dogs_frame_tracker_ids == second_dogs_frame_tracker_ids
    ), "The same image, expected no tracker IDs change"
    assert (
        first_crowd_frame_tracker_ids == second_crowd_frame_tracker_ids
    ), "The same image, expected no tracker IDs change"
    assert first_license_plate_frame_tracker_ids == [
        1,
        2,
        3,
    ], "Expected tracker IDs to be generated sequentially, independent of other trackers"
