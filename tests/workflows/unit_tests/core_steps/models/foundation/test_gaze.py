from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pydantic import BaseModel, ValidationError

from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.models.foundation.gaze.v1 import (
    BlockManifest,
    GazeBlockV1,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


class MockGazePrediction(BaseModel):
    face: dict
    yaw: float
    pitch: float


class MockGazeResponse(BaseModel):
    predictions: List[MockGazePrediction]


@pytest.fixture
def mock_model_manager():
    # Mock a model manager that returns predictable gaze predictions
    mock = MagicMock()
    mock.infer_from_request_sync.return_value = [
        MockGazeResponse(
            predictions=[
                MockGazePrediction(
                    face={
                        "x": 100,
                        "y": 100,
                        "width": 50,
                        "height": 50,
                        "confidence": 0.9,
                        "landmarks": [
                            {"x": 120, "y": 120},
                            {"x": 130, "y": 120},
                        ],
                    },
                    yaw=0.5,  # ~28.6 degrees
                    pitch=-0.2,  # ~-11.5 degrees
                )
            ]
        )
    ]
    return mock


@pytest.fixture
def mock_workflow_image_data():
    # Create a mock image
    start_image = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some"),
        numpy_image=start_image,
    )


@pytest.mark.parametrize("images_field_alias", ["images", "image"])
def test_manifest_parsing_valid(images_field_alias):
    data = {
        "type": "roboflow_core/gaze@v1",
        "name": "my_gaze_step",
        images_field_alias: "$inputs.image",
        "do_run_face_detection": True,
    }

    result = BlockManifest.model_validate(data)
    assert result.type == "roboflow_core/gaze@v1"
    assert result.name == "my_gaze_step"
    assert result.images == "$inputs.image"
    assert result.do_run_face_detection is True


def test_manifest_parsing_invalid_missing_type():
    data = {
        "name": "my_gaze_step",
        "images": "$inputs.image",
        "do_run_face_detection": True,
    }
    with pytest.raises(ValidationError):
        BlockManifest.model_validate(data)


def test_manifest_parsing_invalid_images_type():
    data = {
        "type": "roboflow_core/gaze@v1",
        "name": "my_gaze_step",
        "images": 123,  # invalid type
        "do_run_face_detection": True,
    }
    with pytest.raises(ValidationError):
        BlockManifest.model_validate(data)


def test_manifest_parsing_invalid_do_run_face_detection_type():
    data = {
        "type": "roboflow_core/gaze@v1",
        "name": "my_gaze_step",
        "images": "$inputs.image",
        "do_run_face_detection": "invalid",  # should be boolean
    }
    with pytest.raises(ValidationError):
        BlockManifest.model_validate(data)


def test_run_locally(mock_model_manager, mock_workflow_image_data):
    block = GazeBlockV1(
        model_manager=mock_model_manager,
        api_key=None,
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    result = block.run(
        images=[mock_workflow_image_data],
        do_run_face_detection=True,
    )

    assert len(result) == 1  # One image processed
    assert set(result[0].keys()) == {
        "face_predictions",
        "yaw_degrees",
        "pitch_degrees",
    }

    # Check angles are converted to degrees correctly
    assert len(result[0]["yaw_degrees"]) == 1
    assert len(result[0]["pitch_degrees"]) == 1
    assert np.isclose(result[0]["yaw_degrees"][0], 28.6, rtol=0.1)
    assert np.isclose(result[0]["pitch_degrees"][0], -11.5, rtol=0.1)

    # Verify model manager was called correctly
    mock_model_manager.infer_from_request_sync.assert_called_once()


def test_run_locally_batch_processing(mock_model_manager, mock_workflow_image_data):
    block = GazeBlockV1(
        model_manager=mock_model_manager,
        api_key=None,
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    result = block.run(
        images=[mock_workflow_image_data, mock_workflow_image_data],
        do_run_face_detection=True,
    )

    assert len(result) == 2  # Two images processed
    assert mock_model_manager.infer_from_request_sync.call_count == 2
