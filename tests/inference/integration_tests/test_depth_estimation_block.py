import os

import pytest
import requests
import numpy as np
from PIL import Image

from tests.inference.integration_tests.regression_test import bool_env
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData, ImageParentMetadata

api_key = os.environ.get("API_KEY")


@pytest.mark.skipif(
    bool_env(os.getenv("SKIP_DEPTH_ESTIMATION_TEST", False)),
    reason="Skipping Depth Estimation test",
)
def test_depth_estimation_block_inference(
    server_url: str, clean_loaded_models_every_test_fixture
) -> None:
    # given
    # Create a test image
    test_image = np.zeros((224, 224, 3), dtype=np.uint8)
    test_image[50:150, 50:150] = 255  # Add a white square
    parent_metadata = ImageParentMetadata(parent_id="test_image")
    workflow_image = WorkflowImageData(
        numpy_image=test_image,
        parent_metadata=parent_metadata
    )

    # Create the block configuration
    block_config = {
        "type": "roboflow_core/depth_estimation@v1",
        "name": "depth_estimation",
        "images": {"type": "WorkflowImageData", "value": workflow_image},
        "model_version": "depth-anything-v2/small"
    }

    # when
    response = requests.post(
        f"{server_url}/workflows/execute",
        json={
            "api_key": api_key,
            "blocks": [block_config]
        },
    )

    # then
    response.raise_for_status()
    data = response.json()
    
    # Verify the response structure
    assert "results" in data, "Expected results in response"
    assert len(data["results"]) == 1, "Expected one result"
    result = data["results"][0]
    
    # Verify the depth estimation outputs
    assert "image" in result, "Expected image in result"
    assert "normalized_depth" in result, "Expected normalized_depth in result"
    
    # Verify the depth map is valid
    depth_map = result["normalized_depth"]
    assert isinstance(depth_map, list), "Expected depth map to be a list"
    assert len(depth_map) > 0, "Expected non-empty depth map"
    assert all(0 <= x <= 1 for x in depth_map), "Expected depth values to be normalized between 0 and 1" 