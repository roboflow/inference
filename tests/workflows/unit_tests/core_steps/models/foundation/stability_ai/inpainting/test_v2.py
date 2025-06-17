"""Tests for Stability AI Inpainting v2 block."""

import numpy as np
import pytest
import supervision as sv
from unittest.mock import MagicMock, patch

from inference.core.workflows.core_steps.models.foundation.stability_ai.inpainting.v2 import (
    StabilityAIInpaintingBlockV2,
    BlockManifest,
)
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData


class TestStabilityAIInpaintingBlockV2:
    """Test cases for StabilityAIInpaintingBlockV2."""

    def test_manifest_parsing(self):
        """Test that the manifest can be parsed successfully."""
        manifest = BlockManifest(
            type="roboflow_core/stability_ai_inpainting@v2",
            name="test_step",
            execution_mode="cloud",
            image="$inputs.image",
            segmentation_mask="$steps.segmentation.predictions",
            prompt="Replace with flowers",
            api_key="test_key",
        )
        assert manifest.execution_mode == "cloud"
        assert manifest.prompt == "Replace with flowers"

    def test_cloud_mode_requires_api_key(self):
        """Test that cloud mode requires an API key in run method."""
        # The validation now happens in the run method, not in manifest
        block = StabilityAIInpaintingBlockV2()
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        workflow_image = WorkflowImageData(
            numpy_image=test_image,
            parent_metadata={"parent_id": "test"}
        )
        
        mask = np.zeros((100, 100), dtype=bool)
        segmentation = sv.Detections(
            xyxy=np.array([[0, 0, 100, 100]]),
            mask=np.array([mask]),
        )
        
        with pytest.raises(ValueError, match="API key is required"):
            block.run(
                image=workflow_image,
                segmentation_mask=segmentation,
                prompt="test",
                negative_prompt=None,
                execution_mode="cloud",
                api_key=None,
                num_inference_steps=None,
                guidance_scale=None,
                seed=None,
            )

    def test_local_mode_does_not_require_api_key(self):
        """Test that local mode doesn't require an API key."""
        manifest = BlockManifest(
            type="roboflow_core/stability_ai_inpainting@v2",
            name="test_step",
            execution_mode="local",
            image="$inputs.image",
            segmentation_mask="$steps.segmentation.predictions",
            prompt="test",
            api_key=None,
        )
        assert manifest.api_key is None

    @patch("requests.post")
    def test_cloud_inference(self, mock_post):
        """Test cloud inference execution."""
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"fake_image_data"
        mock_post.return_value = mock_response

        # Create test data
        block = StabilityAIInpaintingBlockV2()
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        workflow_image = WorkflowImageData(numpy_image=test_image)
        
        # Create mock segmentation mask
        mock_mask = np.zeros((100, 100), dtype=bool)
        mock_mask[25:75, 25:75] = True
        segmentation = sv.Detections(
            xyxy=np.array([[25, 25, 75, 75]]),
            mask=np.array([mock_mask]),
        )

        # Run inference
        with patch("cv2.imdecode") as mock_decode:
            mock_decode.return_value = test_image
            result = block.run(
                image=workflow_image,
                segmentation_mask=segmentation,
                prompt="test prompt",
                negative_prompt=None,
                execution_mode="cloud",
                api_key="test_key",
                num_inference_steps=None,
                guidance_scale=None,
                seed=None,
            )

        # Verify API call
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "https://api.stability.ai" in call_args[0][0]
        assert call_args[1]["headers"]["authorization"] == "Bearer test_key"

    def test_describe_outputs(self):
        """Test output description."""
        outputs = BlockManifest.describe_outputs()
        assert len(outputs) == 1
        assert outputs[0].name == "image"

    def test_get_execution_engine_compatibility(self):
        """Test execution engine compatibility."""
        compatibility = BlockManifest.get_execution_engine_compatibility()
        assert compatibility == ">=1.4.0,<2.0.0"
