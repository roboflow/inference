"""Unit tests for VLM blocks remote execution (Florence2, Moondream2, SmolVLM, Qwen)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


@pytest.fixture
def mock_model_manager():
    return MagicMock()


@pytest.fixture
def mock_workflow_image_data():
    start_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some"),
        numpy_image=start_image,
    )


class TestFlorence2Remote:
    """Tests for Florence2 remote execution."""

    @patch(
        "inference.core.workflows.core_steps.models.foundation.florence2.v1.InferenceHTTPClient"
    )
    def test_run_remotely_calls_infer_lmm(
        self, mock_client_cls, mock_model_manager, mock_workflow_image_data
    ):
        from inference.core.workflows.core_steps.models.foundation.florence2.v1 import (
            Florence2BlockV1,
        )

        mock_client = MagicMock()
        mock_client.infer_lmm.return_value = {
            "response": {"<OD>": {"bboxes": [], "labels": []}}
        }
        mock_client_cls.return_value = mock_client

        block = Florence2BlockV1(
            model_manager=mock_model_manager,
            api_key="test_api_key",
            step_execution_mode=StepExecutionMode.REMOTE,
        )

        result = block.run(
            images=[mock_workflow_image_data],
            model_version="florence-2-base",
            task_type="object-detection",
            prompt=None,
            classes=None,
            grounding_detection=None,
            grounding_selection_mode="first",
        )

        assert len(result) == 1
        mock_client.infer_lmm.assert_called_once()

    @patch(
        "inference.core.workflows.core_steps.models.foundation.florence2.v1.InferenceHTTPClient"
    )
    def test_run_remotely_with_custom_prompt(
        self, mock_client_cls, mock_model_manager, mock_workflow_image_data
    ):
        from inference.core.workflows.core_steps.models.foundation.florence2.v1 import (
            Florence2BlockV1,
        )

        mock_client = MagicMock()
        mock_client.infer_lmm.return_value = {
            "response": {"custom_output": "test response"}
        }
        mock_client_cls.return_value = mock_client

        block = Florence2BlockV1(
            model_manager=mock_model_manager,
            api_key="test_api_key",
            step_execution_mode=StepExecutionMode.REMOTE,
        )

        result = block.run(
            images=[mock_workflow_image_data],
            model_version="florence-2-base",
            task_type="custom",
            prompt="Describe this image",
            classes=None,
            grounding_detection=None,
            grounding_selection_mode="first",
        )

        assert len(result) == 1
        call_args = mock_client.infer_lmm.call_args
        assert "Describe this image" in call_args.kwargs.get("prompt", "")


class TestMoondream2Remote:
    """Tests for Moondream2 remote execution."""

    @patch(
        "inference.core.workflows.core_steps.models.foundation.moondream2.v1.InferenceHTTPClient"
    )
    def test_run_remotely_calls_infer_lmm(
        self, mock_client_cls, mock_model_manager, mock_workflow_image_data
    ):
        from inference.core.workflows.core_steps.models.foundation.moondream2.v1 import (
            Moondream2BlockV1,
        )

        mock_client = MagicMock()
        mock_client.infer_lmm.return_value = {
            "predictions": [],
            "image": {"width": 640, "height": 480},
        }
        mock_client_cls.return_value = mock_client

        block = Moondream2BlockV1(
            model_manager=mock_model_manager,
            api_key="test_api_key",
            step_execution_mode=StepExecutionMode.REMOTE,
        )

        result = block.run(
            images=[mock_workflow_image_data],
            model_version="moondream2/moondream2_2b_jul24",
            prompt="Detect objects",
        )

        assert len(result) == 1
        mock_client.infer_lmm.assert_called_once()


class TestSmolVLMRemote:
    """Tests for SmolVLM remote execution."""

    @patch(
        "inference.core.workflows.core_steps.models.foundation.smolvlm.v1.InferenceHTTPClient"
    )
    def test_run_remotely_calls_infer_lmm(
        self, mock_client_cls, mock_model_manager, mock_workflow_image_data
    ):
        from inference.core.workflows.core_steps.models.foundation.smolvlm.v1 import (
            SmolVLM2BlockV1,
        )

        mock_client = MagicMock()
        mock_client.infer_lmm.return_value = {"response": "This is a test image."}
        mock_client_cls.return_value = mock_client

        block = SmolVLM2BlockV1(
            model_manager=mock_model_manager,
            api_key="test_api_key",
            step_execution_mode=StepExecutionMode.REMOTE,
        )

        result = block.run(
            images=[mock_workflow_image_data],
            model_version="smolvlm2/smolvlm-2.2b-instruct",
            prompt="Describe this image",
        )

        assert len(result) == 1
        assert "parsed_output" in result[0]
        mock_client.infer_lmm.assert_called_once()


class TestQwen25VLRemote:
    """Tests for Qwen2.5-VL remote execution."""

    @patch(
        "inference.core.workflows.core_steps.models.foundation.qwen.v1.InferenceHTTPClient"
    )
    def test_run_remotely_calls_infer_lmm(
        self, mock_client_cls, mock_model_manager, mock_workflow_image_data
    ):
        from inference.core.workflows.core_steps.models.foundation.qwen.v1 import (
            Qwen25VLBlockV1,
        )

        mock_client = MagicMock()
        mock_client.infer_lmm.return_value = {"response": "This is a test response."}
        mock_client_cls.return_value = mock_client

        block = Qwen25VLBlockV1(
            model_manager=mock_model_manager,
            api_key="test_api_key",
            step_execution_mode=StepExecutionMode.REMOTE,
        )

        result = block.run(
            images=[mock_workflow_image_data],
            model_version="qwen25-vl-7b",
            prompt="Describe this image",
            system_prompt="You are helpful.",
        )

        assert len(result) == 1
        assert "parsed_output" in result[0]
        mock_client.infer_lmm.assert_called_once()


class TestQwen3VLRemote:
    """Tests for Qwen3-VL remote execution."""

    @patch(
        "inference.core.workflows.core_steps.models.foundation.qwen3vl.v1.InferenceHTTPClient"
    )
    def test_run_remotely_calls_infer_lmm(
        self, mock_client_cls, mock_model_manager, mock_workflow_image_data
    ):
        from inference.core.workflows.core_steps.models.foundation.qwen3vl.v1 import (
            Qwen3VLBlockV1,
        )

        mock_client = MagicMock()
        mock_client.infer_lmm.return_value = {"response": "This is a test response."}
        mock_client_cls.return_value = mock_client

        block = Qwen3VLBlockV1(
            model_manager=mock_model_manager,
            api_key="test_api_key",
            step_execution_mode=StepExecutionMode.REMOTE,
        )

        result = block.run(
            images=[mock_workflow_image_data],
            model_version="qwen3vl-2b-instruct",
            prompt="Describe this image",
            system_prompt="You are helpful.",
        )

        assert len(result) == 1
        assert "parsed_output" in result[0]
        mock_client.infer_lmm.assert_called_once()
