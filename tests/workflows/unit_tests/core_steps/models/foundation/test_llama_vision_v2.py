"""Smoke tests for the Llama Vision v2 block."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.models.foundation.llama_vision.v2 import (
    BlockManifest,
    LlamaVisionBlockV2,
    MODEL_VERSION_MAPPING,
)
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData


def _stub_image() -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=MagicMock(parent_id="root", workflow_root_ancestor_metadata=None),
        numpy_image=np.zeros((10, 10, 3), dtype=np.uint8),
    )


def test_manifest_defaults_include_managed_key_and_deny_privacy():
    manifest = BlockManifest.model_validate(
        {
            "type": "roboflow_core/llama_vision@v2",
            "name": "step",
            "images": "$inputs.image",
            "prompt": "describe",
        }
    )
    assert manifest.api_key == "rf_key:account"
    assert manifest.privacy_level == "deny"
    assert manifest.model_version == "11B (Free) - OpenRouter"


def test_manifest_rejects_invalid_temperature():
    with pytest.raises(ValidationError):
        BlockManifest.model_validate(
            {
                "type": "roboflow_core/llama_vision@v2",
                "name": "step",
                "images": "$inputs.image",
                "prompt": "describe",
                "temperature": 5.0,
            }
        )


@patch.object(LlamaVisionBlockV2, "execute_openrouter_batch")
def test_run_uses_correct_openrouter_slug(mock_execute):
    mock_execute.return_value = ["ok"]
    block = LlamaVisionBlockV2(model_manager=MagicMock(), api_key="ws-key")

    block.run(
        images=[_stub_image()],
        task_type="caption",
        prompt=None,
        output_structure=None,
        classes=None,
        api_key="rf_key:account",
        privacy_level="zdr",
        model_version="90B (Regular) - OpenRouter",
        max_tokens=128,
        temperature=0.2,
        max_concurrent_requests=None,
    )

    assert (
        mock_execute.call_args.kwargs["model"]
        == MODEL_VERSION_MAPPING["90B (Regular) - OpenRouter"]
        == "meta-llama/llama-3.2-90b-vision-instruct"
    )
    assert mock_execute.call_args.kwargs["privacy_level"] == "zdr"
