"""Smoke tests for the Gemma v2 block.

The shared routing/privacy logic is exercised in
``tests/workflows/unit_tests/core_steps/common/test_openrouter.py``;
this file just verifies the v2 manifest plus the block correctly
maps the model_version label to the OpenRouter slug.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.models.foundation.google_gemma.v2 import (
    BlockManifest,
    GoogleGemmaBlockV2,
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
            "type": "roboflow_core/google_gemma@v2",
            "name": "step",
            "images": "$inputs.image",
            "prompt": "describe",
        }
    )
    assert manifest.api_key == "rf_key:account"
    assert manifest.privacy_level == "deny"
    assert manifest.model_version == "Gemma 4 31B - OpenRouter"
    assert manifest.max_tokens == 500


def test_manifest_accepts_custom_user_key():
    manifest = BlockManifest.model_validate(
        {
            "type": "roboflow_core/google_gemma@v2",
            "name": "step",
            "images": "$inputs.image",
            "prompt": "describe",
            "api_key": "sk-or-v1-abc",
            "privacy_level": "zdr",
        }
    )
    assert manifest.api_key == "sk-or-v1-abc"
    assert manifest.privacy_level == "zdr"


def test_manifest_rejects_unconstrained_without_prompt():
    with pytest.raises(ValidationError, match="prompt"):
        BlockManifest.model_validate(
            {
                "type": "roboflow_core/google_gemma@v2",
                "name": "step",
                "images": "$inputs.image",
                "task_type": "unconstrained",
            }
        )


@patch.object(GoogleGemmaBlockV2, "execute_openrouter_batch")
def test_run_translates_label_to_openrouter_slug(mock_execute):
    mock_execute.return_value = ["caption text"]
    block = GoogleGemmaBlockV2(model_manager=MagicMock(), api_key="ws-key")

    result = block.run(
        images=[_stub_image()],
        task_type="caption",
        prompt=None,
        output_structure=None,
        classes=None,
        api_key="rf_key:account",
        privacy_level="deny",
        model_version="Gemma 4 26B A4B - OpenRouter",
        max_tokens=128,
        temperature=0.2,
        max_concurrent_requests=None,
    )

    assert result == [{"output": "caption text", "classes": None}]
    assert mock_execute.call_count == 1
    kwargs = mock_execute.call_args.kwargs
    assert (
        kwargs["model"]
        == MODEL_VERSION_MAPPING["Gemma 4 26B A4B - OpenRouter"]
        == "google/gemma-4-26b-a4b-it"
    )
    assert kwargs["openrouter_api_key"] == "rf_key:account"
    assert kwargs["privacy_level"] == "deny"
