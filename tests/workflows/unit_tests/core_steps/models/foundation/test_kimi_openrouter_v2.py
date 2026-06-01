"""Smoke tests for the Kimi v2 block."""

from unittest.mock import MagicMock, patch

import numpy as np
from pydantic import ValidationError
import pytest

from inference.core.workflows.core_steps.models.foundation.kimi_openrouter.v2 import (
    BlockManifest,
    KimiOpenrouterBlockV2,
    MODEL_VERSION_MAPPING,
)
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData


def _stub_image() -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=MagicMock(parent_id="root", workflow_root_ancestor_metadata=None),
        numpy_image=np.zeros((10, 10, 3), dtype=np.uint8),
    )


def test_manifest_defaults_managed_and_deny():
    manifest = BlockManifest.model_validate(
        {
            "type": "roboflow_core/kimi_openrouter@v2",
            "name": "step",
            "images": "$inputs.image",
            "prompt": "describe",
        }
    )
    assert manifest.api_key == "rf_key:account"
    assert manifest.privacy_level == "deny"
    assert manifest.model_version == "Kimi K2.6 - OpenRouter"


def test_manifest_rejects_object_detection_without_classes():
    with pytest.raises(ValidationError, match="classes"):
        BlockManifest.model_validate(
            {
                "type": "roboflow_core/kimi_openrouter@v2",
                "name": "step",
                "images": "$inputs.image",
                "task_type": "object-detection",
            }
        )


@patch.object(KimiOpenrouterBlockV2, "execute_openrouter_batch")
def test_run_passes_kimi_slug_and_user_key(mock_execute):
    mock_execute.return_value = ["text"]
    block = KimiOpenrouterBlockV2(model_manager=MagicMock(), api_key="ws-key")

    block.run(
        images=[_stub_image()],
        task_type="ocr",
        prompt=None,
        output_structure=None,
        classes=None,
        api_key="sk-or-v1-test",
        privacy_level="allow",
        model_version="Kimi K2.5 - OpenRouter",
        max_tokens=128,
        temperature=0.0,
        max_concurrent_requests=None,
    )

    assert (
        mock_execute.call_args.kwargs["model"]
        == MODEL_VERSION_MAPPING["Kimi K2.5 - OpenRouter"]
        == "moonshotai/kimi-k2.5"
    )
    assert mock_execute.call_args.kwargs["openrouter_api_key"] == "sk-or-v1-test"
    assert mock_execute.call_args.kwargs["privacy_level"] == "allow"
