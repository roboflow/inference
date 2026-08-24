"""Tests for the OpenRouter v2 block (v1 + token-usage outputs).

Shared routing/privacy logic is exercised in
``tests/workflows/unit_tests/core_steps/common/test_openrouter.py``; the
v1 behavior suite lives with the v1 block. This file covers the v2 delta:
``input_tokens`` / ``output_tokens`` outputs.
"""

from unittest.mock import MagicMock, patch

import numpy as np

from inference.core.workflows.core_steps.models.foundation.openrouter.v2 import (
    BlockManifest,
    OpenRouterBlockV2,
)
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData


def _stub_image() -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=MagicMock(
            parent_id="root", workflow_root_ancestor_metadata=None
        ),
        numpy_image=np.zeros((10, 10, 3), dtype=np.uint8),
    )


def test_manifest_declares_token_outputs():
    outputs = {output.name for output in BlockManifest.describe_outputs()}
    assert {"input_tokens", "output_tokens"} <= outputs


@patch.object(OpenRouterBlockV2, "execute_openrouter_batch")
def test_run_surfaces_token_usage(mock_execute):
    mock_execute.return_value = [("caption text", 11, 4)]
    block = OpenRouterBlockV2(model_manager=MagicMock(), api_key="ws-key")

    result = block.run(
        images=[_stub_image()],
        model_id="google/gemma-4-26b-a4b-it",
        task_type="caption",
        prompt=None,
        output_structure=None,
        classes=None,
        api_key="rf_key:account",
        privacy_level="deny",
        max_tokens=128,
        temperature=0.2,
        max_concurrent_requests=None,
    )

    assert result == [
        {
            "output": "caption text",
            "classes": None,
            "input_tokens": 11,
            "output_tokens": 4,
        }
    ]
    assert mock_execute.call_args.kwargs["include_usage"] is True
