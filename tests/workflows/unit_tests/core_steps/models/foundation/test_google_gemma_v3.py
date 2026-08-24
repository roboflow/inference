"""Tests for the Gemma v3 block (v2 + token-usage outputs).

The v2 behavior suite lives in ``test_google_gemma_v2.py``; this file
covers the v3 delta: ``input_tokens`` / ``output_tokens`` outputs.
"""

from unittest.mock import MagicMock, patch

import numpy as np

from inference.core.workflows.core_steps.models.foundation.google_gemma.v3 import (
    MODEL_VERSION_MAPPING,
    GoogleGemmaBlockV3,
)
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData


def _stub_image() -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=MagicMock(
            parent_id="root", workflow_root_ancestor_metadata=None
        ),
        numpy_image=np.zeros((10, 10, 3), dtype=np.uint8),
    )


@patch.object(GoogleGemmaBlockV3, "execute_openrouter_batch")
def test_run_surfaces_token_usage(mock_execute):
    mock_execute.return_value = [("caption text", 11, 4)]
    block = GoogleGemmaBlockV3(model_manager=MagicMock(), api_key="ws-key")

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

    assert result == [
        {
            "output": "caption text",
            "classes": None,
            "input_tokens": 11,
            "output_tokens": 4,
        }
    ]
    kwargs = mock_execute.call_args.kwargs
    assert kwargs["model"] == MODEL_VERSION_MAPPING["Gemma 4 26B A4B - OpenRouter"]
    assert kwargs["include_usage"] is True
