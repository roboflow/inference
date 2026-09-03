"""Tests for the Meta VLM v2 block (v1 + token-usage outputs).

The v1 behavior suite lives in ``test_meta_vlm_v1.py``; this file covers
the v2 delta: ``input_tokens`` / ``output_tokens`` outputs.
"""

from unittest.mock import MagicMock, patch

import numpy as np

from inference.core.workflows.core_steps.common.openrouter import OpenRouterResult
from inference.core.workflows.core_steps.models.foundation.meta_vlm.v2 import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL_VERSION,
    DEFAULT_REASONING_EFFORT,
    BlockManifest,
    MetaVlmBlockV2,
)
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData


def _stub_image() -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=MagicMock(
            parent_id="root", workflow_root_ancestor_metadata=None
        ),
        numpy_image=np.zeros((10, 10, 3), dtype=np.uint8),
    )


def _base_run_kwargs(**overrides):
    kwargs = dict(
        images=[_stub_image()],
        model_version=DEFAULT_MODEL_VERSION,
        task_type="caption",
        prompt=None,
        output_structure=None,
        classes=None,
        reasoning_effort=DEFAULT_REASONING_EFFORT,
        api_key="rf_key:account",
        privacy_level="deny",
        max_tokens=DEFAULT_MAX_TOKENS,
        temperature=None,
        max_concurrent_requests=None,
    )
    kwargs.update(overrides)
    return kwargs


def test_manifest_defaults_to_muse_spark_1_3():
    manifest = BlockManifest.model_validate(
        {
            "type": "roboflow_core/meta_vlm@v2",
            "name": "muse",
            "images": "$inputs.image",
            "task_type": "caption",
        }
    )

    assert manifest.model_version == "Muse Spark 1.3"
    assert DEFAULT_MODEL_VERSION == "Muse Spark 1.3"


@patch(
    "inference.core.workflows.core_steps.models.foundation.meta_vlm.v2."
    "OpenRouterWorkflowBlockBase.execute_openrouter_batch_with_usage"
)
def test_run_openrouter_passes_spark_1_3_slug_and_reasoning(mock_or):
    mock_or.return_value = [
        OpenRouterResult(
            content="boxes", reasoning_trace="trace", input_tokens=20, output_tokens=8
        )
    ]
    block = MetaVlmBlockV2(model_manager=MagicMock(), api_key="rf_key")

    result = block.run(
        **_base_run_kwargs(
            task_type="object-detection",
            classes=["cat"],
            model_version="Muse Spark 1.3",
            reasoning_effort="minimal",
        )
    )

    assert mock_or.call_args.kwargs["model"] == "meta/muse-spark-1.3"
    assert mock_or.call_args.kwargs["reasoning"] == {"effort": "minimal"}
    assert result == [
        {
            "output": "boxes",
            "classes": ["cat"],
            "thinking": "trace",
            "input_tokens": 20,
            "output_tokens": 8,
        }
    ]


@patch(
    "inference.core.workflows.core_steps.models.foundation.meta_vlm.v2."
    "OpenRouterWorkflowBlockBase.execute_openrouter_batch_with_usage"
)
def test_run_surfaces_token_usage(mock_or):
    mock_or.return_value = [
        OpenRouterResult(
            content="boxes", reasoning_trace="trace", input_tokens=20, output_tokens=8
        )
    ]
    block = MetaVlmBlockV2(model_manager=MagicMock(), api_key="rf_key")

    result = block.run(
        **_base_run_kwargs(
            task_type="object-detection",
            classes=["cat"],
            model_version="Muse Glimmer",
        )
    )

    assert result == [
        {
            "output": "boxes",
            "classes": ["cat"],
            "thinking": "trace",
            "input_tokens": 20,
            "output_tokens": 8,
        }
    ]
