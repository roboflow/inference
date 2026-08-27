"""Tests for the Qwen VLM v3 block (v2 + token-usage outputs).

The v2 behavior suite lives in ``test_qwen_vlm_v2.py``; this file covers
the v3 delta: ``input_tokens`` / ``output_tokens`` outputs on the
OpenRouter backend, and ``None`` token counts on the native backend.
"""

from unittest.mock import MagicMock, patch

import numpy as np

from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.common.openrouter import OpenRouterResult
from inference.core.workflows.core_steps.models.foundation.qwen_vlm.v3 import (
    DEFAULT_OPENROUTER_MODEL_VERSION,
    BlockManifest,
    QwenVlmBlockV3,
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
        backend="native",
        model_version="Qwen 3.5 VL 2B",
        fine_tuned_model_id=None,
        openrouter_model_version=DEFAULT_OPENROUTER_MODEL_VERSION,
        task_type="caption",
        prompt=None,
        enable_thinking=False,
        reasoning_effort="none",
        output_structure=None,
        classes=None,
        api_key="rf_key:account",
        privacy_level="deny",
        max_tokens=2048,
        temperature=None,
        max_concurrent_requests=None,
    )
    kwargs.update(overrides)
    return kwargs


@patch.object(QwenVlmBlockV3, "execute_openrouter_batch_with_usage")
def test_run_openrouter_surfaces_token_usage(mock_or):
    mock_or.return_value = [
        OpenRouterResult(
            content="resp", reasoning_trace="trace", input_tokens=11, output_tokens=7
        )
    ]
    block = QwenVlmBlockV3(
        model_manager=MagicMock(),
        api_key="ws-key",
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    result = block.run(**_base_run_kwargs(backend="openrouter"))

    assert result == [
        {
            "output": "resp",
            "classes": None,
            "thinking": "trace",
            "input_tokens": 11,
            "output_tokens": 7,
        }
    ]


def test_manifest_accepts_qwen_3_8_flash_variant():
    manifest = BlockManifest(
        type="roboflow_core/qwen_vlm@v3",
        name="qwen",
        images="$inputs.image",
        backend="openrouter",
        openrouter_model_version="Qwen 3.8 Flash",
        task_type="unconstrained",
        prompt="What is in this image?",
    )

    assert manifest.openrouter_model_version == "Qwen 3.8 Flash"


@patch.object(QwenVlmBlockV3, "execute_openrouter_batch_with_usage")
def test_run_openrouter_qwen_3_8_flash_maps_slug_and_disables_reasoning(mock_or):
    # OpenRouter reports reasoning as non-mandatory for qwen/qwen3.8-flash,
    # so the default effort ("none") must translate to `enabled: false`
    # instead of the low-effort fallback used by reasoning-required models.
    mock_or.return_value = [
        OpenRouterResult(
            content="resp", reasoning_trace="", input_tokens=5, output_tokens=3
        )
    ]
    block = QwenVlmBlockV3(
        model_manager=MagicMock(),
        api_key="ws-key",
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    result = block.run(
        **_base_run_kwargs(
            backend="openrouter",
            openrouter_model_version="Qwen 3.8 Flash",
        )
    )

    call_kwargs = mock_or.call_args.kwargs
    assert call_kwargs["model"] == "qwen/qwen3.8-flash"
    assert call_kwargs["reasoning"] == {"enabled": False}
    assert result[0]["output"] == "resp"


def test_run_native_reports_none_token_usage():
    model_manager = MagicMock()
    fake_prediction = MagicMock()
    fake_prediction.response = "native local answer"
    model_manager.infer_from_request_sync.return_value = fake_prediction

    block = QwenVlmBlockV3(
        model_manager=model_manager,
        api_key="ws-key",
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    result = block.run(**_base_run_kwargs())

    assert result == [
        {
            "output": "native local answer",
            "classes": None,
            "thinking": "",
            "input_tokens": None,
            "output_tokens": None,
        }
    ]
