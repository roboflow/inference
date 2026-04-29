"""Tests for the unified Qwen-VL block."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.models.foundation.qwen_vlm.v1 import (
    BlockManifest,
    MODEL_VARIANTS,
    QwenVlmBlockV1,
    _build_native_prompt,
    _coerce_native_response_to_str,
)
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData


def _stub_image() -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=MagicMock(parent_id="root", workflow_root_ancestor_metadata=None),
        numpy_image=np.zeros((10, 10, 3), dtype=np.uint8),
    )


# ---------------------------------------------------------------------------
# Manifest validation
# ---------------------------------------------------------------------------


def test_manifest_native_default():
    manifest = BlockManifest.model_validate(
        {
            "type": "roboflow_core/qwen_vlm@v1",
            "name": "step",
            "images": "$inputs.image",
            "task_type": "caption",
        }
    )
    assert manifest.backend == "native"
    assert manifest.model_version == "Qwen 3.5 VL 2B - Native"
    assert manifest.api_key == "rf_key:account"
    assert manifest.privacy_level == "deny"


def test_manifest_rejects_native_backend_with_openrouter_model():
    with pytest.raises(ValidationError, match="requires backend 'openrouter'"):
        BlockManifest.model_validate(
            {
                "type": "roboflow_core/qwen_vlm@v1",
                "name": "step",
                "images": "$inputs.image",
                "task_type": "ocr",
                "backend": "native",
                "model_version": "Qwen 3.6 27B - OpenRouter",
            }
        )


def test_manifest_rejects_openrouter_backend_with_native_model():
    with pytest.raises(ValidationError, match="requires backend 'native'"):
        BlockManifest.model_validate(
            {
                "type": "roboflow_core/qwen_vlm@v1",
                "name": "step",
                "images": "$inputs.image",
                "task_type": "ocr",
                "backend": "openrouter",
                "model_version": "Qwen 3.5 VL 2B - Native",
            }
        )


def test_manifest_accepts_selector_for_model_version():
    """Selector references like `$inputs.qwen_model` skip the cross-check."""
    manifest = BlockManifest.model_validate(
        {
            "type": "roboflow_core/qwen_vlm@v1",
            "name": "step",
            "images": "$inputs.image",
            "task_type": "caption",
            "model_version": "$inputs.qwen_model",
            "backend": "openrouter",
        }
    )
    assert manifest.model_version == "$inputs.qwen_model"


def test_all_variants_have_valid_backend():
    for label, info in MODEL_VARIANTS.items():
        assert info["backend"] in ("native", "openrouter"), label
        assert info["model_id"], label


# ---------------------------------------------------------------------------
# Native prompt builder
# ---------------------------------------------------------------------------


def test_build_native_prompt_unconstrained_uses_user_prompt():
    out = _build_native_prompt(
        task_type="unconstrained",
        prompt="What's in here?",
        output_structure=None,
        classes=None,
    )
    user, _, system = out.partition("<system_prompt>")
    assert user == "What's in here?"
    # System prompt identity-primes the model as Qwen, matching the legacy
    # native qwen v1 blocks (some Qwen variants are sensitive to identity priming).
    assert "Qwen" in system


def test_build_native_prompt_classification_uses_classes():
    out = _build_native_prompt(
        task_type="classification",
        prompt=None,
        output_structure=None,
        classes=["dog", "cat"],
    )
    assert "dog, cat" in out
    assert "<system_prompt>" in out
    # System portion should mention single-class JSON schema.
    assert "single-class classification" in out.split("<system_prompt>")[1]


def test_build_native_prompt_unknown_task_raises():
    with pytest.raises(ValueError, match="not supported"):
        _build_native_prompt(
            task_type="garbage",
            prompt=None,
            output_structure=None,
            classes=None,
        )


# ---------------------------------------------------------------------------
# Native response coercion (downstream parsers expect strings)
# ---------------------------------------------------------------------------


def test_coerce_native_response_strings_pass_through():
    assert _coerce_native_response_to_str("hello") == "hello"


def test_coerce_native_response_dict_with_answer_returns_answer():
    out = _coerce_native_response_to_str({"thinking": "...", "answer": "yes"})
    assert out == "yes"


def test_coerce_native_response_dict_without_answer_returns_json():
    out = _coerce_native_response_to_str({"foo": "bar"})
    assert "foo" in out and "bar" in out


def test_coerce_native_response_none_returns_empty_string():
    assert _coerce_native_response_to_str(None) == ""


def test_coerce_native_response_other_types_serialized_to_json():
    out = _coerce_native_response_to_str(42)
    assert out == "42"


# ---------------------------------------------------------------------------
# Backend dispatch
# ---------------------------------------------------------------------------


@patch.object(QwenVlmBlockV1, "execute_openrouter_batch")
def test_run_dispatches_to_openrouter_for_openrouter_backend(mock_or):
    mock_or.return_value = ["resp"]
    block = QwenVlmBlockV1(
        model_manager=MagicMock(),
        api_key="ws-key",
        step_execution_mode=StepExecutionMode.LOCAL,  # ignored for openrouter path
    )

    block.run(
        images=[_stub_image()],
        backend="openrouter",
        model_version="Qwen 3.6 27B - OpenRouter",
        task_type="ocr",
        prompt=None,
        output_structure=None,
        classes=None,
        api_key="rf_key:account",
        privacy_level="deny",
        max_tokens=128,
        temperature=0.1,
        max_concurrent_requests=None,
    )
    assert mock_or.called
    assert mock_or.call_args.kwargs["model"] == "qwen/qwen3.6-27b"
    assert mock_or.call_args.kwargs["privacy_level"] == "deny"


def test_run_dispatches_to_local_native_when_step_mode_local():
    model_manager = MagicMock()
    fake_prediction = MagicMock()
    fake_prediction.response = "native local answer"
    model_manager.infer_from_request_sync.return_value = fake_prediction

    block = QwenVlmBlockV1(
        model_manager=model_manager,
        api_key="ws-key",
        step_execution_mode=StepExecutionMode.LOCAL,
    )
    result = block.run(
        images=[_stub_image()],
        backend="native",
        model_version="Qwen 3.5 VL 2B - Native",
        task_type="caption",
        prompt=None,
        output_structure=None,
        classes=None,
        api_key="rf_key:account",
        privacy_level="deny",
        max_tokens=64,
        temperature=0.1,
        max_concurrent_requests=None,
    )
    assert result == [{"output": "native local answer", "classes": None}]
    model_manager.add_model.assert_called_once_with(
        model_id="qwen3_5-2b", api_key="ws-key"
    )
    assert model_manager.infer_from_request_sync.called


@patch(
    "inference.core.workflows.core_steps.models.foundation.qwen_vlm.v1.InferenceHTTPClient"
)
def test_run_dispatches_to_remote_native_when_step_mode_remote(mock_client_cls):
    fake_client = MagicMock()
    fake_client.infer_lmm.return_value = {"response": "remote answer"}
    mock_client_cls.return_value = fake_client

    block = QwenVlmBlockV1(
        model_manager=MagicMock(),
        api_key="ws-key",
        step_execution_mode=StepExecutionMode.REMOTE,
    )
    result = block.run(
        images=[_stub_image()],
        backend="native",
        model_version="Qwen 3.5 VL 0.8B - Native",
        task_type="ocr",
        prompt=None,
        output_structure=None,
        classes=None,
        api_key="rf_key:account",
        privacy_level="deny",
        max_tokens=64,
        temperature=0.0,
        max_concurrent_requests=None,
    )
    assert result == [{"output": "remote answer", "classes": None}]
    assert fake_client.infer_lmm.called
    call_kwargs = fake_client.infer_lmm.call_args.kwargs
    assert call_kwargs["model_id"] == "qwen3_5-0.8b"
