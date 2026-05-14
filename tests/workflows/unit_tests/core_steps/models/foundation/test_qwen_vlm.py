"""Tests for the unified Qwen-VL block."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.models.foundation.qwen_vlm.v1 import (
    BlockManifest,
    DEFAULT_NATIVE_MODEL_VERSION,
    FINE_TUNED_NATIVE_LABEL,
    MODEL_VARIANTS,
    QwenVlmBlockV1,
    _build_native_prompt,
    _coerce_native_response,
)
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData


def _stub_image() -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=MagicMock(parent_id="root", workflow_root_ancestor_metadata=None),
        numpy_image=np.zeros((10, 10, 3), dtype=np.uint8),
    )


def _base_run_kwargs(**overrides):
    """Default kwargs for QwenVlmBlockV1.run; override specific keys per test."""
    kwargs = dict(
        images=[_stub_image()],
        backend="native",
        model_version="Qwen 3.5 VL 2B",
        fine_tuned_model_id=None,
        openrouter_model_version="Qwen 3.6 27B",
        task_type="caption",
        prompt=None,
        enable_thinking=False,
        output_structure=None,
        classes=None,
        api_key="rf_key:account",
        privacy_level="deny",
        max_tokens=64,
        temperature=0.1,
        max_concurrent_requests=None,
    )
    kwargs.update(overrides)
    return kwargs


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
    # Default lands on a pre-trained variant; fine-tuned is opt-in via the
    # sentinel entry in the same dropdown.
    assert manifest.model_version == "Qwen 3.5 VL 2B"
    assert manifest.fine_tuned_model_id is None
    assert manifest.openrouter_model_version == "Qwen 3.6 27B"
    assert manifest.api_key == "rf_key:account"
    assert manifest.privacy_level == "deny"
    # Thinking mode is off by default; matches legacy qwen3_5vl@v1.
    assert manifest.enable_thinking is False


def test_manifest_native_fine_tuned_requires_model_id():
    """Picking the `Fine-tuned` dropdown entry without a model id is rejected."""
    with pytest.raises(ValidationError, match="fine_tuned_model_id"):
        BlockManifest.model_validate(
            {
                "type": "roboflow_core/qwen_vlm@v1",
                "name": "step",
                "images": "$inputs.image",
                "task_type": "caption",
                "backend": "native",
                "model_version": FINE_TUNED_NATIVE_LABEL,
            }
        )


def test_manifest_native_accepts_fine_tuned_model_id():
    """Workspace fine-tunes (e.g. `your-workspace/3`) flow through as model_id."""
    manifest = BlockManifest.model_validate(
        {
            "type": "roboflow_core/qwen_vlm@v1",
            "name": "step",
            "images": "$inputs.image",
            "task_type": "caption",
            "backend": "native",
            "model_version": FINE_TUNED_NATIVE_LABEL,
            "fine_tuned_model_id": "your-workspace/3",
        }
    )
    assert manifest.model_version == FINE_TUNED_NATIVE_LABEL
    assert manifest.fine_tuned_model_id == "your-workspace/3"


def test_manifest_openrouter_resets_stale_fine_tuned_model_version():
    """Switching backend to OpenRouter while `model_version` still carries the
    stale `FINE_TUNED_NATIVE_LABEL` from a prior native session must reset
    model_version. That way the `fine_tuned_model_id` selector (gated on
    `model_version=FINE_TUNED_NATIVE_LABEL`) hides itself in the UI on
    revalidation."""
    manifest = BlockManifest.model_validate(
        {
            "type": "roboflow_core/qwen_vlm@v1",
            "name": "step",
            "images": "$inputs.image",
            "task_type": "ocr",
            "backend": "openrouter",
            "model_version": FINE_TUNED_NATIVE_LABEL,
            # No fine_tuned_model_id — the validator would normally reject
            # this on native, but the reset makes it irrelevant for openrouter.
        }
    )
    assert manifest.model_version == DEFAULT_NATIVE_MODEL_VERSION


def test_manifest_accepts_selector_for_openrouter_model_version():
    manifest = BlockManifest.model_validate(
        {
            "type": "roboflow_core/qwen_vlm@v1",
            "name": "step",
            "images": "$inputs.image",
            "task_type": "caption",
            "openrouter_model_version": "$inputs.qwen_model",
            "backend": "openrouter",
        }
    )
    assert manifest.openrouter_model_version == "$inputs.qwen_model"


def test_manifest_accepts_enable_thinking():
    manifest = BlockManifest.model_validate(
        {
            "type": "roboflow_core/qwen_vlm@v1",
            "name": "step",
            "images": "$inputs.image",
            "task_type": "unconstrained",
            "prompt": "What is in this image?",
            "backend": "native",
            "model_version": "Qwen 3.5 VL 2B",
            "enable_thinking": True,
        }
    )
    assert manifest.enable_thinking is True


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
    assert _coerce_native_response("hello") == ("hello", "")


def test_coerce_native_response_dict_with_answer_returns_answer_and_thinking():
    out = _coerce_native_response({"thinking": "reasoning trace", "answer": "yes"})
    assert out == ("yes", "reasoning trace")


def test_coerce_native_response_dict_without_thinking_returns_empty_thinking():
    out = _coerce_native_response({"answer": "yes"})
    assert out == ("yes", "")


def test_coerce_native_response_dict_without_answer_returns_json_and_empty_thinking():
    output, thinking = _coerce_native_response({"foo": "bar"})
    assert "foo" in output and "bar" in output
    assert thinking == ""


def test_coerce_native_response_none_returns_empty_strings():
    assert _coerce_native_response(None) == ("", "")


def test_coerce_native_response_other_types_serialized_to_json():
    assert _coerce_native_response(42) == ("42", "")


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

    result = block.run(
        **_base_run_kwargs(
            backend="openrouter",
            task_type="ocr",
            max_tokens=128,
        )
    )
    assert mock_or.called
    assert mock_or.call_args.kwargs["model"] == "qwen/qwen3.6-27b"
    assert mock_or.call_args.kwargs["privacy_level"] == "deny"
    # OpenRouter path always reports empty thinking — that's a native-only field.
    assert result == [{"output": "resp", "classes": None, "thinking": ""}]


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
    result = block.run(**_base_run_kwargs())
    assert result == [
        {"output": "native local answer", "classes": None, "thinking": ""}
    ]
    model_manager.add_model.assert_called_once_with(
        model_id="qwen3_5-2b", api_key="ws-key"
    )
    assert model_manager.infer_from_request_sync.called
    request = model_manager.infer_from_request_sync.call_args.kwargs["request"]
    assert request.enable_thinking is False


def test_run_local_native_with_enable_thinking_splits_response():
    """When the native runtime returns {thinking, answer}, the block splits
    that into the `output` and `thinking` outputs (matches legacy 3.5 block)."""
    model_manager = MagicMock()
    fake_prediction = MagicMock()
    fake_prediction.response = {"thinking": "reasoning...", "answer": "42"}
    model_manager.infer_from_request_sync.return_value = fake_prediction

    block = QwenVlmBlockV1(
        model_manager=model_manager,
        api_key="ws-key",
        step_execution_mode=StepExecutionMode.LOCAL,
    )
    result = block.run(
        **_base_run_kwargs(
            task_type="unconstrained",
            prompt="What is 6 times 7?",
            enable_thinking=True,
        )
    )
    assert result == [
        {"output": "42", "classes": None, "thinking": "reasoning..."}
    ]
    request = model_manager.infer_from_request_sync.call_args.kwargs["request"]
    assert request.enable_thinking is True


def test_run_local_native_enable_thinking_silently_ignored_on_unsupported_model():
    """`enable_thinking=True` on a non-3.5 variant must NOT propagate to the
    LMM request — the field is a no-op there, gated by NATIVE_THINKING_MODEL_VERSIONS."""
    model_manager = MagicMock()
    fake_prediction = MagicMock()
    fake_prediction.response = "ok"
    model_manager.infer_from_request_sync.return_value = fake_prediction

    block = QwenVlmBlockV1(
        model_manager=model_manager,
        api_key="ws-key",
        step_execution_mode=StepExecutionMode.LOCAL,
    )
    block.run(
        **_base_run_kwargs(
            model_version="Qwen 2.5 VL 7B",
            task_type="unconstrained",
            prompt="hi",
            enable_thinking=True,
        )
    )
    request = model_manager.infer_from_request_sync.call_args.kwargs["request"]
    assert request.enable_thinking is False


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
        **_base_run_kwargs(
            model_version="Qwen 3.5 VL 0.8B",
            task_type="ocr",
            temperature=0.0,
        )
    )
    assert result == [
        {"output": "remote answer", "classes": None, "thinking": ""}
    ]
    assert fake_client.infer_lmm.called
    call_kwargs = fake_client.infer_lmm.call_args.kwargs
    assert call_kwargs["model_id"] == "qwen3_5-0.8b"
    # `enable_thinking` is always forwarded to the remote LMM call so the
    # remote side has the bit available; gating is done before this hop.
    assert "enable_thinking" in call_kwargs


def test_run_dispatches_to_local_native_with_fine_tuned_model_id():
    """Fine-tuned source path: the workspace model_id is passed straight to model_manager."""
    model_manager = MagicMock()
    fake_prediction = MagicMock()
    fake_prediction.response = "finetune answer"
    model_manager.infer_from_request_sync.return_value = fake_prediction

    block = QwenVlmBlockV1(
        model_manager=model_manager,
        api_key="ws-key",
        step_execution_mode=StepExecutionMode.LOCAL,
    )
    result = block.run(
        **_base_run_kwargs(
            model_version=FINE_TUNED_NATIVE_LABEL,
            fine_tuned_model_id="your-workspace/3",
        )
    )
    assert result == [
        {"output": "finetune answer", "classes": None, "thinking": ""}
    ]
    model_manager.add_model.assert_called_once_with(
        model_id="your-workspace/3", api_key="ws-key"
    )
