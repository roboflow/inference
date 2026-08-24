"""Tests for the unified Qwen-VL block v2 (Qwen-specific OpenRouter plumbing)."""

import base64
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.models.foundation.qwen_vlm import v2
from inference.core.workflows.core_steps.models.foundation.qwen_vlm.v2 import (
    DEFAULT_NATIVE_MODEL_VERSION,
    DEFAULT_OPENROUTER_MODEL_VERSION,
    FINE_TUNED_NATIVE_LABEL,
    BlockManifest,
    QwenVlmBlockV2,
    build_qwen_openrouter_prompts,
    build_reasoning_config,
    encode_image_for_qwen_openrouter,
)
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData

# Copied literally from vlm-exam's `_NORMALIZED_XYXY_PROMPT_TEMPLATE`
# (the benchmarked Qwen detection contract) so any accidental edit to the
# block's template fails this exact-match test.
EXPECTED_DETECTION_PROMPT_TEMPLATE = (
    "Detect all objects in this image. "
    "Output a JSON list where each entry contains the 2D bounding box "
    'in the key "box_2d" and the text label in the key "label". '
    'The "box_2d" value must be [x_min, y_min, x_max, y_max]: the '
    "top-left and bottom-right corners as integers between 0 and 1000, "
    "normalized to the image width (x) and height (y). "
    "Return only the JSON list, with no extra text. "
    "Only use these labels: {class_list}"
)


def _stub_image() -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=MagicMock(
            parent_id="root", workflow_root_ancestor_metadata=None
        ),
        numpy_image=np.zeros((10, 10, 3), dtype=np.uint8),
    )


def _base_run_kwargs(**overrides):
    """Default kwargs for QwenVlmBlockV2.run; override specific keys per test."""
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


# ---------------------------------------------------------------------------
# Manifest validation
# ---------------------------------------------------------------------------


def test_manifest_defaults():
    manifest = BlockManifest.model_validate(
        {
            "type": "roboflow_core/qwen_vlm@v2",
            "name": "step",
            "images": "$inputs.image",
            "task_type": "caption",
        }
    )
    assert manifest.backend == "native"
    assert manifest.model_version == DEFAULT_NATIVE_MODEL_VERSION
    assert manifest.openrouter_model_version == DEFAULT_OPENROUTER_MODEL_VERSION
    assert manifest.max_tokens == 2048
    assert manifest.temperature is None
    assert manifest.reasoning_effort == "none"


def test_manifest_object_detection_requires_classes():
    with pytest.raises(ValidationError):
        BlockManifest.model_validate(
            {
                "type": "roboflow_core/qwen_vlm@v2",
                "name": "step",
                "images": "$inputs.image",
                "task_type": "object-detection",
                "backend": "openrouter",
            }
        )


def test_manifest_native_fine_tuned_requires_model_id():
    with pytest.raises(ValidationError, match="fine_tuned_model_id"):
        BlockManifest.model_validate(
            {
                "type": "roboflow_core/qwen_vlm@v2",
                "name": "step",
                "images": "$inputs.image",
                "task_type": "caption",
                "backend": "native",
                "model_version": FINE_TUNED_NATIVE_LABEL,
            }
        )


def test_manifest_openrouter_resets_stale_fine_tuned_model_version():
    manifest = BlockManifest.model_validate(
        {
            "type": "roboflow_core/qwen_vlm@v2",
            "name": "step",
            "images": "$inputs.image",
            "task_type": "ocr",
            "backend": "openrouter",
            "model_version": FINE_TUNED_NATIVE_LABEL,
        }
    )
    assert manifest.model_version == DEFAULT_NATIVE_MODEL_VERSION


# ---------------------------------------------------------------------------
# Reasoning config
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "effort,reasoning_required,expected",
    [
        ("none", False, {"enabled": False}),
        ("none", True, {"effort": "low"}),
        ("low", False, {"effort": "low"}),
        ("low", True, {"effort": "low"}),
        ("medium", False, {"effort": "medium"}),
        ("high", False, {"effort": "high"}),
        ("high", True, {"effort": "high"}),
    ],
)
def test_build_reasoning_config(effort, reasoning_required, expected):
    assert (
        build_reasoning_config(effort, reasoning_required=reasoning_required)
        == expected
    )


# ---------------------------------------------------------------------------
# OpenRouter prompt builders
# ---------------------------------------------------------------------------


def test_detection_prompt_matches_benchmarked_template_exactly():
    image = np.zeros((32, 32, 3), dtype=np.uint8)

    prompts = build_qwen_openrouter_prompts(
        images=[image],
        task_type="object-detection",
        prompt=None,
        output_structure=None,
        classes=["dog", "cat"],
    )

    text_part = prompts[0][0]["content"][1]
    assert text_part["type"] == "text"
    assert text_part["text"] == EXPECTED_DETECTION_PROMPT_TEMPLATE.format(
        class_list="dog, cat"
    )


@pytest.mark.parametrize(
    "task_type,kwargs",
    [
        ("unconstrained", {"prompt": "What is this?"}),
        ("ocr", {}),
        ("visual-question-answering", {"prompt": "How many dogs?"}),
        ("caption", {}),
        ("detailed-caption", {}),
        ("classification", {"classes": ["a", "b"]}),
        ("multi-label-classification", {"classes": ["a", "b"]}),
        ("structured-answering", {"output_structure": {"field": "desc"}}),
        ("object-detection", {"classes": ["a", "b"]}),
    ],
)
def test_prompts_are_single_user_message_with_image_first(task_type, kwargs):
    image = np.zeros((16, 16, 3), dtype=np.uint8)

    prompts = build_qwen_openrouter_prompts(
        images=[image],
        task_type=task_type,
        prompt=kwargs.get("prompt"),
        output_structure=kwargs.get("output_structure"),
        classes=kwargs.get("classes"),
    )

    assert len(prompts) == 1
    messages = prompts[0]
    # Qwen contract: single user message, no system role, image part first.
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    content = messages[0]["content"]
    assert len(content) == 2
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert content[1]["type"] == "text"
    assert len(content[1]["text"]) > 0


def test_native_detection_prompt_uses_same_benchmarked_template():
    # Both backends must emit one detection contract so the recommended
    # parser (vlm_as_detector@v2, model_type="qwen") works regardless of
    # backend. Regression guard for the v1-era x_min/0.0-1.0 prompt.
    prompt = v2._build_native_prompt(
        task_type="object-detection",
        prompt=None,
        output_structure=None,
        classes=["dog", "cat"],
    )

    user_text, system_text = prompt.split("<system_prompt>")
    assert user_text == EXPECTED_DETECTION_PROMPT_TEMPLATE.format(class_list="dog, cat")
    # Empty system half: the model server substitutes its default system
    # prompt, the closest native equivalent of the benchmark's
    # single-user-message structure.
    assert system_text == ""


def test_classification_prompt_keeps_json_contract_and_classes():
    image = np.zeros((16, 16, 3), dtype=np.uint8)

    prompts = build_qwen_openrouter_prompts(
        images=[image],
        task_type="classification",
        prompt=None,
        output_structure=None,
        classes=["dog", "cat"],
    )

    text = prompts[0][0]["content"][1]["text"]
    # Output contract consumed by vlm_as_classifier@v2 must be preserved.
    assert '{"class_name": "class-name", "confidence": 0.4}' in text
    assert "List of all classes to be recognised by model: dog, cat" in text


# ---------------------------------------------------------------------------
# Payload-capped image encoding
# ---------------------------------------------------------------------------


def test_encode_image_under_cap_keeps_dimensions():
    image = np.zeros((64, 128, 3), dtype=np.uint8)

    encoded = encode_image_for_qwen_openrouter(image)

    decoded = cv2.imdecode(
        np.frombuffer(base64.b64decode(encoded), dtype=np.uint8), cv2.IMREAD_COLOR
    )
    assert decoded.shape[:2] == (64, 128)


def test_encode_image_over_cap_downscales_until_it_fits():
    rng = np.random.default_rng(42)
    # Random noise compresses poorly, keeping the payload above a tiny cap
    # until the image is downscaled substantially.
    image = rng.integers(0, 256, size=(512, 1024, 3), dtype=np.uint8)

    with patch.object(v2, "OPENROUTER_MAX_BASE64_BYTES", 50_000):
        encoded = encode_image_for_qwen_openrouter(image)

    assert len(encoded) <= 50_000
    decoded = cv2.imdecode(
        np.frombuffer(base64.b64decode(encoded), dtype=np.uint8), cv2.IMREAD_COLOR
    )
    assert decoded is not None
    height, width = decoded.shape[:2]
    assert height < 512 and width < 1024
    # Aspect ratio is preserved through iterative downscaling.
    assert width / height == pytest.approx(2.0, abs=0.1)


# ---------------------------------------------------------------------------
# Backend dispatch
# ---------------------------------------------------------------------------


@patch.object(QwenVlmBlockV2, "execute_openrouter_batch")
def test_run_openrouter_passes_slug_reasoning_and_temperature(mock_or):
    mock_or.return_value = [("resp", "", 11, 7)]
    block = QwenVlmBlockV2(
        model_manager=MagicMock(),
        api_key="ws-key",
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    result = block.run(
        **_base_run_kwargs(
            backend="openrouter",
            task_type="object-detection",
            classes=["dog", "cat"],
        )
    )

    assert mock_or.called
    call_kwargs = mock_or.call_args.kwargs
    assert call_kwargs["model"] == "qwen/qwen3.7-plus"
    assert call_kwargs["reasoning"] == {"enabled": False}
    assert call_kwargs["temperature"] is None
    assert call_kwargs["max_tokens"] == 2048
    assert call_kwargs["include_reasoning"] is True
    assert result == [
        {
            "output": "resp",
            "classes": ["dog", "cat"],
            "thinking": "",
            "input_tokens": 11,
            "output_tokens": 7,
        }
    ]


@patch.object(QwenVlmBlockV2, "execute_openrouter_batch")
def test_run_openrouter_populates_thinking_from_reasoning_trace(mock_or):
    mock_or.return_value = [("the answer", "step-by-step trace", 5, 3)]
    block = QwenVlmBlockV2(
        model_manager=MagicMock(),
        api_key="ws-key",
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    result = block.run(
        **_base_run_kwargs(
            backend="openrouter",
            task_type="ocr",
            reasoning_effort="low",
        )
    )

    assert result == [
        {
            "output": "the answer",
            "classes": None,
            "thinking": "step-by-step trace",
            "input_tokens": 5,
            "output_tokens": 3,
        }
    ]


@patch.object(QwenVlmBlockV2, "execute_openrouter_batch")
def test_run_openrouter_reasoning_required_model_falls_back_to_low_effort(mock_or):
    mock_or.return_value = [("resp", "", 11, 7)]
    block = QwenVlmBlockV2(
        model_manager=MagicMock(),
        api_key="ws-key",
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    block.run(
        **_base_run_kwargs(
            backend="openrouter",
            openrouter_model_version="Qwen 3.8 Max",
            task_type="ocr",
            reasoning_effort="none",
        )
    )

    call_kwargs = mock_or.call_args.kwargs
    assert call_kwargs["model"] == "qwen/qwen3.8-max"
    # Qwen 3.8 Max rejects `enabled: false`; disabled maps to low effort.
    assert call_kwargs["reasoning"] == {"effort": "low"}


@patch.object(QwenVlmBlockV2, "execute_openrouter_batch")
def test_run_openrouter_explicit_max_tokens_overrides_default(mock_or):
    mock_or.return_value = [("resp", "", 11, 7)]
    block = QwenVlmBlockV2(
        model_manager=MagicMock(),
        api_key="ws-key",
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    block.run(
        **_base_run_kwargs(
            backend="openrouter",
            task_type="ocr",
            max_tokens=8192,
        )
    )

    assert mock_or.call_args.kwargs["max_tokens"] == 8192


def test_run_native_default_max_tokens_is_forwarded():
    model_manager = MagicMock()
    fake_prediction = MagicMock()
    fake_prediction.response = "answer"
    model_manager.infer_from_request_sync.return_value = fake_prediction

    block = QwenVlmBlockV2(
        model_manager=model_manager,
        api_key="ws-key",
        step_execution_mode=StepExecutionMode.LOCAL,
    )
    block.run(**_base_run_kwargs())

    request = model_manager.infer_from_request_sync.call_args.kwargs["request"]
    assert request.max_new_tokens == 2048


def test_run_native_explicit_max_tokens_is_forwarded_as_max_new_tokens():
    model_manager = MagicMock()
    fake_prediction = MagicMock()
    fake_prediction.response = "answer"
    model_manager.infer_from_request_sync.return_value = fake_prediction

    block = QwenVlmBlockV2(
        model_manager=model_manager,
        api_key="ws-key",
        step_execution_mode=StepExecutionMode.LOCAL,
    )
    block.run(**_base_run_kwargs(max_tokens=1024))

    request = model_manager.infer_from_request_sync.call_args.kwargs["request"]
    assert request.max_new_tokens == 1024


def test_run_dispatches_to_local_native_when_step_mode_local():
    model_manager = MagicMock()
    fake_prediction = MagicMock()
    fake_prediction.response = "native local answer"
    model_manager.infer_from_request_sync.return_value = fake_prediction

    block = QwenVlmBlockV2(
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
    model_manager.add_model.assert_called_once_with(
        model_id="qwen3_5-2b", api_key="ws-key"
    )


def test_run_local_native_with_enable_thinking_splits_response():
    model_manager = MagicMock()
    fake_prediction = MagicMock()
    fake_prediction.response = {"thinking": "reasoning...", "answer": "42"}
    model_manager.infer_from_request_sync.return_value = fake_prediction

    block = QwenVlmBlockV2(
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
        {
            "output": "42",
            "classes": None,
            "thinking": "reasoning...",
            "input_tokens": None,
            "output_tokens": None,
        }
    ]
    request = model_manager.infer_from_request_sync.call_args.kwargs["request"]
    assert request.enable_thinking is True


@patch(
    "inference.core.workflows.core_steps.models.foundation.qwen_vlm.v2.InferenceHTTPClient"
)
def test_run_dispatches_to_remote_native_when_step_mode_remote(mock_client_cls):
    fake_client = MagicMock()
    fake_client.infer_lmm.return_value = {"response": "remote answer"}
    mock_client_cls.return_value = fake_client

    block = QwenVlmBlockV2(
        model_manager=MagicMock(),
        api_key="ws-key",
        step_execution_mode=StepExecutionMode.REMOTE,
    )
    result = block.run(
        **_base_run_kwargs(
            model_version="Qwen 3.5 VL 0.8B",
            task_type="ocr",
        )
    )
    assert result == [
        {
            "output": "remote answer",
            "classes": None,
            "thinking": "",
            "input_tokens": None,
            "output_tokens": None,
        }
    ]
    assert fake_client.infer_lmm.call_args.kwargs["model_id"] == "qwen3_5-0.8b"


def test_run_dispatches_to_local_native_with_fine_tuned_model_id():
    model_manager = MagicMock()
    fake_prediction = MagicMock()
    fake_prediction.response = "finetune answer"
    model_manager.infer_from_request_sync.return_value = fake_prediction

    block = QwenVlmBlockV2(
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
        {
            "output": "finetune answer",
            "classes": None,
            "thinking": "",
            "input_tokens": None,
            "output_tokens": None,
        }
    ]
    model_manager.add_model.assert_called_once_with(
        model_id="your-workspace/3", api_key="ws-key"
    )
