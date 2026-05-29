import base64
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from inference.models.qwen3_5vl import qwen3_5vl_inference_models as qwen35_module


def test_normalize_vllm_base_url_adds_openai_path() -> None:
    result = qwen35_module._normalize_vllm_base_url("http://vllm:8000")

    assert result == "http://vllm:8000/v1"


def test_normalize_vllm_base_url_keeps_existing_openai_path() -> None:
    result = qwen35_module._normalize_vllm_base_url("http://vllm:8000/v1/")

    assert result == "http://vllm:8000/v1"


def test_split_native_prompt_extracts_system_prompt() -> None:
    user_prompt, system_prompt = qwen35_module._split_native_prompt(
        "OCR<system_prompt>You are helpful."
    )

    assert user_prompt == "OCR"
    assert system_prompt == "You are helpful."


def test_qwen35_adapter_delegates_to_vllm(monkeypatch) -> None:
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="HOME 42"),
                finish_reason="stop",
            )
        ]
    )
    openai_factory = MagicMock(return_value=fake_client)
    from_pretrained = MagicMock()

    monkeypatch.setattr(qwen35_module, "VLLM_LMM_ENABLED", True)
    monkeypatch.setattr(qwen35_module, "VLLM_LMM_BASE_URL", "http://vllm:8000")
    monkeypatch.setattr(qwen35_module, "VLLM_LMM_MODEL_NAME", "vlm-ocr-14")
    monkeypatch.setattr(qwen35_module, "VLLM_LMM_API_KEY", "EMPTY")
    monkeypatch.setattr(qwen35_module, "VLLM_LMM_TIMEOUT_SECONDS", 12.0)
    monkeypatch.setattr(qwen35_module, "VLLM_LMM_TEMPERATURE", 0.0)
    monkeypatch.setattr(qwen35_module, "OpenAI", openai_factory)
    monkeypatch.setattr(qwen35_module.AutoModel, "from_pretrained", from_pretrained)

    adapter = qwen35_module.InferenceModelsQwen35VLAdapter(
        model_id="vlm-ocr/14",
        api_key="rf-api-key",
    )
    responses = adapter.infer(
        image=np.zeros((2, 3, 3), dtype=np.uint8),
        prompt="OCR<system_prompt>You are Qwen.",
        max_new_tokens=16,
        enable_thinking=False,
    )

    assert responses[0].response == "HOME 42"
    assert responses[0].image.width == 3
    assert responses[0].image.height == 2
    openai_factory.assert_called_once_with(
        base_url="http://vllm:8000/v1",
        api_key="EMPTY",
        timeout=12.0,
    )
    from_pretrained.assert_not_called()

    create_kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert create_kwargs["model"] == "vlm-ocr-14"
    assert create_kwargs["max_tokens"] == 16
    assert create_kwargs["temperature"] == 0.0
    assert create_kwargs["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }

    messages = create_kwargs["messages"]
    assert messages[0] == {"role": "system", "content": "You are Qwen."}
    assert messages[1]["role"] == "user"
    assert messages[1]["content"][0] == {"type": "text", "text": "OCR"}
    assert messages[1]["content"][1]["type"] == "image_url"
    assert messages[1]["content"][1]["image_url"]["url"].startswith(
        "data:image/jpeg;base64,"
    )


def test_qwen35_adapter_preserves_base64_request_image_for_vllm(monkeypatch) -> None:
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="HOME 42"),
                finish_reason="stop",
            )
        ]
    )
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    image_base64 = base64.b64encode(
        qwen35_module.encode_image_to_jpeg_bytes(image)
    ).decode("ascii")

    monkeypatch.setattr(qwen35_module, "VLLM_LMM_ENABLED", True)
    monkeypatch.setattr(qwen35_module, "VLLM_LMM_BASE_URL", "http://vllm:8000")
    monkeypatch.setattr(qwen35_module, "VLLM_LMM_MODEL_NAME", "vlm-ocr-14")
    monkeypatch.setattr(qwen35_module, "VLLM_LMM_API_KEY", "EMPTY")
    monkeypatch.setattr(qwen35_module, "VLLM_LMM_TIMEOUT_SECONDS", 12.0)
    monkeypatch.setattr(qwen35_module, "VLLM_LMM_TEMPERATURE", 0.0)
    monkeypatch.setattr(qwen35_module, "OpenAI", MagicMock(return_value=fake_client))
    monkeypatch.setattr(qwen35_module.AutoModel, "from_pretrained", MagicMock())

    adapter = qwen35_module.InferenceModelsQwen35VLAdapter(
        model_id="vlm-ocr/14",
        api_key="rf-api-key",
    )
    responses = adapter.infer(
        image={"type": "base64", "value": image_base64},
        prompt="OCR<system_prompt>You are Qwen.",
    )

    assert responses[0].image.width == 3
    assert responses[0].image.height == 2
    messages = fake_client.chat.completions.create.call_args.kwargs["messages"]
    assert messages[1]["content"][1]["image_url"]["url"] == (
        f"data:image/jpeg;base64,{image_base64}"
    )
