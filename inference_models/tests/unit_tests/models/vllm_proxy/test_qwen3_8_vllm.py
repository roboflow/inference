from unittest.mock import MagicMock

import numpy as np
import pytest

from inference_models.configuration import (
    INFERENCE_MODELS_QWEN3_8_DEFAULT_MAX_NEW_TOKENS,
)
from inference_models.models.vllm_proxy import qwen3_8_vllm as qwen3_8_vllm_module
from inference_models.models.vllm_proxy.adapter_manager import (
    SUPPORTED_MODEL_ARCHITECTURES,
)
from inference_models.models.vllm_proxy.qwen3_5_vllm import (
    IMAGE_PATCH_FACTOR,
    MAX_PIXELS,
    MIN_PIXELS,
    Qwen35VLLMProxy,
)
from inference_models.models.vllm_proxy.qwen3_8_vllm import Qwen38VLLMProxy
from inference_models.weights_providers.entities import ModelMetadata


class _FakeAdapterManager:
    def __init__(self, served_name: str = "qwen3_8-27b"):
        self.client = MagicMock()
        self.served_name = served_name
        self.resolve_calls = []

    def resolve_and_register(self, model_id, **kwargs):
        self.resolve_calls.append({"model_id": model_id, **kwargs})
        return self.served_name

    def invalidate(self, served_name):
        pass

    def get_registration(self, served_name):
        return None


def build_metadata(model_id: str = "qwen3_8-27b") -> ModelMetadata:
    return ModelMetadata(
        model_id=model_id,
        model_architecture="qwen3_8",
        model_packages=[],
        task_type="vlm",
        model_variant="qwen3_8-27b",
    )


@pytest.fixture
def fake_manager(monkeypatch) -> _FakeAdapterManager:
    manager = _FakeAdapterManager()
    monkeypatch.setattr(qwen3_8_vllm_module, "get_adapter_manager", lambda: manager)
    return manager


@pytest.fixture
def model(fake_manager) -> Qwen38VLLMProxy:
    return Qwen38VLLMProxy.from_model_metadata(
        model_id="qwen3_8-27b", metadata=build_metadata(), api_key="some-key"
    )


class TestQwen38ProxyConfiguration:
    def test_pixel_budget_matches_qwen3_5(self) -> None:
        assert Qwen38VLLMProxy.image_patch_factor == IMAGE_PATCH_FACTOR
        assert Qwen38VLLMProxy.min_pixels == MIN_PIXELS
        assert Qwen38VLLMProxy.max_pixels == MAX_PIXELS
        assert (
            Qwen38VLLMProxy.default_system_prompt
            == Qwen35VLLMProxy.default_system_prompt
        )

    def test_generation_defaults_are_family_specific(self) -> None:
        assert (
            Qwen38VLLMProxy.default_max_new_tokens
            == INFERENCE_MODELS_QWEN3_8_DEFAULT_MAX_NEW_TOKENS
        )
        assert Qwen38VLLMProxy.supports_thinking is True

    def test_construction_resolves_served_name(
        self, model: Qwen38VLLMProxy, fake_manager: _FakeAdapterManager
    ) -> None:
        assert len(fake_manager.resolve_calls) == 1
        assert fake_manager.resolve_calls[0]["model_id"] == "qwen3_8-27b"
        assert model._served_name == "qwen3_8-27b"

    def test_architecture_is_not_adapter_servable(self) -> None:
        assert not any(
            "qwen3_8".startswith(architecture)
            for architecture in SUPPORTED_MODEL_ARCHITECTURES
        )


class TestPrompt:
    def test_default_max_tokens_and_thinking_forwarded(
        self, model: Qwen38VLLMProxy, fake_manager: _FakeAdapterManager
    ) -> None:
        fake_manager.client.chat_completion.return_value = {
            "choices": [{"message": {"content": "reasoning</think>final answer"}}]
        }

        result = model.prompt(
            np.zeros((64, 64, 3), dtype=np.uint8), enable_thinking=True
        )

        assert result == [{"thinking": "reasoning", "answer": "final answer"}]
        _, kwargs = fake_manager.client.chat_completion.call_args
        assert kwargs["model"] == "qwen3_8-27b"
        assert kwargs["max_tokens"] == INFERENCE_MODELS_QWEN3_8_DEFAULT_MAX_NEW_TOKENS
        assert kwargs["chat_template_kwargs"] == {"enable_thinking": True}
        assert kwargs["temperature"] == 0


class TestPostProcessText:
    def test_plain_text_is_cleaned(self, model: Qwen38VLLMProxy) -> None:
        assert (
            model.post_process_text("an answer<|im_end|>", enable_thinking=False)
            == "an answer"
        )

    def test_thinking_output_is_parsed(self, model: Qwen38VLLMProxy) -> None:
        result = model.post_process_text(
            "reasoning</think>final answer",
            enable_thinking=True,
        )
        assert result == {"thinking": "reasoning", "answer": "final answer"}
