import base64
import sys
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
import torch

from inference_models.models.vllm_proxy import qwen3_5_vllm as qwen3_5_vllm_module
from inference_models.models.vllm_proxy import qwen_vllm_base as qwen_vllm_base_module
from inference_models.models.vllm_proxy.errors import (
    VLLMConnectionError,
    VLLMHTTPError,
)
from inference_models.models.vllm_proxy.qwen3_5_vllm import (
    MIN_PIXELS,
    Qwen35VLLMProxy,
    post_process_generated_text,
    smart_resize_dimensions,
    split_prompt_and_system_prompt,
)
from inference_models.weights_providers.entities import ModelMetadata


class _FakeAdapterManager:
    def __init__(self, served_name: str = "qwen3_5-0.8b"):
        self.client = MagicMock()
        self.served_name = served_name
        self.resolve_calls = []
        self.invalidate_calls = []

    def resolve_and_register(self, model_id, **kwargs):
        self.resolve_calls.append({"model_id": model_id, **kwargs})
        return self.served_name

    def invalidate(self, served_name):
        self.invalidate_calls.append(served_name)

    def get_registration(self, served_name):
        return None


def build_metadata(model_id: str = "qwen3_5-0.8b") -> ModelMetadata:
    return ModelMetadata(
        model_id=model_id,
        model_architecture="qwen3_5",
        model_packages=[],
        task_type="vlm",
        model_variant="qwen3_5-0.8b",
    )


@pytest.fixture
def fake_manager(monkeypatch) -> _FakeAdapterManager:
    manager = _FakeAdapterManager()
    monkeypatch.setattr(qwen3_5_vllm_module, "get_adapter_manager", lambda: manager)
    return manager


@pytest.fixture
def model(fake_manager) -> Qwen35VLLMProxy:
    return Qwen35VLLMProxy.from_model_metadata(
        model_id="qwen3_5-0.8b",
        metadata=build_metadata(),
        api_key="some-key",
        weights_provider_extra_headers={"X-Extra": "1"},
        device=torch.device("cpu"),
        onnx_execution_providers=None,
    )


def _single_choice(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


def _decode_sent_image(call_kwargs: dict) -> np.ndarray:
    data_uri = call_kwargs["messages"][1]["content"][0]["image_url"]["url"]
    png_bytes = base64.b64decode(data_uri.split(",", 1)[1])
    return cv2.imdecode(np.frombuffer(png_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)


class TestSplitPromptAndSystemPrompt:
    def test_none_prompt_uses_defaults(self) -> None:
        assert split_prompt_and_system_prompt(None) == (
            "Describe what's in this image.",
            "You are a helpful assistant.",
        )

    def test_plain_prompt_uses_default_system_prompt(self) -> None:
        assert split_prompt_and_system_prompt("what is this?") == (
            "what is this?",
            "You are a helpful assistant.",
        )

    def test_prompt_with_system_prompt_marker_is_split(self) -> None:
        assert split_prompt_and_system_prompt(
            "what is this?<system_prompt>You are a vision model."
        ) == ("what is this?", "You are a vision model.")

    def test_empty_segments_fall_back_to_defaults(self) -> None:
        assert split_prompt_and_system_prompt("<system_prompt>") == (
            "Describe what's in this image.",
            "You are a helpful assistant.",
        )


class TestMessages:
    def test_messages_structure(
        self, model: Qwen35VLLMProxy, fake_manager: _FakeAdapterManager
    ) -> None:
        fake_manager.client.chat_completion.return_value = _single_choice("x")
        image = np.zeros((64, 48, 3), dtype=np.uint8)

        model.prompt(image, prompt="what?<system_prompt>You are X.")

        _, kwargs = fake_manager.client.chat_completion.call_args
        messages = kwargs["messages"]
        assert messages[0] == {
            "role": "system",
            "content": [{"type": "text", "text": "You are X."}],
        }
        assert messages[1]["role"] == "user"
        image_part, text_part = messages[1]["content"]
        assert image_part["type"] == "image_url"
        assert image_part["image_url"]["url"].startswith("data:image/png;base64,")
        assert text_part == {"type": "text", "text": "what?"}

    def test_image_is_resized_to_processor_pixel_budget(
        self, model: Qwen35VLLMProxy, fake_manager: _FakeAdapterManager
    ) -> None:
        fake_manager.client.chat_completion.return_value = _single_choice("x")
        image = np.zeros((64, 48, 3), dtype=np.uint8)

        model.prompt(image, prompt="caption")

        _, kwargs = fake_manager.client.chat_completion.call_args
        height, width = _decode_sent_image(kwargs).shape[:2]
        assert (height, width) == smart_resize_dimensions(height=64, width=48)
        assert height % 32 == 0 and width % 32 == 0
        assert height * width >= MIN_PIXELS

    def test_bgr_array_is_encoded_without_channel_swap(
        self, model: Qwen35VLLMProxy, fake_manager: _FakeAdapterManager
    ) -> None:
        fake_manager.client.chat_completion.return_value = _single_choice("x")
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[:, :, 0] = 255

        model.prompt(image, prompt="caption")

        _, kwargs = fake_manager.client.chat_completion.call_args
        assert tuple(_decode_sent_image(kwargs)[0, 0]) == (255, 0, 0)

    def test_tensor_is_treated_as_rgb_chw_like_the_hf_class(
        self, model: Qwen35VLLMProxy, fake_manager: _FakeAdapterManager
    ) -> None:
        fake_manager.client.chat_completion.return_value = _single_choice("x")
        image = torch.zeros((3, 64, 64), dtype=torch.uint8)
        image[0] = 255

        model.prompt(image, prompt="caption")

        _, kwargs = fake_manager.client.chat_completion.call_args
        assert tuple(_decode_sent_image(kwargs)[0, 0]) == (0, 0, 255)

    def test_list_of_images_is_served_one_completion_per_image_in_order(
        self, model: Qwen35VLLMProxy, fake_manager: _FakeAdapterManager
    ) -> None:
        fake_manager.client.chat_completion.side_effect = [
            _single_choice("first"),
            _single_choice("second"),
        ]
        images = [
            np.zeros((64, 64, 3), dtype=np.uint8),
            np.full((64, 64, 3), 255, dtype=np.uint8),
        ]

        result = model.prompt(images, prompt="caption")

        assert result == ["first", "second"]
        assert fake_manager.client.chat_completion.call_count == 2
        first_call, second_call = fake_manager.client.chat_completion.call_args_list
        assert int(_decode_sent_image(first_call.kwargs)[0, 0, 0]) == 0
        assert int(_decode_sent_image(second_call.kwargs)[0, 0, 0]) == 255


class TestPrompt:
    def test_chat_completion_parameters(
        self, model: Qwen35VLLMProxy, fake_manager: _FakeAdapterManager
    ) -> None:
        fake_manager.client.chat_completion.return_value = _single_choice("a cat")
        image = np.zeros((64, 64, 3), dtype=np.uint8)

        result = model.prompt(
            image,
            prompt="caption",
            max_new_tokens=64,
            do_sample=True,
            skip_special_tokens=False,
            enable_thinking=False,
        )

        assert result == ["a cat"]
        _, kwargs = fake_manager.client.chat_completion.call_args
        assert kwargs["model"] == "qwen3_5-0.8b"
        assert kwargs["temperature"] == 0
        assert kwargs["max_tokens"] == 64
        assert kwargs["chat_template_kwargs"] == {"enable_thinking": False}
        assert set(kwargs) == {
            "model",
            "messages",
            "temperature",
            "max_tokens",
            "chat_template_kwargs",
        }

    def test_default_max_tokens_applied(
        self, model: Qwen35VLLMProxy, fake_manager: _FakeAdapterManager
    ) -> None:
        fake_manager.client.chat_completion.return_value = _single_choice("x")

        model.prompt(np.zeros((64, 64, 3), dtype=np.uint8), enable_thinking=True)

        _, kwargs = fake_manager.client.chat_completion.call_args
        assert kwargs["max_tokens"] == 512
        assert kwargs["chat_template_kwargs"] == {"enable_thinking": True}

    def test_empty_content_yields_empty_string(
        self, model: Qwen35VLLMProxy, fake_manager: _FakeAdapterManager
    ) -> None:
        fake_manager.client.chat_completion.return_value = _single_choice(None)

        result = model.prompt(np.zeros((64, 64, 3), dtype=np.uint8))

        assert result == [""]


def _unknown_model_error(served_name: str = "qwen3_5-0.8b") -> VLLMHTTPError:
    """The 404 vLLM's OpenAI server returns for an unknown served model."""
    return VLLMHTTPError(
        message="vLLM sidecar returned HTTP 404 for POST /v1/chat/completions",
        status_code=404,
        response_body=(
            '{"object":"error","message":"The model `'
            + served_name
            + '` does not exist.","type":"NotFoundError","code":404}'
        ),
    )


class TestSelfHealOnUnknownLora:
    """On the unknown-model 404 naming our adapter, prompt must invalidate +
    re-register + retry exactly once."""

    def test_unknown_model_error_triggers_reregister_and_single_retry(
        self,
        monkeypatch,
        model: Qwen35VLLMProxy,
        fake_manager: _FakeAdapterManager,
    ) -> None:
        logger_mock = MagicMock()
        monkeypatch.setattr(qwen_vllm_base_module, "LOGGER", logger_mock)
        fake_manager.client.chat_completion.side_effect = [
            _unknown_model_error(),
            _single_choice("a cat"),
        ]

        result = model.prompt(np.zeros((64, 64, 3), dtype=np.uint8))

        assert result == ["a cat"]
        assert fake_manager.invalidate_calls == ["qwen3_5-0.8b"]
        assert len(fake_manager.resolve_calls) == 2
        heal_call = fake_manager.resolve_calls[1]
        assert heal_call["model_id"] == "qwen3_5-0.8b"
        assert heal_call["api_key"] == "some-key"
        assert heal_call["weights_provider_extra_headers"] == {"X-Extra": "1"}
        assert heal_call.get("metadata") is None
        assert fake_manager.client.chat_completion.call_count == 2
        logger_mock.warning.assert_called_once()

    def test_second_consecutive_unknown_model_error_propagates(
        self, model: Qwen35VLLMProxy, fake_manager: _FakeAdapterManager
    ) -> None:
        fake_manager.client.chat_completion.side_effect = [
            _unknown_model_error(),
            _unknown_model_error(),
        ]

        with pytest.raises(VLLMHTTPError):
            model.prompt(np.zeros((64, 64, 3), dtype=np.uint8))
        assert fake_manager.client.chat_completion.call_count == 2
        assert fake_manager.invalidate_calls == ["qwen3_5-0.8b"]
        assert len(fake_manager.resolve_calls) == 2

    @pytest.mark.parametrize(
        "error",
        [
            VLLMHTTPError(
                message="HTTP 404",
                status_code=404,
                response_body="The model `other-adapter` does not exist.",
            ),
            VLLMHTTPError(
                message="HTTP 404",
                status_code=404,
                response_body="qwen3_5-0.8b: route unavailable",
            ),
            VLLMHTTPError(
                message="HTTP 500",
                status_code=500,
                response_body="The model `qwen3_5-0.8b` does not exist.",
            ),
            VLLMHTTPError(message="HTTP 404", status_code=404, response_body=None),
            VLLMHTTPError(
                message="HTTP 503",
                status_code=503,
                response_body="engine overloaded",
            ),
        ],
    )
    def test_other_http_errors_do_not_retry(
        self,
        model: Qwen35VLLMProxy,
        fake_manager: _FakeAdapterManager,
        error: VLLMHTTPError,
    ) -> None:
        fake_manager.client.chat_completion.side_effect = error

        with pytest.raises(VLLMHTTPError):
            model.prompt(np.zeros((64, 64, 3), dtype=np.uint8))
        assert fake_manager.client.chat_completion.call_count == 1
        assert fake_manager.invalidate_calls == []
        assert len(fake_manager.resolve_calls) == 1

    def test_connection_error_does_not_retry(
        self, model: Qwen35VLLMProxy, fake_manager: _FakeAdapterManager
    ) -> None:
        fake_manager.client.chat_completion.side_effect = VLLMConnectionError(
            "Could not reach vLLM sidecar"
        )

        with pytest.raises(VLLMConnectionError):
            model.prompt(np.zeros((64, 64, 3), dtype=np.uint8))
        assert fake_manager.client.chat_completion.call_count == 1
        assert fake_manager.invalidate_calls == []


class TestPostprocess:
    def test_response_shape_matches_hf_path(
        self, model: Qwen35VLLMProxy, fake_manager: _FakeAdapterManager
    ) -> None:
        fake_manager.client.chat_completion.return_value = _single_choice(
            "a cat<|im_end|>"
        )

        result = model.prompt(np.zeros((64, 64, 3), dtype=np.uint8))

        assert result == ["a cat"]

    def test_thinking_response_shape(
        self, model: Qwen35VLLMProxy, fake_manager: _FakeAdapterManager
    ) -> None:
        fake_manager.client.chat_completion.return_value = _single_choice(
            "reasoning here</think>the answer"
        )

        result = model.prompt(
            np.zeros((64, 64, 3), dtype=np.uint8), enable_thinking=True
        )

        assert result == [{"thinking": "reasoning here", "answer": "the answer"}]


class TestThinkTagParityWithHF:
    """Compares post_process_generated_text against Qwen35HF.post_process_generation."""

    @pytest.fixture
    def hf_post_process(self):
        try:
            transformers_module = sys.modules.get("transformers")
            if transformers_module is None:
                import transformers as transformers_module
            if not hasattr(transformers_module, "Qwen3_5ForConditionalGeneration"):
                transformers_module.Qwen3_5ForConditionalGeneration = MagicMock()
            from inference_models.models.qwen3_5.qwen3_5_hf import Qwen35HF
        except ImportError:
            pytest.skip("inference_models qwen3_5 HF implementation not importable")

        def _run(text: str, enable_thinking: bool):
            processor = MagicMock()
            processor.batch_decode.return_value = [text]
            hf_model = Qwen35HF(
                model=None,
                processor=processor,
                inference_config=None,
                device=torch.device("cpu"),
            )
            return hf_model.post_process_generation(
                generated_ids=None,
                skip_special_tokens=True,
                enable_thinking=enable_thinking,
            )[0]

        return _run

    @pytest.mark.parametrize(
        "text",
        [
            "a plain answer",
            "a cat<|im_end|>",
            "assistant\nthe answer<|endoftext|>",
            "<think>internal</think>final answer",
            "answer with addCriterion\n artifact",
        ],
    )
    def test_parity_without_thinking(self, hf_post_process, text: str) -> None:
        assert post_process_generated_text(
            text, enable_thinking=False
        ) == hf_post_process(text, enable_thinking=False)

    @pytest.mark.parametrize(
        "text",
        [
            "step one\nstep two</think>The answer is 42.",
            "thinking forever and ever",
            "</think>only answer",
            "thinking<|im_end|></think>answer<|endoftext|>",
        ],
    )
    def test_parity_with_thinking(self, hf_post_process, text: str) -> None:
        assert post_process_generated_text(
            text, enable_thinking=True
        ) == hf_post_process(text, enable_thinking=True)


class TestInit:
    def test_from_model_metadata_registers_with_passed_metadata_and_api_key(
        self, fake_manager: _FakeAdapterManager
    ) -> None:
        metadata = build_metadata(model_id="ws/proj/1")

        model = Qwen35VLLMProxy.from_model_metadata(
            model_id="ws/proj/1", metadata=metadata, api_key="secret-key"
        )

        assert isinstance(model, Qwen35VLLMProxy)
        assert len(fake_manager.resolve_calls) == 1
        assert fake_manager.resolve_calls[0]["model_id"] == "ws/proj/1"
        assert fake_manager.resolve_calls[0]["metadata"] is metadata
        assert fake_manager.resolve_calls[0]["api_key"] == "secret-key"
        assert model.default_system_prompt == "You are a helpful assistant."
        assert model._inference_config is None

    def test_proxy_class_does_not_inherit_from_hf_class(self) -> None:
        assert all("HF" not in base.__name__ for base in Qwen35VLLMProxy.__mro__)
