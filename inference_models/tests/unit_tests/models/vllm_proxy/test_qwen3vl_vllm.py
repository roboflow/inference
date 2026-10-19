import base64
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
import torch

from inference_models.models.vllm_proxy import qwen3vl_vllm as qwen3vl_vllm_module
from inference_models.models.vllm_proxy.qwen3vl_vllm import (
    IMAGE_PATCH_FACTOR,
    MAX_PIXELS,
    MIN_PIXELS,
    Qwen3VLVLLMProxy,
    post_process_generated_text,
    smart_resize_dimensions,
    split_prompt_and_system_prompt,
)
from inference_models.weights_providers.entities import ModelMetadata

QWEN3VL_DEFAULT_SYSTEM_PROMPT = (
    "You are a Qwen3-VL a helpful assistant for any visual task."
)


class _FakeAdapterManager:
    def __init__(self, served_name: str = "qwen3vl-2b-instruct"):
        self.client = MagicMock()
        self.served_name = served_name
        self.resolve_calls = []

    def resolve_and_register(self, model_id, **kwargs):
        self.resolve_calls.append({"model_id": model_id, **kwargs})
        return self.served_name

    def get_registration(self, served_name):
        return None


def build_metadata(model_id: str = "qwen3vl-2b-instruct") -> ModelMetadata:
    return ModelMetadata(
        model_id=model_id,
        model_architecture="qwen3vl",
        model_packages=[],
        task_type="vlm",
        model_variant="2b-instruct",
    )


@pytest.fixture
def fake_manager(monkeypatch) -> _FakeAdapterManager:
    manager = _FakeAdapterManager()
    monkeypatch.setattr(qwen3vl_vllm_module, "get_adapter_manager", lambda: manager)
    return manager


@pytest.fixture
def model(fake_manager) -> Qwen3VLVLLMProxy:
    return Qwen3VLVLLMProxy.from_model_metadata(
        model_id="qwen3vl-2b-instruct", metadata=build_metadata(), api_key="some-key"
    )


def _single_choice(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


def _decode_sent_image(call_kwargs: dict) -> np.ndarray:
    data_uri = call_kwargs["messages"][1]["content"][0]["image_url"]["url"]
    png_bytes = base64.b64decode(data_uri.split(",", 1)[1])
    return cv2.imdecode(np.frombuffer(png_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)


class TestFamilyConstants:
    def test_pixel_budget_mirrors_qwen3vl_hf_processor(self) -> None:
        assert MIN_PIXELS == 256 * 28 * 28
        assert MAX_PIXELS == 1280 * 28 * 28
        assert IMAGE_PATCH_FACTOR == 32


class TestSplitPromptAndSystemPrompt:
    def test_none_prompt_uses_qwen3vl_defaults(self) -> None:
        assert split_prompt_and_system_prompt(None) == (
            "Describe what's in this image.",
            QWEN3VL_DEFAULT_SYSTEM_PROMPT,
        )

    def test_plain_prompt_uses_default_system_prompt(self) -> None:
        assert split_prompt_and_system_prompt("what is this?") == (
            "what is this?",
            QWEN3VL_DEFAULT_SYSTEM_PROMPT,
        )

    def test_prompt_with_system_prompt_marker_is_split(self) -> None:
        assert split_prompt_and_system_prompt(
            "what is this?<system_prompt>You are a vision model."
        ) == ("what is this?", "You are a vision model.")

    def test_empty_segments_fall_back_to_defaults(self) -> None:
        assert split_prompt_and_system_prompt("<system_prompt>") == (
            "Describe what's in this image.",
            QWEN3VL_DEFAULT_SYSTEM_PROMPT,
        )


class TestMessages:
    def test_messages_structure(
        self, model: Qwen3VLVLLMProxy, fake_manager: _FakeAdapterManager
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

    def test_default_system_prompt_is_qwen3vl_specific(
        self, model: Qwen3VLVLLMProxy, fake_manager: _FakeAdapterManager
    ) -> None:
        fake_manager.client.chat_completion.return_value = _single_choice("x")

        model.prompt(np.zeros((64, 48, 3), dtype=np.uint8), prompt="caption")

        _, kwargs = fake_manager.client.chat_completion.call_args
        assert (
            kwargs["messages"][0]["content"][0]["text"] == QWEN3VL_DEFAULT_SYSTEM_PROMPT
        )
        assert model.default_system_prompt == QWEN3VL_DEFAULT_SYSTEM_PROMPT

    def test_image_is_resized_to_qwen3vl_pixel_budget(
        self, model: Qwen3VLVLLMProxy, fake_manager: _FakeAdapterManager
    ) -> None:
        fake_manager.client.chat_completion.return_value = _single_choice("x")
        image = np.zeros((64, 48, 3), dtype=np.uint8)

        model.prompt(image, prompt="caption")

        _, kwargs = fake_manager.client.chat_completion.call_args
        height, width = _decode_sent_image(kwargs).shape[:2]
        assert (height, width) == smart_resize_dimensions(height=64, width=48)
        assert height % IMAGE_PATCH_FACTOR == 0 and width % IMAGE_PATCH_FACTOR == 0
        assert MIN_PIXELS <= height * width <= MAX_PIXELS

    def test_list_of_images_is_served_one_completion_per_image_in_order(
        self, model: Qwen3VLVLLMProxy, fake_manager: _FakeAdapterManager
    ) -> None:
        fake_manager.client.chat_completion.side_effect = [
            _single_choice("first"),
            _single_choice("second"),
        ]
        images = [
            np.zeros((64, 64, 3), dtype=np.uint8),
            torch.zeros((3, 64, 64), dtype=torch.uint8),
        ]

        result = model.prompt(images, prompt="caption")

        assert result == ["first", "second"]
        assert fake_manager.client.chat_completion.call_count == 2


class TestPrompt:
    def test_chat_completion_parameters(
        self, model: Qwen3VLVLLMProxy, fake_manager: _FakeAdapterManager
    ) -> None:
        fake_manager.client.chat_completion.return_value = _single_choice("a cat")

        result = model.prompt(
            np.zeros((64, 64, 3), dtype=np.uint8), max_new_tokens=64, do_sample=True
        )

        assert result == ["a cat"]
        _, kwargs = fake_manager.client.chat_completion.call_args
        assert kwargs["model"] == "qwen3vl-2b-instruct"
        assert kwargs["temperature"] == 0
        assert kwargs["max_tokens"] == 64
        assert kwargs["chat_template_kwargs"] is None

    def test_enable_thinking_is_never_forwarded(
        self, model: Qwen3VLVLLMProxy, fake_manager: _FakeAdapterManager
    ) -> None:
        fake_manager.client.chat_completion.return_value = _single_choice("x")

        model.prompt(np.zeros((64, 64, 3), dtype=np.uint8), enable_thinking=True)

        _, kwargs = fake_manager.client.chat_completion.call_args
        assert kwargs["chat_template_kwargs"] is None

    def test_default_max_tokens_applied(
        self, model: Qwen3VLVLLMProxy, fake_manager: _FakeAdapterManager
    ) -> None:
        fake_manager.client.chat_completion.return_value = _single_choice("x")

        model.prompt(np.zeros((64, 64, 3), dtype=np.uint8))

        _, kwargs = fake_manager.client.chat_completion.call_args
        assert kwargs["max_tokens"] == 512


class TestPostprocess:
    def test_response_shape_matches_hf_path(
        self, model: Qwen3VLVLLMProxy, fake_manager: _FakeAdapterManager
    ) -> None:
        fake_manager.client.chat_completion.return_value = _single_choice(
            "a cat<|im_end|>"
        )

        result = model.prompt(np.zeros((64, 64, 3), dtype=np.uint8))

        assert result == ["a cat"]

    def test_no_think_tag_is_prepended_or_parsed(
        self, model: Qwen3VLVLLMProxy, fake_manager: _FakeAdapterManager
    ) -> None:
        fake_manager.client.chat_completion.return_value = _single_choice(
            "reasoning here</think>the answer"
        )

        result = model.prompt(
            np.zeros((64, 64, 3), dtype=np.uint8), enable_thinking=True
        )

        assert result == ["reasoning here</think>the answer"]

    def test_think_blocks_are_left_verbatim(
        self, model: Qwen3VLVLLMProxy, fake_manager: _FakeAdapterManager
    ) -> None:
        fake_manager.client.chat_completion.return_value = _single_choice(
            "<think>internal</think>final answer"
        )

        result = model.prompt(np.zeros((64, 64, 3), dtype=np.uint8))

        assert result == ["<think>internal</think>final answer"]


class TestPostprocessParityWithHF:
    """Compares post_process_generated_text against Qwen3VLHF.post_process_generation."""

    @pytest.fixture
    def hf_post_process(self):
        try:
            from inference_models.models.qwen3vl.qwen3vl_hf import Qwen3VLHF
        except ImportError:
            pytest.skip("inference_models qwen3vl HF implementation not importable")

        def _run(text: str):
            processor = MagicMock()
            processor.batch_decode.return_value = [text]
            hf_model = Qwen3VLHF(
                model=None,
                processor=processor,
                inference_config=None,
                device=torch.device("cpu"),
            )
            return hf_model.post_process_generation(
                generated_ids=None,
                skip_special_tokens=True,
            )[0]

        return _run

    @pytest.mark.parametrize(
        "text",
        [
            "a plain answer",
            "assistant\nthe answer",
            "answer with addCriterion\n artifact",
            "<think>internal</think>final answer",
            "reasoning</think>answer",
        ],
    )
    def test_parity(self, hf_post_process, text: str) -> None:
        assert post_process_generated_text(text) == hf_post_process(text)

    def test_special_tokens_are_stripped_like_skip_special_tokens_decode(
        self,
    ) -> None:
        assert post_process_generated_text("a cat<|im_end|><|endoftext|>") == "a cat"


class TestInit:
    def test_from_model_metadata_registers_with_passed_metadata_and_api_key(
        self, fake_manager: _FakeAdapterManager
    ) -> None:
        metadata = build_metadata(model_id="ws/proj/1")

        model = Qwen3VLVLLMProxy.from_model_metadata(
            model_id="ws/proj/1", metadata=metadata, api_key="secret-key"
        )

        assert isinstance(model, Qwen3VLVLLMProxy)
        assert len(fake_manager.resolve_calls) == 1
        assert fake_manager.resolve_calls[0]["model_id"] == "ws/proj/1"
        assert fake_manager.resolve_calls[0]["metadata"] is metadata
        assert fake_manager.resolve_calls[0]["api_key"] == "secret-key"

    def test_prompt_signature_has_no_enable_thinking_like_the_hf_class(self) -> None:
        import inspect

        assert (
            "enable_thinking"
            not in inspect.signature(Qwen3VLVLLMProxy.prompt).parameters
        )
