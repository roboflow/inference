import base64
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
import torch

from inference.core.entities.responses import LMMInferenceResponse
from inference.models.vllm_proxy import qwen3vl_vllm as qwen3vl_vllm_module
from inference.models.vllm_proxy.qwen3vl_vllm import (
    IMAGE_PATCH_FACTOR,
    MAX_PIXELS,
    MIN_PIXELS,
    Qwen3VLVLLMProxy,
    post_process_generated_text,
    smart_resize_dimensions,
    split_prompt_and_system_prompt,
)

QWEN3VL_DEFAULT_SYSTEM_PROMPT = (
    "You are a Qwen3-VL a helpful assistant for any visual task."
)


class _FakeAdapterManager:
    def __init__(self, served_name: str = "qwen3vl-2b-instruct"):
        self.client = MagicMock()
        self.served_name = served_name
        self.resolve_calls = []

    def resolve_and_register(self, **kwargs):
        self.resolve_calls.append(kwargs)
        return self.served_name

    def get_registration(self, served_name):
        return None


@pytest.fixture
def fake_manager(monkeypatch) -> _FakeAdapterManager:
    manager = _FakeAdapterManager()
    monkeypatch.setattr(qwen3vl_vllm_module, "get_adapter_manager", lambda: manager)
    return manager


@pytest.fixture
def model(fake_manager) -> Qwen3VLVLLMProxy:
    return Qwen3VLVLLMProxy(model_id="qwen3vl-2b-instruct", api_key="some-key")


class TestFamilyConstants:
    def test_pixel_budget_mirrors_qwen3vl_hf_processor(self) -> None:
        # Mirrors AutoProcessor kwargs in Qwen3VLHF.from_pretrained.
        assert MIN_PIXELS == 256 * 28 * 28
        assert MAX_PIXELS == 1280 * 28 * 28
        # patch_size 16 * merge_size 2 from the Qwen3-VL checkpoint's
        # preprocessor config.
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


class TestPreprocess:
    def test_batched_input_is_rejected(self, model: Qwen3VLVLLMProxy) -> None:
        with pytest.raises(ValueError):
            model.preprocess([np.zeros((64, 64, 3), dtype=np.uint8)])

    def test_messages_structure_and_metadata(self, model: Qwen3VLVLLMProxy) -> None:
        # given
        image = np.zeros((64, 48, 3), dtype=np.uint8)

        # when
        messages, metadata = model.preprocess(
            image, prompt="what?<system_prompt>You are X."
        )

        # then
        assert metadata["image_dims"] == (48, 64)
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
        self, model: Qwen3VLVLLMProxy
    ) -> None:
        # when
        messages, _ = model.preprocess(
            np.zeros((64, 48, 3), dtype=np.uint8), prompt="caption"
        )

        # then
        assert messages[0]["content"][0]["text"] == QWEN3VL_DEFAULT_SYSTEM_PROMPT

    def test_image_is_resized_to_qwen3vl_pixel_budget(
        self, model: Qwen3VLVLLMProxy
    ) -> None:
        # given - 64x48 is far below the qwen3vl MIN_PIXELS (256*28*28), so
        # the image must be upscaled exactly as the HF processor's smart
        # resize would.
        image = np.zeros((64, 48, 3), dtype=np.uint8)

        # when
        messages, _ = model.preprocess(image, prompt="caption")

        # then
        data_uri = messages[1]["content"][0]["image_url"]["url"]
        png_bytes = base64.b64decode(data_uri.split(",", 1)[1])
        decoded = cv2.imdecode(
            np.frombuffer(png_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        height, width = decoded.shape[:2]
        assert (height, width) == smart_resize_dimensions(height=64, width=48)
        assert height % IMAGE_PATCH_FACTOR == 0 and width % IMAGE_PATCH_FACTOR == 0
        assert MIN_PIXELS <= height * width <= MAX_PIXELS


class TestPredict:
    def test_chat_completion_parameters(
        self, model: Qwen3VLVLLMProxy, fake_manager: _FakeAdapterManager
    ) -> None:
        # given
        fake_manager.client.chat_completion.return_value = {
            "choices": [{"message": {"content": "a cat"}}]
        }
        messages = [{"role": "user", "content": []}]

        # when
        result = model.predict(messages, max_new_tokens=64)

        # then
        assert result == "a cat"
        fake_manager.client.chat_completion.assert_called_once_with(
            model="qwen3vl-2b-instruct",
            messages=messages,
            temperature=0,
            max_tokens=64,
            chat_template_kwargs=None,
        )

    def test_enable_thinking_is_never_forwarded(
        self, model: Qwen3VLVLLMProxy, fake_manager: _FakeAdapterManager
    ) -> None:
        # given - qwen3vl-instruct has no thinking mode; even an explicit
        # enable_thinking kwarg must not reach the chat template.
        fake_manager.client.chat_completion.return_value = {
            "choices": [{"message": {"content": "x"}}]
        }

        # when
        model.predict([], enable_thinking=True)

        # then
        _, kwargs = fake_manager.client.chat_completion.call_args
        assert kwargs["chat_template_kwargs"] is None

    def test_default_max_tokens_applied(
        self, model: Qwen3VLVLLMProxy, fake_manager: _FakeAdapterManager
    ) -> None:
        # given
        fake_manager.client.chat_completion.return_value = {
            "choices": [{"message": {"content": "x"}}]
        }

        # when
        model.predict([])

        # then
        _, kwargs = fake_manager.client.chat_completion.call_args
        # INFERENCE_MODELS_QWEN3_VL_DEFAULT_MAX_NEW_TOKENS default
        assert kwargs["max_tokens"] == 512


class TestPostprocess:
    def test_response_shape_matches_hf_path(self, model: Qwen3VLVLLMProxy) -> None:
        # given
        metadata = {"image_dims": (48, 64)}

        # when
        responses = model.postprocess("a cat<|im_end|>", metadata)

        # then
        assert len(responses) == 1
        assert isinstance(responses[0], LMMInferenceResponse)
        assert responses[0].response == "a cat"
        assert responses[0].image.width == 48
        assert responses[0].image.height == 64

    def test_no_think_tag_is_prepended_or_parsed(
        self, model: Qwen3VLVLLMProxy
    ) -> None:
        # given - text that the qwen3_5 thinking parser would split; qwen3vl
        # has no thinking mode, so it must come back verbatim even when the
        # caller passes enable_thinking.
        metadata = {"image_dims": (48, 64)}

        # when
        responses = model.postprocess(
            "reasoning here</think>the answer",
            metadata,
            enable_thinking=True,
        )

        # then - a plain string, no {"thinking": ..., "answer": ...} dict and
        # no "<think>" prepended.
        assert responses[0].response == "reasoning here</think>the answer"

    def test_think_blocks_are_left_verbatim(self, model: Qwen3VLVLLMProxy) -> None:
        # given - the HF path performs no think-tag handling, so neither does
        # the proxy.
        metadata = {"image_dims": (48, 64)}

        # when
        responses = model.postprocess("<think>internal</think>final answer", metadata)

        # then
        assert responses[0].response == "<think>internal</think>final answer"


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
            # think tags pass through untouched on both paths
            "<think>internal</think>final answer",
            "reasoning</think>answer",
        ],
    )
    def test_parity(self, hf_post_process, text: str) -> None:
        assert post_process_generated_text(text) == hf_post_process(text)

    def test_special_tokens_are_stripped_like_skip_special_tokens_decode(
        self,
    ) -> None:
        # The HF path decodes with skip_special_tokens=True, so special
        # tokens never appear in its text; the proxy strips them defensively.
        assert post_process_generated_text("a cat<|im_end|><|endoftext|>") == "a cat"


class TestInit:
    def test_init_passes_api_key_through_to_resolution(
        self, fake_manager: _FakeAdapterManager
    ) -> None:
        # when
        model = Qwen3VLVLLMProxy(model_id="ws/proj/1", api_key="secret-key")

        # then
        assert model.task_type == "lmm"
        assert len(fake_manager.resolve_calls) == 1
        assert fake_manager.resolve_calls[0]["model_id"] == "ws/proj/1"
        assert fake_manager.resolve_calls[0]["api_key"] == "secret-key"
