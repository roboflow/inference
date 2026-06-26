import base64
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
import torch

from inference.core.entities.responses import LMMInferenceResponse
from inference.models.vllm_proxy import qwen3_5_vllm as qwen3_5_vllm_module
from inference.models.vllm_proxy import qwen_vllm_base as qwen_vllm_base_module
from inference.models.vllm_proxy.errors import VLLMConnectionError, VLLMHTTPError
from inference.models.vllm_proxy.qwen3_5_vllm import (
    MIN_PIXELS,
    Qwen35VLLMProxy,
    post_process_generated_text,
    smart_resize_dimensions,
    split_prompt_and_system_prompt,
)


class _FakeAdapterManager:
    def __init__(self, served_name: str = "qwen3_5-0.8b"):
        self.client = MagicMock()
        self.served_name = served_name
        self.resolve_calls = []
        self.invalidate_calls = []

    def resolve_and_register(self, **kwargs):
        self.resolve_calls.append(kwargs)
        return self.served_name

    def invalidate(self, served_name):
        self.invalidate_calls.append(served_name)

    def get_registration(self, served_name):
        return None


@pytest.fixture
def fake_manager(monkeypatch) -> _FakeAdapterManager:
    manager = _FakeAdapterManager()
    monkeypatch.setattr(qwen3_5_vllm_module, "get_adapter_manager", lambda: manager)
    return manager


@pytest.fixture
def model(fake_manager) -> Qwen35VLLMProxy:
    return Qwen35VLLMProxy(model_id="qwen3_5-0.8b", api_key="some-key")


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


class TestPreprocess:
    def test_batched_input_is_rejected(self, model: Qwen35VLLMProxy) -> None:
        with pytest.raises(ValueError):
            model.preprocess([np.zeros((64, 64, 3), dtype=np.uint8)])

    def test_messages_structure_and_metadata(self, model: Qwen35VLLMProxy) -> None:
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

    def test_image_is_resized_to_processor_pixel_budget(
        self, model: Qwen35VLLMProxy
    ) -> None:
        # given - 64x48 is below MIN_PIXELS, so the image must be upscaled
        # exactly as the HF processor's smart resize would.
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
        assert height % 32 == 0 and width % 32 == 0
        assert height * width >= MIN_PIXELS


class TestPredict:
    def test_chat_completion_parameters(
        self, model: Qwen35VLLMProxy, fake_manager: _FakeAdapterManager
    ) -> None:
        # given
        fake_manager.client.chat_completion.return_value = {
            "choices": [{"message": {"content": "a cat"}}]
        }
        messages = [{"role": "user", "content": []}]

        # when
        result = model.predict(messages, max_new_tokens=64, enable_thinking=False)

        # then
        assert result == "a cat"
        fake_manager.client.chat_completion.assert_called_once_with(
            model="qwen3_5-0.8b",
            messages=messages,
            temperature=0,
            max_tokens=64,
            chat_template_kwargs={"enable_thinking": False},
        )

    def test_default_max_tokens_applied(
        self, model: Qwen35VLLMProxy, fake_manager: _FakeAdapterManager
    ) -> None:
        # given
        fake_manager.client.chat_completion.return_value = {
            "choices": [{"message": {"content": "x"}}]
        }

        # when
        model.predict([], enable_thinking=True)

        # then
        _, kwargs = fake_manager.client.chat_completion.call_args
        assert kwargs["max_tokens"] == 512
        assert kwargs["chat_template_kwargs"] == {"enable_thinking": True}


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
    """The shared vLLM engine may forget an adapter this worker's map still
    records (engine restart, NUM_WORKERS>1 desync). On the unknown-model
    404 naming our adapter, predict must invalidate + re-register + retry
    exactly once."""

    def test_unknown_model_error_triggers_reregister_and_single_retry(
        self,
        monkeypatch,
        model: Qwen35VLLMProxy,
        fake_manager: _FakeAdapterManager,
    ) -> None:
        # given
        logger_mock = MagicMock()
        monkeypatch.setattr(qwen_vllm_base_module, "logger", logger_mock)
        fake_manager.client.chat_completion.side_effect = [
            _unknown_model_error(),
            {"choices": [{"message": {"content": "a cat"}}]},
        ]

        # when
        result = model.predict([{"role": "user", "content": []}])

        # then - healed: invalidated, re-resolved, retried once, result back
        assert result == "a cat"
        assert fake_manager.invalidate_calls == ["qwen3_5-0.8b"]
        # one resolve at __init__ + one for the self-heal
        assert len(fake_manager.resolve_calls) == 2
        assert fake_manager.resolve_calls[1]["model_id"] == "qwen3_5-0.8b"
        assert fake_manager.resolve_calls[1]["api_key"] == "some-key"
        assert fake_manager.client.chat_completion.call_count == 2
        logger_mock.warning.assert_called_once()

    def test_second_consecutive_unknown_model_error_propagates(
        self, model: Qwen35VLLMProxy, fake_manager: _FakeAdapterManager
    ) -> None:
        # given - re-registration did not help: no retry loop
        fake_manager.client.chat_completion.side_effect = [
            _unknown_model_error(),
            _unknown_model_error(),
        ]

        # when / then
        with pytest.raises(VLLMHTTPError):
            model.predict([{"role": "user", "content": []}])
        assert fake_manager.client.chat_completion.call_count == 2
        assert fake_manager.invalidate_calls == ["qwen3_5-0.8b"]
        assert len(fake_manager.resolve_calls) == 2

    @pytest.mark.parametrize(
        "error",
        [
            # 404 about a DIFFERENT served model
            VLLMHTTPError(
                message="HTTP 404",
                status_code=404,
                response_body="The model `other-adapter` does not exist.",
            ),
            # 404 naming our model but not a not-found phrasing
            VLLMHTTPError(
                message="HTTP 404",
                status_code=404,
                response_body="qwen3_5-0.8b: route unavailable",
            ),
            # not-found phrasing but not a 404
            VLLMHTTPError(
                message="HTTP 500",
                status_code=500,
                response_body="The model `qwen3_5-0.8b` does not exist.",
            ),
            # empty body
            VLLMHTTPError(message="HTTP 404", status_code=404, response_body=None),
            # generic server error
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
        # given
        fake_manager.client.chat_completion.side_effect = error

        # when / then
        with pytest.raises(VLLMHTTPError):
            model.predict([{"role": "user", "content": []}])
        assert fake_manager.client.chat_completion.call_count == 1
        assert fake_manager.invalidate_calls == []
        assert len(fake_manager.resolve_calls) == 1  # __init__ only

    def test_connection_error_does_not_retry(
        self, model: Qwen35VLLMProxy, fake_manager: _FakeAdapterManager
    ) -> None:
        # given
        fake_manager.client.chat_completion.side_effect = VLLMConnectionError(
            "Could not reach vLLM sidecar"
        )

        # when / then
        with pytest.raises(VLLMConnectionError):
            model.predict([{"role": "user", "content": []}])
        assert fake_manager.client.chat_completion.call_count == 1
        assert fake_manager.invalidate_calls == []


class TestPostprocess:
    def test_response_shape_matches_hf_path(self, model: Qwen35VLLMProxy) -> None:
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

    def test_thinking_response_shape(self, model: Qwen35VLLMProxy) -> None:
        # given
        metadata = {"image_dims": (48, 64)}

        # when
        responses = model.postprocess(
            "reasoning here</think>the answer",
            metadata,
            enable_thinking=True,
        )

        # then
        assert responses[0].response == {
            "thinking": "reasoning here",
            "answer": "the answer",
        }


class TestThinkTagParityWithHF:
    """Compares post_process_generated_text against Qwen35HF.post_process_generation."""

    @pytest.fixture
    def hf_post_process(self):
        try:
            # The CPU test environment may ship a transformers version that
            # predates Qwen3_5ForConditionalGeneration. The class is only
            # needed by Qwen35HF.from_pretrained / generate - not by
            # post_process_generation - so a stub keeps the module importable
            # for the parity check.
            import sys

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
            # generated text starts mid-thinking: the chat template prefilled
            # "<think>\n", so the opening tag is absent from the output
            "step one\nstep two</think>The answer is 42.",
            # max-tokens hit before </think> was produced
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
    def test_init_passes_api_key_through_to_resolution(
        self, fake_manager: _FakeAdapterManager
    ) -> None:
        # when
        model = Qwen35VLLMProxy(model_id="ws/proj/1", api_key="secret-key")

        # then
        assert model.task_type == "lmm"
        assert len(fake_manager.resolve_calls) == 1
        assert fake_manager.resolve_calls[0]["model_id"] == "ws/proj/1"
        assert fake_manager.resolve_calls[0]["api_key"] == "secret-key"
