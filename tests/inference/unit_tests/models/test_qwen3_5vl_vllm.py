from unittest import mock

import numpy as np
import pytest

from inference.core.entities.requests.inference import LMMInferenceRequest
from inference.models.qwen3_5vl import qwen3_5vl_inference_models as qwen_module
from inference.models.qwen3_5vl import vllm as qwen_vllm


class FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._payload


def test_qwen35_vllm_backend_sends_openai_compatible_request() -> None:
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    response_payload = {
        "choices": [
            {
                "message": {
                    "content": "The answer",
                    "reasoning_content": "The reasoning",
                }
            }
        ]
    }

    with mock.patch.object(qwen_module, "LMM_QWEN_BACKEND", "vllm"), mock.patch.object(
        qwen_module, "VLLM_MODE", "sidecar"
    ), mock.patch.object(
        qwen_module, "VLLM_BASE_URL", "http://vllm:8000/v1"
    ), mock.patch.object(
        qwen_module, "VLLM_API_KEY", "vllm-key"
    ), mock.patch.object(
        qwen_module, "VLLM_REQUEST_TIMEOUT", 42.0
    ), mock.patch.object(
        qwen_module.AutoModel, "from_pretrained"
    ) as from_pretrained_mock, mock.patch.object(
        qwen_module, "load_image_bgr", return_value=image
    ) as load_image_bgr_mock, mock.patch.object(
        qwen_vllm.requests,
        "post",
        return_value=FakeResponse(response_payload),
    ) as post_mock:
        adapter = qwen_module.InferenceModelsQwen35VLAdapter(
            model_id="vlm-ocr/11",
            api_key="rf-key",
        )
        response = adapter.infer_from_request(
            LMMInferenceRequest(
                model_id="vlm-ocr/11",
                image={"type": "base64", "value": "ignored"},
                prompt="What is in this image?<system_prompt>You are terse.",
                enable_thinking=True,
                max_new_tokens=25,
                id="request-id",
            )
        )

    from_pretrained_mock.assert_not_called()
    load_image_bgr_mock.assert_called_once()
    post_mock.assert_called_once()
    (endpoint,) = post_mock.call_args.args
    payload = post_mock.call_args.kwargs["json"]
    assert endpoint == "http://vllm:8000/v1/chat/completions"
    assert post_mock.call_args.kwargs["headers"] == {"Authorization": "Bearer vllm-key"}
    assert post_mock.call_args.kwargs["timeout"] == 42.0
    assert payload["model"] == "vlm-ocr/11"
    assert payload["max_tokens"] == 25
    assert payload["temperature"] == 0
    assert payload["chat_template_kwargs"] == {"enable_thinking": True}
    assert payload["messages"][0] == {
        "role": "system",
        "content": "You are terse.",
    }
    user_content = payload["messages"][1]["content"]
    assert user_content[0]["type"] == "image_url"
    assert user_content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert user_content[1] == {"type": "text", "text": "What is in this image?"}
    assert response.response == {
        "answer": "The answer",
        "thinking": "The reasoning",
    }
    assert response.image.width == 3
    assert response.image.height == 2
    assert response.inference_id == "request-id"


def test_resolve_qwen_backend() -> None:
    assert (
        qwen_module.resolve_qwen_backend(qwen_backend="native", vllm_mode="disabled")
        == "native"
    )
    assert (
        qwen_module.resolve_qwen_backend(qwen_backend="auto", vllm_mode="disabled")
        == "native"
    )
    assert (
        qwen_module.resolve_qwen_backend(qwen_backend="auto", vllm_mode="remote")
        == "vllm"
    )
    assert (
        qwen_module.resolve_qwen_backend(qwen_backend="vllm", vllm_mode="sidecar")
        == "vllm"
    )
    with pytest.raises(ValueError):
        qwen_module.resolve_qwen_backend(qwen_backend="vllm", vllm_mode="disabled")
    with pytest.raises(ValueError):
        qwen_module.resolve_qwen_backend(qwen_backend="unknown", vllm_mode="remote")


def test_parse_vllm_thinking_tags_when_reasoning_content_is_absent() -> None:
    result = qwen_vllm.parse_chat_completion_output(
        response_payload={
            "choices": [
                {"message": {"content": "I thought about it.</think>Final answer"}}
            ]
        },
        enable_thinking=True,
    )

    assert result == {"thinking": "I thought about it.", "answer": "Final answer"}
