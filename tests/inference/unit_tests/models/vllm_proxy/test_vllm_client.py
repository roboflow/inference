import json
from unittest.mock import MagicMock

import pytest
import requests

from inference.models.vllm_proxy.errors import VLLMConnectionError, VLLMHTTPError
from inference.models.vllm_proxy.vllm_client import VLLMClient, build_image_content_part


def make_response(status_code: int = 200, payload: dict = None, text: str = ""):
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = payload if payload is not None else {}
    response.text = text or (json.dumps(payload) if payload is not None else "")
    return response


@pytest.fixture
def client() -> VLLMClient:
    client = VLLMClient(base_url="http://vllm-test:8000", request_timeout_s=5)
    client._session = MagicMock()
    return client


def test_build_image_content_part_produces_base64_data_uri() -> None:
    # when
    part = build_image_content_part(image_base64="QUJD")

    # then
    assert part == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,QUJD"},
    }


class TestChatCompletion:
    def test_payload_and_response(self, client: VLLMClient) -> None:
        # given
        expected = {"choices": [{"message": {"content": "hello"}}]}
        client._session.request.return_value = make_response(payload=expected)
        messages = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]

        # when
        result = client.chat_completion(
            model="qwen3_5-0.8b",
            messages=messages,
            temperature=0,
            max_tokens=128,
            chat_template_kwargs={"enable_thinking": True},
        )

        # then
        assert result == expected
        args, kwargs = client._session.request.call_args
        assert args == ("POST", "http://vllm-test:8000/v1/chat/completions")
        assert kwargs["timeout"] == 5
        assert kwargs["json"] == {
            "model": "qwen3_5-0.8b",
            "messages": messages,
            "temperature": 0,
            "max_tokens": 128,
            "chat_template_kwargs": {"enable_thinking": True},
        }

    def test_chat_template_kwargs_omitted_when_not_provided(
        self, client: VLLMClient
    ) -> None:
        # given
        client._session.request.return_value = make_response(payload={"choices": []})

        # when
        client.chat_completion(model="m", messages=[])

        # then
        _, kwargs = client._session.request.call_args
        assert "chat_template_kwargs" not in kwargs["json"]
        assert "max_tokens" not in kwargs["json"]

    def test_http_error_raises_typed_error(self, client: VLLMClient) -> None:
        # given
        client._session.request.return_value = make_response(
            status_code=500, text="boom"
        )

        # when / then
        with pytest.raises(VLLMHTTPError) as error:
            client.chat_completion(model="m", messages=[])
        assert error.value.status_code == 500
        assert error.value.response_body == "boom"

    def test_connection_error_raises_typed_error(self, client: VLLMClient) -> None:
        # given
        client._session.request.side_effect = requests.exceptions.ConnectionError()

        # when / then
        with pytest.raises(VLLMConnectionError):
            client.chat_completion(model="m", messages=[])

    def test_timeout_raises_typed_error(self, client: VLLMClient) -> None:
        # given
        client._session.request.side_effect = requests.exceptions.Timeout()

        # when / then
        with pytest.raises(VLLMConnectionError):
            client.chat_completion(model="m", messages=[])


class TestLoraAdapterEndpoints:
    def test_load_lora_adapter_posts_name_and_path(self, client: VLLMClient) -> None:
        # given
        client._session.request.return_value = make_response()

        # when
        client.load_lora_adapter(name="adapter-1", path="/cache/adapter-1")

        # then
        args, kwargs = client._session.request.call_args
        assert args == ("POST", "http://vllm-test:8000/v1/load_lora_adapter")
        assert kwargs["json"] == {
            "lora_name": "adapter-1",
            "lora_path": "/cache/adapter-1",
        }

    def test_load_lora_adapter_is_idempotent_on_already_loaded(
        self, client: VLLMClient
    ) -> None:
        # given
        client._session.request.return_value = make_response(
            status_code=400,
            text="The lora adapter 'adapter-1' has already been loaded.",
        )

        # when - must not raise
        client.load_lora_adapter(name="adapter-1", path="/cache/adapter-1")

    def test_load_lora_adapter_other_400_raises(self, client: VLLMClient) -> None:
        # given
        client._session.request.return_value = make_response(
            status_code=400, text="invalid adapter"
        )

        # when / then
        with pytest.raises(VLLMHTTPError):
            client.load_lora_adapter(name="adapter-1", path="/cache/adapter-1")

    def test_unload_lora_adapter_posts_name(self, client: VLLMClient) -> None:
        # given
        client._session.request.return_value = make_response()

        # when
        client.unload_lora_adapter(name="adapter-1")

        # then
        args, kwargs = client._session.request.call_args
        assert args == ("POST", "http://vllm-test:8000/v1/unload_lora_adapter")
        assert kwargs["json"] == {"lora_name": "adapter-1"}


class TestModelsAndHealth:
    def test_list_models_returns_data(self, client: VLLMClient) -> None:
        # given
        client._session.request.return_value = make_response(
            payload={"object": "list", "data": [{"id": "qwen3_5-0.8b"}]}
        )

        # when
        models = client.list_models()

        # then
        assert models == [{"id": "qwen3_5-0.8b"}]

    def test_health_true_on_200(self, client: VLLMClient) -> None:
        # given
        client._session.get.return_value = make_response(status_code=200)

        # when / then
        assert client.health() is True

    def test_health_false_on_connection_error(self, client: VLLMClient) -> None:
        # given
        client._session.get.side_effect = requests.exceptions.ConnectionError()

        # when / then
        assert client.health() is False
