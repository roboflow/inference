import json
from unittest.mock import MagicMock

import pytest
import requests

from inference_models import configuration
from inference_models.errors import BaseInferenceModelsError, ModelRuntimeError
from inference_models.models.vllm_proxy import vllm_client as vllm_client_module
from inference_models.models.vllm_proxy.errors import (
    VLLMConnectionError,
    VLLMHTTPError,
    VLLMProxyError,
)
from inference_models.models.vllm_proxy.vllm_client import (
    VLLMClient,
    build_image_content_part,
    set_request_id_provider,
)


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


@pytest.fixture
def no_request_id_provider():
    set_request_id_provider(None)
    yield
    set_request_id_provider(None)


def test_build_image_content_part_produces_base64_data_uri() -> None:
    part = build_image_content_part(image_base64="QUJD")

    assert part == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,QUJD"},
    }


def test_error_hierarchy_is_rooted_in_inference_models_errors() -> None:
    assert issubclass(VLLMProxyError, ModelRuntimeError)
    assert issubclass(VLLMConnectionError, VLLMProxyError)
    assert issubclass(VLLMHTTPError, VLLMProxyError)
    assert issubclass(VLLMProxyError, BaseInferenceModelsError)


def test_client_defaults_come_from_configuration(monkeypatch) -> None:
    monkeypatch.setattr(configuration, "VLLM_BASE_URL", "http://sidecar:9000/")
    monkeypatch.setattr(configuration, "VLLM_REQUEST_TIMEOUT_S", 7.5)

    client = VLLMClient()

    assert client.base_url == "http://sidecar:9000"
    assert client._request_timeout_s == 7.5


class TestChatCompletion:
    def test_payload_and_response(self, client: VLLMClient) -> None:
        expected = {"choices": [{"message": {"content": "hello"}}]}
        client._session.request.return_value = make_response(payload=expected)
        messages = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]

        result = client.chat_completion(
            model="qwen3_5-0.8b",
            messages=messages,
            temperature=0,
            max_tokens=128,
            chat_template_kwargs={"enable_thinking": True},
        )

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
        client._session.request.return_value = make_response(payload={"choices": []})

        client.chat_completion(model="m", messages=[])

        _, kwargs = client._session.request.call_args
        assert "chat_template_kwargs" not in kwargs["json"]
        assert "max_tokens" not in kwargs["json"]

    def test_http_error_raises_typed_error(self, client: VLLMClient) -> None:
        client._session.request.return_value = make_response(
            status_code=500, text="boom"
        )

        with pytest.raises(VLLMHTTPError) as error:
            client.chat_completion(model="m", messages=[])
        assert error.value.status_code == 500
        assert error.value.response_body == "boom"

    def test_connection_error_raises_typed_error(self, client: VLLMClient) -> None:
        client._session.request.side_effect = requests.exceptions.ConnectionError()

        with pytest.raises(VLLMConnectionError):
            client.chat_completion(model="m", messages=[])

    def test_timeout_raises_typed_error(self, client: VLLMClient) -> None:
        client._session.request.side_effect = requests.exceptions.Timeout()

        with pytest.raises(VLLMConnectionError):
            client.chat_completion(model="m", messages=[])


class TestLoraAdapterEndpoints:
    def test_load_lora_adapter_posts_name_and_path(self, client: VLLMClient) -> None:
        client._session.request.return_value = make_response()

        client.load_lora_adapter(name="adapter-1", path="/cache/adapter-1")

        args, kwargs = client._session.request.call_args
        assert args == ("POST", "http://vllm-test:8000/v1/load_lora_adapter")
        assert kwargs["json"] == {
            "lora_name": "adapter-1",
            "lora_path": "/cache/adapter-1",
        }

    def test_load_lora_adapter_is_idempotent_on_already_loaded(
        self, client: VLLMClient
    ) -> None:
        client._session.request.return_value = make_response(
            status_code=400,
            text="The lora adapter 'adapter-1' has already been loaded.",
        )

        client.load_lora_adapter(name="adapter-1", path="/cache/adapter-1")

    def test_load_lora_adapter_other_400_raises(self, client: VLLMClient) -> None:
        client._session.request.return_value = make_response(
            status_code=400, text="invalid adapter"
        )

        with pytest.raises(VLLMHTTPError):
            client.load_lora_adapter(name="adapter-1", path="/cache/adapter-1")

    def test_unload_lora_adapter_posts_name(self, client: VLLMClient) -> None:
        client._session.request.return_value = make_response()

        client.unload_lora_adapter(name="adapter-1")

        args, kwargs = client._session.request.call_args
        assert args == ("POST", "http://vllm-test:8000/v1/unload_lora_adapter")
        assert kwargs["json"] == {"lora_name": "adapter-1"}


class TestCorrelationHeaderPassthrough:
    """X-Request-Id passthrough so vLLM logs correlate with platform ids."""

    def test_request_id_from_provider_is_sent_on_chat_completion(
        self, client: VLLMClient, no_request_id_provider
    ) -> None:
        set_request_id_provider(lambda: "exec-123")
        client._session.request.return_value = make_response(payload={"choices": []})

        client.chat_completion(model="m", messages=[])

        _, kwargs = client._session.request.call_args
        assert kwargs["headers"]["X-Request-Id"] == "exec-123"

    def test_request_id_is_sent_on_load_and_unload(
        self, client: VLLMClient, no_request_id_provider
    ) -> None:
        set_request_id_provider(lambda: "exec-123")
        client._session.request.return_value = make_response()

        client.load_lora_adapter(name="adapter-1", path="/cache/adapter-1")
        client.unload_lora_adapter(name="adapter-1")

        for call in client._session.request.call_args_list:
            assert call.kwargs["headers"]["X-Request-Id"] == "exec-123"

    def test_provider_returning_none_sends_no_header(
        self, client: VLLMClient, no_request_id_provider
    ) -> None:
        set_request_id_provider(lambda: None)
        client._session.request.return_value = make_response(payload={"choices": []})

        client.chat_completion(model="m", messages=[])

        _, kwargs = client._session.request.call_args
        assert "X-Request-Id" not in kwargs.get("headers", {})

    def test_failing_provider_sends_no_header(
        self, client: VLLMClient, no_request_id_provider
    ) -> None:
        def _broken_provider() -> str:
            raise LookupError("no request context")

        set_request_id_provider(_broken_provider)
        client._session.request.return_value = make_response(payload={"choices": []})

        client.chat_completion(model="m", messages=[])

        _, kwargs = client._session.request.call_args
        assert "X-Request-Id" not in kwargs.get("headers", {})

    def test_no_header_is_sent_without_provider(
        self, client: VLLMClient, no_request_id_provider
    ) -> None:
        client._session.request.return_value = make_response(payload={"choices": []})

        client.chat_completion(model="m", messages=[])

        _, kwargs = client._session.request.call_args
        assert "X-Request-Id" not in kwargs.get("headers", {})
        assert vllm_client_module.get_request_id() is None


class TestModelsAndHealth:
    def test_list_models_returns_data(self, client: VLLMClient) -> None:
        client._session.request.return_value = make_response(
            payload={"object": "list", "data": [{"id": "qwen3_5-0.8b"}]}
        )

        models = client.list_models()

        assert models == [{"id": "qwen3_5-0.8b"}]

    def test_health_true_on_200(self, client: VLLMClient) -> None:
        client._session.get.return_value = make_response(status_code=200)

        assert client.health() is True

    def test_health_false_on_connection_error(self, client: VLLMClient) -> None:
        client._session.get.side_effect = requests.exceptions.ConnectionError()

        assert client.health() is False
