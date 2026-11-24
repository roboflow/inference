"""In-process fakes for remote chat tests; nothing leaves the process.

``FakeChatServer`` answers OpenAI-style HTTP requests through an
``httpx.MockTransport``; blocks receive ``server.http_client()`` as their
http client resource. ``FakePlatformClient`` stands in for the host's
Roboflow platform client. Both record every request.
"""

import json
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx

Reply = Callable[[str, Dict[str, Any]], Tuple[int, Dict[str, Any]]]
"""``(path, request JSON) -> (HTTP status, response JSON)``."""


@dataclass(frozen=True)
class RecordedRequest:
    """One request the fake received."""

    url: str
    path: str
    headers: Dict[str, str]
    body: Dict[str, Any]


class FakeChatServer:
    """Answers requests with ``reply``; records them in arrival order."""

    def __init__(self, reply: Reply) -> None:
        self._reply = reply
        self._lock = threading.Lock()
        self.requests: List[RecordedRequest] = []

    def http_client(self) -> httpx.Client:
        client = httpx.Client(transport=httpx.MockTransport(self._handle))

        return client

    def _handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b"{}")
        recorded = RecordedRequest(
            url=str(request.url),
            path=request.url.path,
            headers=dict(request.headers),
            body=body,
        )
        with self._lock:
            self.requests.append(recorded)
        status, payload = self._reply(request.url.path, body)
        response = httpx.Response(status, json=payload)

        return response


class FakePlatformClient:
    """A host platform client of the V1 shape; ``reply`` answers ``post``."""

    def __init__(self, reply: Callable[[str, Dict[str, Any]], Dict[str, Any]]) -> None:
        self._reply = reply
        self._lock = threading.Lock()
        self.posts: List[Dict[str, Any]] = []

    def post(
        self,
        endpoint: str,
        api_key: Optional[str],
        payload: Optional[dict] = None,
        params: Optional[list] = None,
        http_errors_handlers: Optional[dict] = None,
    ) -> dict:
        with self._lock:
            self.posts.append(
                {
                    "endpoint": endpoint,
                    "api_key": api_key,
                    "payload": payload,
                    "http_errors_handlers": http_errors_handlers,
                }
            )
        data = self._reply(endpoint, payload)

        return data

    def build_api_headers(self, explicit_headers=None):
        return dict(explicit_headers or {})

    def build_weights_provider_headers(self, countinference=None, service_secret=None):
        return None

    def wrap_url(self, url: str) -> str:
        return url


def responses_body(
    text: str,
    *,
    status: str = "completed",
    input_tokens: Optional[int] = 11,
    output_tokens: Optional[int] = 7,
) -> Dict[str, Any]:
    """A Responses API result with one message holding ``text``."""
    body: Dict[str, Any] = {
        "id": "resp_fake",
        "object": "response",
        "created_at": 0,
        "model": "fake",
        "status": status,
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "output": [
            {
                "id": "msg_fake",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
    }
    if input_tokens is not None:
        body["usage"] = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + (output_tokens or 0),
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        }

    return body


def chat_completion_body(
    content: Optional[str],
    *,
    reasoning: Optional[str] = None,
    finish_reason: str = "stop",
    prompt_tokens: Optional[int] = 13,
    completion_tokens: Optional[int] = 5,
) -> Dict[str, Any]:
    """A chat-completions result with one choice."""
    message: Dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning"] = reasoning
    body: Dict[str, Any] = {
        "id": "chatcmpl_fake",
        "object": "chat.completion",
        "created": 0,
        "model": "fake",
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
    }
    if prompt_tokens is not None:
        body["usage"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + (completion_tokens or 0),
        }

    return body


def error_body(message: str) -> Dict[str, Any]:
    """An OpenAI-style error body."""
    body = {
        "error": {"message": message, "type": "invalid_request_error", "code": None}
    }

    return body
