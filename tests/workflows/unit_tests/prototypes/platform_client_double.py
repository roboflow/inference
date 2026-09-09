"""One `RoboflowPlatformClient` test double for every Phase 9 test.

Records what the block asked for so a migrated test can assert on `posts`,
`headers_calls`, `weights_calls` and `wrapped` instead of patching module-level
functions that no longer exist.

`post` is backed by a `unittest.mock.Mock` (`post_mock`) so a test migrated
from `mock.patch(... "post_to_roboflow_api")` keeps its `mock_post.*`
assertions verbatim - `return_value`, `side_effect` sequences (an error
followed by success), `call_count`, `call_args_list[i].kwargs["payload"]` all
work unchanged (round-3 defect 3).
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest import mock


class RecordingPlatformClient:
    def __init__(
        self,
        post_response: Optional[dict] = None,
        wrap_prefix: str = "",
        weights_headers: Optional[Dict[str, str]] = None,
    ):
        self.post_mock = mock.Mock(
            name="platform_client.post",
            return_value=post_response if post_response is not None else {},
        )
        self.posts: List[Dict[str, Any]] = []
        self.headers_calls: List[Optional[dict]] = []
        self.weights_calls: List[Tuple[Optional[bool], Optional[str]]] = []
        self.wrapped: List[str] = []
        self._wrap_prefix = wrap_prefix
        self._weights_headers = (
            weights_headers if weights_headers is not None else {"X-Test-Weights": "1"}
        )

    def post(
        self,
        endpoint: str,
        api_key: Optional[str],
        payload: Optional[dict] = None,
        params: Optional[List[Tuple[str, str]]] = None,
        http_errors_handlers: Optional[Dict[int, Callable[[Exception], None]]] = None,
    ) -> dict:
        self.posts.append(
            {
                "endpoint": endpoint,
                "api_key": api_key,
                "payload": payload,
                "params": params,
                "http_errors_handlers": http_errors_handlers,
            }
        )
        return self.post_mock(
            endpoint=endpoint,
            api_key=api_key,
            payload=payload,
            params=params,
            http_errors_handlers=http_errors_handlers,
        )

    def build_api_headers(self, explicit_headers: Optional[dict] = None) -> dict:
        self.headers_calls.append(explicit_headers)
        return {"X-Test": "1", **(explicit_headers or {})}

    def build_weights_provider_headers(
        self,
        countinference: Optional[bool] = None,
        service_secret: Optional[str] = None,
    ) -> Optional[dict]:
        self.weights_calls.append((countinference, service_secret))
        return self._weights_headers

    def wrap_url(self, url: str) -> str:
        self.wrapped.append(url)
        return self._wrap_prefix + url

    def reset(self) -> None:
        """Called by the autouse fixture the test-side codemod installs."""
        self.posts.clear()
        self.headers_calls.clear()
        self.weights_calls.clear()
        self.wrapped.clear()
        self.post_mock.reset_mock(return_value=True, side_effect=True)

    def set_post_response(self, value: Any) -> None:
        """An exception instance is raised by `post`; anything else is returned."""
        if isinstance(value, BaseException):
            self.post_mock.side_effect = value
        else:
            self.post_mock.return_value = value
