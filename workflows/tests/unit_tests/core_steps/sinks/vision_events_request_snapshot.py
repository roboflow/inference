"""Record the HTTP requests of a no-video Vision Events run, deterministically.

Shared by `test_roboflow_vision_events_video_clips.py` and the one-off script
that produced `assets/vision_events_no_video_requests.json` from `origin/main`
(before video clips existed). It must only use the sink API that both versions
share, so the recorded requests stay comparable.

Regenerate (only when a request change is intended) from a checkout of the
reference commit's `workflows/` directory, with this file copied anywhere:

    PYTHONPATH=<reference>/workflows python -c "import runpy; \\
        runpy.run_path('vision_events_request_snapshot.py', run_name='__main__')" \\
        > assets/vision_events_no_video_requests.json
"""

import base64
import json
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest import mock

import requests_mock

API_BASE_URL = "https://api.roboflow.test"
EVENTS_URL = f"{API_BASE_URL}/vision-events"
# The requests library version is not part of the sink's contract.
IGNORED_HEADERS = {"user-agent"}
RUN_PARAMETERS: Dict[str, Any] = {
    "input_image": None,
    "output_image": None,
    "predictions": None,
    "event_type": "custom",
    "solution": "my-use-case",
    "custom_metadata": {"line": "a", "shift": 2, "ok": True},
    "fire_and_forget": False,
    "disable_sink": False,
    "cooldown_seconds": 0,
    "custom_value": "clip",
}


class _FixedDatetime(datetime):
    @classmethod
    def now(cls, tz: Any = None) -> datetime:  # type: ignore[override]
        fixed = datetime(2026, 10, 30, 12, 0, 0, tzinfo=timezone.utc)
        return fixed if tz is not None else fixed.replace(tzinfo=None)


def _platform_client() -> mock.MagicMock:
    client = mock.MagicMock()
    client.build_api_headers.side_effect = lambda explicit_headers=None: dict(
        explicit_headers or {}
    )
    client.wrap_url.side_effect = lambda url: url
    return client


def record_no_video_requests(module: Any) -> List[Dict[str, Any]]:
    """Run the sink of `module` once without video fields and record its requests.

    Args:
        module: A Vision Events sink module (`v1` or `v1_tensor`).

    Returns:
        One dict per request: `method`, `url`, sorted `headers` (without
        `IGNORED_HEADERS`) and the base64 `body`.
    """
    with requests_mock.Mocker() as platform, mock.patch.object(
        module, "uuid4", return_value="event-1"
    ), mock.patch.object(module, "datetime", _FixedDatetime), mock.patch.object(
        module, "API_BASE_URL", API_BASE_URL
    ):
        platform.post(EVENTS_URL, json={"success": True}, status_code=201)
        block = module.RoboflowVisionEventsBlockV1(
            api_key="my-api-key",
            background_tasks=None,
            thread_pool_executor=None,
            platform_client=_platform_client(),
        )
        result = block.run(**RUN_PARAMETERS)
        assert result["error_status"] is False, result
        recorded = [
            {
                "method": request.method,
                "url": request.url,
                "headers": sorted(
                    [key, value]
                    for key, value in request.headers.items()
                    if key.lower() not in IGNORED_HEADERS
                ),
                "body": base64.b64encode(_body_bytes(request.body)).decode("ascii"),
            }
            for request in platform.request_history
        ]

    return recorded


def _body_bytes(body: Any) -> bytes:
    if body is None:
        return b""
    if isinstance(body, str):
        return body.encode("utf-8")
    return bytes(body)


if __name__ == "__main__":
    from roboflow_workflows.core_steps.sinks.roboflow.vision_events import v1, v1_tensor

    snapshot = {
        "v1": record_no_video_requests(v1),
        "v1_tensor": record_no_video_requests(v1_tensor),
    }
    json.dump(snapshot, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
