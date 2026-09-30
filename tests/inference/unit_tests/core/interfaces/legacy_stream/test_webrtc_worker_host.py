import asyncio
import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from inference.core import roboflow_api
from inference.core.interfaces.legacy_stream import inference_pipeline
from inference.core.interfaces.legacy_stream.host import LegacyPipelineHost
from inference.core.interfaces.webrtc_worker import utils as worker_utils
from inference.core.utils import mjpeg, url_utils
from inference.usage_tracking import collector


def test_init_workflow_pipeline_forwards_every_argument(monkeypatch) -> None:
    init = MagicMock(return_value="pipeline")
    monkeypatch.setattr(
        inference_pipeline.InferencePipeline, "init_with_workflow", init
    )

    result = LegacyPipelineHost().init_workflow_pipeline(
        video_reference="reference", model_manager="manager", _is_preview=True
    )

    assert result == "pipeline"
    init.assert_called_once_with(
        video_reference="reference", model_manager="manager", _is_preview=True
    )


def test_get_workflow_specification_forwards_to_the_platform_api(monkeypatch) -> None:
    fetch = MagicMock(return_value={"version": "1.0"})
    monkeypatch.setattr(roboflow_api, "get_workflow_specification", fetch)

    result = LegacyPipelineHost().get_workflow_specification(
        api_key="key",
        workspace_id="workspace",
        workflow_id="workflow",
        workflow_version_id="version",
    )

    assert result == {"version": "1.0"}
    fetch.assert_called_once_with(
        api_key="key",
        workspace_id="workspace",
        workflow_id="workflow",
        workflow_version_id="version",
    )


def test_async_push_usage_payloads_awaits_the_usage_collector(monkeypatch) -> None:
    push = AsyncMock()
    monkeypatch.setattr(collector.usage_collector, "async_push_usage_payloads", push)

    asyncio.run(LegacyPipelineHost().async_push_usage_payloads())

    push.assert_awaited_once_with()


def test_open_mjpeg_player_forwards_to_the_guarded_opener(monkeypatch) -> None:
    opener = MagicMock(return_value="player")
    monkeypatch.setattr(mjpeg, "open_mjpeg_player", opener)

    result = LegacyPipelineHost().open_mjpeg_player(
        "http://camera.example/stream", allow_non_global_addresses=True
    )

    assert result == "player"
    opener.assert_called_once_with(
        "http://camera.example/stream", allow_non_global_addresses=True
    )


def test_is_over_quota_forwards_to_the_plan_lookup(monkeypatch) -> None:
    lookup = MagicMock(return_value=True)
    monkeypatch.setattr(worker_utils, "is_over_quota", lookup)

    assert LegacyPipelineHost().is_over_quota("key") is True
    lookup.assert_called_once_with("key")


def test_wrap_url_forwards_to_the_gateway_wrapper(monkeypatch) -> None:
    wrapper = MagicMock(return_value="https://gateway.example/wrapped")
    monkeypatch.setattr(url_utils, "wrap_url", wrapper)

    result = LegacyPipelineHost().wrap_url("https://api.example/heartbeat")

    assert result == "https://gateway.example/wrapped"
    wrapper.assert_called_once_with("https://api.example/heartbeat")


@pytest.mark.parametrize("is_preview", [True, False])
@pytest.mark.parametrize("established, expected_duration", [(True, 90.0), (False, 0)])
def test_record_session_usage_builds_the_pinned_usage_record(
    monkeypatch, established, expected_duration, is_preview
) -> None:
    record = MagicMock()
    monkeypatch.setattr(collector.usage_collector, "record_usage", record)
    push = MagicMock()
    monkeypatch.setattr(collector.usage_collector, "push_usage_payloads", push)
    request = SimpleNamespace(
        api_key="test-key", requested_plan="webrtc-gpu-small", is_preview=is_preview
    )
    started = datetime.datetime(2026, 1, 1, 12, 0, 0)

    LegacyPipelineHost().record_session_usage(
        webrtc_request=request,
        workflow_id="hash-1",
        video_source="rtsp",
        session_started=started,
        session_stopped=started + datetime.timedelta(seconds=90),
        connection_established=established,
    )

    record.assert_called_once_with(
        source="hash-1",
        category="modal",
        api_key="test-key",
        resource_id="hash-1",
        resource_details={
            "plan": "webrtc-gpu-small",
            "billable": True,
            "video_source": "rtsp",
            "is_preview": is_preview,
        },
        execution_duration=expected_duration,
    )
    assert type(record.call_args.kwargs["execution_duration"]) is type(
        expected_duration
    )
    push.assert_not_called()


def test_push_usage_payloads_forwards_to_the_usage_collector(monkeypatch) -> None:
    push = MagicMock()
    monkeypatch.setattr(collector.usage_collector, "push_usage_payloads", push)

    LegacyPipelineHost().push_usage_payloads()

    push.assert_called_once_with()
