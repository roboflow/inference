import asyncio
from unittest.mock import AsyncMock, MagicMock

from inference.core import roboflow_api
from inference.core.interfaces.legacy_stream import inference_pipeline
from inference.core.interfaces.legacy_stream.host import LegacyPipelineHost
from inference.core.utils import mjpeg
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
