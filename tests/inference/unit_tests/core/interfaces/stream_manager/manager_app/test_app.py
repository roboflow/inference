from unittest import mock
from unittest.mock import MagicMock

import pytest

from inference.core.interfaces.stream_manager.manager_app import app
from inference.core.interfaces.stream_manager.manager_app.app import (
    ManagedInferencePipeline,
    get_or_spawn_pipeline_process,
)


def _managed_pipeline(pipeline_id: str, is_idle: bool) -> ManagedInferencePipeline:
    managed_pipeline = ManagedInferencePipeline(
        pipeline_id=pipeline_id,
        pipeline_manager=MagicMock(),
        command_queue=MagicMock(),
        responses_queue=MagicMock(),
        operation_lock=MagicMock(),
        is_idle=is_idle,
    )
    # the RAM guard in get_or_spawn_pipeline_process cannot handle pipelines whose usage
    # has not been sampled yet, so keep the queue populated here
    managed_pipeline.ram_usage_queue.append(1)
    return managed_pipeline


@mock.patch.object(app, "STREAM_MANAGER_MAX_ACTIVE_PIPELINES", 2)
@mock.patch.object(app, "spawn_managed_pipeline_process")
def test_get_or_spawn_pipeline_process_spawns_when_below_limit(
    spawn_managed_pipeline_process_mock: MagicMock,
) -> None:
    # given
    processes_table = {"existing": _managed_pipeline("existing", is_idle=False)}

    def _spawn(processes_table, mark_as_idle):
        processes_table["new"] = _managed_pipeline("new", is_idle=mark_as_idle)
        return "new"

    spawn_managed_pipeline_process_mock.side_effect = _spawn

    # when
    result = get_or_spawn_pipeline_process(processes_table=processes_table)

    # then
    assert result.pipeline_id == "new"


@mock.patch.object(app, "STREAM_MANAGER_MAX_ACTIVE_PIPELINES", 2)
@mock.patch.object(app, "spawn_managed_pipeline_process")
def test_get_or_spawn_pipeline_process_refuses_to_spawn_above_limit(
    spawn_managed_pipeline_process_mock: MagicMock,
) -> None:
    # given
    processes_table = {
        "first": _managed_pipeline("first", is_idle=False),
        "second": _managed_pipeline("second", is_idle=False),
    }

    # when
    with pytest.raises(Exception):
        _ = get_or_spawn_pipeline_process(processes_table=processes_table)

    # then
    spawn_managed_pipeline_process_mock.assert_not_called()


@mock.patch.object(app, "STREAM_MANAGER_MAX_ACTIVE_PIPELINES", 2)
@mock.patch.object(app, "spawn_managed_pipeline_process")
def test_get_or_spawn_pipeline_process_reuses_idle_pipeline_at_limit(
    spawn_managed_pipeline_process_mock: MagicMock,
) -> None:
    # given
    processes_table = {
        "busy": _managed_pipeline("busy", is_idle=False),
        "idle": _managed_pipeline("idle", is_idle=True),
    }

    # when
    result = get_or_spawn_pipeline_process(processes_table=processes_table)

    # then
    assert result.pipeline_id == "idle"
    assert result.is_idle is False
    spawn_managed_pipeline_process_mock.assert_not_called()
