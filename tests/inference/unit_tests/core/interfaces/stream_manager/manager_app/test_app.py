import json
from typing import List, Optional
from unittest import mock
from unittest.mock import MagicMock

import pytest

from inference.core.interfaces.stream_manager.manager_app import app
from inference.core.interfaces.stream_manager.manager_app.app import (
    InferencePipelinesManagerHandler,
    ManagedInferencePipeline,
    ensure_idle_pipelines_warmed_up,
    get_or_spawn_pipeline_process,
)
from inference.core.interfaces.stream_manager.manager_app.host import (
    PipelineHostDescriptor,
)

# Never resolved: these tests replace the process spawn.
HOST_DESCRIPTOR = PipelineHostDescriptor(factory="some.host.module:create_host")


class _StopLoop(Exception):
    """Breaks out of the infinite warm-up loop after a single sweep."""


def _managed_pipeline(
    pipeline_id: str,
    is_idle: bool,
    ram_usage_samples: Optional[List[int]] = None,
) -> ManagedInferencePipeline:
    managed_pipeline = ManagedInferencePipeline(
        pipeline_id=pipeline_id,
        pipeline_manager=MagicMock(),
        command_queue=MagicMock(),
        responses_queue=MagicMock(),
        operation_lock=MagicMock(),
        is_idle=is_idle,
    )
    managed_pipeline.ram_usage_queue.extend(ram_usage_samples or [])
    return managed_pipeline


@mock.patch.object(app, "STREAM_MANAGER_MAX_ACTIVE_PIPELINES", 2)
@mock.patch.object(app, "spawn_managed_pipeline_process")
def test_get_or_spawn_pipeline_process_spawns_when_below_limit(
    spawn_managed_pipeline_process_mock: MagicMock,
) -> None:
    # given
    processes_table = {"existing": _managed_pipeline("existing", is_idle=False)}

    def _spawn(processes_table, mark_as_idle, host_descriptor):
        assert host_descriptor is HOST_DESCRIPTOR
        processes_table["new"] = _managed_pipeline("new", is_idle=mark_as_idle)
        return "new"

    spawn_managed_pipeline_process_mock.side_effect = _spawn

    # when
    result = get_or_spawn_pipeline_process(
        processes_table=processes_table, host_descriptor=HOST_DESCRIPTOR
    )

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
        _ = get_or_spawn_pipeline_process(
            processes_table=processes_table, host_descriptor=HOST_DESCRIPTOR
        )

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
    result = get_or_spawn_pipeline_process(
        processes_table=processes_table, host_descriptor=HOST_DESCRIPTOR
    )

    # then
    assert result.pipeline_id == "idle"
    assert result.is_idle is False
    spawn_managed_pipeline_process_mock.assert_not_called()


@mock.patch.object(app, "STREAM_MANAGER_MAX_ACTIVE_PIPELINES", 8)
@mock.patch.object(app, "STREAM_MANAGER_MAX_RAM_MB", None)
@mock.patch.object(app, "spawn_managed_pipeline_process")
def test_get_or_spawn_pipeline_process_when_ram_usage_not_sampled_yet(
    spawn_managed_pipeline_process_mock: MagicMock,
) -> None:
    # given - a pipeline spawned before check_process_health sampled its RAM usage
    processes_table = {
        "sampled": _managed_pipeline("sampled", is_idle=False, ram_usage_samples=[10]),
        "not_sampled": _managed_pipeline("not_sampled", is_idle=False),
    }

    def _spawn(processes_table, mark_as_idle, host_descriptor):
        assert host_descriptor is HOST_DESCRIPTOR
        processes_table["new"] = _managed_pipeline("new", is_idle=mark_as_idle)
        return "new"

    spawn_managed_pipeline_process_mock.side_effect = _spawn

    # when
    result = get_or_spawn_pipeline_process(
        processes_table=processes_table, host_descriptor=HOST_DESCRIPTOR
    )

    # then
    assert result.pipeline_id == "new"


@mock.patch.object(app, "STREAM_MANAGER_MAX_ACTIVE_PIPELINES", 8)
@mock.patch.object(app, "STREAM_MANAGER_MAX_RAM_MB", 100)
@mock.patch.object(app, "_get_current_process_ram_usage_mb", MagicMock(return_value=10))
@mock.patch.object(app, "spawn_managed_pipeline_process")
def test_get_or_spawn_pipeline_process_refuses_to_spawn_above_ram_limit(
    spawn_managed_pipeline_process_mock: MagicMock,
) -> None:
    # given - 10MB manager + 60MB last sample, peak 80MB predicted for the new pipeline
    processes_table = {
        "busy": _managed_pipeline("busy", is_idle=False, ram_usage_samples=[80, 60]),
    }

    # when
    with pytest.raises(Exception):
        _ = get_or_spawn_pipeline_process(
            processes_table=processes_table, host_descriptor=HOST_DESCRIPTOR
        )

    # then
    spawn_managed_pipeline_process_mock.assert_not_called()


@mock.patch.object(app, "STREAM_MANAGER_MAX_ACTIVE_PIPELINES", 2)
@mock.patch.object(app, "spawn_managed_pipeline_process")
@mock.patch.object(app, "time")
def test_ensure_idle_pipelines_warmed_up_spawns_below_limit(
    time_mock: MagicMock,
    spawn_managed_pipeline_process_mock: MagicMock,
) -> None:
    # given
    time_mock.sleep.side_effect = _StopLoop()
    processes_table = {"busy": _managed_pipeline("busy", is_idle=False)}

    # when
    with mock.patch.object(app, "PROCESSES_TABLE", processes_table):
        with pytest.raises(_StopLoop):
            ensure_idle_pipelines_warmed_up(
                expected_warmed_up_pipelines=1, host_descriptor=HOST_DESCRIPTOR
            )

    # then
    spawn_managed_pipeline_process_mock.assert_called_once_with(
        processes_table=processes_table, host_descriptor=HOST_DESCRIPTOR
    )


@mock.patch.object(app, "STREAM_MANAGER_MAX_ACTIVE_PIPELINES", 2)
@mock.patch.object(app, "spawn_managed_pipeline_process")
@mock.patch.object(app, "time")
def test_ensure_idle_pipelines_warmed_up_respects_active_pipelines_limit(
    time_mock: MagicMock,
    spawn_managed_pipeline_process_mock: MagicMock,
) -> None:
    # given - warm pool is short, but every slot is taken by a busy pipeline
    time_mock.sleep.side_effect = _StopLoop()
    processes_table = {
        "first": _managed_pipeline("first", is_idle=False),
        "second": _managed_pipeline("second", is_idle=False),
    }

    # when
    with mock.patch.object(app, "PROCESSES_TABLE", processes_table):
        with pytest.raises(_StopLoop):
            ensure_idle_pipelines_warmed_up(
                expected_warmed_up_pipelines=1, host_descriptor=HOST_DESCRIPTOR
            )

    # then
    spawn_managed_pipeline_process_mock.assert_not_called()


class DummySocket:
    def __init__(self):
        self._buffer = b""
        self._sent = b""

    def get_data_that_was_sent(self) -> bytes:
        return self._sent

    def fill(self, data: bytes) -> None:
        self._buffer = data

    def recv(self, __bufsize: int) -> bytes:
        chunk = self._buffer[:__bufsize]
        self._buffer = self._buffer[__bufsize:]
        return chunk

    def sendall(self, __data: bytes) -> None:
        self._sent += __data


def _handle_raw_request(payload: bytes) -> dict:
    socket = DummySocket()
    header = len(payload).to_bytes(length=4, byteorder="big")
    socket.fill(header + payload)
    _ = InferencePipelinesManagerHandler(
        request=socket,
        client_address=MagicMock(),
        server=MagicMock(),
        processes_table={},
        host_descriptor=HOST_DESCRIPTOR,
    )
    return json.loads(socket.get_data_that_was_sent()[4:].decode("utf-8"))


# Carried over from the removed enterprise stream manager's handler tests
# (WP-A05): the manager fails closed on requests it cannot trust.
@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "payload",
    [
        b"FOR SURE NOT A JSON",
        json.dumps({"invalid": "data"}).encode("utf-8"),
        json.dumps({"type": "unknown"}).encode("utf-8"),
    ],
)
def test_pipeline_manager_handler_rejects_invalid_requests(payload: bytes) -> None:
    # when
    response = _handle_raw_request(payload)

    # then
    assert (
        response["pipeline_id"] is None
    ), "Pipeline ID cannot be associated to this request"
    assert response["response"]["status"] == "failure", "Operation should failed"
    assert (
        response["response"]["error_type"] == "invalid_payload"
    ), "Wrong payload should be denoted as error cause"


@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "command_type, error_type",
    [
        ("status", "not_found"),
        ("mute", "not_found"),
        ("resume", "not_found"),
        ("consume_result", "not_found"),
        # Unlike the removed enterprise manager (not_found), termination looks
        # the pipeline up before dispatching the command; still a failure.
        ("terminate", "invalid_payload"),
    ],
)
def test_pipeline_manager_handler_when_command_requested_for_unknown_pipeline(
    command_type: str, error_type: str
) -> None:
    # when
    response = _handle_raw_request(
        json.dumps({"type": command_type, "pipeline_id": "unknown"}).encode("utf-8")
    )

    # then
    assert (
        response["pipeline_id"] == "unknown"
    ), "Pipeline ID must be assigned to request"
    assert response["response"]["status"] == "failure", "Operation should failed"
    assert response["response"]["error_type"] == error_type
