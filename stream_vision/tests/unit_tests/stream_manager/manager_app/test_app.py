import json
import time
from functools import partial
from queue import Empty, Queue
from threading import Event, Lock, Thread, Timer
from typing import Callable, Dict, List, Optional, Tuple
from unittest import mock
from unittest.mock import MagicMock

import pytest
from streamvision.stream_manager.manager_app import app
from streamvision.stream_manager.manager_app.app import (
    InferencePipelinesManagerHandler,
    ManagedInferencePipeline,
    ensure_idle_pipelines_warmed_up,
    get_or_spawn_pipeline_process,
    get_response_ignoring_thrash,
    handle_command,
)
from streamvision.stream_manager.manager_app.entities import ErrorType, OperationStatus
from streamvision.stream_manager.manager_app.host import PipelineHostDescriptor

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


def _handle_raw_request(
    payload: bytes,
    processes_table: Optional[Dict[str, ManagedInferencePipeline]] = None,
) -> dict:
    socket = DummySocket()
    header = len(payload).to_bytes(length=4, byteorder="big")
    socket.fill(header + payload)
    _ = InferencePipelinesManagerHandler(
        request=socket,
        client_address=MagicMock(),
        server=MagicMock(),
        processes_table={} if processes_table is None else processes_table,
        host_descriptor=HOST_DESCRIPTOR,
    )
    return json.loads(socket.get_data_that_was_sent()[4:].decode("utf-8"))


# The manager fails closed on requests it cannot trust.
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
        # terminate looks up the pipeline before dispatching, so its error differs.
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


CALL_TIME_LIMIT_SECONDS = 5


def _start_in_daemon_thread(function: Callable[[], dict]) -> Tuple[Thread, List[dict]]:
    results = []
    thread = Thread(target=lambda: results.append(function()), daemon=True)
    thread.start()

    return thread, results


def _call_in_daemon_thread(function: Callable[[], dict]) -> Tuple[Thread, List[dict]]:
    thread, results = _start_in_daemon_thread(function)
    thread.join(timeout=CALL_TIME_LIMIT_SECONDS)

    return thread, results


def _pipeline_process(is_alive: bool) -> MagicMock:
    pipeline_process = MagicMock()
    pipeline_process.is_alive.return_value = is_alive
    return pipeline_process


@mock.patch.object(app, "RESPONSES_QUEUE_POLL_INTERVAL", 0.01)
def test_get_response_ignoring_thrash_when_process_dead_and_nothing_answered() -> None:
    # given
    responses_queue = Queue()

    # when
    thread, results = _call_in_daemon_thread(
        partial(
            get_response_ignoring_thrash,
            responses_queue=responses_queue,
            matching_request_id="request-id",
            pipeline_process=_pipeline_process(is_alive=False),
            pipeline_id="dead-pipeline",
        )
    )

    # then
    assert not thread.is_alive(), "Waiting for a dead process must not block"
    assert results[0]["status"] is OperationStatus.FAILURE
    assert results[0]["error_type"] is ErrorType.NOT_FOUND
    assert "dead-pipeline" in results[0]["public_error_message"]


@mock.patch.object(app, "RESPONSES_QUEUE_POLL_INTERVAL", 0.01)
def test_get_response_ignoring_thrash_when_process_dead_and_response_queued() -> None:
    # given
    responses_queue = Queue()
    responses_queue.put(("request-id", {"status": "success"}))

    # when
    thread, results = _call_in_daemon_thread(
        partial(
            get_response_ignoring_thrash,
            responses_queue=responses_queue,
            matching_request_id="request-id",
            pipeline_process=_pipeline_process(is_alive=False),
            pipeline_id="dead-pipeline",
        )
    )

    # then
    assert not thread.is_alive()
    assert results == [{"status": "success"}]


@mock.patch.object(app, "RESPONSES_QUEUE_POLL_INTERVAL", 0.01)
def test_get_response_ignoring_thrash_when_process_answers_right_before_dying() -> None:
    # given
    responses_queue = Queue()
    pipeline_process = MagicMock()

    def _answer_and_die() -> bool:
        responses_queue.put(("request-id", {"status": "success"}))
        return False

    pipeline_process.is_alive.side_effect = _answer_and_die

    # when
    thread, results = _call_in_daemon_thread(
        partial(
            get_response_ignoring_thrash,
            responses_queue=responses_queue,
            matching_request_id="request-id",
            pipeline_process=pipeline_process,
            pipeline_id="dead-pipeline",
        )
    )

    # then
    assert not thread.is_alive()
    assert results == [{"status": "success"}]


@mock.patch.object(app, "RESPONSES_QUEUE_POLL_INTERVAL", 0.01)
def test_get_response_ignoring_thrash_waits_while_process_alive() -> None:
    # given
    responses_queue = Queue()
    pipeline_process = _pipeline_process(is_alive=True)
    delayed_response = Timer(
        0.1, responses_queue.put, args=(("request-id", {"status": "success"}),)
    )

    # when
    delayed_response.start()
    thread, results = _call_in_daemon_thread(
        partial(
            get_response_ignoring_thrash,
            responses_queue=responses_queue,
            matching_request_id="request-id",
            pipeline_process=pipeline_process,
            pipeline_id="alive-pipeline",
        )
    )

    # then
    assert not thread.is_alive()
    assert results == [{"status": "success"}]
    pipeline_process.is_alive.assert_called()


@mock.patch.object(app, "RESPONSES_QUEUE_POLL_INTERVAL", 0.01)
def test_get_response_ignoring_thrash_drops_responses_for_other_requests() -> None:
    # given
    responses_queue = Queue()
    responses_queue.put(("other-request-id", {"status": "failure"}))
    responses_queue.put(("request-id", {"status": "success"}))

    # when
    thread, results = _call_in_daemon_thread(
        partial(
            get_response_ignoring_thrash,
            responses_queue=responses_queue,
            matching_request_id="request-id",
            pipeline_process=_pipeline_process(is_alive=True),
            pipeline_id="alive-pipeline",
        )
    )

    # then
    assert not thread.is_alive()
    assert results == [{"status": "success"}]


@mock.patch.object(app, "RESPONSES_QUEUE_POLL_INTERVAL", 0.01)
def test_handle_command_when_pipeline_process_dead() -> None:
    # given
    operation_lock = Lock()
    processes_table = {
        "dead-pipeline": ManagedInferencePipeline(
            pipeline_id="dead-pipeline",
            pipeline_manager=_pipeline_process(is_alive=False),
            command_queue=Queue(),
            responses_queue=Queue(),
            operation_lock=operation_lock,
            is_idle=False,
        )
    }

    # when
    thread, results = _call_in_daemon_thread(
        partial(
            handle_command,
            processes_table=processes_table,
            request_id="request-id",
            pipeline_id="dead-pipeline",
            command={"type": "status", "pipeline_id": "dead-pipeline"},
        )
    )

    # then
    assert not thread.is_alive(), "Command to a dead process must not block"
    assert results[0]["status"] is OperationStatus.FAILURE
    assert results[0]["error_type"] is ErrorType.NOT_FOUND
    assert "dead-pipeline" in results[0]["public_error_message"]
    assert operation_lock.acquire(blocking=False), "Operation lock must be released"


@pytest.mark.parametrize("command_type", ["init", "webrtc"])
@mock.patch.object(app, "RESPONSES_QUEUE_POLL_INTERVAL", 0.01)
@mock.patch.object(app, "get_or_spawn_pipeline_process")
def test_pipeline_manager_handler_when_spawned_pipeline_process_dead(
    get_or_spawn_pipeline_process_mock: MagicMock,
    command_type: str,
) -> None:
    # given
    get_or_spawn_pipeline_process_mock.return_value = ManagedInferencePipeline(
        pipeline_id="dead-pipeline",
        pipeline_manager=_pipeline_process(is_alive=False),
        command_queue=Queue(),
        responses_queue=Queue(),
        operation_lock=Lock(),
        is_idle=False,
    )

    # when
    thread, results = _call_in_daemon_thread(
        partial(_handle_raw_request, json.dumps({"type": command_type}).encode("utf-8"))
    )

    # then
    assert not thread.is_alive(), "Command to a dead process must not block"
    assert results[0]["pipeline_id"] == "dead-pipeline"
    assert results[0]["response"]["status"] == "failure"
    assert results[0]["response"]["error_type"] == "not_found"
    assert not get_or_spawn_pipeline_process_mock.return_value.operation_lock.locked()


def _answering_pipeline(pipeline_id: str, response: dict) -> ManagedInferencePipeline:
    responses_queue = Queue()
    command_queue = MagicMock()
    command_queue.put.side_effect = lambda request: responses_queue.put(
        (request[0], response)
    )
    return ManagedInferencePipeline(
        pipeline_id=pipeline_id,
        pipeline_manager=_pipeline_process(is_alive=True),
        command_queue=command_queue,
        responses_queue=responses_queue,
        operation_lock=Lock(),
        is_idle=False,
    )


@mock.patch.object(app, "RESPONSES_QUEUE_POLL_INTERVAL", 0.01)
def test_pipeline_manager_handler_when_termination_fails() -> None:
    # given
    processes_table = {
        "pipeline": _answering_pipeline("pipeline", {"status": OperationStatus.FAILURE})
    }

    # when
    thread, results = _call_in_daemon_thread(
        partial(
            _handle_raw_request,
            json.dumps({"type": "terminate", "pipeline_id": "pipeline"}).encode(
                "utf-8"
            ),
            processes_table=processes_table,
        )
    )

    # then
    assert not thread.is_alive()
    assert results[0]["response"]["status"] == "failure"
    assert "pipeline" in processes_table
    assert (
        processes_table["pipeline"].is_terminating is False
    ), "Health check must be able to reap a pipeline whose termination failed"


@mock.patch.object(app, "RESPONSES_QUEUE_POLL_INTERVAL", 0.01)
def test_pipeline_manager_handler_when_termination_succeeds() -> None:
    # given
    processes_table = {
        "pipeline": _answering_pipeline("pipeline", {"status": OperationStatus.SUCCESS})
    }

    # when
    thread, results = _call_in_daemon_thread(
        partial(
            _handle_raw_request,
            json.dumps({"type": "terminate", "pipeline_id": "pipeline"}).encode(
                "utf-8"
            ),
            processes_table=processes_table,
        )
    )

    # then
    assert not thread.is_alive()
    assert results[0]["response"]["status"] == "success"
    assert "pipeline" not in processes_table


@pytest.mark.parametrize("command_type", ["init", "webrtc"])
@mock.patch.object(app, "RESPONSES_QUEUE_POLL_INTERVAL", 0.01)
@mock.patch.object(app, "get_or_spawn_pipeline_process")
def test_pipeline_manager_handler_exchanges_command_under_operation_lock(
    get_or_spawn_pipeline_process_mock: MagicMock,
    command_type: str,
) -> None:
    # given
    managed_pipeline = _answering_pipeline(
        "pipeline", {"status": OperationStatus.SUCCESS}
    )
    answer = managed_pipeline.command_queue.put.side_effect
    lock_held_on_put = []

    def _record_lock_and_answer(request: tuple) -> None:
        lock_held_on_put.append(managed_pipeline.operation_lock.locked())
        answer(request)

    managed_pipeline.command_queue.put.side_effect = _record_lock_and_answer
    get_or_spawn_pipeline_process_mock.return_value = managed_pipeline

    # when
    thread, results = _call_in_daemon_thread(
        partial(_handle_raw_request, json.dumps({"type": command_type}).encode("utf-8"))
    )

    # then
    assert not thread.is_alive()
    assert results[0]["response"]["status"] == "success"
    assert lock_held_on_put == [True], "Command must be sent under the operation lock"
    assert not managed_pipeline.operation_lock.locked()


def _answer_commands_in_arrival_order(
    managed_pipeline: ManagedInferencePipeline,
    first_command_received: Event,
    stop: Event,
) -> None:
    while not stop.is_set():
        try:
            request_id, command = managed_pipeline.command_queue.get(timeout=0.01)
        except Empty:
            continue

        first_command_received.set()
        time.sleep(0.02)
        managed_pipeline.responses_queue.put(
            (request_id, {"status": OperationStatus.SUCCESS, "type": command["type"]})
        )


@mock.patch.object(app, "RESPONSES_QUEUE_POLL_INTERVAL", 0.001)
@mock.patch.object(app, "get_or_spawn_pipeline_process")
def test_pipeline_manager_handler_when_status_probed_during_initialisation(
    get_or_spawn_pipeline_process_mock: MagicMock,
) -> None:
    # interleaving is random, so a single round would rarely expose a shared reader
    for _ in range(30):
        # given
        managed_pipeline = ManagedInferencePipeline(
            pipeline_id="pipeline",
            pipeline_manager=_pipeline_process(is_alive=True),
            command_queue=Queue(),
            responses_queue=Queue(),
            operation_lock=Lock(),
            is_idle=False,
        )
        get_or_spawn_pipeline_process_mock.return_value = managed_pipeline
        first_command_received, stop = Event(), Event()
        Thread(
            target=_answer_commands_in_arrival_order,
            args=(managed_pipeline, first_command_received, stop),
            daemon=True,
        ).start()

        # when
        init_thread, init_results = _start_in_daemon_thread(
            partial(_handle_raw_request, json.dumps({"type": "init"}).encode("utf-8"))
        )
        first_command_received.wait(timeout=CALL_TIME_LIMIT_SECONDS)
        status_thread, status_results = _start_in_daemon_thread(
            partial(
                handle_command,
                processes_table={"pipeline": managed_pipeline},
                request_id="status-request-id",
                pipeline_id="pipeline",
                command={"type": "status", "pipeline_id": "pipeline"},
            )
        )
        init_thread.join(timeout=CALL_TIME_LIMIT_SECONDS)
        status_thread.join(timeout=CALL_TIME_LIMIT_SECONDS)
        stop.set()

        # then
        assert not init_thread.is_alive(), "INIT must receive its own response"
        assert not status_thread.is_alive(), "STATUS must receive its own response"
        assert init_results[0]["response"] == {"status": "success", "type": "init"}
        assert status_results == [{"status": OperationStatus.SUCCESS, "type": "status"}]


@mock.patch.object(app, "handle_command", side_effect=RuntimeError("boom"))
def test_pipeline_manager_handler_when_termination_raises(
    handle_command_mock: MagicMock,
) -> None:
    # given
    processes_table = {"pipeline": _managed_pipeline("pipeline", is_idle=False)}

    # when
    response = _handle_raw_request(
        json.dumps({"type": "terminate", "pipeline_id": "pipeline"}).encode("utf-8"),
        processes_table=processes_table,
    )

    # then
    assert response["response"]["status"] == "failure"
    assert response["response"]["error_type"] == "internal_error"
    assert "pipeline" in processes_table
    assert (
        processes_table["pipeline"].is_terminating is False
    ), "Health check must be able to reap a pipeline whose termination raised"
