"""WP-A03: every pipeline process builds its own host from a picklable descriptor.

The spawn tests run a driver script in a fresh interpreter with the `spawn`
start method. A spawned child re-imports its parent's main script (as
`__mp_main__`) before it unpickles the process object, so the driver installs
an audit hook there and records, for each watched runtime module, the moment
the child first imports it: the host descriptor and stream configuration
installed at that point and the call stack importing it.
"""

import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from functools import partial
from multiprocessing import Queue, get_context
from pathlib import Path
from queue import Empty
from typing import Callable, Dict, Iterable, List, Optional, Tuple
from unittest import mock
from unittest.mock import AsyncMock, MagicMock

import psutil
import pytest

from inference.core.exceptions import (
    CannotInitialiseModelError,
    MissingApiKeyError,
    RoboflowAPIConnectionError,
    RoboflowAPINotAuthorizedError,
    RoboflowAPINotNotFoundError,
    RoboflowAPITimeoutError,
)
from inference.core.interfaces.camera.video_source import StreamState
from inference.core.interfaces.stream import pipeline as pipeline_module
from inference.core.interfaces.stream_manager.manager_app import (
    inference_pipeline_manager,
)
from inference.core.interfaces.stream_manager.manager_app.bootstrap import (
    run_stream_manager,
)
from inference.core.interfaces.stream_manager.manager_app.entities import (
    ERROR_TYPE_KEY,
    STATUS_KEY,
    TYPE_KEY,
    CommandType,
    ErrorType,
    InitialisePipelinePayload,
    InitialiseWebRTCPipelinePayload,
    OperationStatus,
    VideoConfiguration,
    WebRTCOffer,
    WorkflowConfiguration,
)
from inference.core.interfaces.stream_manager.manager_app.host import (
    PipelineHostDescriptor,
    PipelineHostNotConfiguredError,
    resolve_host_descriptor,
)
from inference.core.interfaces.stream_manager.manager_app.inference_pipeline_manager import (
    InferencePipelineManager,
)
from inference.core.interfaces.streams_configuration import (
    LEGACY_PIPELINE_HOST_DESCRIPTOR,
    server_streams_configuration,
)
from inference.core.workflows.errors import WorkflowSyntaxError

REPO_ROOT = Path(__file__).resolve().parents[7]
VIDEO_PATH = (
    REPO_ROOT / "tests/inference/unit_tests/core/interfaces/assets/example_video.mp4"
)

# Installed as `__mp_main__` in every spawned child of a driver script.
IMPORT_ORDER_PROBE_SOURCE = """
import json as _probe_json
import os as _probe_os
import sys as _probe_sys

_WATCHED_MODULES = frozenset(
    _probe_os.environ["STREAMS_IMPORT_ORDER_WATCHED"].split(",")
)


def _record_watched_import(event, arguments):
    if event != "import" or arguments[0] not in _WATCHED_MODULES:
        return
    host = _probe_sys.modules.get(
        "inference.core.interfaces.stream_manager.manager_app.host"
    )
    configuration = _probe_sys.modules.get(
        "inference.core.interfaces.stream.configuration"
    )
    stack = []
    frame = _probe_sys._getframe(1)
    while frame is not None:
        code = frame.f_code
        stack.append(f"{_probe_os.path.basename(code.co_filename)}:{code.co_name}")
        frame = frame.f_back
    record = {
        "pid": _probe_os.getpid(),
        "module": arguments[0],
        "descriptor": repr(getattr(host, "_DEFAULT_DESCRIPTOR", None)),
        "configuration": repr(getattr(configuration, "_CONFIGURATION", None)),
        "stack": stack,
    }
    with open(_probe_os.environ["STREAMS_IMPORT_ORDER_PROBE"], "a") as probe_file:
        probe_file.write(_probe_json.dumps(record) + "\\n")


if __name__ == "__mp_main__":
    _probe_sys.addaudithook(_record_watched_import)
"""

# Imported by the pipeline manager module. That module itself is loaded with
# `importlib.import_module`, which raises no `import` audit event; import
# statements and unpickling do.
PIPELINE_RUNTIME_MODULES = (
    "inference.core.interfaces.camera.video_source",
    "inference.core.interfaces.stream.environment",
    "inference.core.interfaces.stream.pipeline",
    "inference.core.interfaces.stream_manager.manager_app.entities",
)

TINY_WORKFLOW = {
    "version": "1.0",
    "inputs": [{"type": "InferenceImage", "name": "image"}],
    "steps": [],
    "outputs": [{"type": "JsonField", "name": "image", "selector": "$inputs.image"}],
}


def run_spawn_driver(
    script: str,
    *,
    tmp_path: Path,
    watched_modules: Iterable[str],
    environment: Optional[Dict[str, str]] = None,
) -> Tuple[dict, List[dict]]:
    """Run `script` as a main script under the import-order probe.

    Returns:
        The JSON the script wrote to `$SPAWN_DRIVER_RESULT`, and the probe
        records of its spawned children.
    """
    driver_path = tmp_path / "spawn_driver.py"
    driver_path.write_text(IMPORT_ORDER_PROBE_SOURCE + script)
    probe_path = tmp_path / "import_order.jsonl"
    result_path = tmp_path / "result.json"
    driver_environment = dict(os.environ)
    driver_environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "DISABLE_VERSION_CHECK": "True",
            "PYTHONPATH": os.pathsep.join(
                [
                    str(REPO_ROOT),
                    str(REPO_ROOT / "workflows"),
                    str(REPO_ROOT / "inference_models"),
                ]
            ),
            "STREAMS_IMPORT_ORDER_PROBE": str(probe_path),
            "STREAMS_IMPORT_ORDER_WATCHED": ",".join(watched_modules),
            "SPAWN_DRIVER_RESULT": str(result_path),
        }
    )
    driver_environment.update(environment or {})
    completed = subprocess.run(
        [sys.executable, str(driver_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=driver_environment,
        timeout=540,
    )
    assert completed.returncode == 0, completed.stderr[-6000:]

    result = json.loads(result_path.read_text())
    records = []
    if probe_path.exists():
        records = [json.loads(line) for line in probe_path.read_text().splitlines()]

    return result, records


# ---------------------------------------------------------------------------
# Pipeline processes under spawn
# ---------------------------------------------------------------------------

_PIPELINE_PROCESSES_DRIVER = '''
import json
import os


class RecordingHost:
    """Records, in its own process, when it is built, used and closed."""

    def __init__(self, record_path, nonce):
        from inference.core.interfaces.stream.configuration import get_configuration
        from inference.core.interfaces.stream_manager.manager_app.host import (
            get_default_host_descriptor,
        )

        self._record_path = record_path
        self._record(
            "created",
            parent_pid=os.getppid(),
            nonce=nonce,
            configuration=repr(get_configuration()),
            default_descriptor=repr(get_default_host_descriptor()),
        )

    def prepare_workflow(self, **kwargs):
        from inference.core.exceptions import RoboflowAPINotNotFoundError

        self._record("prepare", api_key=kwargs["api_key"])
        raise RoboflowAPINotNotFoundError("workflow is not available")

    def close(self):
        self._record("closed")

    def _record(self, event, **details):
        with open(self._record_path, "a") as record_file:
            record = {"event": event, "pid": os.getpid(), **details}
            record_file.write(json.dumps(record) + "\\n")


def main():
    import multiprocessing
    import signal
    from unittest import mock

    multiprocessing.set_start_method("spawn", force=True)

    from inference.core.interfaces.stream.configuration import get_configuration
    from inference.core.interfaces.stream_manager.manager_app import app
    from inference.core.interfaces.stream_manager.manager_app.entities import (
        CommandType,
        InitialisePipelinePayload,
        VideoConfiguration,
        WorkflowConfiguration,
    )
    from inference.core.interfaces.stream_manager.manager_app.host import (
        PipelineHostDescriptor,
    )

    class StopLoop(Exception):
        pass

    descriptor = PipelineHostDescriptor(
        factory="__main__:RecordingHost",
        settings={
            "record_path": os.environ["HOST_RECORD"],
            "nonce": os.environ["HOST_NONCE"],
        },
    )
    init_command = InitialisePipelinePayload(
        video_configuration=VideoConfiguration(
            type="VideoConfiguration", video_reference="rtsp://127.0.0.1/unused"
        ),
        processing_configuration=WorkflowConfiguration(
            type="WorkflowConfiguration",
            workflow_specification={"version": "1.0", "inputs": [], "steps": [], "outputs": []},
        ),
        api_key="request-key",
    ).model_dump(mode="json")
    init_command["type"] = CommandType.INIT.value
    processes_table = {}
    result = {"driver_pid": os.getpid(), "configuration": repr(get_configuration())}

    with mock.patch.object(app, "STREAM_MANAGER_MAX_ACTIVE_PIPELINES", 2):
        with mock.patch.object(app, "PROCESSES_TABLE", processes_table):
            with mock.patch.object(app, "time") as time_mock:
                time_mock.sleep.side_effect = StopLoop()
                try:
                    app.ensure_idle_pipelines_warmed_up(1, host_descriptor=descriptor)
                except StopLoop:
                    pass
        (warm_pipeline_id,) = processes_table
        result["warm_pipeline_idle"] = processes_table[warm_pipeline_id].is_idle

        reused = app.get_or_spawn_pipeline_process(
            processes_table, host_descriptor=descriptor
        )
        spawned = app.get_or_spawn_pipeline_process(
            processes_table, host_descriptor=descriptor
        )
        try:
            app.get_or_spawn_pipeline_process(processes_table, host_descriptor=descriptor)
            result["limit_error"] = None
        except Exception as error:
            result["limit_error"] = str(error)
    result["reused_warm_pipeline"] = reused.pipeline_id == warm_pipeline_id
    result["spawned_new_pipeline"] = spawned.pipeline_id != warm_pipeline_id

    result["init_responses"] = [
        app.handle_command(
            processes_table=processes_table,
            request_id=f"init-{index}",
            pipeline_id=managed_pipeline.pipeline_id,
            command=init_command,
        )
        for index, managed_pipeline in enumerate((reused, spawned))
    ]
    managed_pipelines = list(processes_table.values())
    result["pipeline_pids"] = [
        managed_pipeline.pipeline_manager.pid for managed_pipeline in managed_pipelines
    ]
    try:
        app.execute_termination(signal.SIGTERM, None, processes_table)
    except SystemExit as termination:
        result["termination_exit_code"] = termination.code
    result["pipeline_exit_codes"] = [
        managed_pipeline.pipeline_manager.exitcode
        for managed_pipeline in managed_pipelines
    ]

    with open(os.environ["SPAWN_DRIVER_RESULT"], "w") as result_file:
        json.dump(result, result_file, default=lambda value: value.value)


if __name__ == "__main__":
    main()
'''


@pytest.mark.timeout(600)
def test_spawned_pipeline_processes_configure_themselves_before_the_runtime(
    tmp_path: Path,
) -> None:
    nonce = uuid.uuid4().hex
    host_record_path = tmp_path / "host_events.jsonl"

    result, import_records = run_spawn_driver(
        _PIPELINE_PROCESSES_DRIVER,
        tmp_path=tmp_path,
        watched_modules=PIPELINE_RUNTIME_MODULES,
        environment={"HOST_RECORD": str(host_record_path), "HOST_NONCE": nonce},
    )

    # warm pool, reuse, demand spawn and the active-pipelines limit
    assert result["warm_pipeline_idle"] is True
    assert result["reused_warm_pipeline"] is True
    assert result["spawned_new_pipeline"] is True
    assert "active pipelines limit" in result["limit_error"]
    # the host's platform error keeps its manager error mapping
    for response in result["init_responses"]:
        assert response[STATUS_KEY] == OperationStatus.FAILURE.value
        assert response[ERROR_TYPE_KEY] == ErrorType.NOT_FOUND.value
        assert response["error_class"] == "RoboflowAPINotNotFoundError"
    # shutdown: every pipeline process drained its commands and exited cleanly
    assert result["termination_exit_code"] == 0
    assert result["pipeline_exit_codes"] == [0, 0]

    pipeline_pids = set(result["pipeline_pids"])
    assert len(pipeline_pids) == 2
    assert result["driver_pid"] not in pipeline_pids
    host_events: Dict[int, List[dict]] = {}
    for line in host_record_path.read_text().splitlines():
        event = json.loads(line)
        host_events.setdefault(event["pid"], []).append(event)
    # a fresh host per pipeline process, built there and closed once when the
    # process shut down after the failed initialisation
    assert set(host_events) == pipeline_pids
    for events in host_events.values():
        assert [event["event"] for event in events] == ["created", "prepare", "closed"]
        created = events[0]
        assert created["parent_pid"] == result["driver_pid"]
        assert created["nonce"] == nonce
        assert created["configuration"] == result["configuration"]
        assert nonce in created["default_descriptor"]
        assert events[1]["api_key"] == "request-key"

    # settings were installed before the first runtime import in each child
    child_records = [
        record for record in import_records if record["pid"] in pipeline_pids
    ]
    for pid in pipeline_pids:
        imported = {
            record["module"] for record in child_records if record["pid"] == pid
        }
        assert set(PIPELINE_RUNTIME_MODULES) <= imported
    for record in child_records:
        assert "bootstrap.py:run" in record["stack"], record
        assert nonce in record["descriptor"], record
        assert record["configuration"] == result["configuration"], record


# ---------------------------------------------------------------------------
# Host contract of the pipeline manager, in process
# ---------------------------------------------------------------------------

_STUB_HOST_EVENTS: List[Tuple[str, dict]] = []
# Worker threads of real pipelines started by a test, checked when the host
# closes: nothing may still run on a closed host.
_PIPELINE_WORKERS: List[threading.Thread] = []
_STUB_SPECIFICATION = {"version": "1.0", "inputs": [], "steps": [], "outputs": []}
_STUB_INIT_PARAMETERS = {"workflows_core.api_key": "prepared"}
_STUB_STEP_ERROR_HANDLER = object()
_HOST_ERRORS = {
    "MissingApiKeyError": lambda: MissingApiKeyError("no key"),
    "RoboflowAPINotAuthorizedError": lambda: RoboflowAPINotAuthorizedError("denied"),
    "RoboflowAPINotNotFoundError": lambda: RoboflowAPINotNotFoundError("missing"),
    "RoboflowAPITimeoutError": lambda: RoboflowAPITimeoutError("timed out"),
    "RoboflowAPIConnectionError": lambda: RoboflowAPIConnectionError("no network"),
    "WorkflowSyntaxError": lambda: WorkflowSyntaxError(
        public_message="invalid", context="test"
    ),
}


class StubHost:
    def __init__(self, error: Optional[str] = None):
        self._error = error
        _STUB_HOST_EVENTS.append(("created", {}))

    def prepare_workflow(self, **kwargs):
        _STUB_HOST_EVENTS.append(("prepare", kwargs))
        if self._error is not None:
            raise _HOST_ERRORS[self._error]()

        return _STUB_SPECIFICATION, _STUB_INIT_PARAMETERS, _STUB_STEP_ERROR_HANDLER

    def close(self) -> None:
        live_workers = [
            worker.name for worker in _PIPELINE_WORKERS if worker.is_alive()
        ]
        _STUB_HOST_EVENTS.append(("closed", {"live_workers": live_workers}))


_WORKER_OWNING_HOSTS: List["WorkerOwningHost"] = []


class WorkerOwningHost(StubHost):
    """A stub host running a non-daemon worker until it is closed.

    Like a host owning a real background worker: a pipeline process that
    never closes it cannot exit, and the stream manager joining it hangs.
    """

    def __init__(self, error: Optional[str] = None):
        super().__init__(error=error)
        self.released = threading.Event()
        self.worker = threading.Thread(target=self.released.wait, name="host-worker")
        self.worker.start()
        _WORKER_OWNING_HOSTS.append(self)

    def close(self) -> None:
        super().close()
        self.released.set()
        self.worker.join()


class _FakeLocalDescription:
    sdp = "v=0 fake-answer"
    type = "answer"


class _FakePeerConnection:
    localDescription = _FakeLocalDescription()
    video_transform_track = None
    data_output = None
    data_channel = None
    stream_output = None
    _consumers_signalled = True

    async def close(self) -> None:
        return None


@pytest.fixture(autouse=True)
def _reset_stub_host_events():
    _STUB_HOST_EVENTS.clear()
    _PIPELINE_WORKERS.clear()
    _WORKER_OWNING_HOSTS.clear()
    yield
    # Releases the workers of hosts a regression left open, after the test
    # recorded its failure, so they cannot hang the suite.
    for host in _WORKER_OWNING_HOSTS:
        host.released.set()
    _STUB_HOST_EVENTS.clear()
    _PIPELINE_WORKERS.clear()
    _WORKER_OWNING_HOSTS.clear()


def _stub_descriptor(
    error: Optional[str] = None, host: str = "StubHost"
) -> PipelineHostDescriptor:
    settings = {} if error is None else {"error": error}

    return PipelineHostDescriptor(factory=f"{__name__}:{host}", settings=settings)


def _command(command_type: CommandType, named: bool) -> dict:
    if named:
        processing_configuration = WorkflowConfiguration(
            type="WorkflowConfiguration",
            workspace_name="my-workspace",
            workflow_id="my-workflow",
            workflow_version_id="3",
        )
    else:
        processing_configuration = WorkflowConfiguration(
            type="WorkflowConfiguration", workflow_specification=TINY_WORKFLOW
        )
    video_configuration = VideoConfiguration(
        type="VideoConfiguration", video_reference="rtsp://127.0.0.1/unused"
    )
    if command_type is CommandType.WEBRTC:
        payload = InitialiseWebRTCPipelinePayload(
            video_configuration=video_configuration,
            processing_configuration=processing_configuration,
            webrtc_offer=WebRTCOffer(type="offer", sdp="v=0"),
            api_key="request-key",
        ).dict()
    else:
        payload = InitialisePipelinePayload(
            video_configuration=video_configuration,
            processing_configuration=processing_configuration,
            api_key="request-key",
        ).dict()
    payload[TYPE_KEY] = command_type

    return payload


def _run_manager(descriptor: PipelineHostDescriptor, command: dict) -> List[tuple]:
    command_queue, responses_queue = Queue(), Queue()
    manager = InferencePipelineManager.init(
        pipeline_id="p",
        command_queue=command_queue,
        responses_queue=responses_queue,
        host_descriptor=descriptor,
    )
    command_queue.put(("1", command))
    command_queue.put(("2", {TYPE_KEY: CommandType.TERMINATE}))

    manager.run()

    # A failed WebRTC initialisation stops the manager before the TERMINATE.
    responses = []
    while True:
        try:
            responses.append(responses_queue.get(timeout=1))
        except Empty:
            return responses


@pytest.mark.timeout(30)
@pytest.mark.parametrize("command_type", [CommandType.INIT, CommandType.WEBRTC])
@pytest.mark.parametrize("named", [False, True], ids=["inline", "named"])
@mock.patch.object(
    inference_pipeline_manager, "init_rtc_peer_connection", new_callable=AsyncMock
)
@mock.patch.object(inference_pipeline_manager.InferencePipeline, "init_with_workflow")
def test_host_prepared_workflow_reaches_the_host_neutral_pipeline(
    pipeline_init_mock: MagicMock,
    init_rtc_peer_connection_mock: AsyncMock,
    named: bool,
    command_type: CommandType,
) -> None:
    init_rtc_peer_connection_mock.return_value = _FakePeerConnection()

    responses = _run_manager(_stub_descriptor(), _command(command_type, named=named))

    assert all(
        response[STATUS_KEY] == OperationStatus.SUCCESS for _, response in responses
    )
    assert [event for event, _ in _STUB_HOST_EVENTS] == ["created", "prepare", "closed"]
    prepare_arguments = _STUB_HOST_EVENTS[1][1]
    assert prepare_arguments["api_key"] == "request-key"
    if named:
        assert prepare_arguments["workflow_specification"] is None
        assert prepare_arguments["workspace_name"] == "my-workspace"
        assert prepare_arguments["workflow_id"] == "my-workflow"
    else:
        assert prepare_arguments["workflow_specification"] == TINY_WORKFLOW
        assert prepare_arguments["workspace_name"] is None
    # the WebRTC path never forwarded a workflow version; that is preserved
    expected_version = "3" if named and command_type is CommandType.INIT else None
    assert prepare_arguments["workflow_version_id"] == expected_version
    init_arguments = pipeline_init_mock.call_args.kwargs
    assert init_arguments["workflow_specification"] is _STUB_SPECIFICATION
    assert init_arguments["workflow_init_parameters"] is _STUB_INIT_PARAMETERS
    assert init_arguments["step_error_handler"] is _STUB_STEP_ERROR_HANDLER
    # one profiler records the host's fetch and the pipeline's runs
    assert init_arguments["profiler"] is prepare_arguments["profiler"]
    assert "api_key" not in init_arguments
    assert "workspace_name" not in init_arguments


@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "command_type, error, expected_error_type",
    [
        (CommandType.INIT, "MissingApiKeyError", ErrorType.INVALID_PAYLOAD),
        (
            CommandType.INIT,
            "RoboflowAPINotAuthorizedError",
            ErrorType.AUTHORISATION_ERROR,
        ),
        (CommandType.INIT, "RoboflowAPINotNotFoundError", ErrorType.NOT_FOUND),
        (CommandType.INIT, "RoboflowAPITimeoutError", ErrorType.OPERATION_ERROR),
        (CommandType.INIT, "RoboflowAPIConnectionError", ErrorType.OPERATION_ERROR),
        (CommandType.INIT, "WorkflowSyntaxError", ErrorType.INVALID_PAYLOAD),
        (CommandType.WEBRTC, "MissingApiKeyError", ErrorType.INVALID_PAYLOAD),
        (
            CommandType.WEBRTC,
            "RoboflowAPINotAuthorizedError",
            ErrorType.AUTHORISATION_ERROR,
        ),
        (CommandType.WEBRTC, "RoboflowAPINotNotFoundError", ErrorType.NOT_FOUND),
        (CommandType.WEBRTC, "RoboflowAPITimeoutError", ErrorType.INTERNAL_ERROR),
        (CommandType.WEBRTC, "RoboflowAPIConnectionError", ErrorType.INTERNAL_ERROR),
        (CommandType.WEBRTC, "WorkflowSyntaxError", ErrorType.INVALID_PAYLOAD),
    ],
)
@mock.patch.object(
    inference_pipeline_manager, "init_rtc_peer_connection", new_callable=AsyncMock
)
@mock.patch.object(inference_pipeline_manager.InferencePipeline, "init_with_workflow")
def test_host_errors_keep_their_mapping_and_close_the_host(
    pipeline_init_mock: MagicMock,
    init_rtc_peer_connection_mock: AsyncMock,
    command_type: CommandType,
    error: str,
    expected_error_type: ErrorType,
) -> None:
    init_rtc_peer_connection_mock.return_value = _FakePeerConnection()

    responses = _run_manager(
        _stub_descriptor(error), _command(command_type, named=True)
    )

    failures = [
        response
        for request_id, response in responses
        if request_id == "1" and response[STATUS_KEY] == OperationStatus.FAILURE
    ]
    assert len(failures) == 1
    assert failures[0][ERROR_TYPE_KEY] == expected_error_type
    assert failures[0]["error_class"] == error
    assert [event for event, _ in _STUB_HOST_EVENTS] == ["created", "prepare", "closed"]
    pipeline_init_mock.assert_not_called()


class SignalDuringPrepareHost:
    """Reproduces a termination signal arriving before a pipeline exists.

    `prepare_workflow` sends this process a real SIGTERM - the manager's
    `run()` installs its termination handler before dispatching any command,
    so the handler runs synchronously inside this call, exactly as in the
    confirmed repro.
    """

    def __init__(self, fail: bool = False):
        self._fail = fail
        _STUB_HOST_EVENTS.append(("created", {}))

    def prepare_workflow(self, **kwargs):
        _STUB_HOST_EVENTS.append(("prepare", kwargs))
        os.kill(os.getpid(), signal.SIGTERM)
        if self._fail:
            raise RoboflowAPINotNotFoundError("workflow is not available")
        return _STUB_SPECIFICATION, _STUB_INIT_PARAMETERS, _STUB_STEP_ERROR_HANDLER

    def close(self) -> None:
        _STUB_HOST_EVENTS.append(("closed", {}))


def _signal_during_prepare_descriptor(fail: bool = False) -> PipelineHostDescriptor:
    return PipelineHostDescriptor(
        factory=f"{__name__}:SignalDuringPrepareHost", settings={"fail": fail}
    )


def _recording_pipeline() -> MagicMock:
    pipeline = MagicMock()
    pipeline.terminate.side_effect = lambda: _STUB_HOST_EVENTS.append(("terminate", {}))
    pipeline.join.side_effect = lambda: _STUB_HOST_EVENTS.append(("join", {}))

    return pipeline


@pytest.mark.timeout(30)
@mock.patch.object(inference_pipeline_manager.InferencePipeline, "init_with_workflow")
def test_signal_during_host_prepare_drains_and_closes_the_late_pipeline(
    pipeline_init_mock: MagicMock,
) -> None:
    """WP-A03: SIGTERM during `host.prepare_workflow` must not leak the
    pipeline and host that initialisation creates afterwards."""
    pipeline_instance = _recording_pipeline()
    pipeline_init_mock.return_value = pipeline_instance

    command_queue, responses_queue = Queue(), Queue()
    manager = InferencePipelineManager.init(
        pipeline_id="p",
        command_queue=command_queue,
        responses_queue=responses_queue,
        host_descriptor=_signal_during_prepare_descriptor(),
    )
    command_queue.put(("1", _command(CommandType.INIT, named=False)))

    manager.run()

    request_id, response = responses_queue.get(timeout=1)
    assert request_id == "1"
    assert response[STATUS_KEY] == OperationStatus.SUCCESS
    pipeline_instance.start.assert_called_once()
    # the pipeline created after the signal was drained, then the host closed,
    # instead of being left running past the termination already recorded
    assert [event for event, _ in _STUB_HOST_EVENTS] == [
        "created",
        "prepare",
        "terminate",
        "join",
        "closed",
    ]
    assert manager._inference_pipeline is None
    assert manager._host is None


@pytest.mark.timeout(30)
@mock.patch.object(inference_pipeline_manager.InferencePipeline, "init_with_workflow")
def test_signal_during_host_prepare_before_failed_init_still_closes_host_once(
    pipeline_init_mock: MagicMock,
) -> None:
    """A signal during prepare, followed by prepare itself failing, must
    still close the host exactly once and let `run()` return."""
    command_queue, responses_queue = Queue(), Queue()
    manager = InferencePipelineManager.init(
        pipeline_id="p",
        command_queue=command_queue,
        responses_queue=responses_queue,
        host_descriptor=_signal_during_prepare_descriptor(fail=True),
    )
    command_queue.put(("1", _command(CommandType.INIT, named=False)))

    manager.run()

    request_id, response = responses_queue.get(timeout=1)
    assert request_id == "1"
    assert response[STATUS_KEY] == OperationStatus.FAILURE
    assert response[ERROR_TYPE_KEY] == ErrorType.NOT_FOUND
    pipeline_init_mock.assert_not_called()
    assert manager._host is None
    assert [event for event, _ in _STUB_HOST_EVENTS] == ["created", "prepare", "closed"]


@pytest.mark.timeout(30)
@mock.patch.object(inference_pipeline_manager.InferencePipeline, "init_with_workflow")
def test_signal_after_successful_init_terminates_pipeline_exactly_once(
    pipeline_init_mock: MagicMock,
) -> None:
    """SIGTERM after a pipeline exists only stops the command loop; `run()`
    then drains the pipeline once and closes the host after it."""
    pipeline_init_mock.return_value = _recording_pipeline()

    command_queue, responses_queue = Queue(), Queue()
    manager = InferencePipelineManager.init(
        pipeline_id="p",
        command_queue=command_queue,
        responses_queue=responses_queue,
        host_descriptor=_stub_descriptor(),
    )
    command_queue.put(("1", _command(CommandType.INIT, named=False)))

    collected: Dict[str, tuple] = {}

    def send_signal_once_initialised():
        # Waiting for the init response ensures the signal lands on an
        # already-running pipeline, not during `host.prepare_workflow`.
        collected["response"] = responses_queue.get(timeout=10)
        os.kill(os.getpid(), signal.SIGTERM)

    sender = threading.Thread(target=send_signal_once_initialised, daemon=True)
    sender.start()
    manager.run()
    sender.join(timeout=10)

    assert collected["response"][1][STATUS_KEY] == OperationStatus.SUCCESS
    assert [event for event, _ in _STUB_HOST_EVENTS] == [
        "created",
        "prepare",
        "terminate",
        "join",
        "closed",
    ]
    assert manager._inference_pipeline is None
    assert manager._host is None


@pytest.mark.timeout(60)
@pytest.mark.parametrize("command_type", [CommandType.INIT, CommandType.WEBRTC])
@mock.patch.object(
    inference_pipeline_manager, "init_rtc_peer_connection", new_callable=AsyncMock
)
@mock.patch.object(inference_pipeline_manager.InferencePipeline, "init_with_workflow")
def test_signal_at_pipeline_start_drains_real_workers_before_closing_host(
    pipeline_init_mock: MagicMock,
    init_rtc_peer_connection_mock: AsyncMock,
    command_type: CommandType,
) -> None:
    """WP-A03: SIGTERM once the pipeline is assigned but before `start()`
    runs - its source still NOT_STARTED - must not leave the workers that
    `start()` then launches running on a closed host."""
    init_rtc_peer_connection_mock.return_value = _FakePeerConnection()
    created = {}

    def init_real_pipeline(**kwargs):
        # A real pipeline on an endless source, keeping the manager's sink and
        # watchdog; only the workflow is replaced by a trivial frame handler.
        pipeline = inference_pipeline_manager.InferencePipeline.init_with_custom_logic(
            video_reference="TestPatternStreamProducer",
            on_video_frame=lambda video_frames: [{} for _ in video_frames],
            on_prediction=kwargs["on_prediction"],
            watchdog=kwargs["watchdog"],
        )
        start = pipeline.start

        def start_after_signal(use_main_thread: bool = True) -> None:
            os.kill(os.getpid(), signal.SIGTERM)
            start(use_main_thread=use_main_thread)
            _PIPELINE_WORKERS.extend(
                [pipeline._inference_thread, pipeline._dispatching_thread]
            )

        pipeline.start = start_after_signal
        created["pipeline"] = pipeline
        return pipeline

    pipeline_init_mock.side_effect = init_real_pipeline
    command_queue, responses_queue = Queue(), Queue()
    manager = InferencePipelineManager.init(
        pipeline_id="p",
        command_queue=command_queue,
        responses_queue=responses_queue,
        host_descriptor=_stub_descriptor(),
    )
    command_queue.put(("1", _command(command_type, named=False)))

    try:
        manager.run()

        request_id, response = responses_queue.get(timeout=1)
        (video_source,) = created["pipeline"]._video_sources
        # the source was started by the inference worker, after the signal
        assert video_source._stream_consumption_thread is not None
        _PIPELINE_WORKERS.append(video_source._stream_consumption_thread)
        assert request_id == "1"
        assert response[STATUS_KEY] == OperationStatus.SUCCESS
        assert [worker.name for worker in _PIPELINE_WORKERS if worker.is_alive()] == []
        assert video_source.describe_source().state is StreamState.ENDED
        assert _STUB_HOST_EVENTS[2:] == [("closed", {"live_workers": []})]
        assert manager._inference_pipeline is None
        assert manager._host is None
    finally:
        # Stops what a regression leaves running, so it cannot hang the suite.
        if any(worker.is_alive() for worker in _PIPELINE_WORKERS):
            created["pipeline"].terminate()
            created["pipeline"].join()


@pytest.mark.timeout(30)
@pytest.mark.parametrize("loop_end", ["sentinel", "exception"])
@mock.patch.object(inference_pipeline_manager.InferencePipeline, "init_with_workflow")
def test_run_drains_pipeline_before_closing_host_however_the_loop_ends(
    pipeline_init_mock: MagicMock,
    loop_end: str,
) -> None:
    pipeline_init_mock.return_value = _recording_pipeline()
    command_queue = MagicMock()
    last_command = None if loop_end == "sentinel" else RuntimeError("queue broken")
    command_queue.get.side_effect = [
        ("1", _command(CommandType.INIT, named=False)),
        last_command,
    ]
    manager = InferencePipelineManager.init(
        pipeline_id="p",
        command_queue=command_queue,
        responses_queue=Queue(),
        host_descriptor=_stub_descriptor(),
    )

    if loop_end == "sentinel":
        manager.run()
    else:
        with pytest.raises(RuntimeError):
            manager.run()

    assert [event for event, _ in _STUB_HOST_EVENTS] == [
        "created",
        "prepare",
        "terminate",
        "join",
        "closed",
    ]
    assert manager._inference_pipeline is None
    assert manager._host is None


def _real_pipeline_builder(
    video_reference: List[str],
    created: dict,
    start_error: Optional[Exception] = None,
) -> Callable[..., object]:
    """Replaces `init_with_workflow` with a real pipeline on `video_reference`.

    It keeps the manager's sink and watchdog; only the workflow is replaced by
    a trivial frame handler. Its workers - including the capture worker of
    every source that starts - are registered in `_PIPELINE_WORKERS`, so the
    stub host sees whether any still runs when it closes.
    """

    def init_real_pipeline(**kwargs):
        pipeline = inference_pipeline_manager.InferencePipeline.init_with_custom_logic(
            video_reference=video_reference,
            on_video_frame=lambda video_frames: [{} for _ in video_frames],
            on_prediction=kwargs["on_prediction"],
            on_pipeline_end=lambda: _STUB_HOST_EVENTS.append(("pipeline_end", {})),
            watchdog=kwargs["watchdog"],
        )
        for video_source in pipeline._video_sources:
            _register_capture_worker(video_source)
        start = pipeline.start

        def start_registering_workers(use_main_thread: bool = True) -> None:
            start(use_main_thread=use_main_thread)
            _PIPELINE_WORKERS.extend(
                [pipeline._inference_thread, pipeline._dispatching_thread]
            )
            if start_error is not None:
                raise start_error

        pipeline.start = start_registering_workers
        created["pipeline"] = pipeline
        return pipeline

    return init_real_pipeline


def _register_capture_worker(video_source) -> None:
    start = video_source.start

    def start_registering_worker() -> None:
        start()
        _PIPELINE_WORKERS.append(video_source._stream_consumption_thread)

    video_source.start = start_registering_worker


def _stop_leaked_pipeline(pipeline) -> None:
    # Stops what a regression leaves running, after the test recorded its
    # failure, so it cannot hang the suite.
    pipeline._stop = True
    for video_source in pipeline._video_sources:
        worker = video_source._stream_consumption_thread
        if worker is not None and worker.is_alive():
            video_source.terminate(
                wait_on_frames_consumption=False, purge_frames_buffer=True
            )


def _run_manager_between_commands(
    descriptor: PipelineHostDescriptor,
    commands: List[Optional[tuple]],
    before_next_command: Optional[Callable[[], None]] = None,
) -> Tuple[InferencePipelineManager, List[tuple], List[dict]]:
    """Run a manager on `commands`, recording its state between them.

    The state is recorded whenever the manager asks for its next command -
    once the previous one was handled, before the next one is.
    """
    command_queue, responses_queue = MagicMock(), Queue()
    manager = InferencePipelineManager.init(
        pipeline_id="p",
        command_queue=command_queue,
        responses_queue=responses_queue,
        host_descriptor=descriptor,
    )
    pending = list(commands)
    states = []

    def next_command(timeout: float) -> Optional[tuple]:
        if len(pending) < len(commands):
            if before_next_command is not None:
                before_next_command()
            states.append(
                {
                    "host_events": [event for event, _ in _STUB_HOST_EVENTS],
                    "host_released": manager._host is None,
                    "pipeline_released": manager._inference_pipeline is None,
                    "host_workers_alive": [
                        host.worker.is_alive() for host in _WORKER_OWNING_HOSTS
                    ],
                }
            )
        return pending.pop(0)

    command_queue.get.side_effect = next_command
    manager.run()

    responses = []
    while True:
        try:
            responses.append(responses_queue.get(timeout=1))
        except Empty:
            return manager, responses, states


@pytest.mark.timeout(90)
@pytest.mark.parametrize("loop_end", ["terminate", "sentinel"])
@pytest.mark.parametrize(
    "references",
    [
        ("missing", "TestPatternStreamProducer"),
        ("TestPatternStreamProducer", "missing", "TestPatternStreamProducer"),
    ],
    ids=["first-source-missing", "source-missing-after-a-started-one"],
)
@mock.patch.object(inference_pipeline_manager.InferencePipeline, "init_with_workflow")
def test_pipeline_whose_sources_failed_to_start_is_drained_and_its_host_closed(
    pipeline_init_mock: MagicMock,
    references: Tuple[str, ...],
    loop_end: str,
    tmp_path: Path,
) -> None:
    """WP-A03: a source failing to start ends the pipeline's workers and
    leaves the sources after it NOT_STARTED for good. Termination must stop
    the sources that did start, join the pipeline and close the host,
    instead of waiting for sources that can no longer start."""
    missing_index = references.index("missing")
    video_reference = [
        str(tmp_path / "missing.mp4") if reference == "missing" else reference
        for reference in references
    ]
    created = {}
    pipeline_init_mock.side_effect = _real_pipeline_builder(video_reference, created)
    commands = [("1", _command(CommandType.INIT, named=False))]
    if loop_end == "terminate":
        commands.append(("2", {TYPE_KEY: CommandType.TERMINATE}))
    commands.append(None)
    startup_failure = {}

    def await_startup_failure() -> None:
        if startup_failure:
            return None
        pipeline = created["pipeline"]
        for worker in (pipeline._inference_thread, pipeline._dispatching_thread):
            worker.join(timeout=10)
        startup_failure["workers_alive"] = [
            pipeline._inference_thread.is_alive(),
            pipeline._dispatching_thread.is_alive(),
        ]
        startup_failure["states"] = [
            video_source.describe_source().state
            for video_source in pipeline._video_sources
        ]

    try:
        manager, responses, _ = _run_manager_between_commands(
            _stub_descriptor(host="WorkerOwningHost"),
            commands,
            before_next_command=await_startup_failure,
        )

        # the reproduced state: the pipeline's workers finished, the missing
        # source is in ERROR and the one after it was never started
        assert startup_failure["workers_alive"] == [False, False]
        assert startup_failure["states"][missing_index] is StreamState.ERROR
        assert startup_failure["states"][-1] is StreamState.NOT_STARTED
        assert [
            (request_id, response[STATUS_KEY]) for request_id, response in responses
        ] == [(command[0], OperationStatus.SUCCESS) for command in commands[:-1]]
        # sources that started were stopped, the others left as they were
        expected_states = [StreamState.ENDED] * len(references)
        expected_states[missing_index] = StreamState.ERROR
        expected_states[-1] = StreamState.NOT_STARTED
        assert [
            video_source.describe_source().state
            for video_source in created["pipeline"]._video_sources
        ] == expected_states
        assert [event for event, _ in _STUB_HOST_EVENTS] == [
            "created",
            "prepare",
            "pipeline_end",
            "closed",
        ]
        assert _STUB_HOST_EVENTS[-1] == ("closed", {"live_workers": []})
        assert [host.worker.is_alive() for host in _WORKER_OWNING_HOSTS] == [False]
        assert manager._inference_pipeline is None
        assert manager._host is None
    finally:
        if "pipeline" in created:
            _stop_leaked_pipeline(created["pipeline"])


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "command_type, error",
    [(CommandType.INIT, error) for error in _HOST_ERRORS]
    + [
        (CommandType.WEBRTC, "RoboflowAPITimeoutError"),
        (CommandType.WEBRTC, "RoboflowAPIConnectionError"),
    ],
)
@mock.patch.object(
    inference_pipeline_manager, "init_rtc_peer_connection", new_callable=AsyncMock
)
@mock.patch.object(inference_pipeline_manager.InferencePipeline, "init_with_workflow")
def test_failed_initialisation_closes_the_host_before_the_next_command(
    pipeline_init_mock: MagicMock,
    init_rtc_peer_connection_mock: AsyncMock,
    command_type: CommandType,
    error: str,
) -> None:
    """WP-A03: a failed initialisation leaves the pipeline process serving
    commands, so it must close the host at once - not only on a later
    TERMINATE or shutdown - and a retried initialisation builds a new one."""
    init_rtc_peer_connection_mock.return_value = _FakePeerConnection()

    manager, responses, states = _run_manager_between_commands(
        _stub_descriptor(error, host="WorkerOwningHost"),
        [
            ("1", _command(command_type, named=True)),
            ("2", _command(command_type, named=True)),
            None,
        ],
    )

    for request_id in ("1", "2"):
        failures = [
            response
            for response_request_id, response in responses
            if response_request_id == request_id
            and response[STATUS_KEY] == OperationStatus.FAILURE
        ]
        assert len(failures) == 1
        assert failures[0]["error_class"] == error
    assert states == [
        {
            "host_events": ["created", "prepare", "closed"],
            "host_released": True,
            "pipeline_released": True,
            "host_workers_alive": [False],
        },
        {
            "host_events": ["created", "prepare", "closed"] * 2,
            "host_released": True,
            "pipeline_released": True,
            "host_workers_alive": [False, False],
        },
    ]
    pipeline_init_mock.assert_not_called()


@pytest.mark.timeout(60)
@pytest.mark.parametrize("failure", ["pipeline_init", "pipeline_start"])
@mock.patch.object(inference_pipeline_manager.InferencePipeline, "init_with_workflow")
def test_failed_pipeline_creation_or_start_is_drained_before_the_host_closes(
    pipeline_init_mock: MagicMock,
    failure: str,
) -> None:
    """WP-A03: initialisation failing once the host prepared the workflow -
    creating the pipeline, or starting it after its workers were launched -
    drains what exists and closes the host before the next command."""
    created = {}
    if failure == "pipeline_init":
        pipeline_init_mock.side_effect = CannotInitialiseModelError("no dependency")
    else:
        pipeline_init_mock.side_effect = _real_pipeline_builder(
            ["TestPatternStreamProducer"],
            created,
            start_error=RuntimeError("could not start the dispatching worker"),
        )

    try:
        manager, responses, states = _run_manager_between_commands(
            _stub_descriptor(host="WorkerOwningHost"),
            [("1", _command(CommandType.INIT, named=False)), None],
        )

        ((request_id, response),) = responses
        assert request_id == "1"
        assert response[STATUS_KEY] == OperationStatus.FAILURE
        assert response[ERROR_TYPE_KEY] == ErrorType.INTERNAL_ERROR
        expected_events = ["created", "prepare", "closed"]
        if failure == "pipeline_start":
            expected_events.insert(2, "pipeline_end")
            (video_source,) = created["pipeline"]._video_sources
            assert video_source.describe_source().state is StreamState.ENDED
        assert states == [
            {
                "host_events": expected_events,
                "host_released": True,
                "pipeline_released": True,
                "host_workers_alive": [False],
            }
        ]
        assert _STUB_HOST_EVENTS[-1] == ("closed", {"live_workers": []})
    finally:
        if "pipeline" in created:
            _stop_leaked_pipeline(created["pipeline"])


_PIPELINE_WORKER_TARGETS = {
    "inference": "_execute_inference",
    "dispatcher": "_dispatch_inference_results",
}


def _thread_start_failing_pipeline_builder(
    created: dict, failing_worker: str
) -> Callable[..., object]:
    """Replaces `init_with_workflow` with real pipelines on an endless source.

    The first pipeline cannot start `failing_worker`: its `Thread.start` raises
    at the actual position the pipeline starts it. A failing dispatcher raises
    only once the inference worker - started for real before it - filled the
    single-slot results queue and computed the result it is blocked putting.
    Every pipeline built later starts normally.
    """
    pipelines = created.setdefault("pipelines", [])

    def init_real_pipeline(**kwargs):
        first = not pipelines
        results = []

        def on_video_frame(video_frames):
            results.append(len(video_frames))

            return [{} for _ in video_frames]

        pipeline = inference_pipeline_manager.InferencePipeline.init_with_custom_logic(
            video_reference=["TestPatternStreamProducer"],
            on_video_frame=on_video_frame,
            on_prediction=kwargs["on_prediction"],
            on_pipeline_end=lambda: _STUB_HOST_EVENTS.append(("pipeline_end", {})),
            watchdog=kwargs["watchdog"],
            predictions_queue_size=1,
        )
        for video_source in pipeline._video_sources:
            _register_capture_worker(video_source)
        pipelines.append(pipeline)

        def thread_factory(target, **thread_kwargs) -> threading.Thread:
            worker = threading.Thread(target=target, **thread_kwargs)
            _PIPELINE_WORKERS.append(worker)
            if not first or target.__name__ != _PIPELINE_WORKER_TARGETS[failing_worker]:
                return worker

            def failing_start() -> None:
                if failing_worker == "dispatcher":
                    deadline = time.monotonic() + 10
                    while not (
                        pipeline._predictions_queue.full() and len(results) >= 2
                    ):
                        assert time.monotonic() < deadline, "queue never filled"
                        time.sleep(0.01)
                raise RuntimeError(f"could not start the {failing_worker} worker")

            worker.start = failing_start
            created["unstarted_worker"] = worker

            return worker

        start = pipeline.start

        def start_with_thread_factory(use_main_thread: bool = True) -> None:
            with mock.patch.object(pipeline_module, "Thread", thread_factory):
                start(use_main_thread=use_main_thread)

        pipeline.start = start_with_thread_factory

        return pipeline

    return init_real_pipeline


def _rescue_leaked_pipeline(pipeline) -> None:
    # Stops what a regression leaves running, after the test recorded its
    # failure: an inference worker blocked on a results queue nobody consumes
    # is released by draining the queue until it finishes.
    _stop_leaked_pipeline(pipeline)
    deadline = time.monotonic() + 10
    worker = pipeline._inference_thread
    while worker is not None and worker.is_alive() and time.monotonic() < deadline:
        try:
            pipeline._predictions_queue.get_nowait()
        except Empty:
            time.sleep(0.01)


@pytest.mark.timeout(60)
@pytest.mark.parametrize("failing_worker", ["inference", "dispatcher"])
@mock.patch.object(inference_pipeline_manager.InferencePipeline, "init_with_workflow")
def test_pipeline_whose_worker_failed_to_start_is_drained_and_its_host_closed(
    pipeline_init_mock: MagicMock,
    failing_worker: str,
) -> None:
    """WP-A03: a pipeline worker whose `Thread.start` raised never ran, so it
    cannot be joined; and an inference worker started before the dispatcher
    failed to start has nobody consuming its results, so a full queue blocks
    it - and the join of it - forever. The failed initialisation must still
    end every worker and source that did start, run the end hook, close the
    host once after them, answer once, and let the next INIT start afresh."""
    created = {}
    pipeline_init_mock.side_effect = _thread_start_failing_pipeline_builder(
        created, failing_worker=failing_worker
    )

    try:
        manager, responses, states = _run_manager_between_commands(
            _stub_descriptor(host="WorkerOwningHost"),
            [
                ("1", _command(CommandType.INIT, named=False)),
                ("2", _command(CommandType.INIT, named=False)),
                ("3", {TYPE_KEY: CommandType.TERMINATE}),
                None,
            ],
        )

        assert [
            (request_id, response[STATUS_KEY]) for request_id, response in responses
        ] == [
            ("1", OperationStatus.FAILURE),
            ("2", OperationStatus.SUCCESS),
            ("3", OperationStatus.SUCCESS),
        ]
        assert responses[0][1][ERROR_TYPE_KEY] == ErrorType.INTERNAL_ERROR
        failed_pipeline, retried_pipeline = created["pipelines"]
        assert created["unstarted_worker"].is_alive() is False
        assert failed_pipeline._inference_thread is None
        assert failed_pipeline._dispatching_thread is None
        (video_source,) = failed_pipeline._video_sources
        expected_source_state = {
            "inference": StreamState.NOT_STARTED,
            "dispatcher": StreamState.ENDED,
        }[failing_worker]
        assert video_source.describe_source().state is expected_source_state
        assert [worker.name for worker in _PIPELINE_WORKERS if worker.is_alive()] == []
        failed_events = ["created", "prepare", "pipeline_end", "closed"]
        # the TERMINATE stops the manager, so no state follows it
        assert states == [
            {
                "host_events": failed_events,
                "host_released": True,
                "pipeline_released": True,
                "host_workers_alive": [False],
            },
            {
                "host_events": failed_events + ["created", "prepare"],
                "host_released": False,
                "pipeline_released": False,
                "host_workers_alive": [False, True],
            },
        ]
        assert [event for event, _ in _STUB_HOST_EVENTS] == failed_events * 2
        assert [event for event in _STUB_HOST_EVENTS if event[0] == "closed"] == [
            ("closed", {"live_workers": []})
        ] * 2
        assert [host.worker.is_alive() for host in _WORKER_OWNING_HOSTS] == [
            False,
            False,
        ]
        (retried_source,) = retried_pipeline._video_sources
        assert retried_source.describe_source().state is StreamState.ENDED
        assert manager._inference_pipeline is None
        assert manager._host is None
    finally:
        for pipeline in created.get("pipelines", []):
            _rescue_leaked_pipeline(pipeline)


def test_host_resolution_needs_an_explicit_or_installed_descriptor(monkeypatch) -> None:
    from inference.core.interfaces.stream_manager.manager_app import host

    monkeypatch.setattr(host, "_DEFAULT_DESCRIPTOR", None)
    explicit = _stub_descriptor()

    assert resolve_host_descriptor(explicit) is explicit
    with pytest.raises(PipelineHostNotConfiguredError):
        resolve_host_descriptor(None)
    with pytest.raises(PipelineHostNotConfiguredError):
        InferencePipelineManager.init(
            pipeline_id="p", command_queue=Queue(), responses_queue=Queue()
        )
    with pytest.raises(ValueError):
        PipelineHostDescriptor(factory="module.without.attribute")


def test_legacy_configuration_installs_the_legacy_host_as_process_default() -> None:
    from inference.core.interfaces.stream_manager.manager_app.host import (
        get_default_host_descriptor,
    )

    assert get_default_host_descriptor() == LEGACY_PIPELINE_HOST_DESCRIPTOR
    assert LEGACY_PIPELINE_HOST_DESCRIPTOR.factory == (
        "inference.core.interfaces.legacy_stream.host:LegacyPipelineHost"
    )


# ---------------------------------------------------------------------------
# A real manager on localhost, with the legacy host and a tiny workflow
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _receive_exactly(connection: socket.socket, size: int) -> bytes:
    received = bytearray()
    while len(received) < size:
        chunk = connection.recv(size - len(received))
        assert chunk, "manager closed the connection"
        received.extend(chunk)

    return bytes(received)


def _send_command(port: int, command: dict) -> dict:
    data = json.dumps(command).encode("utf-8")
    with socket.create_connection(("127.0.0.1", port), timeout=180) as connection:
        connection.sendall(len(data).to_bytes(4, byteorder="big") + data)
        header = _receive_exactly(connection, 4)
        payload = _receive_exactly(connection, int.from_bytes(header, byteorder="big"))

    return json.loads(payload)


def _wait_for_manager(process, port: int) -> None:
    deadline = time.monotonic() + 240
    while time.monotonic() < deadline:
        assert process.is_alive(), f"manager exited with {process.exitcode}"
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return None
        except OSError:
            time.sleep(0.5)
    pytest.fail("stream manager did not start listening")


def _init_command(specification: dict) -> dict:
    command = InitialisePipelinePayload(
        video_configuration=VideoConfiguration(
            type="VideoConfiguration", video_reference=str(VIDEO_PATH), max_fps=1
        ),
        processing_configuration=WorkflowConfiguration(
            type="WorkflowConfiguration", workflow_specification=specification
        ),
    ).model_dump(mode="json")
    command[TYPE_KEY] = CommandType.INIT.value

    return command


def _new_pipeline_pid(manager_process: psutil.Process, known: set) -> int:
    # A manager using the spawn start method also starts multiprocessing's
    # resource tracker when it creates its first pipeline queues.
    new_children = [
        child for child in manager_process.children() if child.pid not in known
    ]
    (pid,) = [
        child.pid
        for child in new_children
        if "resource_tracker" not in " ".join(child.cmdline())
    ]
    known.update(child.pid for child in new_children)

    return pid


@pytest.mark.timeout(600)
def test_real_manager_runs_and_cleans_up_a_tiny_workflow_pipeline(monkeypatch) -> None:
    port = _free_port()
    monkeypatch.setenv("STREAM_MANAGER_HOST", "127.0.0.1")
    monkeypatch.setenv("STREAM_MANAGER_PORT", str(port))
    manager = get_context("spawn").Process(
        target=partial(
            run_stream_manager,
            configuration=server_streams_configuration(),
            host_descriptor=LEGACY_PIPELINE_HOST_DESCRIPTOR,
        )
    )
    manager.start()
    pipeline_pids = []
    try:
        _wait_for_manager(manager, port)
        manager_process = psutil.Process(manager.pid)
        known_children = {child.pid for child in manager_process.children()}

        # normal lifecycle: init, status, terminate
        response = _send_command(port, _init_command(TINY_WORKFLOW))
        assert (
            response["response"][STATUS_KEY] == OperationStatus.SUCCESS.value
        ), response
        pipeline_id = response["pipeline_id"]
        pipeline_pids.append(_new_pipeline_pid(manager_process, known_children))
        status = _send_command(
            port, {TYPE_KEY: CommandType.STATUS.value, "pipeline_id": pipeline_id}
        )
        assert status["response"][STATUS_KEY] == OperationStatus.SUCCESS.value, status
        assert status["response"]["report"]["sources_metadata"]
        terminated = _send_command(
            port, {TYPE_KEY: CommandType.TERMINATE.value, "pipeline_id": pipeline_id}
        )
        assert terminated["response"][STATUS_KEY] == OperationStatus.SUCCESS.value
        assert not psutil.pid_exists(pipeline_pids[-1])

        # failed initialisation leaves a process that terminates cleanly too
        invalid_workflow = {
            **TINY_WORKFLOW,
            "steps": [{"type": "NoSuchBlock", "name": "x"}],
        }
        response = _send_command(port, _init_command(invalid_workflow))
        assert (
            response["response"][STATUS_KEY] == OperationStatus.FAILURE.value
        ), response
        failed_pipeline_id = response["pipeline_id"]
        pipeline_pids.append(_new_pipeline_pid(manager_process, known_children))
        terminated = _send_command(
            port,
            {TYPE_KEY: CommandType.TERMINATE.value, "pipeline_id": failed_pipeline_id},
        )
        assert terminated["response"][STATUS_KEY] == OperationStatus.SUCCESS.value
        assert not psutil.pid_exists(pipeline_pids[-1])

        listed = _send_command(port, {TYPE_KEY: CommandType.LIST_PIPELINES.value})
        assert listed["response"]["pipelines"] == []
    finally:
        manager.terminate()
        manager.join(timeout=60)

    assert manager.exitcode == 0
    assert not any(psutil.pid_exists(pid) for pid in pipeline_pids)
