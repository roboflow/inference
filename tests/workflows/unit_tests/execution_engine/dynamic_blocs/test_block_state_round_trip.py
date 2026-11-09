"""Custom Python ``self`` state survives remote (Modal) execution.

The engine-side block instance owns the state: every call sends its user
attributes as JSON, the sandbox seeds ``self`` with them and echoes the state
the function left behind. The end-to-end tests drive the real engine
executors against the real ``modal/modal_app.py`` HTTP and websocket handlers,
loaded through the shared harness in ``conftest.py``.
"""

import asyncio
import json
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace

import msgpack
import pytest
from roboflow_workflows.core_steps.formatters.expression.v1 import BlockManifest
from roboflow_workflows.execution_engine.v1.dynamic_blocks import block_scaffolding
from roboflow_workflows.execution_engine.v1.dynamic_blocks import (
    block_state as engine_state,
)
from roboflow_workflows.execution_engine.v1.dynamic_blocks import (
    modal_executor as client,
)
from roboflow_workflows.execution_engine.v1.dynamic_blocks.block_state import (
    apply_block_state,
    collect_block_state,
    decode_echoed_block_state,
    encode_block_state,
)
from roboflow_workflows.execution_engine.v1.dynamic_blocks.entities import PythonCode

MIB = 1024 * 1024

_OPEN_WS_EXECUTORS: list = []


@pytest.fixture(autouse=True)
def _forget_engine_warnings():
    engine_state._WARNED.clear()
    yield
    engine_state._WARNED.clear()


def _portal_thread_count() -> int:
    count = sum(1 for thread in threading.enumerate() if "-portal-" in thread.name)

    return count


@pytest.fixture(autouse=True)
def _close_ws_sessions_when_the_test_ends():
    portal_threads_before = _portal_thread_count()
    yield
    while _OPEN_WS_EXECUTORS:
        _OPEN_WS_EXECUTORS.pop().close()
    assert _portal_thread_count() == portal_threads_before


@pytest.fixture()
def engine_warnings(monkeypatch) -> list:
    warned = []
    monkeypatch.setattr(
        engine_state.logger,
        "warning",
        lambda message, *args, **kwargs: warned.append(message % args),
    )

    return warned


class _FakeRequest:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode()
        self.headers = {}

    async def body(self) -> bytes:
        return self._body


def _http_server(modal_app):
    server = modal_app.Executor.__new__(modal_app.Executor)
    server._code_namespaces = {}
    server._shared_globals = {}

    return server


class _InProcessHttpExecutor:
    """The engine's HTTP executor, its POST answered by the real handler."""

    def __init__(self, modal_app):
        self.server = _http_server(modal_app)
        self.requests = []
        self.responses = []
        self.drop_echo = False
        self.old_server = False
        self._client = client.ModalExecutor(workspace_id="test-ws")
        self._client._get_endpoint_url = lambda workspace: "http://modal.test"
        self._client._post_execute = self._post

    def _post(self, **payload):
        request = {
            "code_hash": payload["code_hash"],
            "run_function_name": payload["python_code"].run_function_name,
            "inputs_json": payload["inputs_json"],
            "workflow_context": payload["workflow_context"],
        }
        if payload.get("block_state_json") is not None:
            request["block_state_json"] = payload["block_state_json"]
        if payload["send_full_code"]:
            request["code_str"] = payload["python_code"].run_function_code
            request["imports"] = payload["python_code"].imports or []
        self.requests.append(request)
        served_request = dict(request)
        if self.old_server:
            served_request.pop("block_state_json", None)
        response = asyncio.run(self.server.execute_block(_FakeRequest(served_request)))
        if self.drop_echo or self.old_server:
            response.pop("block_state", None)
        self.responses.append(response)

        return response

    def execute_remote_with_state(self, **kwargs):
        return self._client.execute_remote_with_state(**kwargs)

    def execute_remote(self, **kwargs):
        return self._client.execute_remote(**kwargs)


def _ws_client():
    executor = client.WebSocketModalExecutor(workspace_id="test-ws")
    executor._server = client._ServerInfo(
        proto=2, idle_timeout=10.0, container_id="container-1"
    )

    return executor


class _InProcessWsExecutor:
    """The engine's websocket executor talking to the real websocket ASGI app."""

    def __init__(self, modal_app):
        from fastapi.testclient import TestClient

        cls = modal_app.Executor
        user_cls = cls._get_user_cls() if hasattr(cls, "_get_user_cls") else cls
        server = user_cls.__new__(user_cls)
        server.workspace_id = "test-ws"
        user_cls.identify(server)
        self.server = server
        self._session = TestClient(user_cls.wsapp(server)).websocket_connect("/ws")
        self._ws = self._session.__enter__()
        _OPEN_WS_EXECUTORS.append(self)
        self.requests = []
        self.responses = []
        self.drop_echo = False
        self.old_server = False
        self._client = _ws_client()
        self._client._send_recv_with_retry = self._send_recv

    def close(self):
        if self._session is None:
            return

        self._session.__exit__(None, None, None)
        self._session = None

    def _send_recv(self, frame_bytes, workspace, request_id=None):
        request = msgpack.unpackb(frame_bytes, raw=False)
        self.requests.append(request)
        if self.old_server:
            served_request = {
                key: value
                for key, value in request.items()
                if key != "block_state_json"
            }
            frame_bytes = msgpack.packb(served_request, use_bin_type=True)
        self._ws.send_bytes(frame_bytes)
        response = msgpack.unpackb(self._ws.receive_bytes(), raw=False)
        if self.drop_echo or self.old_server:
            response.pop("block_state", None)
        self.responses.append(response)
        response_bytes = msgpack.packb(response, use_bin_type=True)

        return response_bytes

    def execute_remote_with_state(self, **kwargs):
        return self._client.execute_remote_with_state(**kwargs)

    def execute_remote(self, **kwargs):
        return self._client.execute_remote(**kwargs)


def _route_to(monkeypatch, executor) -> None:
    @contextmanager
    def acquire(workspace_id):
        yield executor

    monkeypatch.setattr(block_scaffolding, "_acquire_modal_executor", acquire)


class _WireSocket:
    """The socket the engine's websocket executor holds, backed by the real ASGI app."""

    def __init__(self, harness, session, ws):
        self._harness = harness
        self._session = session
        self._ws = ws

    def send_binary(self, frame: bytes) -> None:
        self._harness.sent.append(frame)
        self._ws.send_bytes(frame)

    def recv(self) -> bytes:
        data = self._ws.receive_bytes()
        if self._harness.lose_next_response:
            self._harness.lose_next_response = False
            self._harness.lost.append(data)
            raise ConnectionError("connection dropped after the server answered")

        if self._harness.foreign_request_id is not None:
            response = msgpack.unpackb(data, raw=False)
            if "request_id" in response:
                response["request_id"] = self._harness.foreign_request_id
                data = msgpack.packb(response, use_bin_type=True)
        self._harness.received.append(data)

        return data

    def gettimeout(self):
        return None

    def settimeout(self, value) -> None:
        pass

    def close(self) -> None:
        if self._session is None:
            return

        self._session.__exit__(None, None, None)
        self._session = None


class _WireWsExecutor:
    """The engine's websocket executor with its own send/receive retry loop.

    Only the connection is replaced: each (re)connect opens a new socket to the
    real websocket ASGI app and runs the executor's real handshake.
    """

    def __init__(self, modal_app):
        from fastapi.testclient import TestClient

        cls = modal_app.Executor
        user_cls = cls._get_user_cls() if hasattr(cls, "_get_user_cls") else cls
        self.server = user_cls.__new__(user_cls)
        self.server.workspace_id = "test-ws"
        user_cls.identify(self.server)
        self._app = user_cls.wsapp(self.server)
        self._test_client_class = TestClient
        self._sockets = []
        self.sent = []
        self.received = []
        self.lost = []
        self.lose_next_response = False
        self.foreign_request_id = None
        self._client = _ws_client()
        self._client._connect = self._connect
        _OPEN_WS_EXECUTORS.append(self)

    def _connect(self, workspace_id) -> None:
        session = self._test_client_class(self._app).websocket_connect("/ws")
        socket = _WireSocket(self, session, session.__enter__())
        self._sockets.append(socket)
        self._client._ws = socket
        self._client._hashes_sent_on_ws = set()
        self._client._handshake()

    def close(self) -> None:
        for socket in self._sockets:
            socket.close()

    def execution_frames(self) -> list:
        frames = [frame for frame in self.sent if b"request_id" in frame]

        return frames

    def execute_remote_with_state(self, **kwargs):
        return self._client.execute_remote_with_state(**kwargs)


def _build_block(monkeypatch, modal_app, code: str, transport: str, executor=None):
    monkeypatch.setattr(client, "MODAL_AVAILABLE", True)
    monkeypatch.setattr(
        block_scaffolding, "WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE", "modal"
    )
    monkeypatch.setattr(client, "validate_code_in_modal", lambda *a, **k: None)
    if executor is None and transport == "http":
        executor = _InProcessHttpExecutor(modal_app)
    elif executor is None:
        executor = _InProcessWsExecutor(modal_app)
    _route_to(monkeypatch, executor)
    block_class = block_scaffolding.assembly_custom_python_block(
        block_type_name="StateBlock",
        unique_identifier=f"state-{transport}",
        manifest=BlockManifest,
        python_code=PythonCode(
            type="PythonCode",
            run_function_code=code,
            run_function_name="run",
            imports=[],
        ),
    )
    block = block_class(
        api_key="key",
        workspace_resolver=SimpleNamespace(resolve_workspace=lambda key: "test-ws"),
    )
    block._workflow_step_name = "state_step"
    block._workflow_step_type = "StateBlock"
    block._workflow_step_selector = "$steps.state_step"

    return block, executor


def _values(block, calls: int) -> list:
    values = [block.run(image=frame)["value"] for frame in range(calls)]

    return values


TRANSPORTS = ["http", "ws"]

COUNTER_CODE = (
    "def run(self, image):\n"
    "    self.calls = getattr(self, 'calls', 0) + 1\n"
    "    first = not hasattr(self, 'has_first_frame')\n"
    "    self.has_first_frame = True\n"
    "    if self.calls == 1:\n"
    "        self.scratch = 1\n"
    "    elif self.calls == 2:\n"
    "        del self.scratch\n"
    "    return {'value': [first, self.calls, hasattr(self, 'scratch')]}\n"
)


@pytest.mark.parametrize("transport", TRANSPORTS)
def test_counter_and_first_frame_flag_persist_and_deletes_propagate(
    monkeypatch, modal_app, transport
):
    block, _ = _build_block(monkeypatch, modal_app, COUNTER_CODE, transport)

    first = block.run(image=1)["value"]
    assert first == [True, 1, True]
    assert block.scratch == 1

    second = block.run(image=2)["value"]
    assert second == [False, 2, False]
    assert not hasattr(block, "scratch")

    third = block.run(image=3)["value"]
    assert third == [False, 3, False]
    assert collect_block_state(block) == {"calls": 3, "has_first_frame": True}
    assert block._api_key == "key"


HELPER_CODE = (
    "class Helper:\n"
    "    built = 0\n"
    "    def __init__(self):\n"
    "        Helper.built += 1\n"
    "        self.n = Helper.built\n"
    "def run(self, image):\n"
    "    if not hasattr(self, 'initialized'):\n"
    "        self.helper = Helper()\n"
    "        self.initialized = True\n"
    "    return {'value': self.helper.n}\n"
)


@pytest.mark.parametrize("transport", TRANSPORTS)
def test_flag_guarded_helper_is_rebuilt_every_call_with_a_warning(
    monkeypatch, modal_app, capsys, transport
):
    block, _ = _build_block(monkeypatch, modal_app, HELPER_CODE, transport)

    values = _values(block, 3)

    assert values == [1, 2, 3]
    assert collect_block_state(block) == {}
    relayed_stderr = capsys.readouterr().err
    assert "self.helper (Helper)" in relayed_stderr
    assert "convert numpy values" in relayed_stderr


REFUSED_VALUES = {
    "tuple": "(1, 2)",
    "counter": "collections.Counter(a=1)",
    "int_enum": "enum.IntEnum('Color', 'RED').RED",
    "np_float32": "np.float32(1.5)",
    "ndarray": "np.zeros(3)",
    "datetime": "datetime.datetime(2024, 1, 1)",
    "nan": "float('nan')",
    "int_over_max": "2**63",
    "int_under_min": "-(2**63) - 1",
    "non_str_key": "{1: 'a'}",
    "depth_33": "functools.reduce(lambda v, _: [v], range(32), [])",
    "self_cycle": "(lambda v: (v.append(v), v)[1])([])",
    "over_1_mib": f"'x' * {MIB}",
    "alias_expansion": "functools.reduce(lambda v, _: [v, v], range(28), [0])",
}

ACCEPTED_VALUES = {
    "depth_32": "functools.reduce(lambda v, _: [v], range(31), [])",
    "int_max": "2**63 - 1",
    "int_min": "-(2**63)",
    "nested": "{'a': [1, 2.5, None, True, 'x', {'b': []}]}",
}


def _state_value_code(expression: str) -> str:
    code = (
        "import collections\n"
        "import datetime\n"
        "import enum\n"
        "import functools\n"
        "import numpy as np\n"
        "def run(self, image):\n"
        "    self.count = getattr(self, 'count', 0) + 1\n"
        f"    self.value_under_test = {expression}\n"
        "    return {'value': self.count}\n"
    )

    return code


@pytest.mark.parametrize("transport", TRANSPORTS)
@pytest.mark.parametrize("name", sorted(REFUSED_VALUES))
def test_a_refused_value_withholds_the_whole_state_and_names_the_attribute(
    monkeypatch, modal_app, capsys, transport, name
):
    code = _state_value_code(REFUSED_VALUES[name])
    block, _ = _build_block(monkeypatch, modal_app, code, transport)

    values = _values(block, 2)

    assert values == [1, 1]
    assert collect_block_state(block) == {}
    relayed_stderr = capsys.readouterr().err
    assert "self.value_under_test (" in relayed_stderr
    assert relayed_stderr.count("not carried") == 1


@pytest.mark.parametrize("transport", TRANSPORTS)
@pytest.mark.parametrize("name", sorted(ACCEPTED_VALUES))
def test_a_boundary_value_inside_the_contract_is_carried(
    monkeypatch, modal_app, capsys, transport, name
):
    code = _state_value_code(ACCEPTED_VALUES[name])
    block, _ = _build_block(monkeypatch, modal_app, code, transport)

    values = _values(block, 2)

    assert values == [1, 2]
    assert "not carried" not in capsys.readouterr().err


def test_state_just_under_one_mib_is_carried_over_http(monkeypatch, modal_app):
    filler = MIB - len('{"count":1,"value_under_test":""}')
    code = _state_value_code(f"'x' * {filler}")
    block, executor = _build_block(monkeypatch, modal_app, code, "http")

    values = _values(block, 2)

    assert values == [1, 2]
    assert len(executor.requests[1]["block_state_json"].encode()) == MIB


def _nested_lists(depth: int) -> list:
    value: list = []
    for _ in range(depth - 1):
        value = [value]

    return value


def test_sandbox_encoder_boundaries(modal_app):
    under = {"a": "x" * (MIB - len('{"a":""}'))}
    over = {"a": "x" * (MIB - len('{"a":""}') + 1)}

    assert modal_app._encode_block_state(under)[1] is None
    assert modal_app._encode_block_state(over) == (
        "{}",
        ("", "state is over 1 MiB encoded"),
    )
    assert modal_app._encode_block_state({"a": _nested_lists(32)})[1] is None
    assert modal_app._encode_block_state({"a": _nested_lists(33)})[1] == (
        "a",
        "nested deeper than 32 levels",
    )
    assert modal_app._encode_block_state({"a": 2**63 - 1, "b": -(2**63)})[1] is None
    assert modal_app._encode_block_state({"a": 2**63})[1] is not None


def test_sandbox_refuses_alias_expansion_before_encoding(modal_app):
    value = [0]
    for _ in range(28):
        value = [value, value]

    started_at = time.perf_counter()
    encoded, problem = modal_app._encode_block_state({"count": 1, "a": value})
    elapsed = time.perf_counter() - started_at

    assert encoded == "{}"
    assert problem == ("a", "state is too large")
    assert elapsed < 1.0


def test_sandbox_warning_for_a_huge_attribute_name_is_bounded(modal_app):
    name = "a" * (2 * MIB)
    stderr_buf = modal_app.StringIO()

    modal_app._echo_block_state(
        None, {name: (1,)}, code_hash="hash", stderr_buf=stderr_buf
    )

    assert len(stderr_buf.getvalue().encode()) < 400
    assert all(
        len(str(part).encode()) < 100
        for key in modal_app._BLOCK_STATE_WARNED
        for part in key
    )


MARKER_CODE = (
    "def run(self, image):\n"
    "    seen = getattr(self, 'markers', None)\n"
    "    self.markers = [\n"
    "        {'_type': 'datetime', 'value': 'x'},\n"
    "        {'type': 'base64', 'value': 'aGk='},\n"
    "    ]\n"
    "    return {'value': seen == self.markers}\n"
)


@pytest.mark.parametrize("transport", TRANSPORTS)
def test_marker_shaped_dicts_round_trip_as_plain_dicts(
    monkeypatch, modal_app, transport
):
    block, _ = _build_block(monkeypatch, modal_app, MARKER_CODE, transport)

    values = _values(block, 2)

    assert values == [False, True]
    assert block.markers == [
        {"_type": "datetime", "value": "x"},
        {"type": "base64", "value": "aGk="},
    ]


USER_STATE_CODE = (
    "def run(self, image):\n"
    "    self.user_state = getattr(self, 'user_state', 0) + 1\n"
    "    return {'value': self.user_state}\n"
)


@pytest.mark.parametrize("transport", TRANSPORTS)
def test_an_attribute_named_user_state_persists(monkeypatch, modal_app, transport):
    block, _ = _build_block(monkeypatch, modal_app, USER_STATE_CODE, transport)

    values = _values(block, 3)

    assert values == [1, 2, 3]
    assert block.user_state == 3


class _ScaffoldedBlock:
    _usage_block_kind = "custom_python"
    _usage_block_type = "Scaffolded"

    def __init__(self):
        self._api_key = "key"
        self._workflow_step_name = "step"
        self.counter = 1

    def run(self):
        return "run"


_ENGINE_ATTRIBUTES = {"_api_key": "key", "_workflow_step_name": "step"}

REFUSED_ECHOES = {
    "usage_field": '{"counter":2,"_usage_block_kind":"x"}',
    "method": '{"counter":2,"run":"x"}',
    "dunder_class": '{"counter":2,"__class__":"x"}',
    "engine_attribute": '{"counter":2,"_api_key":"junk"}',
    "non_dict": "[1]",
    "bytes": b'{"counter":2}',
    "over_2_mib": '{"counter":"' + "x" * (2 * MIB) + '"}',
    "invalid_json": '{"counter":',
    "nan": '{"counter":NaN}',
    "inf_literal": '{"counter":1e400}',
    "int_over_max": '{"counter":9223372036854775808}',
    "int_under_min": '{"counter":-9223372036854775809}',
    "depth_33": '{"counter":' + "[" * 33 + "]" * 33 + "}",
}


@pytest.mark.parametrize("name", sorted(REFUSED_ECHOES))
def test_engine_clears_state_on_a_refused_echo(engine_warnings, name):
    block = _ScaffoldedBlock()

    apply_block_state(
        block, decode_echoed_block_state("Scaffolded", REFUSED_ECHOES[name])
    )

    assert vars(block) == _ENGINE_ATTRIBUTES
    assert block.run() == "run"
    assert _ScaffoldedBlock._usage_block_kind == "custom_python"
    assert len(engine_warnings) == 1
    assert "state reset" in engine_warnings[0]


def test_engine_bounds_the_cost_of_a_huge_integer_echo(engine_warnings):
    block = _ScaffoldedBlock()
    echo = '{"counter":' + "1" * 1_000_000 + "}"

    started_at = time.perf_counter()
    apply_block_state(block, decode_echoed_block_state("Scaffolded", echo))
    elapsed = time.perf_counter() - started_at

    assert elapsed < 1.0
    assert vars(block) == _ENGINE_ATTRIBUTES
    assert len(engine_warnings) == 1


def test_engine_refuses_a_non_str_attribute_name(engine_warnings):
    block = _ScaffoldedBlock()

    apply_block_state(block, {1: "a", "counter": 2})

    assert vars(block) == _ENGINE_ATTRIBUTES
    assert "not a string" in engine_warnings[0]


@pytest.mark.parametrize(
    "echo, expected",
    [
        ('{"counter":2,"_calls":3}', {"counter": 2, "_calls": 3}),
        ('{"mro":1}', {"mro": 1}),
        (
            '{"a":9223372036854775807,"b":-9223372036854775808}',
            {"a": 2**63 - 1, "b": -(2**63)},
        ),
        ('{"a":' + "[" * 32 + "]" * 32 + "}", {"a": _nested_lists(32)}),
    ],
)
def test_engine_applies_an_echo_inside_the_contract(engine_warnings, echo, expected):
    block = _ScaffoldedBlock()

    apply_block_state(block, decode_echoed_block_state("Scaffolded", echo))

    assert vars(block) == {**_ENGINE_ATTRIBUTES, **expected}
    assert engine_warnings == []


def test_engine_warning_for_a_huge_attribute_name_is_bounded(engine_warnings):
    block = _ScaffoldedBlock()
    name = "__" + "a" * (2 * MIB) + "__"

    apply_block_state(block, {name: 1})

    assert vars(block) == _ENGINE_ATTRIBUTES
    assert len(engine_warnings[0].encode()) < 300
    (key,) = engine_state._WARNED
    assert sum(len(part.encode()) for part in key) < 300


def test_engine_encoder_boundaries(engine_warnings):
    under = {"a": "x" * (MIB - len('{"a":""}'))}
    over = {"a": "x" * (MIB - len('{"a":""}') + 1)}

    assert json.loads(encode_block_state("B", under)) == under
    assert encode_block_state("B", over) == "{}"
    assert encode_block_state("B", {"a": float("nan")}) == "{}"
    assert encode_block_state("B", {}) == "{}"
    assert len(engine_warnings) == 2


@pytest.mark.parametrize("transport", TRANSPORTS)
def test_old_server_without_echo_clears_state_and_returns_the_result(
    monkeypatch, modal_app, engine_warnings, transport
):
    block, executor = _build_block(monkeypatch, modal_app, USER_STATE_CODE, transport)

    values = []
    for drop_echo in (False, False, True, False):
        executor.drop_echo = drop_echo
        values.append(block.run(image=1)["value"])

    assert values == [1, 2, 3, 1]
    assert any("server did not return state" in w for w in engine_warnings)


@pytest.mark.parametrize("transport", TRANSPORTS)
def test_old_client_without_state_gets_no_echo(monkeypatch, modal_app, transport):
    _, executor = _build_block(monkeypatch, modal_app, USER_STATE_CODE, transport)

    executor.execute_remote(
        block_type_name="StateBlock",
        python_code=SimpleNamespace(
            run_function_code=USER_STATE_CODE, run_function_name="run", imports=[]
        ),
        inputs={"image": 1},
        workspace_id="test-ws",
        workflow_context={},
    )

    assert "block_state_json" not in executor.requests[-1]
    assert executor.responses[-1]["success"] is True
    assert "block_state" not in executor.responses[-1]


@pytest.mark.parametrize("transport", TRANSPORTS)
def test_fresh_block_sends_an_empty_state_object(monkeypatch, modal_app, transport):
    block, executor = _build_block(monkeypatch, modal_app, USER_STATE_CODE, transport)

    block.run(image=1)

    assert executor.requests[0]["block_state_json"] == "{}"
    assert json.loads(executor.responses[0]["block_state"]) == {"user_state": 1}


FAILING_CODE = (
    "def run(self, image):\n"
    "    self.count = getattr(self, 'count', 0) + 1\n"
    "    if image == 'fail':\n"
    "        raise ValueError('boom')\n"
    "    return {'value': self.count}\n"
)


@pytest.mark.parametrize("transport", TRANSPORTS)
def test_user_code_failure_keeps_the_previous_state(monkeypatch, modal_app, transport):
    block, _ = _build_block(monkeypatch, modal_app, FAILING_CODE, transport)

    assert block.run(image=1)["value"] == 1
    assert block.run(image=2)["value"] == 2
    with pytest.raises(Exception, match="boom"):
        block.run(image="fail")
    assert block.count == 2
    assert block.run(image=3)["value"] == 3


def test_unknown_code_hash_resend_carries_the_same_state(monkeypatch, modal_app):
    block, executor = _build_block(monkeypatch, modal_app, USER_STATE_CODE, "http")
    block.run(image=1)
    executor.server._code_namespaces.clear()

    value = block.run(image=2)["value"]

    first_try, resend = executor.requests[1], executor.requests[2]
    assert "code_str" not in first_try
    assert executor.responses[1]["error_type"] == "UnknownCodeHash"
    assert "code_str" in resend
    assert first_try["block_state_json"] == resend["block_state_json"]
    assert json.loads(resend["block_state_json"]) == {"user_state": 1}
    assert value == 2


UNINSPECTABLE_CODE = (
    "class Meta(type):\n"
    "    @property\n"
    "    def __name__(cls):\n"
    "        raise RuntimeError('no name')\n"
    "class Odd(metaclass=Meta):\n"
    "    pass\n"
    "def run(self, image):\n"
    "    self.count = getattr(self, 'count', 0) + 1\n"
    "    self.odd = Odd()\n"
    "    return {'value': self.count}\n"
)


@pytest.mark.parametrize("transport", TRANSPORTS)
def test_state_that_cannot_be_inspected_does_not_fail_a_successful_call(
    monkeypatch, modal_app, capsys, transport
):
    block, executor = _build_block(
        monkeypatch, modal_app, UNINSPECTABLE_CODE, transport
    )

    values = _values(block, 2)

    assert values == [1, 1]
    assert all(response["success"] is True for response in executor.responses)
    assert collect_block_state(block) == {}
    assert "state could not be inspected" in capsys.readouterr().err


CLEARING_CODE = (
    "def run(self, image):\n"
    "    self.__dict__.clear()\n"
    "    context = self.get_workflow_context()\n"
    "    self.seen = 1\n"
    "    return {'value': context}\n"
)


@pytest.mark.parametrize("transport", TRANSPORTS)
def test_workflow_context_survives_user_code_clearing_the_instance_dict(
    monkeypatch, modal_app, transport
):
    block, executor = _build_block(monkeypatch, modal_app, CLEARING_CODE, transport)

    value = block.run(image=1)["value"]

    assert value == executor.requests[0]["workflow_context"]
    assert json.loads(executor.responses[0]["block_state"]) == {"seen": 1}


@pytest.mark.parametrize("transport", TRANSPORTS)
def test_old_client_can_clear_the_instance_dict_and_read_the_workflow_context(
    monkeypatch, modal_app, transport
):
    _, executor = _build_block(monkeypatch, modal_app, CLEARING_CODE, transport)

    result = executor.execute_remote(
        block_type_name="StateBlock",
        python_code=SimpleNamespace(
            run_function_code=CLEARING_CODE, run_function_name="run", imports=[]
        ),
        inputs={"image": 1},
        workspace_id="test-ws",
        workflow_context={"k": 1},
    )

    assert result == {"value": {"k": 1}}
    assert "block_state" not in executor.responses[-1]


def test_sandbox_refuses_too_many_top_level_attributes_before_encoding(
    monkeypatch, modal_app
):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("json.dumps must not run for over-budget state")

    monkeypatch.setattr(modal_app.json, "dumps", fail_if_called)
    state = {f"{index:x}": 1 for index in range(200_001)}

    started_at = time.perf_counter()
    encoded, problem = modal_app._encode_block_state(state)
    elapsed = time.perf_counter() - started_at

    assert encoded == "{}"
    assert problem is not None
    assert problem[1] == "state is too large"
    assert elapsed < 1.0


def _pooled_ws_executor(modal_app):
    inner = _InProcessWsExecutor(modal_app)
    pooled = client.PooledWebSocketModalExecutor("test-ws")
    pooled._executors = [inner._client]
    pooled._active_counts = [0]

    return pooled, inner


@pytest.mark.parametrize("transport", TRANSPORTS)
def test_two_blocks_sharing_one_executor_keep_separate_state(
    monkeypatch, modal_app, transport
):
    if transport == "http":
        shared = _InProcessHttpExecutor(modal_app)
    else:
        shared, _ = _pooled_ws_executor(modal_app)
    first, _ = _build_block(
        monkeypatch, modal_app, USER_STATE_CODE, transport, executor=shared
    )
    second, _ = _build_block(
        monkeypatch, modal_app, USER_STATE_CODE, transport, executor=shared
    )

    values = []
    for block in (first, second, first, second, first):
        values.append(block.run(image=1)["value"])

    assert values == [1, 1, 2, 2, 3]
    assert first.user_state == 3
    assert second.user_state == 2


def test_a_block_keeps_its_state_when_a_different_pooled_executor_serves_it(
    monkeypatch, modal_app
):
    first_pool, first_inner = _pooled_ws_executor(modal_app)
    second_pool, second_inner = _pooled_ws_executor(modal_app)
    block, _ = _build_block(
        monkeypatch, modal_app, USER_STATE_CODE, "ws", executor=first_pool
    )

    values = _values(block, 2)
    _route_to(monkeypatch, second_pool)
    values += _values(block, 2)
    _route_to(monkeypatch, first_pool)
    values += _values(block, 1)

    assert values == [1, 2, 3, 4, 5]
    assert json.loads(second_inner.requests[0]["block_state_json"]) == {"user_state": 2}
    assert len(first_inner.requests) == 3
    assert len(second_inner.requests) == 2


def test_websocket_code_hash_retry_frame_carries_the_same_state(monkeypatch, modal_app):
    block, executor = _build_block(monkeypatch, modal_app, USER_STATE_CODE, "ws")
    block.run(image=1)
    executor.server._code_namespaces.clear()

    value = block.run(image=2)["value"]

    first_try, retry = executor.requests[1], executor.requests[2]
    assert "code_str" not in first_try
    assert executor.responses[1]["error_type"] == "UnknownCodeHash"
    assert "code_str" in retry
    assert first_try["request_id"] != retry["request_id"]
    assert first_try["block_state_json"] == retry["block_state_json"]
    assert json.loads(retry["block_state_json"]) == {"user_state": 1}
    assert value == 2


RUN_COUNTING_CODE = (
    "RUNS = []\n"
    "def run(self, image):\n"
    "    RUNS.append(image)\n"
    "    self.user_state = getattr(self, 'user_state', 0) + 1\n"
    "    return {'value': [self.user_state, len(RUNS)]}\n"
)


def test_resend_after_a_dropped_connection_is_answered_from_the_response_cache(
    monkeypatch, modal_app
):
    executor = _WireWsExecutor(modal_app)
    block, _ = _build_block(
        monkeypatch, modal_app, RUN_COUNTING_CODE, "ws", executor=executor
    )
    first = block.run(image=1)["value"]
    executor.lose_next_response = True

    resent = block.run(image=2)["value"]
    after = block.run(image=3)["value"]

    original_frame, resent_frame = executor.execution_frames()[1:3]
    request = msgpack.unpackb(original_frame, raw=False)
    assert original_frame == resent_frame
    assert json.loads(request["block_state_json"]) == {"user_state": 1}
    assert len(executor.lost) == 1
    assert executor.received[-2] == executor.lost[0]
    cached_response = executor.server._ws_response_cache.get(request["request_id"])
    assert cached_response == executor.lost[0]
    assert json.loads(msgpack.unpackb(cached_response, raw=False)["block_state"]) == {
        "user_state": 2
    }
    assert [first, resent, after] == [[1, 1], [2, 2], [3, 3]]
    assert block.user_state == 3


def test_a_response_for_another_request_is_rejected_and_its_state_never_applied(
    monkeypatch, modal_app
):
    executor = _WireWsExecutor(modal_app)
    block, _ = _build_block(
        monkeypatch, modal_app, RUN_COUNTING_CODE, "ws", executor=executor
    )
    _values(block, 2)
    executor.foreign_request_id = "someone-elses-request"

    with pytest.raises(Exception, match="did not match the request in flight"):
        block.run(image=3)

    assert block.user_state == 2
    assert collect_block_state(block) == {"user_state": 2}
    executor.foreign_request_id = None
    assert block.run(image=4)["value"][0] == 3


def test_websocket_state_just_under_one_mib_is_carried_in_chunked_frames(
    monkeypatch, modal_app
):
    filler = MIB - len('{"count":1,"value_under_test":""}')
    executor = _WireWsExecutor(modal_app)
    block, _ = _build_block(
        monkeypatch,
        modal_app,
        _state_value_code(f"'x' * {filler}"),
        "ws",
        executor=executor,
    )

    values = _values(block, 3)

    chunk_control_prefix = msgpack.packb({"_chunked": 1}, use_bin_type=True)[:-1]
    sent_controls = [f for f in executor.sent if f.startswith(chunk_control_prefix)]
    received_controls = [
        f for f in executor.received if f.startswith(chunk_control_prefix)
    ]
    assert values == [1, 2, 3]
    assert len(sent_controls) == 2
    assert len(received_controls) == 3
    assert max(len(frame) for frame in executor.sent) == 1024 * 1024
    assert len(block.value_under_test) == filler


NO_OP_CODE = "def run(self, image):\n    return {'value': 1}\n"


class _ClockedTime:
    def __init__(self, now: list):
        self._now = now

    def perf_counter(self) -> float:
        return self._now[0]

    def __getattr__(self, name):
        return getattr(time, name)


@pytest.mark.parametrize("transport", TRANSPORTS)
def test_reported_runtime_excludes_the_sandbox_state_decode_and_encode(
    monkeypatch, modal_app, transport
):
    now = [0.0]
    real_encode = modal_app._encode_block_state
    real_decode = modal_app._decode_block_state_json

    def slow_encode(state):
        now[0] += 0.25

        return real_encode(state)

    def slow_decode(block_state_json):
        now[0] += 0.5

        return real_decode(block_state_json)

    monkeypatch.setattr(modal_app, "time", _ClockedTime(now))
    monkeypatch.setattr(modal_app, "_encode_block_state", slow_encode)
    monkeypatch.setattr(modal_app, "_decode_block_state_json", slow_decode)
    block, executor = _build_block(monkeypatch, modal_app, NO_OP_CODE, transport)

    block.run(image=1)

    assert now[0] == 0.75
    assert executor.responses[0]["execution_time_seconds"] == 0.0


@pytest.mark.parametrize("transport", TRANSPORTS)
def test_counter_over_new_and_old_servers_restarts_on_every_old_one(
    monkeypatch, modal_app, engine_warnings, transport
):
    block, executor = _build_block(monkeypatch, modal_app, USER_STATE_CODE, transport)

    values = []
    for old_server in (False, False, True, True, False, False):
        executor.old_server = old_server
        values.append(block.run(image=1)["value"])

    assert values == [1, 2, 1, 1, 1, 2]
    assert json.loads(executor.requests[2]["block_state_json"]) == {"user_state": 2}
    assert all("block_state" not in executor.responses[i] for i in (2, 3))
    assert any("server did not return state" in w for w in engine_warnings)


def test_engine_warning_set_stays_capped_after_many_distinct_refused_names(
    engine_warnings,
):
    block = _ScaffoldedBlock()

    for index in range(5000):
        apply_block_state(block, {f"__refused_{index}__": 1})

    assert engine_state._MAX_WARNINGS < 5000
    assert 0 < len(engine_state._WARNED) <= engine_state._MAX_WARNINGS
    assert len(engine_warnings) == 5000


def test_sandbox_warning_set_stays_capped_after_many_distinct_refused_names(
    modal_app,
):
    stderr_buf = modal_app.StringIO()

    for index in range(5000):
        modal_app._echo_block_state(
            None, {f"refused_{index}": (1,)}, code_hash="hash", stderr_buf=stderr_buf
        )

    assert modal_app._BLOCK_STATE_MAX_WARNINGS < 5000
    assert 0 < len(modal_app._BLOCK_STATE_WARNED) <= modal_app._BLOCK_STATE_MAX_WARNINGS
    assert stderr_buf.getvalue().count("not carried") == 5000
