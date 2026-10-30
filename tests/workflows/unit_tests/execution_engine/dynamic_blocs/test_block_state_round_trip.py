"""Custom Python ``self`` state survives remote (Modal) execution.

Under ``WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE=modal`` the sandbox used to build
a throwaway ``self`` for every call, so a block that wrote
``self.has_first_frame = True`` on frame 1 saw a bare object again on frame 2.
The engine-side block instance lives for the whole run, so it now owns the
state: every call ships the block's user attributes to the sandbox, which
seeds ``self`` with them and echoes the mutated set back.

The server half of these tests loads the real ``modal/modal_app.py`` through
the shared harness in ``conftest.py``; the client half drives the real
executors with the network call stubbed, and the end-to-end test wires the
engine-side block through the real HTTP handler in-process.
"""

import asyncio
import io
import json
import weakref
from contextlib import contextmanager
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import msgpack
import numpy as np
import pytest
import supervision as sv
from roboflow_workflows.errors import DynamicBlockError
from roboflow_workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)
from roboflow_workflows.execution_engine.v1.dynamic_blocks import block_scaffolding
from roboflow_workflows.execution_engine.v1.dynamic_blocks import (
    modal_executor as client,
)
from roboflow_workflows.execution_engine.v1.dynamic_blocks.block_state import (
    BLOCK_STATE_MAX_ATTRIBUTES,
    BLOCK_STATE_MAX_BYTES,
    apply_block_state,
    collect_block_state,
    partition_block_state,
)
from roboflow_workflows.execution_engine.v1.dynamic_blocks.modal_executor import (
    deserialize_for_modal_remote_execution,
    serialize_for_modal_remote_execution,
    serialize_inputs_for_msgpack,
)

from .conftest import build_ws_app


@pytest.fixture(autouse=True)
def _forget_logged_block_state_warnings():
    # The client logs each block state warning once per process.
    client._WARNED_BLOCK_STATE_MESSAGES.clear()
    yield
    client._WARNED_BLOCK_STATE_MESSAGES.clear()


FIRST_FRAME_CODE = (
    "def run(self, image):\n"
    "    if not hasattr(self, '_ffc_calls'):\n"
    "        self._ffc_calls = 0\n"
    "    self._ffc_calls += 1\n"
    "    if hasattr(self, 'has_first_frame'):\n"
    "        is_first = False\n"
    "    else:\n"
    "        self.has_first_frame = True\n"
    "        is_first = True\n"
    "    return {'is_first_frame': is_first, 'calls': self._ffc_calls}\n"
)


# --------------------------------------------------------------------------- #
# engine side: which attributes are state, and how they are written back
# --------------------------------------------------------------------------- #


def _engine_block(**user_attributes):
    block = SimpleNamespace(
        _api_key="key",
        _workspace_resolver=object(),
        _execution_observer=object(),
        _manifest_description=None,
        _init_results={},
        _workflow_step_name="first_frame",
        _workflow_step_type="FirstFrameCheck",
        _workflow_step_selector="$steps.first_frame",
    )
    for name, value in user_attributes.items():
        setattr(block, name, value)
    return block


def test_collect_block_state_keeps_user_attributes_including_underscored_ones():
    # Lou's block counts calls in ``self._ffc_calls``: a leading underscore is
    # not an engine marker, only the engine's own attribute names are.
    block = _engine_block(has_first_frame=True, _ffc_calls=3)

    state = collect_block_state(block)

    assert state == {"has_first_frame": True, "_ffc_calls": 3}


def test_collect_block_state_is_empty_for_a_fresh_block():
    assert collect_block_state(_engine_block()) == {}


def test_apply_block_state_replaces_user_attributes_and_keeps_engine_ones():
    block = _engine_block(stale="gone", counter=1)

    apply_block_state(block, {"counter": 2, "fresh": "new"})

    assert block.counter == 2
    assert block.fresh == "new"
    assert not hasattr(block, "stale"), "attributes the sandbox deleted must go"
    assert block._workflow_step_name == "first_frame"
    assert block._api_key == "key"


# --------------------------------------------------------------------------- #
# server side, HTTP transport
# --------------------------------------------------------------------------- #


class _FakeRequest:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode()
        self.headers = {}

    async def body(self) -> bytes:
        return self._body


def _http_executor(modal_app):
    executor = modal_app.Executor.__new__(modal_app.Executor)
    executor._code_namespaces = {}
    executor._shared_globals = {}
    return executor


def _run_http(executor, code: str, inputs: dict, block_state=None) -> dict:
    payload = {
        "code_str": code,
        "imports": [],
        "run_function_name": "run",
        "inputs_json": serialize_for_modal_remote_execution(inputs),
    }
    if block_state is not None:
        payload["block_state_json"] = serialize_for_modal_remote_execution(block_state)
    response = asyncio.run(executor.execute_block(_FakeRequest(payload)))
    return response


def test_http_server_seeds_self_from_block_state_and_echoes_the_mutated_state(
    modal_app,
):
    executor = _http_executor(modal_app)

    first = _run_http(executor, FIRST_FRAME_CODE, {"image": 1}, block_state={})
    assert first["success"] is True, first.get("error")
    assert json.loads(first["result"]) == {"is_first_frame": True, "calls": 1}
    state = deserialize_for_modal_remote_execution(first["block_state"])
    assert state == {"has_first_frame": True, "_ffc_calls": 1}

    second = _run_http(executor, FIRST_FRAME_CODE, {"image": 2}, block_state=state)
    assert second["success"] is True, second.get("error")
    assert json.loads(second["result"]) == {"is_first_frame": False, "calls": 2}
    assert deserialize_for_modal_remote_execution(second["block_state"]) == {
        "has_first_frame": True,
        "_ffc_calls": 2,
    }


def test_http_server_omits_block_state_for_a_client_that_sent_none(modal_app):
    # Pre-upgrade clients never send state and must get the old response shape.
    executor = _http_executor(modal_app)

    response = _run_http(executor, FIRST_FRAME_CODE, {"image": 1})

    assert response["success"] is True, response.get("error")
    assert "block_state" not in response


TRACKER_CODE = (
    "class Tracker:\n"
    "    pass\n"
    "def run(self, image):\n"
    "    self.tracker = Tracker()\n"
    "    self.count = getattr(self, 'count', 0) + 1\n"
    "    return {'ok': True}\n"
)

OVERSIZED_CODE = (
    "import numpy as np\n"
    "def run(self, image):\n"
    f"    self.buffer = np.zeros(({2 * BLOCK_STATE_MAX_BYTES},), dtype=np.uint8)\n"
    "    self.count = getattr(self, 'count', 0) + 1\n"
    "    return {'ok': True}\n"
)


def test_http_server_drops_a_non_transportable_self_attribute_and_warns(modal_app):
    executor = _http_executor(modal_app)

    response = _run_http(executor, TRACKER_CODE, {"image": 1}, block_state={})

    assert response["success"] is True
    assert deserialize_for_modal_remote_execution(response["block_state"]) == {}
    assert "self.tracker" in response["stderr"]
    assert "Tracker" in response["stderr"]
    assert "not carried" in response["stderr"]


def test_http_server_warns_about_a_dropped_attribute_once_per_container(modal_app):
    executor = _http_executor(modal_app)

    first = _run_http(executor, TRACKER_CODE, {"image": 1}, block_state={})
    second = _run_http(executor, TRACKER_CODE, {"image": 1}, block_state={"count": 1})

    assert "self.tracker" in first["stderr"]
    assert second["success"] is True
    assert "self.tracker" not in (second["stderr"] or "")
    assert deserialize_for_modal_remote_execution(second["block_state"]) == {}


def test_http_server_drops_an_attribute_over_the_size_cap_and_withholds_the_rest(
    modal_app,
):
    executor = _http_executor(modal_app)

    response = _run_http(executor, OVERSIZED_CODE, {"image": 1}, block_state={})

    assert response["success"] is True
    assert deserialize_for_modal_remote_execution(response["block_state"]) == {}
    assert "self.buffer" in response["stderr"]
    assert "limit" in response["stderr"]


def test_http_server_round_trips_an_ndarray_attribute(modal_app):
    executor = _http_executor(modal_app)
    code = (
        "import numpy as np\n"
        "def run(self, image):\n"
        "    self.history = np.append(getattr(self, 'history', np.array([])), image)\n"
        "    return {'n': int(self.history.size)}\n"
    )

    first = _run_http(executor, code, {"image": 1.0}, block_state={})
    state = deserialize_for_modal_remote_execution(first["block_state"])
    second = _run_http(executor, code, {"image": 2.0}, block_state=state)

    assert json.loads(second["result"]) == {"n": 2}
    history = deserialize_for_modal_remote_execution(second["block_state"])["history"]
    assert isinstance(history, np.ndarray)
    assert history.tolist() == [1.0, 2.0]


# --------------------------------------------------------------------------- #
# server side, websocket transport
# --------------------------------------------------------------------------- #


def _run_ws(executor, modal_app, code: str, inputs: dict, block_state=None) -> dict:
    # Both directions go through msgpack bytes, as on the real wire: an
    # in-memory dict would keep types (an IntEnum, a tuple) the wire loses.
    kwargs = {}
    if block_state is not None:
        kwargs["block_state"] = modal_app.Executor._deserialize_msgpack_inputs(
            _through_msgpack(serialize_inputs_for_msgpack(block_state))
        )
    response = modal_app.Executor._run_user_code_ws(
        executor,
        code,
        [],
        "run",
        modal_app.Executor._deserialize_msgpack_inputs(
            _through_msgpack(serialize_inputs_for_msgpack(inputs))
        ),
        "",
        {},
        **kwargs,
    )
    return _through_msgpack(response)


def _through_msgpack(value):
    unpacked = msgpack.unpackb(msgpack.packb(value, use_bin_type=True), raw=False)
    return unpacked


def test_ws_server_seeds_self_from_block_state_and_echoes_the_mutated_state(
    modal_app,
):
    executor = _http_executor(modal_app)

    first = _run_ws(executor, modal_app, FIRST_FRAME_CODE, {"image": 1}, {})
    assert first["success"] is True, first.get("error")
    assert first["result"] == {"is_first_frame": True, "calls": 1}
    state = client._deserialize_msgpack_result(first["block_state"])
    assert state == {"has_first_frame": True, "_ffc_calls": 1}

    second = _run_ws(executor, modal_app, FIRST_FRAME_CODE, {"image": 2}, state)
    assert second["result"] == {"is_first_frame": False, "calls": 2}
    assert client._deserialize_msgpack_result(second["block_state"]) == {
        "has_first_frame": True,
        "_ffc_calls": 2,
    }


def test_ws_server_omits_block_state_for_a_client_that_sent_none(modal_app):
    executor = _http_executor(modal_app)

    response = _run_ws(executor, modal_app, FIRST_FRAME_CODE, {"image": 1})

    assert response["success"] is True, response.get("error")
    assert "block_state" not in response


def test_ws_server_drops_a_non_transportable_self_attribute_and_warns(modal_app):
    executor = _http_executor(modal_app)

    response = _run_ws(executor, modal_app, TRACKER_CODE, {"image": 1}, {})

    assert response["success"] is True
    assert response["block_state"] == {}
    assert "self.tracker" in response["stderr"]
    assert "Tracker" in response["stderr"]
    assert "not carried" in response["stderr"]


def test_ws_server_warns_about_a_dropped_attribute_once_per_container(modal_app):
    executor = _http_executor(modal_app)

    first = _run_ws(executor, modal_app, TRACKER_CODE, {"image": 1}, {})
    second = _run_ws(executor, modal_app, TRACKER_CODE, {"image": 1}, {"count": 1})

    assert "self.tracker" in first["stderr"]
    assert second["success"] is True
    assert "self.tracker" not in (second["stderr"] or "")
    assert second["block_state"] == {}


def test_ws_server_drops_an_attribute_over_the_size_cap_and_withholds_the_rest(
    modal_app,
):
    executor = _http_executor(modal_app)

    response = _run_ws(executor, modal_app, OVERSIZED_CODE, {"image": 1}, {})

    assert response["success"] is True
    assert response["block_state"] == {}
    assert "self.buffer" in response["stderr"]
    assert "limit" in response["stderr"]


def test_ws_frame_block_state_is_read_by_the_server(modal_app):
    # The wire key the client packs must be the key the server reads, all the
    # way through the real websocket handler.
    from fastapi.testclient import TestClient

    seen = {}

    def run_user_code(
        self,
        code_str,
        imports,
        run_function_name,
        inputs,
        client_code_hash,
        workflow_context,
        block_state=None,
    ):
        seen["block_state"] = block_state
        return {"success": True, "result": {}, "block_state": {"n": 2}}

    _, app = build_ws_app(modal_app, run_user_code)
    frame = client.WebSocketModalExecutor._build_ws_frame(
        python_code=SimpleNamespace(
            run_function_code="def run(self, x):\n    return x\n",
            run_function_name="run",
            imports=[],
        ),
        packed_inputs={"x": 1},
        code_hash="hash-1",
        send_full_code=True,
        msgpack=msgpack,
        workflow_context={},
        request_id="req-1",
        block_state={"n": 1},
    )

    with TestClient(app).websocket_connect("/ws") as ws:
        ws.send_bytes(frame)
        response = msgpack.unpackb(ws.receive_bytes(), raw=False)

    assert seen == {"block_state": {"n": 1}}
    assert response["block_state"] == {"n": 2}


# --------------------------------------------------------------------------- #
# client side executors
# --------------------------------------------------------------------------- #


def _python_code(code: str = FIRST_FRAME_CODE):
    return SimpleNamespace(
        run_function_code=code,
        run_function_name="run",
        imports=[],
    )


def _ws_executor():
    executor = client.WebSocketModalExecutor(workspace_id="test-ws")
    executor._server = client._ServerInfo(
        proto=2, idle_timeout=10.0, container_id="container-1"
    )
    return executor


def test_ws_client_sends_block_state_and_returns_the_echoed_state(monkeypatch):
    monkeypatch.setattr(client, "MODAL_AVAILABLE", True)
    executor = _ws_executor()
    sent = []

    def fake_send_recv(frame_bytes, workspace, request_id=None):
        sent.append(msgpack.unpackb(frame_bytes, raw=False))
        return msgpack.packb(
            {
                "success": True,
                "result": {"is_first_frame": False},
                "block_state": {"has_first_frame": True, "_ffc_calls": 2},
                "request_id": request_id,
            },
            use_bin_type=True,
        )

    executor._send_recv_with_retry = fake_send_recv

    result, state = executor.execute_remote_with_state(
        block_type_name="FirstFrameCheck",
        python_code=_python_code(),
        inputs={"image": 1},
        workspace_id="test-ws",
        workflow_context={},
        block_state={"has_first_frame": True, "_ffc_calls": 1},
    )

    assert sent[0]["block_state"] == {"has_first_frame": True, "_ffc_calls": 1}
    assert result == {"is_first_frame": False}
    assert state == {"has_first_frame": True, "_ffc_calls": 2}


def test_ws_client_clears_state_with_a_warning_when_the_server_drops_it(monkeypatch):
    # A pre-upgrade container (a mixed fleet, or a rollback mid-stream)
    # ignores ``block_state``. Failing would fail every later call of the
    # stream for good, since the engine instance keeps its state; the call
    # goes on with an empty state instead, as before state was carried.
    monkeypatch.setattr(client, "MODAL_AVAILABLE", True)
    warned = _capture_client_warnings(monkeypatch)
    executor = _ws_executor()
    executor._send_recv_with_retry = lambda frame_bytes, workspace, request_id=None: (
        msgpack.packb(
            {"success": True, "result": {}, "request_id": request_id},
            use_bin_type=True,
        )
    )

    _, state = executor.execute_remote_with_state(
        block_type_name="FirstFrameCheck",
        python_code=_python_code(),
        inputs={"image": 1},
        workspace_id="test-ws",
        workflow_context={},
        block_state={"has_first_frame": True},
    )

    assert state == {}
    assert any("older" in message and "reset" in message for message in warned)


def test_ws_client_execute_remote_without_state_sends_no_state_key(monkeypatch):
    monkeypatch.setattr(client, "MODAL_AVAILABLE", True)
    executor = _ws_executor()
    sent = []

    def fake_send_recv(frame_bytes, workspace, request_id=None):
        sent.append(msgpack.unpackb(frame_bytes, raw=False))
        return msgpack.packb(
            {"success": True, "result": {"ok": 1}, "request_id": request_id},
            use_bin_type=True,
        )

    executor._send_recv_with_retry = fake_send_recv

    result = executor.execute_remote(
        block_type_name="FirstFrameCheck",
        python_code=_python_code(),
        inputs={"image": 1},
        workspace_id="test-ws",
        workflow_context={},
    )

    assert result == {"ok": 1}
    assert "block_state" not in sent[0]


def test_http_client_sends_block_state_and_returns_the_echoed_state(monkeypatch):
    monkeypatch.setattr(client, "MODAL_AVAILABLE", True)
    executor = client.ModalExecutor(workspace_id="test-ws")
    executor._get_endpoint_url = lambda workspace: "http://modal.test/execute"
    sent = []

    def fake_post(
        endpoint_url,
        python_code,
        inputs_json,
        code_hash,
        send_full_code,
        workflow_context,
        block_state_json=None,
    ):
        sent.append(block_state_json)
        return {
            "success": True,
            "result": json.dumps({"is_first_frame": False}),
            "block_state": serialize_for_modal_remote_execution(
                {"has_first_frame": True, "_ffc_calls": 2}
            ),
        }

    executor._post_execute = fake_post

    result, state = executor.execute_remote_with_state(
        block_type_name="FirstFrameCheck",
        python_code=_python_code(),
        inputs={"image": 1},
        workspace_id="test-ws",
        workflow_context={},
        block_state={"has_first_frame": True, "_ffc_calls": 1},
    )

    assert json.loads(sent[0]) == {"has_first_frame": True, "_ffc_calls": 1}
    assert result == {"is_first_frame": False}
    assert state == {"has_first_frame": True, "_ffc_calls": 2}


def test_http_client_clears_state_with_a_warning_when_the_server_drops_it(monkeypatch):
    monkeypatch.setattr(client, "MODAL_AVAILABLE", True)
    warned = _capture_client_warnings(monkeypatch)
    executor = client.ModalExecutor(workspace_id="test-ws")
    executor._get_endpoint_url = lambda workspace: "http://modal.test/execute"
    executor._post_execute = lambda **kwargs: {"success": True, "result": "{}"}

    _, state = executor.execute_remote_with_state(
        block_type_name="FirstFrameCheck",
        python_code=_python_code(),
        inputs={"image": 1},
        workspace_id="test-ws",
        workflow_context={},
        block_state={"has_first_frame": True},
    )

    assert state == {}
    assert any("older" in message and "reset" in message for message in warned)


def _serialize_block_state_json(block_state, *, block_type_name):
    """What ``ModalExecutor._execute_http`` sends as ``block_state_json``."""
    carried_state = client._carry_block_state(
        block_state,
        block_type_name=block_type_name,
        workflow_context=None,
        measure=client._json_state_size,
    )
    block_state_json = serialize_for_modal_remote_execution(carried_state)
    return block_state_json


def _capture_client_warnings(monkeypatch) -> list:
    warned = []
    monkeypatch.setattr(
        client.logger,
        "warning",
        lambda message, *args, **kwargs: warned.append(
            message % args if args else message
        ),
    )
    return warned


def test_ws_client_drops_state_over_the_size_cap_with_a_warning(monkeypatch):
    monkeypatch.setattr(client, "MODAL_AVAILABLE", True)
    warned = _capture_client_warnings(monkeypatch)
    executor = _ws_executor()
    sent = []

    def fake_send_recv(frame_bytes, workspace, request_id=None):
        sent.append(msgpack.unpackb(frame_bytes, raw=False))
        return msgpack.packb(
            {
                "success": True,
                "result": {"is_first_frame": True},
                "block_state": {"count": 1},
                "request_id": request_id,
            },
            use_bin_type=True,
        )

    executor._send_recv_with_retry = fake_send_recv
    oversized = {
        "buffer": np.zeros((2 * BLOCK_STATE_MAX_BYTES,), dtype=np.uint8),
        "count": 1,
    }

    executor.execute_remote_with_state(
        block_type_name="FirstFrameCheck",
        python_code=_python_code(),
        inputs={"image": 1},
        workspace_id="test-ws",
        workflow_context={},
        block_state=oversized,
    )

    assert sent[0]["block_state"] == {}
    assert any("buffer" in message and "limit" in message for message in warned)
    assert any("withheld" in message and "self.count" in message for message in warned)


def test_http_client_drops_a_non_transportable_attribute_with_a_warning(monkeypatch):
    warned = _capture_client_warnings(monkeypatch)

    block_state_json = _serialize_block_state_json(
        {"tracker": object(), "count": 1}, block_type_name="FirstFrameCheck"
    )

    assert json.loads(block_state_json) == {}
    assert any("tracker" in message and "object" in message for message in warned)


def test_pooled_ws_executor_delegates_execute_remote_with_state(monkeypatch):
    monkeypatch.setattr(client, "MODAL_AVAILABLE", True)
    pooled = client.PooledWebSocketModalExecutor(workspace_id="test-ws")
    inner = pooled._executors[0]
    inner._server = client._ServerInfo(
        proto=2, idle_timeout=10.0, container_id="container-1"
    )
    inner._send_recv_with_retry = lambda frame_bytes, workspace, request_id=None: (
        msgpack.packb(
            {
                "success": True,
                "result": {},
                "block_state": {"n": 1},
                "request_id": request_id,
            },
            use_bin_type=True,
        )
    )

    _, state = pooled.execute_remote_with_state(
        block_type_name="B",
        python_code=_python_code(),
        inputs={},
        workflow_context={},
        block_state={},
    )

    assert state == {"n": 1}


# --------------------------------------------------------------------------- #
# end to end: engine-side block instance through the real HTTP handler
# --------------------------------------------------------------------------- #


class _InProcessHttpExecutor:
    """The engine's executor, with the POST replaced by a direct call into the
    real ``modal_app.Executor.execute_block`` handler."""

    def __init__(self, modal_app):
        self._server = _http_executor(modal_app)
        self._client = client.ModalExecutor(workspace_id="test-ws")
        self._client._get_endpoint_url = lambda workspace: "http://modal.test"
        self._client._post_execute = self._post

    def _post(self, **payload):
        request = {
            "code_hash": payload["code_hash"],
            "run_function_name": payload["python_code"].run_function_name,
            "inputs_json": payload["inputs_json"],
            "workflow_context": payload["workflow_context"],
            "code_str": payload["python_code"].run_function_code,
            "imports": payload["python_code"].imports or [],
        }
        if payload.get("block_state_json") is not None:
            request["block_state_json"] = payload["block_state_json"]
        return asyncio.run(self._server.execute_block(_FakeRequest(request)))

    def execute_remote_with_state(self, **kwargs):
        return self._client.execute_remote_with_state(**kwargs)

    def execute_remote(self, **kwargs):
        return self._client.execute_remote(**kwargs)


class _InProcessWsExecutor:
    """The engine's websocket executor talking to the real websocket ASGI app
    of ``modal_app``, frames as real msgpack bytes."""

    def __init__(self, modal_app):
        from fastapi.testclient import TestClient

        cls = modal_app.Executor
        user_cls = cls._get_user_cls() if hasattr(cls, "_get_user_cls") else cls
        server = user_cls.__new__(user_cls)
        server.workspace_id = "test-ws"
        user_cls.identify(server)
        self._session = TestClient(user_cls.wsapp(server)).websocket_connect("/ws")
        self._ws = self._session.__enter__()
        # Closed when the block (and with it this executor) is collected.
        weakref.finalize(self, self._session.__exit__, None, None, None)
        self._client = _ws_executor()
        self._client._send_recv_with_retry = self._send_recv

    def _send_recv(self, frame_bytes, workspace, request_id=None):
        self._ws.send_bytes(frame_bytes)
        return self._ws.receive_bytes()

    def execute_remote_with_state(self, **kwargs):
        return self._client.execute_remote_with_state(**kwargs)

    def execute_remote(self, **kwargs):
        return self._client.execute_remote(**kwargs)


def _build_first_frame_block(
    monkeypatch, modal_app, code=FIRST_FRAME_CODE, transport="http"
):
    from roboflow_workflows.execution_engine.v1.dynamic_blocks.block_assembler import (
        compile_dynamic_blocks,
    )

    monkeypatch.setattr(client, "MODAL_AVAILABLE", True)
    monkeypatch.setattr(
        block_scaffolding, "WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE", "modal"
    )
    monkeypatch.setattr(client, "validate_code_in_modal", lambda *a, **k: None)
    if transport == "http":
        executor = _InProcessHttpExecutor(modal_app)
    else:
        executor = _InProcessWsExecutor(modal_app)

    @contextmanager
    def acquire(workspace_id):
        yield executor

    monkeypatch.setattr(block_scaffolding, "_acquire_modal_executor", acquire)

    definition = {
        "type": "DynamicBlockDefinition",
        "manifest": {
            "type": "ManifestDescription",
            "block_type": "FirstFrameCheck",
            "inputs": {
                "image": {
                    "type": "DynamicInputDefinition",
                    "selector_types": ["input_parameter"],
                }
            },
            "outputs": {
                "is_first_frame": {"type": "DynamicOutputDefinition", "kind": []},
                "calls": {"type": "DynamicOutputDefinition", "kind": []},
            },
        },
        "code": {"type": "PythonCode", "run_function_code": code},
    }
    specification = compile_dynamic_blocks([definition])[0]
    block = specification.block_class(
        api_key="key",
        workspace_resolver=SimpleNamespace(resolve_workspace=lambda key: "test-ws"),
    )
    block._workflow_step_name = "first_frame"
    block._workflow_step_type = "FirstFrameCheck"
    block._workflow_step_selector = "$steps.first_frame"
    return block


def test_block_instance_keeps_self_state_across_remote_calls(monkeypatch, modal_app):
    block = _build_first_frame_block(monkeypatch, modal_app)

    results = [block.run(image=frame) for frame in (1, 2, 3)]

    assert [r["is_first_frame"] for r in results] == [True, False, False]
    assert [r["calls"] for r in results] == [1, 2, 3]
    assert block.has_first_frame is True
    assert block._ffc_calls == 3


def test_block_instance_drops_an_engine_side_non_transportable_attribute(
    monkeypatch, modal_app
):
    warned = _capture_client_warnings(monkeypatch)
    block = _build_first_frame_block(monkeypatch, modal_app)
    block.tracker = object()

    result = block.run(image=1)

    assert result["is_first_frame"] is True
    assert not hasattr(block, "tracker")
    assert any("tracker" in message for message in warned)


CACHED_HELPER_CODE = (
    "class Helper:\n"
    "    pass\n"
    "def run(self, image):\n"
    "    if not hasattr(self, 'helper'):\n"
    "        self.helper = Helper()\n"
    "    self._ffc_calls = getattr(self, '_ffc_calls', 0) + 1\n"
    "    return {'is_first_frame': self._ffc_calls == 1, 'calls': self._ffc_calls}\n"
)


def test_block_instance_keeps_working_when_it_caches_a_non_transportable_object(
    monkeypatch, modal_app, capsys
):
    # Pre-PR behaviour for such a block: nothing persists, nothing breaks.
    block = _build_first_frame_block(monkeypatch, modal_app, code=CACHED_HELPER_CODE)

    results = [block.run(image=frame) for frame in (1, 2, 3)]

    assert [r["is_first_frame"] for r in results] == [True, True, True]
    assert [r["calls"] for r in results] == [1, 1, 1]
    assert not hasattr(block, "_ffc_calls")
    assert not hasattr(block, "helper")
    relayed_stderr = capsys.readouterr().err
    assert "self.helper" in relayed_stderr
    assert "Helper" in relayed_stderr


# --------------------------------------------------------------------------- #
# the echo is user output and must not reach engine attributes
# --------------------------------------------------------------------------- #


class _ScaffoldedBlock:
    """Shape of a compiled dynamic block: usage metadata lives on the class."""

    _usage_block_kind = "custom_python"

    def __init__(self):
        self._api_key = "key"
        self._workflow_step_name = "first_frame"


@pytest.mark.parametrize(
    "echoed_state, offending_name",
    [
        ({"_api_key": "junk"}, "_api_key"),
        ({"_workspace_resolver": None}, "_workspace_resolver"),
        ({"__dict__": {"counter": 9}}, "__dict__"),
        ({"__class__": object}, "__class__"),
        ({"_usage_block_kind": "x"}, "_usage_block_kind"),
        ({"run": "shadowed"}, "run"),
    ],
)
def test_apply_block_state_refuses_reserved_names_and_leaves_the_block_untouched(
    echoed_state, offending_name
):
    _ScaffoldedBlock.run = lambda self: None
    block = _ScaffoldedBlock()
    block.counter = 1
    block.stale = "kept until a valid echo replaces it"
    state_before = dict(vars(block))

    with pytest.raises(DynamicBlockError) as error:
        apply_block_state(block, {"counter": 2, **echoed_state})

    assert offending_name in str(error.value)
    assert vars(block) == state_before, "a refused echo must not partially apply"
    assert type(block)._usage_block_kind == "custom_python"
    assert block._api_key == "key"


def test_apply_block_state_refuses_non_string_keys():
    block = _engine_block(counter=1)

    with pytest.raises(DynamicBlockError):
        apply_block_state(block, {1: "a"})

    assert block.counter == 1


def test_apply_block_state_keeps_a_user_attribute_named_like_a_base_attribute():
    # Regression guard for the filter: a user name that is NOT on the class
    # and NOT an engine attribute still applies, underscore or not.
    block = _ScaffoldedBlock()

    apply_block_state(block, {"_calls": 2, "has_first_frame": True})

    assert block._calls == 2
    assert block.has_first_frame is True
    assert block._api_key == "key"


RESERVED_NAME_CODE = (
    "def run(self, image):\n" "    self._api_key = 'junk'\n" "    return {'ok': True}\n"
)


def test_http_server_drops_an_engine_attribute_name_from_the_echo(modal_app):
    executor = _http_executor(modal_app)

    response = _run_http(executor, RESERVED_NAME_CODE, {"image": 1}, block_state={})

    assert response["success"] is True
    assert deserialize_for_modal_remote_execution(response["block_state"]) == {}
    assert "self._api_key" in response["stderr"]


def test_ws_server_drops_an_engine_attribute_name_from_the_echo(modal_app):
    executor = _http_executor(modal_app)

    response = _run_ws(executor, modal_app, RESERVED_NAME_CODE, {"image": 1}, {})

    assert response["success"] is True
    assert response["block_state"] == {}
    assert "self._api_key" in response["stderr"]


SHADOW_USAGE_CODE = (
    "def run(self, image):\n"
    "    self._usage_block_kind = 'x'\n"
    "    return {'is_first_frame': True, 'calls': 1}\n"
)


def test_block_instance_cannot_shadow_class_attributes_from_the_sandbox(
    monkeypatch, modal_app
):
    block = _build_first_frame_block(monkeypatch, modal_app, code=SHADOW_USAGE_CODE)
    kind_before = type(block)._usage_block_kind

    with pytest.raises(DynamicBlockError) as error:
        block.run(image=1)

    assert "_usage_block_kind" in str(error.value)
    assert "_usage_block_kind" not in vars(block)
    assert block._usage_block_kind == kind_before


# --------------------------------------------------------------------------- #
# numpy scalars keep their numpy type on both transports
# --------------------------------------------------------------------------- #


NUMPY_SCALAR_CODE = (
    "import numpy as np\n"
    "def run(self, image):\n"
    "    self.seen = np.bool_(False)\n"
    "    self.score = np.float32(0.5)\n"
    "    self.n = np.int64(3)\n"
    "    return {'ok': True}\n"
)


def _assert_numpy_scalars(state):
    assert type(state["seen"]) is np.bool_ and not state["seen"], state
    assert type(state["score"]) is np.float32 and state["score"] == 0.5
    assert type(state["n"]) is np.int64 and state["n"] == 3


def test_http_server_echoes_numpy_scalars_with_their_numpy_type(modal_app):
    executor = _http_executor(modal_app)

    response = _run_http(executor, NUMPY_SCALAR_CODE, {"image": 1}, block_state={})

    assert response["success"] is True, response.get("error")
    _assert_numpy_scalars(
        client._read_echoed_block_state_json(
            response, block_state={"x": 1}, block_type_name="b"
        )
    )


def test_ws_server_echoes_numpy_scalars_with_their_numpy_type(modal_app):
    executor = _http_executor(modal_app)

    response = _run_ws(executor, modal_app, NUMPY_SCALAR_CODE, {"image": 1}, {})

    assert response["success"] is True, response.get("error")
    _assert_numpy_scalars(
        client._read_echoed_block_state_msgpack(
            response, block_state={"x": 1}, block_type_name="b"
        )
    )


def test_http_client_sends_numpy_scalars_with_their_numpy_type():
    state = {"seen": np.bool_(False), "score": np.float32(0.5), "n": np.int64(3)}

    block_state_json = _serialize_block_state_json(state, block_type_name="b")

    _assert_numpy_scalars(
        client._read_echoed_block_state_json(
            {"block_state": block_state_json}, block_state=state, block_type_name="b"
        )
    )


def test_ws_client_packs_numpy_scalars_with_their_numpy_type():
    state = {"seen": np.bool_(False), "score": np.float32(0.5), "n": np.int64(3)}

    packed = client.WebSocketModalExecutor._pack_block_state(
        state, block_type_name="b", msgpack=msgpack
    )

    _assert_numpy_scalars(
        client._read_echoed_block_state_msgpack(
            {"block_state": _through_msgpack(packed)},
            block_state=state,
            block_type_name="b",
        )
    )


def _detections():
    detections = sv.Detections(
        xyxy=np.array([[0, 0, 1, 1]], dtype=np.float32),
        class_id=np.array([0]),
        confidence=np.array([0.5], dtype=np.float32),
        data={
            "class_name": np.array(["cat"]),
            "detection_id": np.array(["d0"]),
            "image_dimensions": np.array([[20, 30]]),
            "parent_id": np.array(["frame-1"]),
            "root_parent_id": np.array(["root-frame"]),
        },
    )
    return detections


def test_http_client_carries_a_tuple_holding_detections_as_a_tuple():
    block_state_json = _serialize_block_state_json(
        {"prev": (_detections(), 1)}, block_type_name="b"
    )

    prev = client._read_echoed_block_state_json(
        {"block_state": block_state_json}, block_state={"x": 1}, block_type_name="b"
    )["prev"]

    assert type(prev) is tuple
    assert isinstance(prev[0], sv.Detections)
    assert prev[1] == 1


def test_http_server_echoes_a_tuple_holding_detections_as_a_tuple(modal_app):
    executor = _http_executor(modal_app)
    code = (
        "def run(self, image):\n"
        "    self.prev = (self.prev[0], self.prev[1] + 1)\n"
        "    return {'n': len(self.prev[0])}\n"
    )
    seeded = _serialize_block_state_json(
        {"prev": (_detections(), 1)}, block_type_name="b"
    )
    payload = {
        "code_str": code,
        "imports": [],
        "run_function_name": "run",
        "inputs_json": serialize_for_modal_remote_execution({"image": 1}),
        "block_state_json": seeded,
    }

    response = asyncio.run(executor.execute_block(_FakeRequest(payload)))

    assert response["success"] is True, response.get("error")
    prev = client._read_echoed_block_state_json(
        response, block_state={"x": 1}, block_type_name="b"
    )["prev"]
    assert type(prev) is tuple
    assert isinstance(prev[0], sv.Detections)
    assert prev[1] == 2


# --------------------------------------------------------------------------- #
# dropping must keep the block working, and never fail the call
# --------------------------------------------------------------------------- #

FLAG_GUARDED_HELPER_CODE = (
    "class Annotator:\n"
    "    pass\n"
    "def run(self, image):\n"
    "    if not hasattr(self, 'initialized'):\n"
    "        self.annotator = Annotator()\n"
    "        self.initialized = True\n"
    "    self.annotator\n"
    "    return {'is_first_frame': True, 'calls': 1}\n"
)


def test_http_server_withholds_the_whole_state_when_an_attribute_is_dropped(
    modal_app,
):
    # A flag carried without the helper it guards would make the next call
    # skip the helper's construction and crash; when anything is dropped the
    # call carries nothing, as before state was carried at all.
    executor = _http_executor(modal_app)

    response = _run_http(executor, TRACKER_CODE, {"image": 1}, block_state={})

    assert response["success"] is True
    assert deserialize_for_modal_remote_execution(response["block_state"]) == {}
    assert "self.tracker" in response["stderr"]
    assert "self.count" in response["stderr"]
    assert "withheld" in response["stderr"]


def test_ws_server_withholds_the_whole_state_when_an_attribute_is_dropped(modal_app):
    executor = _http_executor(modal_app)

    response = _run_ws(executor, modal_app, TRACKER_CODE, {"image": 1}, {})

    assert response["success"] is True
    assert response["block_state"] == {}
    assert "self.tracker" in response["stderr"]
    assert "self.count" in response["stderr"]
    assert "withheld" in response["stderr"]


def test_ws_server_warns_about_withheld_state_once_per_container(modal_app):
    executor = _http_executor(modal_app)

    first = _run_ws(executor, modal_app, TRACKER_CODE, {"image": 1}, {})
    second = _run_ws(executor, modal_app, TRACKER_CODE, {"image": 1}, {})

    assert "withheld" in first["stderr"]
    assert "withheld" not in (second["stderr"] or "")


def test_block_instance_survives_a_flag_guarded_helper(monkeypatch, modal_app, capsys):
    block = _build_first_frame_block(
        monkeypatch, modal_app, code=FLAG_GUARDED_HELPER_CODE
    )

    results = [block.run(image=frame) for frame in (1, 2, 3)]

    assert [r["is_first_frame"] for r in results] == [True, True, True]
    assert not hasattr(block, "initialized")
    assert not hasattr(block, "annotator")
    relayed_stderr = capsys.readouterr().err
    assert "self.annotator" in relayed_stderr
    assert "self.initialized" in relayed_stderr


def test_engine_partition_withholds_everything_when_an_attribute_is_dropped():
    partition = partition_block_state(
        {"tracker": object(), "count": 1, "initialized": True},
        measure=lambda attribute: len(json.dumps(attribute)),
    )

    assert partition.carried == {}
    assert [name for name, _ in partition.dropped] == ["tracker"]
    assert any("withheld" in note and "self.count" in note for note in partition.notes)


UNCARRIABLE_VALUE_EXPRESSIONS = {
    "complex_array": "np.array([1 + 2j])",
    "datetime64_array": "np.array(['2020-01-01'], dtype='datetime64[D]')",
    "object_array": "np.array([object()], dtype=object)",
    "bytes_array": "np.array([b'a'])",
    "longdouble": "np.longdouble(1.5)",
    "timedelta64": "np.timedelta64(1, 's')",
    "huge_int": "2 ** 64",
    "uuid_int": "uuid.uuid4().int",
    "nested_huge_int": "{'ids': [2 ** 64]}",
    "detections_without_confidence": (
        "sv.Detections(xyxy=np.array([[0, 0, 10, 10]], dtype=float), "
        "class_id=np.array([1]))"
    ),
    "detections_without_image_dimensions": (
        "sv.Detections(xyxy=np.array([[0, 0, 10, 10]], dtype=float), "
        "class_id=np.array([1]), confidence=np.array([0.5]), "
        "data={'class_name': np.array(['a']), 'detection_id': np.array(['d'])})"
    ),
    "dict_with_type_marker": "{'_type': 'datetime', 'value': '2020-01-01T00:00:00'}",
    "base64_image_shaped_dict": "{'type': 'base64', 'value': 'abc'}",
    "nested_base64_image_shaped_dict": "[{'type': 'base64', 'value': 'abc'}]",
    "defaultdict": "collections.defaultdict(int, {'k1': 1})",
    "counter": "collections.Counter({'k1': 1})",
    "ordered_dict": "collections.OrderedDict(k1=1)",
    "namedtuple": "collections.namedtuple('Point', 'x')(1)",
    "int_enum": "enum.IntEnum('Color', 'RED').RED",
    "str_subclass": "type('Name', (str,), {})('x')",
    "numpy_int_subclass": "type('Count', (np.int64,), {})(1)",
    "numpy_str_with_trailing_nul": "np.str_('a' + chr(0))",
    "datetime_with_keyless_zoneinfo": (
        "datetime.datetime(2020, 1, 1, tzinfo=__import__('zoneinfo').ZoneInfo"
        ".from_file(open(__import__('zoneinfo').TZPATH[0] + '/UTC', 'rb')))"
    ),
    "datetime_with_uncarried_tzinfo": (
        "datetime.datetime(2020, 1, 1, tzinfo=type('Tz', (datetime.tzinfo,), "
        "{'utcoffset': lambda self, dt: datetime.timedelta(0), "
        "'tzname': lambda self, dt: 'X', 'dst': lambda self, dt: None})())"
    ),
    "detections_numpy_uint64_keypoint_id": (
        "sv.Detections(xyxy=np.array([[0, 0, 10, 10]], dtype=float), "
        "class_id=np.array([1]), confidence=np.array([0.5]), "
        "data={'class_name': np.array(['a']), 'detection_id': np.array(['d']), "
        "'image_dimensions': np.array([[20, 20]]), "
        "'keypoints_class_id': np.array([[np.uint64(2 ** 63)]], dtype=object), "
        "'keypoints_class_name': np.array([['k']], dtype=object), "
        "'keypoints_confidence': np.array([[0.5]]), "
        "'keypoints_xy': np.array([[[1.0, 1.0]]])})"
    ),
    "detections_tuple_in_data_payload": (
        "sv.Detections(xyxy=np.array([[0, 0, 10, 10]], dtype=float), "
        "class_id=np.array([1]), confidence=np.array([0.5]), "
        "data={'class_name': np.array(['a']), 'detection_id': np.array(['d']), "
        "'image_dimensions': np.array([[20, 20]]), "
        "'data': np.array([{'pair': (1, 2)}], dtype=object)})"
    ),
    "detections_marker_in_data_payload": (
        "sv.Detections(xyxy=np.array([[0, 0, 10, 10]], dtype=float), "
        "class_id=np.array([1]), confidence=np.array([0.5]), "
        "data={'class_name': np.array(['a']), 'detection_id': np.array(['d']), "
        "'image_dimensions': np.array([[20, 20]]), "
        "'data': np.array([{'_type': 'datetime', 'value': 'bad'}], dtype=object)})"
    ),
    "detections_image_shaped_dict_in_data_payload": (
        "sv.Detections(xyxy=np.array([[0, 0, 10, 10]], dtype=float), "
        "class_id=np.array([1]), confidence=np.array([0.5]), "
        "data={'class_name': np.array(['a']), 'detection_id': np.array(['d']), "
        "'image_dimensions': np.array([[20, 20]]), "
        "'data': np.array([{'type': 'base64', 'value': 'abc'}], dtype=object)})"
    ),
    "detections_object_keypoint_id_beyond_int64": (
        "sv.Detections(xyxy=np.array([[0, 0, 10, 10]], dtype=float), "
        "class_id=np.array([1]), confidence=np.array([0.5]), "
        "data={'class_name': np.array(['a']), 'detection_id': np.array(['d']), "
        "'image_dimensions': np.array([[20, 20]]), "
        "'keypoints_class_id': np.array([[2 ** 63]], dtype=object), "
        "'keypoints_class_name': np.array([['k']], dtype=object), "
        "'keypoints_confidence': np.array([[0.5]]), "
        "'keypoints_xy': np.array([[[1.0, 1.0]]])})"
    ),
    "list_subclass": "type('Items', (list,), {})([1])",
    "datetime_subclass": "type('Stamp', (datetime.datetime,), {})(2020, 1, 1)",
    "detections_uint64_class_id": (
        "sv.Detections(xyxy=np.array([[0, 0, 10, 10]], dtype=float), "
        "class_id=np.array([2 ** 63], dtype=np.uint64), confidence=np.array([0.5]), "
        "data={'class_name': np.array(['a']), 'detection_id': np.array(['d']), "
        "'image_dimensions': np.array([[20, 20]])})"
    ),
    "detections_uint64_tracker_id": (
        "sv.Detections(xyxy=np.array([[0, 0, 10, 10]], dtype=float), "
        "class_id=np.array([1]), confidence=np.array([0.5]), "
        "tracker_id=np.array([2 ** 63], dtype=np.uint64), "
        "data={'class_name': np.array(['a']), 'detection_id': np.array(['d']), "
        "'image_dimensions': np.array([[20, 20]])})"
    ),
    "detections_rle_without_size": (
        "sv.Detections(xyxy=np.array([[0, 0, 10, 10]], dtype=float), "
        "class_id=np.array([1]), confidence=np.array([0.5]), "
        "data={'class_name': np.array(['a']), 'detection_id': np.array(['d']), "
        "'rle_mask': np.array([{'counts': '04'}], dtype=object)})"
    ),
    "array_over_the_allocation_budget": "np.zeros(1, dtype='U3000000')",
}


def _uncarriable_value_code(expression: str) -> str:
    code = (
        "import collections\n"
        "import datetime\n"
        "import enum\n"
        "import uuid\n"
        "import numpy as np\n"
        "import supervision as sv\n"
        "def run(self, image):\n"
        f"    self.value = {expression}\n"
        "    return {'ok': True}\n"
    )
    return code


@pytest.mark.parametrize(
    "expression",
    UNCARRIABLE_VALUE_EXPRESSIONS.values(),
    ids=UNCARRIABLE_VALUE_EXPRESSIONS,
)
def test_http_server_drops_a_value_the_transport_cannot_carry(modal_app, expression):
    executor = _http_executor(modal_app)

    response = _run_http(
        executor, _uncarriable_value_code(expression), {"image": 1}, block_state={}
    )

    assert response["success"] is True, response.get("error")
    assert deserialize_for_modal_remote_execution(response["block_state"]) == {}
    assert "self.value" in response["stderr"]


@pytest.mark.parametrize(
    "expression",
    UNCARRIABLE_VALUE_EXPRESSIONS.values(),
    ids=UNCARRIABLE_VALUE_EXPRESSIONS,
)
def test_ws_server_drops_a_value_the_transport_cannot_carry(modal_app, expression):
    executor = _http_executor(modal_app)

    response = _run_ws(
        executor, modal_app, _uncarriable_value_code(expression), {"image": 1}, {}
    )

    assert response["success"] is True, response.get("error")
    assert response["block_state"] == {}
    assert "self.value" in response["stderr"]
    msgpack.packb(response, use_bin_type=True)


def _uncarriable_values() -> dict:
    import collections
    import datetime as datetime_module
    import enum
    import uuid

    namespace = {
        "np": np,
        "sv": sv,
        "uuid": uuid,
        "collections": collections,
        "datetime": datetime_module,
        "enum": enum,
    }
    values = {
        name: eval(expression, namespace)
        for name, expression in UNCARRIABLE_VALUE_EXPRESSIONS.items()
    }
    return values


def _carriable_values() -> dict:
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="p"),
        numpy_image=np.zeros((2, 2, 3), dtype=np.uint8),
    )
    values = {
        "flag": True,
        "count": 3,
        "big_but_fits": 2**63 - 1,
        "ratio": 0.5,
        "name": "x",
        "nothing": None,
        "raw": b"\x00",
        "when": datetime(2026, 10, 26),
        "array": np.zeros((2,)),
        "str_array": np.array(["a"]),
        "scalar": np.int64(4),
        "flag_scalar": np.bool_(True),
        "ratio_scalar": np.float32(0.25),
        "history": [1, (2, 3), {"nested": [None]}],
        "shape": (4, 4, 3),
        "label": np.array(["cat"])[0],
        "wide_int": np.longlong(5),
        "empty_label": np.str_(""),
        "zoned": datetime(2026, 3, 28, 12, tzinfo=ZoneInfo("Europe/Warsaw")),
        "raw_label": np.array([b"cat"])[0],
        "per_class": {np.array(["cat"])[0]: 1},
        "detections": _detections(),
        "image": image,
    }
    return values


@pytest.mark.parametrize("name", list(UNCARRIABLE_VALUE_EXPRESSIONS))
def test_engine_partition_drops_a_value_the_transport_cannot_carry(name):
    value = _uncarriable_values()[name]

    partition = partition_block_state(
        {"value": value},
        measure=lambda attribute: len(serialize_for_modal_remote_execution(attribute)),
    )

    assert partition.carried == {}
    assert [dropped_name for dropped_name, _ in partition.dropped] == ["value"]


def test_engine_partition_carries_every_transportable_value():
    partition = partition_block_state(
        _carriable_values(),
        measure=lambda attribute: len(serialize_for_modal_remote_execution(attribute)),
    )

    assert partition.dropped == []
    decoded = client._read_echoed_block_state_json(
        {"block_state": serialize_for_modal_remote_execution(partition.carried)},
        block_state={"sent": True},
        block_type_name="B",
    )
    assert set(decoded) == set(_carriable_values())
    for name in ("flag", "count", "big_but_fits", "ratio", "name", "raw", "when"):
        assert decoded[name] == _carriable_values()[name], name
    assert decoded["history"] == [1, (2, 3), {"nested": [None]}]
    assert type(decoded["history"][1]) is tuple


def _comparable(state: dict) -> dict:
    # Serializing an image stamps a fresh frame timestamp, so images compare
    # by pixels and parentage; everything else by its serialized form.
    comparable = {
        name: (
            (value.parent_metadata.parent_id, value.numpy_image.tobytes())
            if isinstance(value, WorkflowImageData)
            else serialize_for_modal_remote_execution({name: value})
        )
        for name, value in state.items()
    }
    return comparable


def test_engine_and_sandbox_partitions_agree_on_every_value(modal_app):
    def measure(attribute):
        return len(serialize_for_modal_remote_execution(attribute))

    for name, value in {**_carriable_values(), **_uncarriable_values()}.items():
        engine = partition_block_state({name: value}, measure=measure)
        sandbox_carried, sandbox_dropped, sandbox_notes = (
            modal_app._partition_block_state({name: value}, measure)
        )

        assert _comparable(engine.carried) == _comparable(sandbox_carried), name
        assert engine.dropped == sandbox_dropped, name
        if not engine.dropped:
            # Withheld notes are worded per side on purpose.
            assert engine.notes == sandbox_notes, name


def test_engine_partition_names_the_offending_path_inside_a_container():
    class Tracker:
        pass

    partition = partition_block_state(
        {"history": [1, {"tracker": Tracker()}]}, measure=lambda attribute: 0
    )

    reason = dict(partition.dropped)["history"]
    assert "history[1]['tracker']" in reason
    assert "Tracker" in reason


def test_engine_partition_drops_a_dict_with_non_string_keys():
    partition = partition_block_state({"lookup": {1: "a"}}, measure=lambda a: 0)

    assert "lookup" in dict(partition.dropped)


def test_engine_partition_drops_a_value_only_the_encoder_refuses():
    # The validator cannot foresee every encoder failure; whatever ``measure``
    # raises is a drop, not a failed call.
    def measure(attribute):
        raise TypeError("Object of type Weird is not JSON serializable")

    partition = partition_block_state({"weird": 1}, measure=measure)

    assert partition.carried == {}
    assert "not JSON serializable" in dict(partition.dropped)["weird"]


def test_http_client_fails_as_a_dynamic_block_error_when_the_echo_cannot_be_decoded():
    broken = json.dumps(
        {"prev": {"_type": "sv_detections", "predictions": [{}], "image": {}}}
    )

    with pytest.raises(DynamicBlockError) as error:
        client._read_echoed_block_state_json(
            {"block_state": broken}, block_state={}, block_type_name="B"
        )

    assert "B" in str(error.value)


def test_ws_client_fails_as_a_dynamic_block_error_when_the_echo_cannot_be_decoded():
    broken = {"prev": {"_type": "sv_detections", "predictions": [{}], "image": {}}}

    with pytest.raises(DynamicBlockError) as error:
        client._read_echoed_block_state_msgpack(
            {"block_state": broken}, block_state={}, block_type_name="B"
        )

    assert "B" in str(error.value)


@pytest.mark.parametrize(
    "reader", ["_read_echoed_block_state_json", "_read_echoed_block_state_msgpack"]
)
def test_client_refuses_a_non_dict_echo(reader):
    echoed = json.dumps(["count"]) if reader.endswith("json") else ["count"]

    with pytest.raises(DynamicBlockError):
        getattr(client, reader)(
            {"block_state": echoed}, block_state={}, block_type_name="B"
        )


USER_STATE_NAME_CODE = (
    "def run(self, image):\n"
    "    self.user_state = {'frames': 1}\n"
    "    return {'ok': True}\n"
)


def test_http_server_lets_a_block_keep_an_attribute_named_user_state(modal_app):
    executor = _http_executor(modal_app)

    response = _run_http(executor, USER_STATE_NAME_CODE, {"image": 1}, block_state={})

    assert response["success"] is True, response.get("error")
    assert deserialize_for_modal_remote_execution(response["block_state"]) == {
        "user_state": {"frames": 1}
    }


def test_ws_server_lets_a_block_keep_an_attribute_named_user_state(modal_app):
    executor = _http_executor(modal_app)

    response = _run_ws(executor, modal_app, USER_STATE_NAME_CODE, {"image": 1}, {})

    assert response["success"] is True, response.get("error")
    assert response["block_state"] == {"user_state": {"frames": 1}}


def test_client_logs_a_block_state_warning_once_per_process(monkeypatch):
    warned = _capture_client_warnings(monkeypatch)

    for _ in range(3):
        _serialize_block_state_json(
            {"tracker": object()}, block_type_name="FirstFrameCheck"
        )

    assert len(warned) == 2, warned  # the drop and the withheld note, once each


def test_ws_client_tolerates_a_missing_echo_when_it_sent_no_state(monkeypatch):
    # A stateless block against a pre-upgrade server keeps working; only a
    # block that already holds state is refused (see the "fails loudly" test).
    monkeypatch.setattr(client, "MODAL_AVAILABLE", True)
    warned = _capture_client_warnings(monkeypatch)
    executor = _ws_executor()
    executor._send_recv_with_retry = lambda frame_bytes, workspace, request_id=None: (
        msgpack.packb(
            {"success": True, "result": {"ok": 1}, "request_id": request_id},
            use_bin_type=True,
        )
    )

    result, state = executor.execute_remote_with_state(
        block_type_name="FirstFrameCheck",
        python_code=_python_code(),
        inputs={"image": 1},
        workspace_id="test-ws",
        workflow_context={},
        block_state={},
    )

    assert result == {"ok": 1}
    assert state == {}
    assert any("older" in message for message in warned)


def test_http_client_tolerates_a_missing_echo_when_it_sent_no_state(monkeypatch):
    monkeypatch.setattr(client, "MODAL_AVAILABLE", True)
    warned = _capture_client_warnings(monkeypatch)
    executor = client.ModalExecutor(workspace_id="test-ws")
    executor._get_endpoint_url = lambda workspace: "http://modal.test/execute"
    executor._post_execute = lambda **kwargs: {"success": True, "result": "{}"}

    result, state = executor.execute_remote_with_state(
        block_type_name="FirstFrameCheck",
        python_code=_python_code(),
        inputs={"image": 1},
        workspace_id="test-ws",
        workflow_context={},
        block_state={},
    )

    assert result == {}
    assert state == {}
    assert any("older" in message for message in warned)


MARKER_SHAPED_STATE_CODE = (
    "def run(self, image):\n"
    "    self._type = 'datetime'\n"
    "    self.value = '2020-01-01T00:00:00'\n"
    "    return {'seeded': type(getattr(self, 'seeded', None)).__name__}\n"
)


def test_http_server_keeps_attributes_named_like_a_wire_marker_as_attributes(
    modal_app,
):
    executor = _http_executor(modal_app)

    response = _run_http(
        executor,
        MARKER_SHAPED_STATE_CODE,
        {"image": 1},
        block_state={
            "_type": "datetime",
            "value": "2020-01-01T00:00:00",
            "seeded": "x",
        },
    )

    assert response["success"] is True, response.get("error")
    assert json.loads(response["result"]) == {"seeded": "str"}
    assert client._read_echoed_block_state_json(
        response, block_state={}, block_type_name="B"
    ) == {
        "_type": "datetime",
        "value": "2020-01-01T00:00:00",
        "seeded": "x",
    }


def test_ws_server_keeps_attributes_named_like_a_wire_marker_as_attributes(modal_app):
    executor = _http_executor(modal_app)

    response = _run_ws(
        executor,
        modal_app,
        MARKER_SHAPED_STATE_CODE,
        {"image": 1},
        {"_type": "datetime", "value": "2020-01-01T00:00:00", "seeded": "x"},
    )

    assert response["success"] is True, response.get("error")
    assert response["result"] == {"seeded": "str"}
    assert response["block_state"] == {
        "_type": "datetime",
        "value": "2020-01-01T00:00:00",
        "seeded": "x",
    }


def test_http_client_decodes_an_echo_shaped_like_a_wire_marker_as_attributes():
    echoed = json.dumps({"_type": "datetime", "value": "2020-01-01T00:00:00"})

    state = client._read_echoed_block_state_json(
        {"block_state": echoed}, block_state={}, block_type_name="B"
    )

    assert state == {"_type": "datetime", "value": "2020-01-01T00:00:00"}


def test_ws_client_decodes_an_echo_shaped_like_a_wire_marker_as_attributes():
    echoed = {"_type": "datetime", "value": "2020-01-01T00:00:00"}

    state = client._read_echoed_block_state_msgpack(
        {"block_state": echoed}, block_state={}, block_type_name="B"
    )

    assert state == {"_type": "datetime", "value": "2020-01-01T00:00:00"}


def _detections_with_custom_data():
    detections = _detections()
    detections.data["custom"] = np.array(["kept?"])
    return detections


def test_engine_partition_notes_detections_data_keys_the_transport_loses():
    partition = partition_block_state(
        {"prev": _detections_with_custom_data()},
        measure=lambda attribute: len(serialize_for_modal_remote_execution(attribute)),
    )

    assert "prev" in partition.carried
    assert any("self.prev" in note and "custom" in note for note in partition.notes)


def test_http_server_notes_detections_data_keys_the_transport_loses(modal_app):
    executor = _http_executor(modal_app)
    code = (
        "import numpy as np\n"
        "def run(self, image):\n"
        "    self.prev.data['custom'] = np.array(['x'])\n"
        "    return {'ok': True}\n"
    )
    seeded = _serialize_block_state_json({"prev": _detections()}, block_type_name="b")
    payload = {
        "code_str": code,
        "imports": [],
        "run_function_name": "run",
        "inputs_json": serialize_for_modal_remote_execution({"image": 1}),
        "block_state_json": seeded,
    }

    response = asyncio.run(executor.execute_block(_FakeRequest(payload)))

    assert response["success"] is True, response.get("error")
    prev = deserialize_for_modal_remote_execution(response["block_state"])["prev"]
    assert "custom" not in prev.data
    assert "self.prev" in response["stderr"]
    assert "custom" in response["stderr"]


def test_apply_block_state_refuses_a_non_dict_echo_and_leaves_the_block_untouched():
    block = _engine_block(count=7)

    with pytest.raises(DynamicBlockError):
        apply_block_state(block, ["count"])

    assert block.count == 7


def test_apply_block_state_allows_names_only_the_metaclass_defines():
    from abc import ABC

    class AbstractBlock(ABC):
        pass

    block = AbstractBlock()
    block._api_key = "key"

    apply_block_state(block, {"register": 1, "mro": 2})

    assert block.register == 1
    assert block.mro == 2


def test_engine_and_sandbox_agree_on_the_carried_detections_data_keys(modal_app):
    from roboflow_workflows.execution_engine.v1.dynamic_blocks import block_state

    assert (
        modal_app._CARRIED_DETECTIONS_DATA_KEYS
        == block_state.CARRIED_DETECTIONS_DATA_KEYS
    )
    assert modal_app._ENGINE_ATTRIBUTE_NAMES == block_state.ENGINE_ATTRIBUTE_NAMES


# --------------------------------------------------------------------------- #
# flag-guarded helpers, numpy dates, masks, images and parent ids
# --------------------------------------------------------------------------- #


def test_ws_server_withholds_a_flag_whose_guarded_helper_is_dropped(modal_app):
    executor = _http_executor(modal_app)

    first = _run_ws(executor, modal_app, FLAG_GUARDED_HELPER_CODE, {"image": 1}, {})
    second = _run_ws(
        executor,
        modal_app,
        FLAG_GUARDED_HELPER_CODE,
        {"image": 1},
        first["block_state"],
    )

    assert first["block_state"] == {}
    assert second["success"] is True, second.get("error")


def test_http_server_withholds_a_flag_whose_guarded_helper_is_dropped(modal_app):
    executor = _http_executor(modal_app)

    first = _run_http(executor, FLAG_GUARDED_HELPER_CODE, {"image": 1}, block_state={})
    state = client._read_echoed_block_state_json(
        first, block_state={}, block_type_name="B"
    )
    second = _run_http(executor, FLAG_GUARDED_HELPER_CODE, {"image": 1}, state)

    assert state == {}
    assert second["success"] is True, second.get("error")


@pytest.mark.parametrize("partition_side", ["engine", "sandbox"])
def test_partition_drops_a_timedelta64_scalar(modal_app, partition_side):
    # np.timedelta64 subclasses np.signedinteger, so an integer check admits it.
    def measure(attribute):
        return 0

    state = {"gap": np.timedelta64(1, "s")}
    if partition_side == "engine":
        carried, dropped, _ = partition_block_state(state, measure=measure)
    else:
        carried, dropped, _ = modal_app._partition_block_state(state, measure)

    assert carried == {}
    assert [name for name, _ in dropped] == ["gap"]
    assert "timedelta64" in dropped[0][1]


def _detections_with_an_empty_mask():
    detections = _detections()
    detections.mask = np.zeros((1, 20, 30), dtype=bool)
    return detections


def test_engine_partition_notes_that_masks_travel_as_polygons():
    partition = partition_block_state(
        {"prev": _detections_with_an_empty_mask()}, measure=lambda attribute: 0
    )

    assert any("self.prev" in note and "polygon" in note for note in partition.notes)


def test_sandbox_partition_notes_that_masks_travel_as_polygons(modal_app):
    _, _, notes = modal_app._partition_block_state(
        {"prev": _detections_with_an_empty_mask()}, lambda attribute: 0
    )

    assert any("self.prev" in note and "polygon" in note for note in notes)


IMAGE_ON_SELF_CODE = (
    "def run(self, image):\n" "    self.last = image\n" "    return {'ok': True}\n"
)


def _image():
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="frame-1"),
        numpy_image=np.tile(
            np.linspace(40, 200, 16, dtype=np.uint8)[None, :, None], (16, 1, 3)
        ),
    )
    return image


def test_http_server_carries_an_image_kept_on_self_and_notes_the_loss(modal_app):
    executor = _http_executor(modal_app)

    response = _run_http(executor, IMAGE_ON_SELF_CODE, {"image": _image()}, {})

    assert response["success"] is True, response.get("error")
    state = client._read_echoed_block_state_json(
        response, block_state={}, block_type_name="B"
    )
    assert isinstance(state["last"], WorkflowImageData)
    assert state["last"].numpy_image.shape == (16, 16, 3)
    pixel_error = np.abs(
        state["last"].numpy_image.astype(int) - _image().numpy_image.astype(int)
    )
    # A gradient, so JPEG has edges to smear (a flat image would pass any
    # tolerance); 8 levels covers one encode at the default quality.
    assert pixel_error.max() <= 8
    assert "self.last" in response["stderr"] and "JPEG" in response["stderr"]


def test_ws_server_carries_an_image_kept_on_self_and_notes_the_loss(modal_app):
    executor = _http_executor(modal_app)

    response = _run_ws(executor, modal_app, IMAGE_ON_SELF_CODE, {"image": _image()}, {})

    assert response["success"] is True, response.get("error")
    state = client._read_echoed_block_state_msgpack(
        response,
        block_state={},
        block_type_name="B",
    )
    assert isinstance(state["last"], WorkflowImageData)
    assert state["last"].numpy_image.shape == (16, 16, 3)
    pixel_error = np.abs(
        state["last"].numpy_image.astype(int) - _image().numpy_image.astype(int)
    )
    # A gradient, so JPEG has edges to smear (a flat image would pass any
    # tolerance); 8 levels covers one encode at the default quality.
    assert pixel_error.max() <= 8
    assert "self.last" in response["stderr"] and "JPEG" in response["stderr"]


@pytest.mark.parametrize("transport", ["http", "ws"])
def test_detections_parent_ids_survive_a_server_round_trip(modal_app, transport):
    executor = _http_executor(modal_app)
    code = "def run(self, image):\n    return {'ok': True}\n"
    seeded = {"prev": _detections()}

    if transport == "http":
        response = _run_http(executor, code, {"image": 1}, seeded)
        state = client._read_echoed_block_state_json(
            response, block_state=seeded, block_type_name="B"
        )
    else:
        response = _run_ws(executor, modal_app, code, {"image": 1}, seeded)
        state = client._read_echoed_block_state_msgpack(
            msgpack.unpackb(msgpack.packb(response, use_bin_type=True), raw=False),
            block_state=seeded,
            block_type_name="B",
        )

    assert state["prev"].data["parent_id"].tolist() == ["frame-1"]
    assert state["prev"].data["root_parent_id"].tolist() == ["root-frame"]


# --------------------------------------------------------------------------- #
# exact types, image-shaped dicts, Detections ranges, rollback, warnings
# --------------------------------------------------------------------------- #


SUBCLASS_STATE_CODE = (
    "import collections\n"
    "def run(self, image):\n"
    "    if not hasattr(self, 'seen'):\n"
    "        self.seen = collections.defaultdict(int)\n"
    "    self.seen['k' + str(image)] += 1\n"
    "    return {'is_first_frame': True, 'calls': len(self.seen)}\n"
)


@pytest.mark.parametrize("transport", ["http", "ws"])
def test_block_instance_keeps_working_with_a_defaultdict_on_self(
    monkeypatch, modal_app, transport
):
    block = _build_first_frame_block(
        monkeypatch, modal_app, code=SUBCLASS_STATE_CODE, transport=transport
    )

    results = [block.run(image=frame) for frame in (1, 2, 3)]

    assert [r["calls"] for r in results] == [1, 1, 1]
    assert not hasattr(block, "seen")


@pytest.mark.parametrize("transport", ["http", "ws"])
def test_block_instance_keeps_self_state_across_calls_on_both_transports(
    monkeypatch, modal_app, transport
):
    block = _build_first_frame_block(monkeypatch, modal_app, transport=transport)

    results = [block.run(image=frame) for frame in (1, 2, 3)]

    assert [r["is_first_frame"] for r in results] == [True, False, False]
    assert [r["calls"] for r in results] == [1, 2, 3]


def test_engine_partition_survives_detections_with_mixed_data_key_types():
    detections = _detections()
    detections.data["custom"] = np.array(["x"])
    detections.data[42] = np.array(["y"])

    partition = partition_block_state({"prev": detections}, measure=lambda attribute: 0)

    assert "prev" in partition.carried
    assert any("custom" in note and "42" in note for note in partition.notes)


def test_sandbox_partition_survives_detections_with_mixed_data_key_types(modal_app):
    detections = _detections()
    detections.data["custom"] = np.array(["x"])
    detections.data[42] = np.array(["y"])

    carried, _, notes = modal_app._partition_block_state(
        {"prev": detections}, lambda attribute: 0
    )

    assert "prev" in carried
    assert any("custom" in note and "42" in note for note in notes)


def _detections_with_equal_parent_ids():
    detections = _detections()
    detections.data["root_parent_id"] = np.array(["frame-1"])
    return detections


def test_engine_partition_notes_a_root_parent_id_equal_to_the_parent_id():
    partition = partition_block_state(
        {"prev": _detections_with_equal_parent_ids()}, measure=lambda attribute: 0
    )

    assert any("root_parent_id" in note for note in partition.notes)


def test_sandbox_partition_notes_a_root_parent_id_equal_to_the_parent_id(modal_app):
    _, _, notes = modal_app._partition_block_state(
        {"prev": _detections_with_equal_parent_ids()}, lambda attribute: 0
    )

    assert any("root_parent_id" in note for note in notes)


GROWING_BUFFER_CODE = (
    "import numpy as np\n"
    "def run(self, image):\n"
    "    self.buffer = np.zeros((2 * 1024 * 1024 + image,), dtype=np.uint8)\n"
    "    return {'ok': True}\n"
)


def test_http_server_warns_once_about_an_oversized_attribute_whose_size_changes(
    modal_app,
):
    executor = _http_executor(modal_app)

    responses = [
        _run_http(executor, GROWING_BUFFER_CODE, {"image": frame}, block_state={})
        for frame in (1, 2, 3, 4)
    ]

    stderrs = [response["stderr"] or "" for response in responses]
    assert "self.buffer" in stderrs[0]
    assert all("self.buffer" not in stderr for stderr in stderrs[1:])
    assert len(executor._block_state_drop_warnings) == 2


def test_sandbox_warning_set_is_bounded(modal_app):
    executor = _http_executor(modal_app)
    stderr = io.StringIO()

    for index in range(modal_app._MAX_BLOCK_STATE_WARNINGS + 10):
        modal_app._warn_dropped_block_state(
            executor, f"hash-{index}", [("x", "reason")], [], stderr
        )

    assert (
        len(executor._block_state_drop_warnings) <= modal_app._MAX_BLOCK_STATE_WARNINGS
    )


def test_engine_withheld_note_says_the_sandbox_starts_this_call_empty():
    partition = partition_block_state(
        {"tracker": object(), "count": 1}, measure=lambda attribute: 0
    )

    assert any("this call" in note and "empty" in note for note in partition.notes)


def test_sandbox_withheld_note_says_the_next_call_starts_empty(modal_app):
    _, _, notes = modal_app._partition_block_state(
        {"tracker": object(), "count": 1}, lambda attribute: 0
    )

    assert any("next call" in note and "empty" in note for note in notes)


@pytest.mark.parametrize("partition_side", ["engine", "sandbox"])
@pytest.mark.parametrize("size_over_cap, carried", [(0, True), (1, False)])
def test_partition_size_cap_boundary(modal_app, partition_side, size_over_cap, carried):
    def measure(attribute):
        return BLOCK_STATE_MAX_BYTES + size_over_cap

    if partition_side == "engine":
        carried_state, _, _ = partition_block_state({"x": 1}, measure=measure)
    else:
        carried_state, _, _ = modal_app._partition_block_state({"x": 1}, measure)

    assert ("x" in carried_state) is carried


def test_client_decode_error_names_the_failing_attribute():
    broken = json.dumps(
        {"ok": 1, "prev": {"_type": "sv_detections", "predictions": [{}], "image": {}}}
    )

    with pytest.raises(DynamicBlockError) as error:
        client._read_echoed_block_state_json(
            {"block_state": broken}, block_state={}, block_type_name="B"
        )

    assert "self.prev" in str(error.value)


def test_client_warns_once_per_step_not_once_per_block_type(monkeypatch):
    warned = _capture_client_warnings(monkeypatch)

    for step_name in ("first", "second", "second"):
        client._carry_block_state(
            {"tracker": object()},
            block_type_name="FirstFrameCheck",
            workflow_context={"step_selector": f"$steps.{step_name}"},
            measure=client._json_state_size,
        )

    tracker_warnings = [m for m in warned if "self.tracker" in m and "rebuilt" in m]
    assert len(tracker_warnings) == 2


# --------------------------------------------------------------------------- #
# tuples, numpy strings, Detections payloads, total cap, warning keys
# --------------------------------------------------------------------------- #


TUPLE_IN_SET_CODE = (
    "def run(self, image):\n"
    "    if not hasattr(self, 'last'):\n"
    "        self.last = (1, 2)\n"
    "    seen = {self.last}\n"
    "    resets = getattr(self, 'resets', 0)\n"
    "    if getattr(self, 'shape', None) != (4, 4, 3):\n"
    "        resets += 1\n"
    "        self.shape = (4, 4, 3)\n"
    "    self.resets = resets\n"
    "    return {'is_first_frame': len(seen) == 1, 'calls': resets}\n"
)


@pytest.mark.parametrize("transport", ["http", "ws"])
def test_block_instance_keeps_tuples_on_self_as_tuples(
    monkeypatch, modal_app, transport
):
    block = _build_first_frame_block(
        monkeypatch, modal_app, code=TUPLE_IN_SET_CODE, transport=transport
    )

    results = [block.run(image=frame) for frame in (1, 2, 3)]

    assert [r["calls"] for r in results] == [1, 1, 1]
    assert type(block.last) is tuple
    assert block.shape == (4, 4, 3)


NP_STR_COUNTER_CODE = (
    "import numpy as np\n"
    "def run(self, image):\n"
    "    labels = np.array(['cat', 'dog'])\n"
    "    self.counts = getattr(self, 'counts', {})\n"
    "    for label in labels:\n"
    "        self.counts[label] = self.counts.get(label, 0) + 1\n"
    "    self.last_label = labels[0]\n"
    "    return {'is_first_frame': True, 'calls': self.counts['cat']}\n"
)


@pytest.mark.parametrize("transport", ["http", "ws"])
def test_block_instance_keeps_numpy_string_keys_and_values(
    monkeypatch, modal_app, transport
):
    block = _build_first_frame_block(
        monkeypatch, modal_app, code=NP_STR_COUNTER_CODE, transport=transport
    )

    results = [block.run(image=frame) for frame in (1, 2, 3)]

    assert [r["calls"] for r in results] == [1, 2, 3]
    assert block.counts == {"cat": 3, "dog": 3}
    assert type(block.last_label) is np.str_
    assert [type(key) for key in block.counts] == [str, str]


NUMPY_SCALAR_SUBCLASS_CODE = (
    "import numpy as np\n"
    "class Count(np.int64):\n"
    "    def next(self):\n"
    "        return Count(self + 1)\n"
    "def run(self, image):\n"
    "    self.count = self.count.next() if hasattr(self, 'count') else Count(1)\n"
    "    return {'is_first_frame': True, 'calls': int(self.count)}\n"
)


@pytest.mark.parametrize("transport", ["http", "ws"])
def test_block_instance_survives_a_numpy_scalar_subclass(
    monkeypatch, modal_app, transport
):
    block = _build_first_frame_block(
        monkeypatch, modal_app, code=NUMPY_SCALAR_SUBCLASS_CODE, transport=transport
    )

    results = [block.run(image=frame) for frame in (1, 2, 3)]

    assert [r["calls"] for r in results] == [1, 1, 1]


@pytest.mark.parametrize("transport", ["http", "ws"])
def test_carriable_values_come_back_with_their_types(modal_app, transport):
    executor = _http_executor(modal_app)
    code = "def run(self, image):\n    return {'ok': True}\n"
    values = _carriable_values()

    # Sent exactly as the clients send it, so the round trip covers both
    # encoders and both decoders on each transport.
    if transport == "http":
        payload = {
            "code_str": code,
            "imports": [],
            "run_function_name": "run",
            "inputs_json": serialize_for_modal_remote_execution({"image": 1}),
            "block_state_json": _serialize_block_state_json(
                values, block_type_name="B"
            ),
        }
        response = asyncio.run(executor.execute_block(_FakeRequest(payload)))
        state = client._read_echoed_block_state_json(
            response, block_state=values, block_type_name="B"
        )
    else:
        packed = client.WebSocketModalExecutor._pack_block_state(
            values, block_type_name="B", msgpack=msgpack
        )
        response = _through_msgpack(
            modal_app.Executor._run_user_code_ws(
                executor,
                code,
                [],
                "run",
                modal_app.Executor._deserialize_msgpack_inputs(
                    _through_msgpack(serialize_inputs_for_msgpack({"image": 1}))
                ),
                "",
                {},
                block_state=modal_app.Executor._deserialize_msgpack_inputs(
                    _through_msgpack(packed)
                ),
            )
        )
        state = client._read_echoed_block_state_msgpack(
            response, block_state=values, block_type_name="B"
        )

    assert response["success"] is True, response.get("error")
    assert set(state) == set(values)
    for name in ("flag", "count", "big_but_fits", "ratio", "name", "raw", "when"):
        assert state[name] == values[name], name
        assert type(state[name]) is type(values[name]), name
    assert state["shape"] == (4, 4, 3) and type(state["shape"]) is tuple
    assert state["history"] == [1, (2, 3), {"nested": [None]}]
    assert type(state["label"]) is np.str_ and state["label"] == "cat"
    assert type(state["raw_label"]) is np.bytes_ and state["raw_label"] == b"cat"
    for name in ("scalar", "flag_scalar", "ratio_scalar"):
        assert type(state[name]) is type(values[name]), name
        assert state[name] == values[name], name
    assert type(state["zoned"].tzinfo).__name__ == "ZoneInfo"
    assert state["zoned"] == values["zoned"]
    assert state["per_class"] == {"cat": 1}
    assert [type(key) for key in state["per_class"]] == [str]


def test_ws_total_cap_is_measured_on_the_combined_payload():
    import msgpack as packer

    def measure(attribute):
        return len(
            packer.packb(serialize_inputs_for_msgpack(attribute), use_bin_type=True)
        )

    state = {"a": "x" * (BLOCK_STATE_MAX_BYTES - 11), "b": ""}
    assert measure(state) == BLOCK_STATE_MAX_BYTES

    partition = partition_block_state(state, measure=measure)

    assert set(partition.carried) == {"a", "b"}


def test_ws_total_cap_one_byte_over_drops_the_largest_attribute():
    import msgpack as packer

    def measure(attribute):
        return len(
            packer.packb(serialize_inputs_for_msgpack(attribute), use_bin_type=True)
        )

    state = {"a": "x" * (BLOCK_STATE_MAX_BYTES - 10), "b": ""}
    assert measure(state) == BLOCK_STATE_MAX_BYTES + 1

    partition = partition_block_state(state, measure=measure)

    assert partition.carried == {}
    assert [name for name, _ in partition.dropped] == ["a"]


def test_client_warns_separately_about_attributes_whose_names_differ_by_digits(
    monkeypatch,
):
    warned = _capture_client_warnings(monkeypatch)

    for name in ("buf1", "buf2", "buf2"):
        client._carry_block_state(
            {name: object()},
            block_type_name="FirstFrameCheck",
            workflow_context={"step_selector": "$steps.a"},
            measure=client._json_state_size,
        )

    rebuilt = [m for m in warned if "rebuilt" in m]
    assert len(rebuilt) == 2


def test_client_warns_once_about_an_oversized_attribute_whose_size_changes(
    monkeypatch,
):
    warned = _capture_client_warnings(monkeypatch)

    for extra in (1, 2, 3):
        client._carry_block_state(
            {"buffer": "x" * (BLOCK_STATE_MAX_BYTES + extra)},
            block_type_name="FirstFrameCheck",
            workflow_context={"step_selector": "$steps.a"},
            measure=client._json_state_size,
        )

    assert len([m for m in warned if "self.buffer" in m and "rebuilt" in m]) == 1


def test_sandbox_warns_separately_about_attributes_whose_names_differ_by_digits(
    modal_app,
):
    executor = _http_executor(modal_app)
    stderr = io.StringIO()

    for name in ("buf1", "buf2"):
        modal_app._warn_dropped_block_state(
            executor, "hash", [(name, "it is a Tracker")], [], stderr
        )

    assert "self.buf1" in stderr.getvalue()
    assert "self.buf2" in stderr.getvalue()


def test_client_scopes_the_missing_echo_warning_per_step(monkeypatch):
    warned = _capture_client_warnings(monkeypatch)

    for step in ("a", "b"):
        client._read_echoed_block_state_json(
            {"success": True},
            block_state={},
            block_type_name="FirstFrameCheck",
            workflow_context={"step_selector": f"$steps.{step}"},
        )

    assert len([m for m in warned if "older" in m]) == 2


class _OldThenNewHttpExecutor(_InProcessHttpExecutor):
    """An HTTP executor whose server stops echoing state while ``old`` is set."""

    old = False

    def _post(self, **payload):
        response = super()._post(**payload)
        if self.old:
            response.pop("block_state", None)
        return response


def test_block_instance_survives_new_old_new_webexec(monkeypatch, modal_app):
    block = _build_first_frame_block(monkeypatch, modal_app)
    executor = _OldThenNewHttpExecutor(modal_app)

    @contextmanager
    def acquire(workspace_id):
        yield executor

    monkeypatch.setattr(block_scaffolding, "_acquire_modal_executor", acquire)
    calls = []
    for old in (False, False, True, True, False, False):
        executor.old = old
        calls.append(block.run(image=1)["calls"])

    # The first call against the old server still receives the state; only its
    # echo is lost, so the engine clears the state from the next call on.
    assert calls == [1, 2, 3, 1, 1, 2]


def test_engine_notes_a_grayscale_image_gaining_channels():
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="frame-1"),
        numpy_image=np.zeros((4, 4), dtype=np.uint8),
    )

    partition = partition_block_state({"last": image}, measure=lambda a: 0)

    assert any("three channels" in note for note in partition.notes)


def test_engine_notes_integer_class_names_becoming_strings():
    detections = _detections()
    detections.data["class_name"] = np.array([7])

    partition = partition_block_state({"prev": detections}, measure=lambda a: 0)

    assert any("strings" in note for note in partition.notes)


def test_root_parent_note_is_skipped_for_placeholder_ids():
    detections = _detections()
    detections.data["parent_id"] = np.array(["input"])
    detections.data["root_parent_id"] = np.array(["input"])

    partition = partition_block_state({"prev": detections}, measure=lambda a: 0)

    assert not any("root_parent_id" in note for note in partition.notes)


def test_detections_rle_size_may_be_an_array():
    detections = _detections()
    del detections.data["image_dimensions"]
    detections.data["rle_mask"] = np.array(
        [{"size": np.array([20, 30]), "counts": "04"}], dtype=object
    )

    partition = partition_block_state({"prev": detections}, measure=lambda a: 0)

    assert "prev" in partition.carried


def test_custom_detections_data_keys_are_not_range_checked():
    detections = _detections()
    detections.data["hash"] = np.array([2**63], dtype=np.uint64)

    partition = partition_block_state({"prev": detections}, measure=lambda a: 0)

    assert "prev" in partition.carried


# --------------------------------------------------------------------------- #
# numpy scalar behaviour, time zones, image parentage, warning keys
# --------------------------------------------------------------------------- #


NUMPY_SCALAR_OPS_CODE = (
    "import numpy as np\n"
    "def run(self, image):\n"
    "    if not hasattr(self, 'best'):\n"
    "        self.best = np.array([0.25, 0.75]).max()\n"
    "        self.flag = np.bool_(True)\n"
    "        self.counter = np.uint8(254)\n"
    "    else:\n"
    "        self.flag = ~self.flag\n"
    "        self.counter = self.counter + np.uint8(1)\n"
    "    ops = [\n"
    "        self.best.item(),\n"
    "        float(self.best.round(1)),\n"
    "        float(self.best / np.float64(0)),\n"
    "        bool(self.flag),\n"
    "        int(self.counter),\n"
    "        isinstance(self.best, np.floating),\n"
    "    ]\n"
    "    return {'is_first_frame': True, 'calls': ops}\n"
)


@pytest.mark.parametrize("transport", ["http", "ws"])
def test_block_instance_keeps_numpy_scalar_behaviour(monkeypatch, modal_app, transport):
    block = _build_first_frame_block(
        monkeypatch, modal_app, code=NUMPY_SCALAR_OPS_CODE, transport=transport
    )

    with np.errstate(divide="ignore"):
        results = [block.run(image=frame)["calls"] for frame in (1, 2, 3)]

    assert [result[0] for result in results] == [0.75, 0.75, 0.75]
    assert [result[1] for result in results] == [0.8, 0.8, 0.8]
    assert [result[2] for result in results] == [float("inf")] * 3
    assert [result[3] for result in results] == [True, False, True]
    assert [result[4] for result in results] == [254, 255, 0]
    assert [result[5] for result in results] == [True, True, True]


ZONED_DATETIME_CODE = (
    "import datetime\n"
    "from zoneinfo import ZoneInfo\n"
    "def run(self, image):\n"
    "    if not hasattr(self, 'start'):\n"
    "        self.start = datetime.datetime(\n"
    "            2026, 3, 28, 12, tzinfo=ZoneInfo('Europe/Warsaw')\n"
    "        )\n"
    "    next_day = self.start + datetime.timedelta(days=1)\n"
    "    return {'is_first_frame': True, 'calls': str(next_day.utcoffset())}\n"
)


@pytest.mark.parametrize("transport", ["http", "ws"])
def test_block_instance_keeps_a_zoneinfo_time_zone(monkeypatch, modal_app, transport):
    block = _build_first_frame_block(
        monkeypatch, modal_app, code=ZONED_DATETIME_CODE, transport=transport
    )

    results = [block.run(image=frame)["calls"] for frame in (1, 2, 3)]

    assert results == ["2:00:00", "2:00:00", "2:00:00"]
    assert type(block.start.tzinfo).__name__ == "ZoneInfo"


def _root_image():
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="cam-1"),
        numpy_image=np.zeros((4, 4, 3), dtype=np.uint8),
    )
    return image


@pytest.mark.parametrize("partition_side", ["engine", "sandbox"])
def test_partition_notes_a_root_image_losing_its_parent_id(modal_app, partition_side):
    if partition_side == "engine":
        _, _, notes = partition_block_state(
            {"last": _root_image()}, measure=lambda a: 0
        )
    else:
        _, _, notes = modal_app._partition_block_state(
            {"last": _root_image()}, lambda a: 0
        )

    assert any("self.last" in note and "parent_id" in note for note in notes)


ROOT_IMAGE_ON_SELF_CODE = (
    "import numpy as np\n"
    "from inference.core.workflows.execution_engine.entities.base import (\n"
    "    ImageParentMetadata, WorkflowImageData,\n"
    ")\n"
    "def run(self, image):\n"
    "    self.last = WorkflowImageData(\n"
    "        parent_metadata=ImageParentMetadata(parent_id='cam-1'),\n"
    "        numpy_image=np.zeros((4, 4, 3), dtype=np.uint8),\n"
    "    )\n"
    "    return {'ok': True}\n"
)


def test_http_server_returns_a_root_image_with_a_placeholder_parent_id(modal_app):
    # An image the block builds itself keeps its own parent id locally; an
    # image that came in as an input already holds the placeholder.
    executor = _http_executor(modal_app)

    response = _run_http(executor, ROOT_IMAGE_ON_SELF_CODE, {"image": 1}, {})

    state = client._read_echoed_block_state_json(
        response, block_state={}, block_type_name="B"
    )
    assert state["last"].parent_metadata.parent_id in ("input", "result")
    assert "parent_id" in response["stderr"]


@pytest.mark.parametrize("partition_side", ["engine", "sandbox"])
def test_partition_notes_empty_detections_losing_data_and_metadata(
    modal_app, partition_side
):
    empty = sv.Detections.empty()
    empty.metadata["image_dimensions"] = [20, 30]
    if partition_side == "engine":
        _, _, notes = partition_block_state({"prev": empty}, measure=lambda a: 0)
    else:
        _, _, notes = modal_app._partition_block_state({"prev": empty}, lambda a: 0)

    assert any("self.prev" in note and "empty" in note for note in notes)


@pytest.mark.parametrize("partition_side", ["engine", "sandbox"])
def test_partition_notes_use_attribute_paths_not_wire_markers(
    modal_app, partition_side
):
    state = {"v": (_image(), 1)}
    if partition_side == "engine":
        _, _, notes = partition_block_state(state, measure=lambda a: 0)
    else:
        _, _, notes = modal_app._partition_block_state(state, lambda a: 0)

    assert any("self.v[0]" in note for note in notes)
    assert not any("'value'" in note for note in notes)


FIRST_SEEN_CODE = (
    "import numpy as np\n"
    "class Tracker:\n"
    "    pass\n"
    "def run(self, image):\n"
    "    self.first_seen = {str(image): Tracker()}\n"
    "    return {'ok': True}\n"
)


def test_http_server_warns_once_about_a_key_that_changes_every_frame(modal_app):
    executor = _http_executor(modal_app)

    responses = [
        _run_http(executor, FIRST_SEEN_CODE, {"image": frame}, block_state={})
        for frame in range(6)
    ]

    lines = [
        line
        for response in responses
        for line in (response["stderr"] or "").splitlines()
        if "self.first_seen" in line and "rebuilt" in line
    ]
    assert len(lines) == 1


def test_client_warns_once_about_a_key_that_changes_every_frame(monkeypatch):
    warned = _capture_client_warnings(monkeypatch)

    for frame in range(6):
        client._carry_block_state(
            {"first_seen": {str(frame): object()}},
            block_type_name="FirstFrameCheck",
            workflow_context={"step_selector": "$steps.a"},
            measure=client._json_state_size,
        )

    assert len([m for m in warned if "self.first_seen" in m and "rebuilt" in m]) == 1


class _OldThenNewWsExecutor(_InProcessWsExecutor):
    """A websocket executor whose server stops echoing state while ``old`` is set."""

    old = False

    def _send_recv(self, frame_bytes, workspace, request_id=None):
        response_bytes = super()._send_recv(frame_bytes, workspace, request_id)
        if not self.old:
            return response_bytes
        response = msgpack.unpackb(response_bytes, raw=False)
        response.pop("block_state", None)
        return msgpack.packb(response, use_bin_type=True)


def test_block_instance_survives_new_old_new_webexec_over_websocket(
    monkeypatch, modal_app
):
    block = _build_first_frame_block(monkeypatch, modal_app, transport="ws")
    executor = _OldThenNewWsExecutor(modal_app)

    @contextmanager
    def acquire(workspace_id):
        yield executor

    monkeypatch.setattr(block_scaffolding, "_acquire_modal_executor", acquire)
    calls = []
    for old in (False, False, True, True, False, False):
        executor.old = old
        calls.append(block.run(image=1)["calls"])

    assert calls == [1, 2, 3, 1, 1, 2]


# --------------------------------------------------------------------------- #
# zoned datetimes at the edges, and the echo as untrusted input
# --------------------------------------------------------------------------- #


ZONED_EDGES_CODE = (
    "import datetime\n"
    "from zoneinfo import ZoneInfo\n"
    "def run(self, image):\n"
    "    if not hasattr(self, 'values'):\n"
    "        warsaw, new_york = ZoneInfo('Europe/Warsaw'), ZoneInfo('America/New_York')\n"
    "        self.values = [\n"
    "            datetime.datetime.min.replace(tzinfo=warsaw),\n"
    "            datetime.datetime.max.replace(tzinfo=new_york),\n"
    "            datetime.datetime(2024, 3, 31, 2, 30, tzinfo=warsaw),\n"
    "            datetime.datetime(2024, 10, 27, 2, 30, fold=1, tzinfo=warsaw),\n"
    "        ]\n"
    "    described = [\n"
    "        [value.isoformat(), value.fold, value.tzinfo.key, value.hour]\n"
    "        for value in self.values\n"
    "    ]\n"
    "    return {'is_first_frame': True, 'calls': described}\n"
)


@pytest.mark.parametrize("transport", ["http", "ws"])
def test_block_instance_keeps_zoned_datetimes_at_the_edges(
    monkeypatch, modal_app, transport
):
    block = _build_first_frame_block(
        monkeypatch, modal_app, code=ZONED_EDGES_CODE, transport=transport
    )

    results = [block.run(image=frame)["calls"] for frame in (1, 2, 3)]

    assert results[1] == results[0]
    assert results[2] == results[0]
    assert results[0][3][1] == 1


def _read_forged_echo(transport, echoed_attribute):
    if transport == "http":
        echo = {"block_state": json.dumps({"x": echoed_attribute})}
        reader = client._read_echoed_block_state_json
    else:
        echo = {"block_state": {"x": echoed_attribute}}
        reader = client._read_echoed_block_state_msgpack
    state = reader(echo, block_state={"x": 1}, block_type_name="B")
    return state


FORGED_ATTRIBUTES = {
    "numpy_bytes_from_an_int": {
        "_type": "numpy_scalar",
        "dtype": "S1",
        "value": 50_000_000,
    },
    "numpy_scalar_wrong_value_type": {
        "_type": "numpy_scalar",
        "dtype": "int64",
        "value": "1",
    },
    "numpy_scalar_out_of_range": {
        "_type": "numpy_scalar",
        "dtype": "uint8",
        "value": 300,
    },
    "numpy_scalar_object_dtype": {"_type": "numpy_scalar", "dtype": "O", "value": 1},
    "numpy_scalar_longdouble": {"_type": "numpy_scalar", "dtype": "g", "value": 1.0},
    "numpy_scalar_uint64_beyond_int64": {
        "_type": "numpy_scalar",
        "dtype": "Q",
        "value": 2**64 - 1,
    },
    "numpy_scalar_huge_str_dtype": {
        "_type": "numpy_scalar",
        "dtype": "<U99999999",
        "value": "a",
    },
    "ndarray_huge_dtype": {
        "_type": "ndarray",
        "dtype": "S50000000",
        "shape": [1],
        "value": ["a"],
    },
    "ndarray_object_dtype": {
        "_type": "ndarray",
        "dtype": "O",
        "shape": [1],
        "value": [1],
    },
    "image_from_a_file": {
        "_type": "workflow_image",
        "type": "file",
        "value": "/etc/hosts",
    },
    "untagged_base64_image": {"type": "base64", "value": "abc"},
    "unknown_marker": {"_type": "batch", "value": [1]},
    "zoned_datetime_bad_fold": {
        "_type": "zoned_datetime",
        "value": "2020-01-01T00:00:00",
        "fold": 7,
        "zone": "UTC",
    },
}


@pytest.mark.parametrize("transport", ["http", "ws"])
@pytest.mark.parametrize(
    "forged", FORGED_ATTRIBUTES.values(), ids=list(FORGED_ATTRIBUTES)
)
def test_client_refuses_a_forged_echo_before_building_values(transport, forged):
    with pytest.raises(DynamicBlockError) as error:
        _read_forged_echo(transport, forged)

    assert "self.x" in str(error.value)


@pytest.mark.parametrize("transport", ["http", "ws"])
def test_client_refuses_a_deeply_nested_echo(transport):
    nested = 1
    for _ in range(5000):
        nested = [nested]
    if transport == "http":
        echo = {"block_state": '{"x": ' + "[" * 5000 + "1" + "]" * 5000 + "}"}
        reader = client._read_echoed_block_state_json
    else:
        echo = {"block_state": {"x": nested}}
        reader = client._read_echoed_block_state_msgpack

    with pytest.raises(DynamicBlockError):
        reader(echo, block_state={"x": 1}, block_type_name="B")


@pytest.mark.parametrize("transport", ["http", "ws"])
def test_client_refuses_an_echo_with_too_many_attributes(transport):
    attributes = {f"a{index}": 1 for index in range(BLOCK_STATE_MAX_ATTRIBUTES + 1)}
    if transport == "http":
        echo = {"block_state": json.dumps(attributes)}
        reader = client._read_echoed_block_state_json
    else:
        echo = {"block_state": attributes}
        reader = client._read_echoed_block_state_msgpack

    with pytest.raises(DynamicBlockError):
        reader(echo, block_state={"x": 1}, block_type_name="B")


@pytest.mark.parametrize("partition_side", ["engine", "sandbox"])
def test_partition_withholds_state_with_too_many_attributes(modal_app, partition_side):
    state = {f"a{index}": 1 for index in range(BLOCK_STATE_MAX_ATTRIBUTES + 1)}
    if partition_side == "engine":
        carried, dropped, _ = partition_block_state(state, measure=lambda a: 0)
    else:
        carried, dropped, _ = modal_app._partition_block_state(state, lambda a: 0)

    assert carried == {}
    assert dropped and "attributes" in dropped[0][1]


NUMPY_KEY_CODE = (
    "import numpy as np\n"
    "def run(self, image):\n"
    "    self.first_seen = {np.int64(image): 1}\n"
    "    return {'ok': True}\n"
)


def test_http_server_warns_once_about_a_non_string_key_that_changes(modal_app):
    executor = _http_executor(modal_app)

    responses = [
        _run_http(executor, NUMPY_KEY_CODE, {"image": frame}, block_state={})
        for frame in range(6)
    ]

    lines = [
        line
        for response in responses
        for line in (response["stderr"] or "").splitlines()
        if "self.first_seen" in line and "rebuilt" in line
    ]
    assert len(lines) == 1


TYPE_AND_VALUE_DICTS_CODE = (
    "def run(self, image):\n"
    "    if not hasattr(self, 'last_event'):\n"
    "        self.last_event = {'type': 'line_crossed', 'value': 3}\n"
    "        self.events = [\n"
    "            {'type': 'line_crossed', 'value': 3},\n"
    "            {'type': 'zone_left', 'value': None},\n"
    "        ]\n"
    "        self.pair = (\n"
    "            {'type': 'line_crossed', 'value': 3},\n"
    "            {'type': 'file', 'value': '/etc/hosts'},\n"
    "        )\n"
    "        self.calls = 0\n"
    "    self.calls += 1\n"
    "    return {'is_first_frame': self.calls == 1, 'calls': self.calls}\n"
)


@pytest.mark.parametrize("transport", ["http", "ws"])
def test_block_instance_keeps_dicts_with_type_and_value_keys(
    monkeypatch, modal_app, transport
):
    block = _build_first_frame_block(
        monkeypatch, modal_app, code=TYPE_AND_VALUE_DICTS_CODE, transport=transport
    )

    results = [block.run(image=frame) for frame in (1, 2, 3)]

    assert [r["calls"] for r in results] == [1, 2, 3]
    assert block.last_event == {"type": "line_crossed", "value": 3}
    assert block.events == [
        {"type": "line_crossed", "value": 3},
        {"type": "zone_left", "value": None},
    ]
    assert type(block.pair) is tuple
    assert block.pair == (
        {"type": "line_crossed", "value": 3},
        {"type": "file", "value": "/etc/hosts"},
    )


@pytest.mark.parametrize("transport", ["http", "ws"])
def test_client_decodes_an_untagged_dict_naming_a_file_as_a_plain_dict(transport):
    state = _read_forged_echo(transport, {"type": "file", "value": "/etc/hosts"})

    assert state == {"x": {"type": "file", "value": "/etc/hosts"}}


DEEP_STATE_CODE = (
    "import numpy as np\n"
    "import supervision as sv\n"
    "def build(kind, extra):\n"
    "    if kind == 'lists':\n"
    "        value = 1\n"
    "        for _ in range(64):\n"
    "            value = [value]\n"
    "    elif kind == 'dicts':\n"
    "        value = 1\n"
    "        for _ in range(64):\n"
    "            value = {'k': value}\n"
    "    elif kind == 'numpy_bytes':\n"
    "        value = np.bytes_(b'x')\n"
    "        for _ in range(62):\n"
    "            value = [value]\n"
    "    elif kind == 'tuples':\n"
    "        value = 1\n"
    "        for _ in range(32):\n"
    "            value = (value,)\n"
    "    else:\n"
    "        value = [sv.Detections(\n"
    "            xyxy=np.array([[0.0, 0.0, 8.0, 8.0]]),\n"
    "            mask=np.ones((1, 8, 8), dtype=bool),\n"
    "            class_id=np.array([0]),\n"
    "            confidence=np.array([0.5]),\n"
    "            data={\n"
    "                'class_name': np.array(['cat']),\n"
    "                'detection_id': np.array(['d0']),\n"
    "                'image_dimensions': np.array([[8, 8]]),\n"
    "            },\n"
    "        )]\n"
    "        for _ in range(29):\n"
    "            value = (value,)\n"
    "    for _ in range(extra):\n"
    "        value = [value]\n"
    "    return value\n"
    "def run(self, image):\n"
    "    if not hasattr(self, 'deep'):\n"
    "        self.deep = build(KIND, EXTRA)\n"
    "        self.calls = 0\n"
    "    self.calls += 1\n"
    "    return {'is_first_frame': self.calls == 1, 'calls': self.calls}\n"
)


def _unwrap_deep_state(kind, value):
    if kind == "lists":
        for _ in range(64):
            assert type(value) is list
            value = value[0]
    elif kind == "dicts":
        for _ in range(64):
            assert type(value) is dict
            value = value["k"]
    elif kind == "numpy_bytes":
        for _ in range(62):
            assert type(value) is list
            value = value[0]
        assert type(value) is np.bytes_
        value = 1 if value == b"x" else value
    elif kind == "tuples":
        for _ in range(32):
            assert type(value) is tuple
            value = value[0]
    else:
        for _ in range(29):
            assert type(value) is tuple
            value = value[0]
        assert type(value) is list
        value = value[0]
    return value


@pytest.mark.parametrize("transport", ["http", "ws"])
@pytest.mark.parametrize(
    "kind", ["lists", "dicts", "numpy_bytes", "tuples", "tuple_holding_detections"]
)
def test_block_instance_keeps_the_deepest_carried_state(
    monkeypatch, modal_app, capsys, transport, kind
):
    code = DEEP_STATE_CODE.replace("KIND", repr(kind)).replace("EXTRA", "0")
    block = _build_first_frame_block(
        monkeypatch, modal_app, code=code, transport=transport
    )

    results = [block.run(image=frame) for frame in (1, 2, 3)]

    assert [r["calls"] for r in results] == [1, 2, 3]
    bottom = _unwrap_deep_state(kind, block.deep)
    if kind == "tuple_holding_detections":
        assert isinstance(bottom, sv.Detections) and len(bottom) == 1
        assert bottom.mask is not None
    else:
        assert bottom == 1
    assert "withheld" not in capsys.readouterr().err


@pytest.mark.parametrize("transport", ["http", "ws"])
@pytest.mark.parametrize(
    "kind", ["lists", "dicts", "numpy_bytes", "tuples", "tuple_holding_detections"]
)
def test_block_instance_withholds_state_one_level_deeper_than_carried(
    monkeypatch, modal_app, capsys, transport, kind
):
    code = DEEP_STATE_CODE.replace("KIND", repr(kind)).replace("EXTRA", "1")
    block = _build_first_frame_block(
        monkeypatch, modal_app, code=code, transport=transport
    )

    results = [block.run(image=frame) for frame in (1, 2, 3)]

    assert [r["calls"] for r in results] == [1, 1, 1]
    assert not hasattr(block, "deep")
    assert "self.deep" in capsys.readouterr().err


@pytest.mark.parametrize("partition_side", ["engine", "sandbox"])
def test_partition_drops_state_nested_deeper_than_the_echo_accepts(
    modal_app, partition_side
):
    deep = 1
    for _ in range(65):
        deep = [deep]
    if partition_side == "engine":
        carried, dropped, _ = partition_block_state({"deep": deep}, measure=lambda a: 0)
    else:
        carried, dropped, _ = modal_app._partition_block_state(
            {"deep": deep}, lambda a: 0
        )

    assert carried == {}
    assert [name for name, _ in dropped] == ["deep"]


def test_engine_and_sandbox_agree_on_the_state_depth_limit(modal_app):
    from roboflow_workflows.execution_engine.v1.dynamic_blocks import block_state

    assert modal_app._MAX_ECHO_DEPTH == block_state._MAX_ECHO_DEPTH


@pytest.fixture
def image_loading_calls(monkeypatch, tmp_path):
    import os

    import cv2
    from roboflow_workflows.core_steps.common import deserializers

    calls = []
    real_exists = os.path.exists
    real_imread = cv2.imread

    def exists(path):
        calls.append(("exists", path))
        return real_exists(path)

    def imread(path, *args, **kwargs):
        calls.append(("imread", path))
        return real_imread(path, *args, **kwargs)

    class RecordingCodec:
        def fetch_url(self, value, *args, **kwargs):
            calls.append(("fetch_url", value))
            return np.zeros((2, 2, 3), dtype=np.uint8)

        def decode_string(self, value, *args, **kwargs):
            calls.append(("decode_string", value))
            return np.zeros((2, 2, 3), dtype=np.uint8), True

        def ensure_local_file_load_allowed(self, path):
            calls.append(("ensure_local_file_load_allowed", path))

        def load_image(self, value, *args, **kwargs):
            calls.append(("load_image", value))
            return np.zeros((2, 2, 3), dtype=np.uint8), True

    monkeypatch.setattr(os.path, "exists", exists)
    monkeypatch.setattr(cv2, "imread", imread)
    monkeypatch.setattr(deserializers, "get_image_codec", lambda: RecordingCodec())
    return calls


def _forged_image_payloads(tmp_path) -> dict:
    import cv2

    host_file = tmp_path / "host.jpg"
    cv2.imwrite(str(host_file), np.zeros((4, 4, 3), dtype=np.uint8))
    payloads = {
        "url": "http://169.254.169.254/latest/meta-data/image.jpg",
        "host_file": str(host_file),
        "etc_hosts": "/etc/hosts",
        "int": 7,
        "list": [1, 2],
    }
    return payloads


FORGED_IMAGE_PAYLOAD_IDS = ["url", "host_file", "etc_hosts", "int", "list"]


def _loading_calls_for(calls, payload):
    loading_calls = [
        call
        for call in calls
        if call[0] != "exists" or (isinstance(payload, str) and call[1] == payload)
    ]
    return loading_calls


@pytest.mark.parametrize("transport", ["http", "ws"])
@pytest.mark.parametrize("payload_id", FORGED_IMAGE_PAYLOAD_IDS)
def test_client_refuses_a_forged_echo_image_without_loading_it(
    transport, payload_id, image_loading_calls, tmp_path
):
    payload = _forged_image_payloads(tmp_path)[payload_id]
    forged = {"_type": "workflow_image", "type": "base64", "value": payload}

    with pytest.raises(DynamicBlockError) as error:
        _read_forged_echo(transport, forged)

    assert "self.x" in str(error.value)
    assert _loading_calls_for(image_loading_calls, payload) == []


@pytest.mark.parametrize("transport", ["http", "ws"])
@pytest.mark.parametrize("payload_id", FORGED_IMAGE_PAYLOAD_IDS)
def test_client_refuses_a_forged_result_image_without_loading_it(
    transport, payload_id, image_loading_calls, tmp_path
):
    payload = _forged_image_payloads(tmp_path)[payload_id]
    forged = {"_type": "workflow_image", "type": "base64", "value": payload}

    with pytest.raises(Exception):
        if transport == "http":
            deserialize_for_modal_remote_execution(json.dumps({"out": forged}))
        else:
            client._deserialize_msgpack_result(_through_msgpack({"out": forged}))

    assert _loading_calls_for(image_loading_calls, payload) == []


@pytest.mark.parametrize("payload_id", FORGED_IMAGE_PAYLOAD_IDS)
def test_http_client_refuses_a_forged_untagged_result_image_without_loading_it(
    payload_id, image_loading_calls, tmp_path
):
    payload = _forged_image_payloads(tmp_path)[payload_id]

    with pytest.raises(Exception):
        deserialize_for_modal_remote_execution(
            json.dumps({"out": {"type": "base64", "value": payload}})
        )

    assert _loading_calls_for(image_loading_calls, payload) == []


def _crop_image():
    from roboflow_workflows.execution_engine.entities.base import (
        OriginCoordinatesSystem,
        VideoMetadata,
    )

    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(
            parent_id="crop-1",
            origin_coordinates=OriginCoordinatesSystem(
                left_top_x=2, left_top_y=3, origin_width=32, origin_height=24
            ),
        ),
        workflow_root_ancestor_metadata=ImageParentMetadata(
            parent_id="root-1",
            origin_coordinates=OriginCoordinatesSystem(
                left_top_x=5, left_top_y=7, origin_width=64, origin_height=48
            ),
        ),
        numpy_image=_image().numpy_image,
        video_metadata=VideoMetadata(
            video_identifier="cam-7",
            frame_number=9,
            frame_timestamp=datetime(2026, 1, 2, 3, 4, 5),
            fps=25,
        ),
    )
    return image


def _assert_crop_lineage(image):
    expected = _crop_image()
    assert isinstance(image, WorkflowImageData)
    assert image.parent_metadata == expected.parent_metadata
    assert (
        image.workflow_root_ancestor_metadata
        == expected.workflow_root_ancestor_metadata
    )
    assert image.video_metadata.video_identifier == "cam-7"
    assert image.video_metadata.frame_number == 9
    assert image.video_metadata.frame_timestamp == datetime(2026, 1, 2, 3, 4, 5)
    assert image.numpy_image.shape == (16, 16, 3)


@pytest.mark.parametrize("transport", ["http", "ws"])
def test_image_lineage_and_video_metadata_survive_a_server_round_trip(
    modal_app, transport
):
    executor = _http_executor(modal_app)
    code = "def run(self, image):\n    return {'image': self.last}\n"
    seeded = {"last": _crop_image()}

    if transport == "http":
        response = _run_http(executor, code, {"image": 1}, seeded)
        result = deserialize_for_modal_remote_execution(response["result"])
        state = client._read_echoed_block_state_json(
            response, block_state=seeded, block_type_name="B"
        )
    else:
        response = modal_app.Executor._run_user_code_ws(
            executor,
            code,
            [],
            "run",
            modal_app.Executor._deserialize_msgpack_inputs(
                _through_msgpack(serialize_inputs_for_msgpack({"image": 1}))
            ),
            "",
            {},
            block_state=modal_app.Executor._deserialize_msgpack_inputs(
                _through_msgpack(serialize_inputs_for_msgpack(seeded))
            ),
        )
        response["result"] = modal_app.Executor._serialize_msgpack_result(
            response["result"]
        )
        response = _through_msgpack(response)
        result = client._deserialize_msgpack_result(response["result"])
        state = client._read_echoed_block_state_msgpack(
            response, block_state=seeded, block_type_name="B"
        )

    assert response["success"] is True, response.get("error")
    _assert_crop_lineage(state["last"])
    _assert_crop_lineage(result["image"])


FORGED_ARRAYS = {
    "zero_width_string_dtype": {
        "_type": "ndarray",
        "dtype": "U0",
        "shape": [2048],
        "value": ["a" * 2048] + [""] * 2047,
    },
    "empty_shape_with_a_huge_dtype": {
        "_type": "ndarray",
        "dtype": "U10000000",
        "shape": [0],
        "value": [""],
    },
    "more_values_than_the_shape": {
        "_type": "ndarray",
        "dtype": "U1000000",
        "shape": [1],
        "value": [""] * 5,
    },
}


def _peak_traced_bytes(action) -> int:
    import tracemalloc

    tracemalloc.start()
    try:
        action()
    finally:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return peak


@pytest.mark.parametrize("transport", ["http", "ws"])
@pytest.mark.parametrize("forged", FORGED_ARRAYS.values(), ids=list(FORGED_ARRAYS))
def test_client_refuses_a_forged_echo_array_without_allocating_it(transport, forged):
    def read():
        with pytest.raises(DynamicBlockError):
            _read_forged_echo(transport, forged)

    assert _peak_traced_bytes(read) < 3 * 1024 * 1024


@pytest.mark.parametrize("transport", ["http", "ws"])
@pytest.mark.parametrize("forged", FORGED_ARRAYS.values(), ids=list(FORGED_ARRAYS))
def test_client_refuses_a_forged_result_array_without_allocating_it(transport, forged):
    def read():
        with pytest.raises(ValueError):
            if transport == "http":
                deserialize_for_modal_remote_execution(json.dumps({"out": forged}))
            else:
                client._deserialize_msgpack_result(_through_msgpack({"out": forged}))

    assert _peak_traced_bytes(read) < 3 * 1024 * 1024


def _honest_arrays() -> dict:
    arrays = {
        "zero_d": np.array(3.5),
        "empty_rows": np.zeros((0, 4)),
        "three_d": np.arange(24, dtype=np.int32).reshape(2, 3, 4),
        "strings": np.array([["a", "bcd"], ["", "e"]]),
        "empty_strings": np.array(["", ""]),
        "no_strings": np.array([], dtype=str),
    }
    return arrays


@pytest.mark.parametrize("transport", ["http", "ws"])
def test_honest_arrays_round_trip_as_results(transport):
    arrays = _honest_arrays()

    if transport == "http":
        decoded = deserialize_for_modal_remote_execution(
            serialize_for_modal_remote_execution(arrays)
        )
    else:
        decoded = client._deserialize_msgpack_result(
            _through_msgpack(serialize_inputs_for_msgpack(arrays))
        )

    for name, array in arrays.items():
        assert decoded[name].shape == array.shape, name
        assert decoded[name].dtype == array.dtype, name
        assert np.array_equal(decoded[name], array), name


@pytest.mark.parametrize("transport", ["http", "ws"])
def test_honest_arrays_round_trip_as_state_through_the_server(modal_app, transport):
    executor = _http_executor(modal_app)
    code = "def run(self, image):\n    return {'ok': True}\n"
    arrays = _honest_arrays()

    if transport == "http":
        response = _run_http(executor, code, {"image": 1}, arrays)
        state = client._read_echoed_block_state_json(
            response, block_state=arrays, block_type_name="B"
        )
    else:
        response = _run_ws(executor, modal_app, code, {"image": 1}, arrays)
        state = client._read_echoed_block_state_msgpack(
            response, block_state=arrays, block_type_name="B"
        )

    for name, array in arrays.items():
        assert state[name].shape == array.shape, name
        assert np.array_equal(state[name], array), name


def test_http_client_refuses_an_oversized_echo_before_parsing_it(monkeypatch):
    cap = client._MAX_ECHOED_BLOCK_STATE_JSON_LENGTH
    echo = {"block_state": '{"x": "' + "a" * cap + '"}'}
    parsed = []
    real_loads = client.json.loads

    def loads(value, *args, **kwargs):
        parsed.append(len(value))
        return real_loads(value, *args, **kwargs)

    monkeypatch.setattr(client.json, "loads", loads)

    with pytest.raises(DynamicBlockError):
        client._read_echoed_block_state_json(
            echo, block_state={"x": 1}, block_type_name="B"
        )

    assert parsed == []


def test_http_client_accepts_an_echo_at_the_length_cap():
    cap = client._MAX_ECHOED_BLOCK_STATE_JSON_LENGTH
    filler = cap - len('{"x": ""}')
    echo = {"block_state": '{"x": "' + "a" * filler + '"}'}

    state = client._read_echoed_block_state_json(
        echo, block_state={"x": 1}, block_type_name="B"
    )

    assert len(echo["block_state"]) == cap
    assert len(state["x"]) == filler


def test_http_server_echo_of_the_largest_carried_state_fits_the_cap(modal_app):
    executor = _http_executor(modal_app)
    length = BLOCK_STATE_MAX_BYTES - len('{"big": ""}')
    code = f"def run(self, image):\n    self.big = 'a' * {length}\n    return {{}}\n"

    response = _run_http(executor, code, {"image": 1}, block_state={})
    state = client._read_echoed_block_state_json(
        response, block_state={}, block_type_name="B"
    )

    assert response["success"] is True, response.get("error")
    assert len(response["block_state"]) == BLOCK_STATE_MAX_BYTES
    assert BLOCK_STATE_MAX_BYTES <= client._MAX_ECHOED_BLOCK_STATE_JSON_LENGTH
    assert len(state["big"]) == length


ALLOCATION_BUDGET = 8 * BLOCK_STATE_MAX_BYTES


def _huge_string_array_marker():
    marker = {
        "_type": "ndarray",
        "value": ["a"],
        "dtype": "<U2000000",
        "shape": [1],
    }
    return marker


def _huge_string_scalar_marker():
    marker = {"_type": "numpy_scalar", "dtype": "<U2000000", "value": "a"}
    return marker


FORGED_ALLOCATION_ECHOES = {
    "arrays_in_one_list": lambda: {"x": [_huge_string_array_marker()] * 5},
    "arrays_across_attributes": lambda: {
        f"x{index}": _huge_string_array_marker() for index in range(5)
    },
    "numpy_scalars_in_one_list": lambda: {"x": [_huge_string_scalar_marker()] * 5},
}


def _read_forged_echo_state(transport, echoed_state):
    if transport == "http":
        echo = {"block_state": json.dumps(echoed_state)}
        reader = client._read_echoed_block_state_json
    else:
        echo = {"block_state": _through_msgpack(echoed_state)}
        reader = client._read_echoed_block_state_msgpack
    state = reader(echo, block_state={"x": 1}, block_type_name="B")
    return state


@pytest.mark.parametrize("transport", ["http", "ws"])
@pytest.mark.parametrize(
    "forged", FORGED_ALLOCATION_ECHOES.values(), ids=list(FORGED_ALLOCATION_ECHOES)
)
def test_client_refuses_an_echo_whose_values_together_exceed_the_allocation_budget(
    transport, forged
):
    echoed_state = forged()

    def read():
        with pytest.raises(DynamicBlockError):
            _read_forged_echo_state(transport, echoed_state)

    assert _peak_traced_bytes(read) < 3 * 1024 * 1024


def test_engine_and_sandbox_agree_on_the_state_allocation_limit(modal_app):
    from roboflow_workflows.execution_engine.v1.dynamic_blocks import block_state

    assert (
        modal_app._MAX_ECHO_ALLOCATION_BYTES
        == block_state._MAX_ECHO_ALLOCATION_BYTES
        == ALLOCATION_BUDGET
    )


OVER_ALLOCATION_VALUES = {
    "one_array": lambda: np.zeros(1, dtype="U3000000"),
    "two_arrays_together": lambda: [np.zeros(1, dtype="U1500000")] * 2,
    "detections_field": lambda: sv.Detections(
        xyxy=np.array([[0, 0, 10, 10]], dtype=float),
        class_id=np.array([1]),
        confidence=np.array([0.5]),
        data={
            "class_name": np.array(["a"]),
            "detection_id": np.array(["d"]),
            "image_dimensions": np.array([[20, 20]]),
            "data": np.zeros((1, 1), dtype="U3000000"),
        },
    ),
}


@pytest.mark.parametrize("partition_side", ["engine", "sandbox"])
@pytest.mark.parametrize(
    "build", OVER_ALLOCATION_VALUES.values(), ids=list(OVER_ALLOCATION_VALUES)
)
def test_partition_drops_state_that_would_allocate_over_the_budget(
    modal_app, partition_side, build
):
    state = {"small": 1, "big": build()}
    if partition_side == "engine":
        carried, dropped, _ = partition_block_state(state, measure=_json_state_size)
    else:
        carried, dropped, _ = modal_app._partition_block_state(state, _json_state_size)

    assert carried == {}
    assert [name for name, _ in dropped] == ["big"]


def _json_state_size(attribute):
    size = len(serialize_for_modal_remote_execution(attribute))
    return size


def test_partition_drops_attributes_that_together_allocate_over_the_budget(
    modal_app,
):
    state = {
        "first": np.zeros(1, dtype=f"U{ALLOCATION_BUDGET // 8}"),
        "second": np.zeros(1, dtype=f"U{ALLOCATION_BUDGET // 8 + 1}"),
    }

    engine = partition_block_state(state, measure=_json_state_size)
    sandbox_carried, sandbox_dropped, _ = modal_app._partition_block_state(
        state, _json_state_size
    )

    assert engine.carried == {} and sandbox_carried == {}
    assert [name for name, _ in engine.dropped] == ["second"]
    assert engine.dropped == sandbox_dropped


@pytest.mark.parametrize("partition_side", ["engine", "sandbox"])
def test_partition_carries_state_at_the_allocation_budget(modal_app, partition_side):
    state = {"big": np.zeros(1, dtype=f"U{ALLOCATION_BUDGET // 4}")}
    if partition_side == "engine":
        carried, dropped, _ = partition_block_state(state, measure=_json_state_size)
    else:
        carried, dropped, _ = modal_app._partition_block_state(state, _json_state_size)

    assert dropped == []
    assert list(carried) == ["big"]


OVER_ALLOCATION_CODE = (
    "import numpy as np\n"
    "def run(self, image):\n"
    "    self.calls = getattr(self, 'calls', 0) + 1\n"
    "    self.big = np.zeros(1, dtype='U3000000')\n"
    "    return {'is_first_frame': self.calls == 1, 'calls': self.calls}\n"
)


@pytest.mark.parametrize("transport", ["http", "ws"])
def test_block_instance_withholds_sandbox_state_over_the_allocation_budget(
    monkeypatch, modal_app, capsys, transport
):
    block = _build_first_frame_block(
        monkeypatch, modal_app, code=OVER_ALLOCATION_CODE, transport=transport
    )

    results = [block.run(image=frame) for frame in (1, 2, 3)]

    assert [r["calls"] for r in results] == [1, 1, 1]
    assert not hasattr(block, "big")
    assert "self.big" in capsys.readouterr().err


@pytest.mark.parametrize("transport", ["http", "ws"])
def test_block_instance_withholds_engine_state_over_the_allocation_budget(
    monkeypatch, modal_app, transport
):
    warned = _capture_client_warnings(monkeypatch)
    block = _build_first_frame_block(monkeypatch, modal_app, transport=transport)
    block.big = np.zeros(1, dtype="U3000000")

    results = [block.run(image=frame) for frame in (1, 2, 3)]

    assert [r["calls"] for r in results] == [1, 2, 3]
    assert not hasattr(block, "big")
    assert any("self.big" in message for message in warned)


AT_ALLOCATION_BUDGET_CODE = (
    "import numpy as np\n"
    "def run(self, image):\n"
    "    if not hasattr(self, 'big'):\n"
    f"        self.big = np.full(1, 'ab', dtype='U{ALLOCATION_BUDGET // 4}')\n"
    "    self.calls = getattr(self, 'calls', 0) + 1\n"
    "    return {'is_first_frame': self.calls == 1, 'calls': self.calls}\n"
)


@pytest.mark.parametrize("transport", ["http", "ws"])
def test_block_instance_keeps_state_at_the_allocation_budget(
    monkeypatch, modal_app, capsys, transport
):
    block = _build_first_frame_block(
        monkeypatch, modal_app, code=AT_ALLOCATION_BUDGET_CODE, transport=transport
    )

    results = [block.run(image=frame) for frame in (1, 2, 3)]

    assert [r["calls"] for r in results] == [1, 2, 3]
    assert block.big.dtype == np.dtype(f"U{ALLOCATION_BUDGET // 4}")
    assert block.big.tolist() == ["ab"]
    assert "withheld" not in capsys.readouterr().err


def _jpeg_base64():
    import base64

    import cv2

    pixels = np.zeros((64, 64, 3), dtype=np.uint8)
    pixels[16:48, 16:48] = (40, 160, 220)
    _, encoded = cv2.imencode(".jpg", pixels)
    return base64.encodebytes(encoded.tobytes()).decode("ascii")


def _whitespace_images() -> dict:
    wrapped = _jpeg_base64()
    images = {
        "encodebytes_wrapped": wrapped,
        "trailing_newline": wrapped.replace("\n", "") + "\n",
        "crlf_and_tabs": wrapped.replace("\n", "\r\n\t "),
    }
    return images


def _image_marker(value):
    marker = {"_type": "workflow_image", "type": "base64", "value": value}
    return marker


def _decode_result(transport, result):
    if transport == "http":
        decoded = deserialize_for_modal_remote_execution(json.dumps(result))
    else:
        decoded = client._deserialize_msgpack_result(_through_msgpack(result))
    return decoded


@pytest.mark.parametrize("transport", ["http", "ws"])
@pytest.mark.parametrize("name", list(_whitespace_images()))
def test_image_with_base64_whitespace_decodes_as_a_result(transport, name):
    value = _whitespace_images()[name]

    decoded = _decode_result(transport, {"out": _image_marker(value)})["out"]

    assert isinstance(decoded, WorkflowImageData)
    assert decoded.numpy_image.shape == (64, 64, 3)


@pytest.mark.parametrize("transport", ["http", "ws"])
@pytest.mark.parametrize("name", list(_whitespace_images()))
def test_image_with_base64_whitespace_decodes_as_state(transport, name):
    value = _whitespace_images()[name]

    state = _read_forged_echo(transport, _image_marker(value))

    assert isinstance(state["x"], WorkflowImageData)
    assert state["x"].numpy_image.shape == (64, 64, 3)


def _non_base64_payloads_with_whitespace(tmp_path) -> list:
    host_file = tmp_path / "host.jpg"
    payloads = [
        "http://169.254.169.254/latest/image.jpg\n",
        " /etc/hosts\n",
        f"{host_file}\n",
        "abcd:efgh\n",
        _jpeg_base64().replace("\n", ":\n"),
    ]
    return payloads


@pytest.mark.parametrize("transport", ["http", "ws"])
@pytest.mark.parametrize("index", range(5))
def test_image_with_non_base64_characters_and_whitespace_is_refused_without_loading(
    transport, index, image_loading_calls, tmp_path
):
    import cv2

    cv2.imwrite(str(tmp_path / "host.jpg"), np.zeros((4, 4, 3), dtype=np.uint8))
    payload = _non_base64_payloads_with_whitespace(tmp_path)[index]

    with pytest.raises(DynamicBlockError):
        _read_forged_echo(transport, _image_marker(payload))
    with pytest.raises(ValueError):
        _decode_result(transport, {"out": _image_marker(payload)})

    assert _loading_calls_for(image_loading_calls, payload.strip()) == []


def _object_arrays() -> dict:
    two_by_two = np.empty((2, 2), dtype=object)
    two_by_two[0, 0], two_by_two[0, 1] = 1, "a"
    two_by_two[1, 0], two_by_two[1, 1] = None, {"k": [1, 2]}
    arrays = {
        "ragged_lists": np.array([[1, 2], [3]], dtype=object),
        "dicts": np.array([{"a": 1}, {"b": [2]}], dtype=object),
        "none_and_strings": np.array([None, "x", "yz"], dtype=object),
        "two_d": two_by_two,
        "empty": np.empty((0, 3), dtype=object),
    }
    return arrays


def _decode_result_with_base_decoder(transport, encoded):
    if transport == "http":
        parsed = json.loads(encoded)
    else:
        parsed = encoded
    decoded = {
        name: np.array(marker["value"], dtype=marker["dtype"]).reshape(marker["shape"])
        for name, marker in parsed.items()
    }
    return decoded


@pytest.mark.parametrize("transport", ["http", "ws"])
def test_object_arrays_round_trip_as_results_through_the_server(modal_app, transport):
    executor = _http_executor(modal_app)
    code = (
        "import numpy as np\n"
        "def run(self, image):\n"
        "    two_by_two = np.empty((2, 2), dtype=object)\n"
        "    two_by_two[0, 0], two_by_two[0, 1] = 1, 'a'\n"
        "    two_by_two[1, 0], two_by_two[1, 1] = None, {'k': [1, 2]}\n"
        "    return {\n"
        "        'ragged_lists': np.array([[1, 2], [3]], dtype=object),\n"
        "        'dicts': np.array([{'a': 1}, {'b': [2]}], dtype=object),\n"
        "        'none_and_strings': np.array([None, 'x', 'yz'], dtype=object),\n"
        "        'two_d': two_by_two,\n"
        "        'empty': np.empty((0, 3), dtype=object),\n"
        "    }\n"
    )

    if transport == "http":
        response = _run_http(executor, code, {"image": 1})
        encoded = response["result"]
        decoded = deserialize_for_modal_remote_execution(encoded)
    else:
        response = _run_ws_result(executor, modal_app, code)
        encoded = response["result"]
        decoded = client._deserialize_msgpack_result(encoded)
    expected = _decode_result_with_base_decoder(transport, encoded)

    assert response["success"] is True, response.get("error")
    for name, array in _object_arrays().items():
        assert decoded[name].dtype == array.dtype == expected[name].dtype, name
        assert decoded[name].shape == array.shape == expected[name].shape, name
        assert decoded[name].tolist() == expected[name].tolist(), name
        assert decoded[name].tolist() == array.tolist(), name


def _run_ws_result(executor, modal_app, code, block_state=None):
    kwargs = {}
    if block_state is not None:
        kwargs["block_state"] = modal_app.Executor._deserialize_msgpack_inputs(
            _through_msgpack(serialize_inputs_for_msgpack(block_state))
        )
    response = modal_app.Executor._run_user_code_ws(
        executor,
        code,
        [],
        "run",
        modal_app.Executor._deserialize_msgpack_inputs(
            _through_msgpack(serialize_inputs_for_msgpack({"image": 1}))
        ),
        "",
        {},
        **kwargs,
    )
    response["result"] = modal_app.Executor._serialize_msgpack_result(
        response["result"]
    )
    return _through_msgpack(response)


def test_object_array_with_values_off_its_outer_shape_is_refused():
    marker = {
        "_type": "ndarray",
        "dtype": "object",
        "shape": [3],
        "value": [[1, 2], [3]],
    }

    with pytest.raises(ValueError):
        deserialize_for_modal_remote_execution(json.dumps({"out": marker}))
    with pytest.raises(ValueError):
        client._deserialize_msgpack_result(_through_msgpack({"out": marker}))


def _ragged_polygon_detections():
    polygons = np.empty(2, dtype=object)
    polygons[0] = np.array([[0, 0], [8, 0], [8, 8]], dtype=np.int32)
    polygons[1] = np.array([[10, 10], [18, 10], [18, 18], [10, 18]], dtype=np.int32)
    detections = sv.Detections(
        xyxy=np.array([[0, 0, 8, 8], [10, 10, 18, 18]], dtype=float),
        class_id=np.array([0, 1]),
        confidence=np.array([0.5, 0.6]),
        data={
            "class_name": np.array(["a", "b"]),
            "detection_id": np.array(["d0", "d1"]),
            "image_dimensions": np.array([[20, 20], [20, 20]]),
            "polygon": polygons,
        },
    )
    return detections


def _assert_ragged_polygons(detections):
    expected = _ragged_polygon_detections().data["polygon"]
    polygons = detections.data["polygon"]
    assert polygons.dtype == object
    assert len(polygons) == len(expected)
    for polygon, expected_polygon in zip(polygons, expected):
        assert np.array_equal(polygon, expected_polygon)


RAGGED_POLYGON_STATE_CODE = (
    "def run(self, image):\n"
    "    self.calls = getattr(self, 'calls', 0) + 1\n"
    "    return {'is_first_frame': self.calls == 1, 'calls': self.calls}\n"
)


@pytest.mark.parametrize("transport", ["http", "ws"])
def test_block_instance_keeps_detections_with_ragged_polygons_on_self(
    monkeypatch, modal_app, capsys, transport
):
    block = _build_first_frame_block(
        monkeypatch, modal_app, code=RAGGED_POLYGON_STATE_CODE, transport=transport
    )
    block.detections = _ragged_polygon_detections()

    results = [block.run(image=frame) for frame in (1, 2, 3)]

    assert [r["calls"] for r in results] == [1, 2, 3]
    _assert_ragged_polygons(block.detections)
    assert "withheld" not in capsys.readouterr().err


@pytest.mark.parametrize("transport", ["http", "ws"])
def test_detections_with_ragged_polygons_round_trip_as_a_result(modal_app, transport):
    executor = _http_executor(modal_app)
    code = "def run(self, image):\n    return {'out': self.detections}\n"
    seeded = {"detections": _ragged_polygon_detections()}

    if transport == "http":
        response = _run_http(executor, code, {"image": 1}, seeded)
        result = deserialize_for_modal_remote_execution(response["result"])
    else:
        response = _run_ws_result(executor, modal_app, code, seeded)
        result = client._deserialize_msgpack_result(response["result"])

    assert response["success"] is True, response.get("error")
    _assert_ragged_polygons(result["out"])


def _detections_with_data_field(key, field):
    detections = sv.Detections(
        xyxy=np.array([[0, 0, 10, 10]], dtype=float),
        class_id=np.array([1]),
        confidence=np.array([0.5]),
        data={
            "class_name": np.array(["a"]),
            "detection_id": np.array(["d"]),
            "image_dimensions": np.array([[20, 20]]),
            key: field,
        },
    )
    return detections


def _nested_object_rows():
    rows = np.empty(1, dtype=object)
    rows[0] = np.array(["x", "y"], dtype=object)
    return rows


OBJECT_ROW_DETECTIONS = {
    "data_2d_object": lambda: _detections_with_data_field(
        "data", np.array([["x", "y"]], dtype=object)
    ),
    "time_in_zone_2d_object": lambda: _detections_with_data_field(
        "time_in_zone", np.array([[1.0, 2.0]], dtype=object)
    ),
    "data_object_rows_in_object_field": lambda: _detections_with_data_field(
        "data", _nested_object_rows()
    ),
}


@pytest.mark.parametrize("transport", ["http", "ws"])
@pytest.mark.parametrize(
    "build", OBJECT_ROW_DETECTIONS.values(), ids=list(OBJECT_ROW_DETECTIONS)
)
def test_block_instance_withholds_detections_whose_rows_travel_as_object_arrays(
    monkeypatch, modal_app, transport, build
):
    warned = _capture_client_warnings(monkeypatch)
    block = _build_first_frame_block(
        monkeypatch, modal_app, code=RAGGED_POLYGON_STATE_CODE, transport=transport
    )
    block.detections = build()

    results = [block.run(image=frame) for frame in (1, 2, 3)]

    assert [r["calls"] for r in results] == [1, 2, 3]
    assert not hasattr(block, "detections")
    assert any("self.detections" in message for message in warned)


@pytest.mark.parametrize(
    "build", OBJECT_ROW_DETECTIONS.values(), ids=list(OBJECT_ROW_DETECTIONS)
)
def test_engine_and_sandbox_partitions_drop_detections_with_object_array_rows(
    modal_app, build
):
    state = {"detections": build()}

    engine = partition_block_state(state, measure=_json_state_size)
    sandbox_carried, sandbox_dropped, _ = modal_app._partition_block_state(
        state, _json_state_size
    )

    assert engine.carried == {} and sandbox_carried == {}
    assert [name for name, _ in engine.dropped] == ["detections"]
    assert engine.dropped == sandbox_dropped


def _model_shaped_detections() -> dict:
    def base_data(count):
        data = {
            "class_name": np.array([f"c{index}" for index in range(count)]),
            "detection_id": np.array([f"d{index}" for index in range(count)]),
            "image_dimensions": np.array([[64, 96]] * count),
            "parent_id": np.array(["frame"] * count),
            "root_parent_id": np.array(["frame"] * count),
        }
        return data

    def boxes():
        return sv.Detections(
            xyxy=np.array([[0, 0, 10, 10], [5, 5, 30, 20]], dtype=float),
            class_id=np.array([0, 1]),
            confidence=np.array([0.5, 0.9]),
            data=base_data(2),
        )

    masks = boxes()
    masks.xyxy = np.array([[2, 2, 30, 20], [40, 30, 90, 60]], dtype=float)
    masks.mask = np.zeros((2, 64, 96), dtype=bool)
    masks.mask[0, 2:20, 2:30] = True
    masks.mask[1, 30:60, 40:90] = True
    keypoints = sv.Detections(
        xyxy=np.array([[0, 0, 10, 10]], dtype=float),
        class_id=np.array([0]),
        confidence=np.array([0.7]),
        data={
            **base_data(1),
            "keypoints_class_id": np.array([[0, 1]]),
            "keypoints_class_name": np.array([["nose", "eye"]], dtype=object),
            "keypoints_confidence": np.array([[0.9, 0.8]]),
            "keypoints_xy": np.array([[[1.0, 2.0], [3.0, 4.0]]]),
        },
    )
    tracked = boxes()
    tracked.tracker_id = np.array([7, 8])
    tracked.data["time_in_zone"] = np.array([1.5, 2.0])
    uniform_polygons = boxes()
    uniform_polygons.data["polygon"] = np.array(
        [[[0, 0], [10, 0], [10, 10]], [[5, 5], [30, 5], [30, 20]]]
    )
    ragged_polygons = boxes()
    ragged_polygons.data["polygon"] = _ragged_polygon_detections().data["polygon"]
    detections = {
        "boxes": boxes(),
        "masks": masks,
        "keypoints": keypoints,
        "tracked": tracked,
        "uniform_polygons": uniform_polygons,
        "ragged_polygons": ragged_polygons,
        "empty": sv.Detections.empty(),
    }
    return detections


@pytest.mark.parametrize("transport", ["http", "ws"])
def test_block_instance_keeps_model_shaped_detections_on_self(
    monkeypatch, modal_app, capsys, transport
):
    warned = _capture_client_warnings(monkeypatch)
    block = _build_first_frame_block(
        monkeypatch, modal_app, code=RAGGED_POLYGON_STATE_CODE, transport=transport
    )
    originals = _model_shaped_detections()
    block.detections = dict(originals)

    results = [block.run(image=frame) for frame in (1, 2, 3)]

    assert [r["calls"] for r in results] == [1, 2, 3]
    assert "withheld" not in capsys.readouterr().err
    assert not any("withheld" in message for message in warned)
    assert set(block.detections) == set(originals)
    for name, original in originals.items():
        carried = block.detections[name]
        assert len(carried) == len(original), name
        assert np.allclose(carried.xyxy, original.xyxy), name
        if len(original):
            assert np.array_equal(carried.class_id, original.class_id), name
            assert np.allclose(carried.confidence, original.confidence), name
            assert carried.data["class_name"].tolist() == (
                original.data["class_name"].tolist()
            ), name
    assert np.array_equal(block.detections["tracked"].tracker_id, [7, 8])
    assert np.allclose(block.detections["tracked"].data["time_in_zone"], [1.5, 2.0])
    assert block.detections["keypoints"].data["keypoints_class_name"].tolist() == [
        ["nose", "eye"]
    ]
    assert block.detections["masks"].mask is not None
    _assert_ragged_polygons(block.detections["ragged_polygons"])
    assert np.array_equal(
        block.detections["uniform_polygons"].data["polygon"],
        originals["uniform_polygons"].data["polygon"],
    )


def test_engine_and_sandbox_agree_on_the_passed_through_detections_data_keys(
    modal_app,
):
    from roboflow_workflows.execution_engine.v1.dynamic_blocks import block_state

    assert (
        modal_app._PASSED_THROUGH_DETECTIONS_DATA_KEYS
        == block_state._PASSED_THROUGH_DETECTIONS_DATA_KEYS
    )
