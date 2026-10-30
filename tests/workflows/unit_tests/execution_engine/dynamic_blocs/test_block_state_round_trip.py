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
import json
from contextlib import contextmanager
from datetime import datetime
from types import SimpleNamespace

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
    kwargs = {}
    if block_state is not None:
        kwargs["block_state"] = modal_app.Executor._deserialize_msgpack_inputs(
            serialize_inputs_for_msgpack(block_state)
        )
    response = modal_app.Executor._run_user_code_ws(
        executor,
        code,
        [],
        "run",
        modal_app.Executor._deserialize_msgpack_inputs(
            serialize_inputs_for_msgpack(inputs)
        ),
        "",
        {},
        **kwargs,
    )
    return response


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


def test_ws_client_fails_loudly_when_the_server_drops_the_state(monkeypatch):
    # A pre-upgrade container ignores ``block_state`` and answers without it.
    # Silently continuing would reset ``self`` on every frame — the bug this
    # feature fixes — so the client refuses.
    monkeypatch.setattr(client, "MODAL_AVAILABLE", True)
    executor = _ws_executor()
    executor._send_recv_with_retry = lambda frame_bytes, workspace, request_id=None: (
        msgpack.packb(
            {"success": True, "result": {}, "request_id": request_id},
            use_bin_type=True,
        )
    )

    with pytest.raises(DynamicBlockError) as error:
        executor.execute_remote_with_state(
            block_type_name="FirstFrameCheck",
            python_code=_python_code(),
            inputs={"image": 1},
            workspace_id="test-ws",
            workflow_context={},
            block_state={"has_first_frame": True},
        )

    assert "block_state" in str(error.value)


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


def test_http_client_fails_loudly_when_the_server_drops_the_state(monkeypatch):
    monkeypatch.setattr(client, "MODAL_AVAILABLE", True)
    executor = client.ModalExecutor(workspace_id="test-ws")
    executor._get_endpoint_url = lambda workspace: "http://modal.test/execute"
    executor._post_execute = lambda **kwargs: {"success": True, "result": "{}"}

    with pytest.raises(DynamicBlockError) as error:
        executor.execute_remote_with_state(
            block_type_name="FirstFrameCheck",
            python_code=_python_code(),
            inputs={"image": 1},
            workspace_id="test-ws",
            workflow_context={},
            block_state={"has_first_frame": True},
        )

    assert "block_state" in str(error.value)


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

    block_state_json = client._serialize_block_state_json(
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


def _build_first_frame_block(monkeypatch, modal_app, code=FIRST_FRAME_CODE):
    from roboflow_workflows.execution_engine.v1.dynamic_blocks.block_assembler import (
        compile_dynamic_blocks,
    )

    monkeypatch.setattr(client, "MODAL_AVAILABLE", True)
    monkeypatch.setattr(
        block_scaffolding, "WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE", "modal"
    )
    monkeypatch.setattr(client, "validate_code_in_modal", lambda *a, **k: None)
    executor = _InProcessHttpExecutor(modal_app)

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
# review round 1: the echo is user output and must not reach engine attributes
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
# review round 1: numpy scalars and tuples must survive both transports
# --------------------------------------------------------------------------- #


NUMPY_SCALAR_CODE = (
    "import numpy as np\n"
    "def run(self, image):\n"
    "    self.seen = np.bool_(False)\n"
    "    self.score = np.float32(0.5)\n"
    "    self.n = np.int64(3)\n"
    "    return {'ok': True}\n"
)


def _assert_python_scalars(state):
    assert state["seen"] is False, state
    assert type(state["score"]) is float and state["score"] == 0.5
    assert type(state["n"]) is int and state["n"] == 3


def test_http_server_echoes_numpy_scalars_as_python_scalars(modal_app):
    executor = _http_executor(modal_app)

    response = _run_http(executor, NUMPY_SCALAR_CODE, {"image": 1}, block_state={})

    assert response["success"] is True, response.get("error")
    _assert_python_scalars(
        deserialize_for_modal_remote_execution(response["block_state"])
    )


def test_ws_server_echoes_numpy_scalars_as_python_scalars(modal_app):
    executor = _http_executor(modal_app)

    response = _run_ws(executor, modal_app, NUMPY_SCALAR_CODE, {"image": 1}, {})

    assert response["success"] is True, response.get("error")
    _assert_python_scalars(client._deserialize_msgpack_result(response["block_state"]))


def test_http_client_sends_numpy_scalars_as_python_scalars():
    state = {"seen": np.bool_(False), "score": np.float32(0.5), "n": np.int64(3)}

    block_state_json = client._serialize_block_state_json(state, block_type_name="b")

    _assert_python_scalars(json.loads(block_state_json))


def test_ws_client_packs_numpy_scalars_as_python_scalars():
    state = {"seen": np.bool_(False), "score": np.float32(0.5), "n": np.int64(3)}

    packed = client.WebSocketModalExecutor._pack_block_state(
        state, block_type_name="b", msgpack=msgpack
    )

    _assert_python_scalars(msgpack.unpackb(msgpack.packb(packed, use_bin_type=True)))


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


def test_http_client_carries_a_tuple_holding_detections_as_a_list():
    block_state_json = client._serialize_block_state_json(
        {"prev": (_detections(), 1)}, block_type_name="b"
    )

    prev = deserialize_for_modal_remote_execution(block_state_json)["prev"]

    assert isinstance(prev, list)
    assert isinstance(prev[0], sv.Detections)
    assert prev[1] == 1


def test_http_server_echoes_a_tuple_holding_detections_as_a_list(modal_app):
    executor = _http_executor(modal_app)
    code = (
        "def run(self, image):\n"
        "    self.prev = (self.prev[0], self.prev[1] + 1)\n"
        "    return {'n': len(self.prev[0])}\n"
    )
    seeded = client._serialize_block_state_json(
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
    prev = deserialize_for_modal_remote_execution(response["block_state"])["prev"]
    assert isinstance(prev, list)
    assert isinstance(prev[0], sv.Detections)
    assert prev[1] == 2


# --------------------------------------------------------------------------- #
# review round 3: dropping must keep the block working, and never fail the call
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
}


def _uncarriable_value_code(expression: str) -> str:
    code = (
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
    import uuid

    values = {
        name: eval(expression, {"np": np, "sv": sv, "uuid": uuid})
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
    assert set(partition.carried) == set(_carriable_values())


def test_engine_and_sandbox_partitions_agree_on_every_value(modal_app):
    def measure(attribute):
        return len(serialize_for_modal_remote_execution(attribute))

    for name, value in {**_carriable_values(), **_uncarriable_values()}.items():
        engine = partition_block_state({name: value}, measure=measure)
        sandbox_carried, sandbox_dropped, _ = modal_app._partition_block_state(
            {name: value}, measure
        )

        assert set(engine.carried) == set(sandbox_carried), name
        assert [n for n, _ in engine.dropped] == [n for n, _ in sandbox_dropped], name


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
        client._serialize_block_state_json(
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
    seeded = client._serialize_block_state_json(
        {"prev": _detections()}, block_type_name="b"
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
# review round 3, follow-up: gaps found re-checking the review against 889228f5e
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
        numpy_image=np.full((8, 8, 3), 100, dtype=np.uint8),
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
    assert state["last"].numpy_image.shape == (8, 8, 3)
    assert "self.last" in response["stderr"] and "JPEG" in response["stderr"]


def test_ws_server_carries_an_image_kept_on_self_and_notes_the_loss(modal_app):
    executor = _http_executor(modal_app)

    response = _run_ws(executor, modal_app, IMAGE_ON_SELF_CODE, {"image": _image()}, {})

    assert response["success"] is True, response.get("error")
    state = client._read_echoed_block_state_msgpack(
        msgpack.unpackb(msgpack.packb(response, use_bin_type=True), raw=False),
        block_state={},
        block_type_name="B",
    )
    assert isinstance(state["last"], WorkflowImageData)
    assert state["last"].numpy_image.shape == (8, 8, 3)
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
