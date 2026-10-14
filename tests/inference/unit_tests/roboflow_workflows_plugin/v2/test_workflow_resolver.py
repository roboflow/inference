"""RoboflowWorkflowResolver against a loopback Roboflow API and the real V2 compiler.

Every request goes to an in-process HTTP server on 127.0.0.1 through the real
``roboflow_api`` code. A socket guard fails any other connection, and the API
key is invented.
"""

import json
import socket
import threading
import traceback
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple
from urllib.parse import parse_qs, urlsplit

import pytest
from roboflow_workflows.v2.data.kinds import FLOAT_KIND
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref
from roboflow_workflows.v2.engine.errors import (
    CycleError,
    NestedWorkflowError,
    WorkflowCompileError,
)
from roboflow_workflows.v2.engine.plan import CompileOptions

import inference.core.roboflow_api as roboflow_api
from inference.core.cache.memory import MemoryCache
from inference.roboflow_workflows_plugin.v2.workflow_resolver import (
    RoboflowWorkflowResolver,
)

INVENTED_API_KEY = "invented-key-NOT-REAL-0123456789"
WORKSPACE = "demo-workspace"
LOOPBACK_HOSTS = ("127.0.0.1", "::1")


class Scale(Block):
    """Multiplies a number by a factor."""

    type = "test/scale@v1"
    outputs = {"scaled": Output(FLOAT_KIND)}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)
        factor: float | Ref(FLOAT_KIND) = 2.0

    def run(self, *, value: float, factor: float) -> Dict[str, Any]:
        return {"scaled": value * factor}


CATALOGUE = Catalogue([Scale], kinds=[FLOAT_KIND])


def workflow(steps: List[dict], outputs: Dict[str, str]) -> dict:
    definition = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowParameter", "name": "x", "kind": ["float"]}],
        "steps": steps,
        "outputs": [
            {"type": "JsonField", "name": name, "selector": selector}
            for name, selector in outputs.items()
        ],
    }

    return definition


def saved(name: str, workflow_id: str, **fields: Any) -> dict:
    step = {
        "type": "roboflow_core/inner_workflow@v1",
        "name": name,
        "workflow_workspace_id": WORKSPACE,
        "workflow_id": workflow_id,
        "parameter_bindings": {"x": "$inputs.x"},
        **fields,
    }

    return step


TRIPLE = workflow(
    [{"type": "test/scale@v1", "name": "scale", "value": "$inputs.x", "factor": 3}],
    {"y": "$steps.scale.scaled"},
)
PARENT = workflow([saved("child", "triple")], {"result": "$steps.child.y"})


def api_response(specification: Any, *, record_id: str = "record-1") -> dict:
    """The platform response shape: the specification is JSON inside JSON."""
    response = {
        "workflow": {
            "id": record_id,
            "config": json.dumps({"specification": specification}),
        }
    }

    return response


class LoopbackApi:
    """Answers ``GET /<workspace>/workflows/<id>`` from a route table."""

    def __init__(self) -> None:
        self.routes: Dict[str, Tuple[int, bytes]] = {}
        self.requests: List[Tuple[str, Dict[str, List[str]]]] = []
        api = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                parts = urlsplit(self.path)
                api.requests.append((parts.path, parse_qs(parts.query)))
                status, body = api.routes.get(parts.path, (404, b'{"error": "nf"}'))
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args: Any) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def save(self, workflow_id: str, specification: Any) -> None:
        body = json.dumps(api_response(specification)).encode()
        self.routes[f"/{WORKSPACE}/workflows/{workflow_id}"] = (200, body)

    def answer(self, workflow_id: str, status: int, body: bytes) -> None:
        self.routes[f"/{WORKSPACE}/workflows/{workflow_id}"] = (status, body)


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[LoopbackApi]:
    real_connect = socket.socket.connect

    def loopback_only(sock: socket.socket, address: Any) -> None:
        if not (isinstance(address, tuple) and address[0] in LOOPBACK_HOSTS):
            raise AssertionError(f"non-loopback connection attempted: {address!r}")
        real_connect(sock, address)

    loopback = LoopbackApi()
    thread = threading.Thread(target=loopback.server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(socket.socket, "connect", loopback_only)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setattr(roboflow_api, "API_BASE_URL", loopback.url)
    monkeypatch.setattr(roboflow_api, "MODEL_CACHE_DIR", str(tmp_path))
    yield loopback
    loopback.server.shutdown()
    loopback.server.server_close()
    thread.join()


def resolver(**overrides: Any) -> RoboflowWorkflowResolver:
    settings = {"api_key": INVENTED_API_KEY, "ephemeral_cache": MemoryCache()}
    settings.update(overrides)
    built = RoboflowWorkflowResolver(**settings)

    return built


def run_once(definition: dict, reference_resolver: Any, x: float) -> dict:
    plan = compile_workflow(
        definition, catalogue=CATALOGUE, reference_resolver=reference_resolver
    )
    session = plan.create_session()
    try:
        [row] = session.run({"x": x}).rows()
    finally:
        session.close()

    return row


def compile_error(definition: dict, reference_resolver: Any, **kwargs: Any) -> str:
    with pytest.raises(WorkflowCompileError) as info:
        compile_workflow(
            definition,
            catalogue=CATALOGUE,
            reference_resolver=reference_resolver,
            **kwargs,
        )
    message = str(info.value)

    return message


def error_chain_text(error: Optional[BaseException]) -> str:
    texts = []
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        texts.append(f"{type(error).__name__}: {error!r} {error}")
        error = error.__cause__ or error.__context__

    return "\n".join(texts)


def test_saved_reference_is_fetched_compiled_and_run(api: LoopbackApi) -> None:
    api.save("triple", TRIPLE)

    row = run_once(PARENT, resolver(), x=7.0)

    assert row == {"result": 21.0}
    assert api.requests == [
        (f"/{WORKSPACE}/workflows/triple", {"api_key": [INVENTED_API_KEY]})
    ]


def test_pinned_version_is_requested(api: LoopbackApi) -> None:
    api.save("triple", TRIPLE)
    parent = workflow(
        [saved("child", "triple", workflow_version_id="v7")],
        {"result": "$steps.child.y"},
    )

    run_once(parent, resolver(), x=1.0)

    [(_, query)] = api.requests
    assert query == {"api_key": [INVENTED_API_KEY], "workflow_version": ["v7"]}


def test_one_compilation_fetches_each_reference_once(api: LoopbackApi) -> None:
    api.save("triple", TRIPLE)
    parent = workflow(
        [saved("first", "triple"), saved("second", "triple")],
        {"a": "$steps.first.y", "b": "$steps.second.y"},
    )

    row = run_once(parent, resolver(use_cache=False), x=2.0)

    assert row == {"a": 6.0, "b": 6.0}
    assert len(api.requests) == 1


def test_host_cache_serves_later_compilations_like_v1(api: LoopbackApi) -> None:
    api.save("triple", TRIPLE)
    shared = resolver()

    run_once(PARENT, shared, x=1.0)
    run_once(PARENT, shared, x=1.0)
    run_once(PARENT, resolver(use_cache=False), x=1.0)

    assert len(api.requests) == 2


def test_missing_api_key_fails_before_any_request(api: LoopbackApi) -> None:
    api.save("triple", TRIPLE)

    message = compile_error(PARENT, resolver(api_key=None))

    assert "$steps.child" in message
    assert f"resolving saved workflow {WORKSPACE}/triple failed" in message
    assert "RoboflowApiKeyRequiredError" in message
    assert api.requests == []


def test_local_workspace_reads_the_model_cache_without_a_key(
    api: LoopbackApi, tmp_path: Path
) -> None:
    folder = tmp_path / "workflow" / "local"
    folder.mkdir(parents=True)
    file_name = f"{sha256(b'triple').hexdigest()}.json"
    (folder / file_name).write_text(json.dumps(api_response(TRIPLE)["workflow"]))
    parent = workflow(
        [saved("child", "triple", workflow_workspace_id="local")],
        {"result": "$steps.child.y"},
    )

    row = run_once(parent, resolver(api_key=None), x=5.0)

    assert row == {"result": 15.0}
    assert api.requests == []


@pytest.mark.parametrize(
    "status, error_name",
    [
        (401, "RoboflowAPINotAuthorizedError"),
        (403, "RoboflowAPIForbiddenError"),
        (404, "RoboflowAPINotNotFoundError"),
        (500, "RoboflowAPIUnsuccessfulRequestError"),
    ],
)
def test_http_errors_name_the_step_and_never_the_key(
    api: LoopbackApi, status: int, error_name: str
) -> None:
    api.answer("triple", status, b'{"error": "refused"}')

    with pytest.raises(NestedWorkflowError) as info:
        compile_workflow(PARENT, catalogue=CATALOGUE, reference_resolver=resolver())

    message = str(info.value)
    assert message.startswith("$steps.child: resolving saved workflow")
    assert error_name in message
    assert INVENTED_API_KEY not in error_chain_text(info.value)


@pytest.mark.parametrize(
    "body, error_name",
    [
        (b"not json", "MalformedRoboflowAPIResponseError"),
        (b'{"unexpected": {}}', "MalformedWorkflowResponseError"),
        (b'{"workflow": {"config": "{broken"}}', "MalformedWorkflowResponseError"),
        (b'{"workflow": {"config": "{}"}}', "MalformedWorkflowResponseError"),
        (
            json.dumps(api_response(["not", "a", "mapping"])).encode(),
            "MalformedWorkflowResponseError",
        ),
    ],
)
def test_malformed_responses_fail_compilation(
    api: LoopbackApi, body: bytes, error_name: str
) -> None:
    api.answer("triple", 200, body)

    message = compile_error(PARENT, resolver())

    assert error_name in message


def test_unreachable_api_is_a_connection_error(api: LoopbackApi) -> None:
    api.server.shutdown()
    api.server.server_close()

    with pytest.raises(NestedWorkflowError) as info:
        compile_workflow(PARENT, catalogue=CATALOGUE, reference_resolver=resolver())

    rendered = "".join(traceback.format_exception(info.value))
    assert "$steps.child" in rendered
    assert "RoboflowAPIConnectionError" in rendered
    assert INVENTED_API_KEY not in error_chain_text(info.value)
    assert INVENTED_API_KEY not in rendered
    assert info.value.__cause__.__cause__ is None
    assert info.value.__cause__.__context__ is None


def test_saved_v1_definition_is_rejected_not_converted(api: LoopbackApi) -> None:
    api.save("triple", {**TRIPLE, "version": "1.0"})

    message = compile_error(PARENT, resolver())

    assert "version must be '2.0' for the V2 engine, got '1.0'" in message


def test_saved_definition_still_meets_v2_validation(api: LoopbackApi) -> None:
    api.save("triple", {**TRIPLE, "unexpected_section": []})

    message = compile_error(PARENT, resolver())

    assert "unexpected_section" in message


def test_cycles_of_saved_workflows_are_rejected(api: LoopbackApi) -> None:
    api.save("ping", workflow([saved("pong", "pong")], {"y": "$steps.pong.y"}))
    api.save("pong", workflow([saved("ping", "ping")], {"y": "$steps.ping.y"}))
    parent = workflow([saved("start", "ping")], {"y": "$steps.start.y"})

    with pytest.raises(CycleError, match="demo-workspace/ping -> demo-workspace/pong"):
        compile_workflow(parent, catalogue=CATALOGUE, reference_resolver=resolver())


def chain_of_saved_workflows(api: LoopbackApi, length: int) -> dict:
    """Saved ``level-1`` embeds ``level-2`` ... down to ``level-<length>``."""
    api.save(f"level-{length}", TRIPLE)
    for level in range(length - 1, 0, -1):
        api.save(
            f"level-{level}",
            workflow([saved("inner", f"level-{level + 1}")], {"y": "$steps.inner.y"}),
        )
    parent = workflow([saved("outer", "level-1")], {"y": "$steps.outer.y"})

    return parent


def test_default_depth_limit_of_four_holds(api: LoopbackApi) -> None:
    within = chain_of_saved_workflows(api, length=4)
    assert run_once(within, resolver(), x=1.0) == {"y": 3.0}

    beyond = chain_of_saved_workflows(api, length=5)
    message = compile_error(beyond, resolver())

    assert "nested workflow depth 5 exceeds the limit of 4" in message


def test_custom_depth_limit_holds(api: LoopbackApi) -> None:
    parent = chain_of_saved_workflows(api, length=3)

    message = compile_error(
        parent, resolver(), options=CompileOptions(max_nested_depth=2)
    )

    assert "nested workflow depth 3 exceeds the limit of 2" in message


def test_default_count_limit_of_thirty_two_holds(api: LoopbackApi) -> None:
    api.save("triple", TRIPLE)
    steps = [saved(f"child_{index}", "triple") for index in range(33)]
    parent = workflow(steps, {"y": "$steps.child_0.y"})

    message = compile_error(parent, resolver())

    assert "33 nested workflow steps exceed the limit of 32" in message


def test_remote_dispatch_stays_unsupported(api: LoopbackApi) -> None:
    api.save("triple", TRIPLE)
    parent = workflow(
        [saved("child", "triple", execution_mode="remote_dispatch")],
        {"result": "$steps.child.y"},
    )

    message = compile_error(parent, resolver())

    assert "uses execution_mode 'remote_dispatch'" in message
    assert api.requests == []


def test_repr_hides_the_api_key() -> None:
    text = repr(RoboflowWorkflowResolver(api_key=INVENTED_API_KEY))

    assert INVENTED_API_KEY not in text
    assert text == "RoboflowWorkflowResolver(use_cache=True)"


def test_socket_guard_blocks_non_loopback_connections(api: LoopbackApi) -> None:
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(AssertionError, match="non-loopback"):
            client.connect(("192.0.2.1", 80))
    finally:
        client.close()
