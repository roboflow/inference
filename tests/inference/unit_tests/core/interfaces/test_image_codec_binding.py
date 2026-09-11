"""One codec, both paths, at every server entry point into the engine.

Round-1 Defect 7 / round-2 Defect 2 / round-3 Defect 2: counting installer calls
anywhere in a file accepts a call placed AFTER `ExecutionEngine.init`, or inside
a nested function that is never reached, and proves nothing about the
dictionary that actually reaches the engine; and a root test that replaces the
engine proves nothing about image behaviour. So this file has two halves:

* STRUCTURE - an AST test that resolves each `ExecutionEngine.init` call to its
  innermost enclosing function, scans that function WITHOUT descending into
  nested functions, and requires a `bind_image_codec(<X>)` call earlier in it
  whose argument is the very Name passed as `init_parameters=`.
* EXECUTION - every root runs the REAL `ExecutionEngine.init` (the fixture only
  records the `init_parameters` object on its way through), compiles a
  model-free workflow, and runs it: the HTTP run route on a base64 input and on
  URL inputs that the server's URL policy must accept / refuse, the validate
  route through compilation, the pipeline root through the `on_video_frame`
  callable it hands to `init_with_custom_logic`, and the CLI root end to end
  including a caller override and a conflicting override.

Fixtures reused from the repository:
`tests/inference/unit_tests/core/interfaces/http/test_http_api.py`
(`_build_plain_interface`), the `InferencePipeline.init_with_workflow` pattern
from `tests/inference/unit_tests/core/interfaces/stream/test_interface_pipeline.py`,
and the real CLI function `_run_workflow_for_single_image_with_inference`.
"""

import ast
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from requests_mock import Mocker

from inference.core.interfaces.workflows_image_codec import (
    GUARDED_IMAGE_CODEC,
    install_guarded_image_codec,
)
from inference.core.utils import image_utils
from inference.core.workflows.errors import WorkflowEnvironmentConfigurationError
from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference.core.workflows.execution_engine.entities.types import IMAGE_KIND
from inference.core.workflows.prototypes.image_codec import (
    WorkflowsLocalImageCodec,
    get_image_codec,
    reset_image_codec,
)
from tests.inference.unit_tests.core.interfaces.http.test_http_api import (
    _build_plain_interface,
)

# tests/inference/unit_tests/core/interfaces/<this file> -> five levels up is the repo root
REPO_ROOT = Path(__file__).resolve().parents[5]
COMPOSITION_ROOTS = [
    "inference/core/interfaces/http/http_api.py",
    "inference/core/interfaces/stream/inference_pipeline.py",
    "inference_cli/lib/workflows/local_image_adapter.py",
]
CODEC_INIT_PARAMETER = "workflows_core.image_codec"
ALLOWED_HOST = "cdn.allowed.example.com"
DENIED_HOST = "metadata.internal.example.com"

# Model-free, so every root can compile and run it with a MagicMock model manager.
BLUR_WORKFLOW = {
    "version": "1.0",
    "inputs": [{"type": "WorkflowImage", "name": "image"}],
    "steps": [
        {
            "type": "roboflow_core/image_blur@v1",
            "name": "blur",
            "image": "$inputs.image",
            "blur_type": "gaussian",
            "kernel_size": 5,
        }
    ],
    "outputs": [
        {"type": "JsonField", "name": "blurred", "selector": "$steps.blur.image"}
    ],
}


@pytest.fixture(autouse=True)
def _clean_registry():
    reset_image_codec()
    yield
    reset_image_codec()


@pytest.fixture
def forwarded_engine_init(monkeypatch) -> dict:
    """Record what reaches `ExecutionEngine.init`, then run the REAL initializer.

    Round-3 Defect 2: a mocked engine proves the dict was built, not that the
    engine compiled with it or that images flow through the bound codec. The
    patch lands on the class, which is the same object `http_api`,
    `inference_pipeline` (function-local import) and `local_image_adapter`
    all resolve.
    """
    real_init = ExecutionEngine.init  # bound classmethod, captured before patching
    captured = {"init_parameters": [], "engines": []}

    def _forwarding_init(**kwargs):
        captured["init_parameters"].append(kwargs["init_parameters"])
        engine = real_init(**kwargs)
        captured["engines"].append(engine)
        return engine

    monkeypatch.setattr(ExecutionEngine, "init", _forwarding_init)
    return captured


def _png_bytes() -> bytes:
    image = np.zeros((16, 24, 3), dtype=np.uint8)
    image[..., 1] = 200
    return cv2.imencode(".png", image)[1].tobytes()


def _png_base64() -> str:
    return base64.b64encode(_png_bytes()).decode("ascii")


def _decode_serialised_image(payload: dict) -> np.ndarray:
    assert payload["type"] == "base64", payload
    return cv2.imdecode(
        np.frombuffer(base64.b64decode(payload["value"]), np.uint8), cv2.IMREAD_COLOR
    )


def _get_urls(requests_mock: Mocker) -> list:
    return [r.url for r in requests_mock.request_history if r.method == "GET"]


def _assert_one_object_on_both_paths(captured: dict, expected) -> None:
    assert len(captured["init_parameters"]) == 1, "expected exactly one engine init"
    # Path A: the SAME dict object the root handed to the engine carries the codec...
    assert captured["init_parameters"][0][CODEC_INIT_PARAMETER] is expected
    # ...and the engine really rebound its image deserializer to that object...
    engine = captured["engines"][0]
    bound = engine._engine._compiled_workflow.kinds_deserializers[IMAGE_KIND.name]
    assert bound.keywords == {"image_codec": expected}
    # ...while Path B (the process registry) holds the identical object.
    assert get_image_codec() is expected


# --------------------------------------------------------------------------
# Structure: the binding must precede the engine and target the SAME dict
# --------------------------------------------------------------------------


def _parent_map(tree: ast.AST) -> dict:
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _enclosing_function(node, parents):
    current = parents.get(node)
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return current
        current = parents.get(current)
    return None


def _nodes_in_scope(scope):
    """Nodes belonging to `scope`, NOT descending into nested functions.

    `ast.walk` would happily accept a binding buried in an uncalled inner
    helper; this generator stops at every function/lambda boundary.
    """
    stack = list(ast.iter_child_nodes(scope))
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        stack.extend(ast.iter_child_nodes(node))


def _engine_init_calls(tree: ast.AST) -> list:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "init"
        and getattr(node.func.value, "id", None) == "ExecutionEngine"
    ]


def _bind_calls_in_scope(scope) -> list:
    return [
        node
        for node in _nodes_in_scope(scope)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "bind_image_codec"
    ]


@pytest.mark.parametrize("relative", COMPOSITION_ROOTS)
def test_each_engine_init_is_preceded_by_a_binding_of_its_own_parameters(
    relative: str,
) -> None:
    tree = ast.parse((REPO_ROOT / relative).read_text(encoding="utf-8"))
    parents = _parent_map(tree)
    engine_inits = _engine_init_calls(tree)
    assert engine_inits, relative

    for call in engine_inits:
        scope = _enclosing_function(call, parents)
        assert scope is not None, (relative, call.lineno)

        init_parameters_kwarg = next(
            (k for k in call.keywords if k.arg == "init_parameters"), None
        )
        assert init_parameters_kwarg is not None, (relative, call.lineno)
        assert isinstance(init_parameters_kwarg.value, ast.Name), (
            f"{relative}:{call.lineno} - init_parameters must be a named dict so "
            f"the binding can be matched against it"
        )
        parameters_name = init_parameters_kwarg.value.id

        matching = [
            bind
            for bind in _bind_calls_in_scope(scope)
            if bind.lineno < call.lineno
            and bind.args
            and isinstance(bind.args[0], ast.Name)
            and bind.args[0].id == parameters_name
        ]
        assert matching, (
            f"{relative}:{call.lineno} - no bind_image_codec({parameters_name}) "
            f"before ExecutionEngine.init inside {scope.name}"
        )


@pytest.mark.parametrize("relative", COMPOSITION_ROOTS)
def test_every_composition_root_imports_the_binder(relative: str) -> None:
    tree = ast.parse((REPO_ROOT / relative).read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "inference.core.interfaces.workflows_image_codec"
        for alias in node.names
    }
    assert "bind_image_codec" in imported, (relative, sorted(imported))


def test_the_scope_scan_rejects_a_binding_hidden_in_a_nested_function() -> None:
    # Guards the guard (round-2 Defect 2): `ast.walk` accepted this shape.
    source = (
        "def root():\n"
        "    def never_called():\n"
        "        bind_image_codec(params)\n"
        "    params = {}\n"
        "    ExecutionEngine.init(init_parameters=params)\n"
    )
    tree = ast.parse(source)
    scope = tree.body[0]
    assert _bind_calls_in_scope(scope) == []
    assert [
        n
        for n in ast.walk(scope)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "bind_image_codec"
    ]


def test_http_api_binds_at_both_engine_entry_points() -> None:
    tree = ast.parse(
        (REPO_ROOT / "inference/core/interfaces/http/http_api.py").read_text(
            encoding="utf-8"
        )
    )
    assert len(_engine_init_calls(tree)) == 2


# --------------------------------------------------------------------------
# Execution: the HTTP run route, with a real engine and real image loading
# --------------------------------------------------------------------------


def test_http_run_route_runs_a_real_engine_and_loads_the_input_through_the_bound_codec(
    monkeypatch, forwarded_engine_init
) -> None:
    interface, _ = _build_plain_interface(monkeypatch)
    payload = _png_base64()

    # `ServerImageCodec.decode_string` calls the MODULE attribute, so a spy
    # installed on `image_utils` sees exactly the calls that went through the
    # bound codec - and still runs the real decoder.
    with mock.patch.object(
        image_utils,
        "attempt_loading_image_from_string",
        wraps=image_utils.attempt_loading_image_from_string,
    ) as server_decoder, TestClient(interface.app) as client:
        response = client.post(
            "/workflows/run",
            headers={"Authorization": "Bearer header-key"},
            json={"specification": BLUR_WORKFLOW, "inputs": {"image": payload}},
        )

    assert response.status_code == 200, response.text
    _assert_one_object_on_both_paths(forwarded_engine_init, GUARDED_IMAGE_CODEC)
    server_decoder.assert_called_once()
    assert server_decoder.call_args.kwargs["value"] == payload
    blurred = _decode_serialised_image(response.json()["outputs"][0]["blurred"])
    assert blurred.shape == (16, 24, 3)


@mock.patch.object(image_utils, "ALLOW_URL_INPUT", True)
@mock.patch.object(image_utils, "ALLOW_NON_HTTPS_URL_INPUT", False)
@mock.patch.object(image_utils, "ALLOW_URL_INPUT_WITHOUT_FQDN", False)
@mock.patch.object(image_utils, "WHITELISTED_DESTINATIONS_FOR_URL_INPUT", None)
@mock.patch.object(image_utils, "BLACKLISTED_DESTINATIONS_FOR_URL_INPUT", {DENIED_HOST})
def test_http_run_route_refuses_a_deny_listed_url_input_through_the_bound_codec(
    monkeypatch, forwarded_engine_init
) -> None:
    # The SSRF deny-list is the server's; Path A must reach it for a URL that
    # arrives as a workflow input. The engine is real, the request is refused
    # before any transport, and the refusal surfaces as a client error.
    interface, _ = _build_plain_interface(monkeypatch)

    with TestClient(interface.app) as client:
        response = client.post(
            "/workflows/run",
            headers={"Authorization": "Bearer header-key"},
            json={
                "specification": BLUR_WORKFLOW,
                "inputs": {"image": f"https://{DENIED_HOST}/latest/meta-data"},
            },
        )

    assert response.status_code == 400, response.text
    assert "blacklisted" in response.text
    _assert_one_object_on_both_paths(forwarded_engine_init, GUARDED_IMAGE_CODEC)


@mock.patch.object(image_utils, "VALIDATE_IMAGE_URL_REDIRECTS", False)
@mock.patch.object(image_utils, "ALLOW_URL_TO_NON_GLOBAL_ADDRESSES", True)
@mock.patch.object(image_utils, "ALLOW_URL_INPUT", True)
@mock.patch.object(image_utils, "ALLOW_NON_HTTPS_URL_INPUT", False)
@mock.patch.object(image_utils, "ALLOW_URL_INPUT_WITHOUT_FQDN", False)
@mock.patch.object(image_utils, "BLACKLISTED_DESTINATIONS_FOR_URL_INPUT", None)
@mock.patch.object(
    image_utils, "WHITELISTED_DESTINATIONS_FOR_URL_INPUT", {ALLOWED_HOST}
)
def test_http_run_route_fetches_an_allow_listed_url_input_through_the_bound_codec(
    monkeypatch, forwarded_engine_init, requests_mock: Mocker
) -> None:
    # `requests_mock` replaces the transport adapter (so the SSRF *address*
    # adapter is not exercised here - the parity suite covers it); the
    # allow-list check runs before the transport and is what this proves.
    # `TestClient` speaks httpx, so `requests_mock` does not intercept it.
    url = f"https://{ALLOWED_HOST}/image.png"
    requests_mock.get(url, content=_png_bytes())
    interface, _ = _build_plain_interface(monkeypatch)

    with TestClient(interface.app) as client:
        response = client.post(
            "/workflows/run",
            headers={"Authorization": "Bearer header-key"},
            json={"specification": BLUR_WORKFLOW, "inputs": {"image": url}},
        )

    assert response.status_code == 200, response.text
    # GETs only: the process-global usage collector may POST through the same
    # mocked transport (order-dependent).
    assert _get_urls(requests_mock) == [url]
    _assert_one_object_on_both_paths(forwarded_engine_init, GUARDED_IMAGE_CODEC)
    blurred = _decode_serialised_image(response.json()["outputs"][0]["blurred"])
    assert blurred.shape == (16, 24, 3)


def test_http_validate_route_compiles_a_real_engine_with_the_guarded_codec(
    monkeypatch, forwarded_engine_init
) -> None:
    interface, _ = _build_plain_interface(monkeypatch)

    with TestClient(interface.app) as client:
        response = client.post(
            "/workflows/validate?api_key=some-key", json=BLUR_WORKFLOW
        )

    assert response.status_code == 200, response.text
    assert response.json() == {"status": "ok"}
    _assert_one_object_on_both_paths(forwarded_engine_init, GUARDED_IMAGE_CODEC)


# --------------------------------------------------------------------------
# Execution: the pipeline root
# --------------------------------------------------------------------------


def test_pipeline_root_runs_a_real_engine_with_the_guarded_codec(
    monkeypatch, forwarded_engine_init
) -> None:
    # `init_with_custom_logic` is the only thing replaced (it would open
    # `video.mp4`); the engine, the WorkflowRunner and the frame are real.
    from inference.core.interfaces.camera.entities import VideoFrame
    from inference.core.interfaces.stream.inference_pipeline import InferencePipeline

    init_with_custom_logic = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(
        InferencePipeline, "init_with_custom_logic", init_with_custom_logic
    )

    InferencePipeline.init_with_workflow(
        video_reference="video.mp4",
        workflow_specification=BLUR_WORKFLOW,
        model_manager=MagicMock(),
        image_input_name="image",
    )

    _assert_one_object_on_both_paths(forwarded_engine_init, GUARDED_IMAGE_CODEC)
    on_video_frame = init_with_custom_logic.call_args.kwargs["on_video_frame"]
    frame = VideoFrame(
        image=np.zeros((16, 24, 3), dtype=np.uint8),
        frame_id=1,
        frame_timestamp=datetime.now(),
    )
    results = on_video_frame([frame])
    assert results[0]["blurred"].numpy_image.shape == (16, 24, 3)


# --------------------------------------------------------------------------
# Execution: the CLI root, including the override that split the paths (R2-1)
# --------------------------------------------------------------------------


def _run_cli_root(tmp_path, init_params=None) -> dict:
    from inference_cli.lib.workflows import local_image_adapter

    image_path = str(tmp_path / "frame.png")
    assert cv2.imwrite(image_path, np.zeros((16, 24, 3), dtype=np.uint8))
    # A real executor: the engine runs its steps on the executor it is given.
    with ThreadPoolExecutor(max_workers=1) as executor:
        return local_image_adapter._run_workflow_for_single_image_with_inference(
            model_manager=MagicMock(),
            image_path=image_path,
            workflow_specification=BLUR_WORKFLOW,
            workflow_id=None,
            image_input_name="image",
            workflow_parameters=None,
            api_key=None,
            thread_pool_executor=executor,
            max_concurrent_workflows_steps=1,
            workflows_execution_engine_init_params=init_params,
        )


def test_cli_root_runs_a_real_engine_with_the_guarded_codec(
    tmp_path, forwarded_engine_init
) -> None:
    result = _run_cli_root(tmp_path)
    _assert_one_object_on_both_paths(forwarded_engine_init, GUARDED_IMAGE_CODEC)
    assert _decode_serialised_image(result["blurred"]).shape == (16, 24, 3)


def test_cli_override_moves_both_paths_together(
    tmp_path, forwarded_engine_init
) -> None:
    # Round-2 Defect 1: `local_image_adapter.py` merges caller overrides AFTER
    # the dict is built, so a codec written before that point would change
    # Path A only. Binding after the merge keeps the two paths identical, and
    # the real engine still compiles and runs with the override.
    override = WorkflowsLocalImageCodec()
    result = _run_cli_root(tmp_path, init_params={CODEC_INIT_PARAMETER: override})
    _assert_one_object_on_both_paths(forwarded_engine_init, override)
    assert _decode_serialised_image(result["blurred"]).shape == (16, 24, 3)


def test_cli_override_conflicting_with_an_install_is_refused_before_the_engine_starts(
    tmp_path, forwarded_engine_init
) -> None:
    install_guarded_image_codec()
    with pytest.raises(WorkflowEnvironmentConfigurationError):
        _run_cli_root(
            tmp_path, init_params={CODEC_INIT_PARAMETER: WorkflowsLocalImageCodec()}
        )
    assert (
        forwarded_engine_init["init_parameters"] == []
    ), "refused after the engine started"
    assert get_image_codec() is GUARDED_IMAGE_CODEC


# --------------------------------------------------------------------------
# Behaviour (appended by Task 10.8, after `base.py` and the block files are
# repointed): ONE image goes through both paths, so the two must agree
# --------------------------------------------------------------------------


PASSTHROUGH_WORKFLOW = {
    "version": "1.0",
    "inputs": [{"type": "WorkflowImage", "name": "image"}],
    "steps": [],
    "outputs": [{"type": "JsonField", "name": "image", "selector": "$inputs.image"}],
}


@mock.patch.object(image_utils, "VALIDATE_IMAGE_URL_REDIRECTS", False)
@mock.patch.object(image_utils, "ALLOW_URL_TO_NON_GLOBAL_ADDRESSES", True)
@mock.patch.object(image_utils, "ALLOW_URL_INPUT", True)
@mock.patch.object(image_utils, "ALLOW_NON_HTTPS_URL_INPUT", False)
@mock.patch.object(image_utils, "ALLOW_URL_INPUT_WITHOUT_FQDN", False)
@mock.patch.object(image_utils, "BLACKLISTED_DESTINATIONS_FOR_URL_INPUT", None)
@mock.patch.object(
    image_utils, "WHITELISTED_DESTINATIONS_FOR_URL_INPUT", {ALLOWED_HOST}
)
def test_one_image_uses_both_paths_and_they_must_agree(requests_mock: Mocker) -> None:
    # The mechanism (round-4 Defect 2): Path A deserializes the URL input and
    # keeps the decoded pixels AND the reference (deserializers.py:133-138);
    # `numpy_image` then serves the cached pixels (base.py:549) - no reload
    # there. The reload happens downstream: `to_inference_format()` preserves
    # the URL (base.py:733-740) and a VLM block calls `load_image` on that dict
    # (openai/v1.py:288, before the endpoint call at :293). Both stages must go
    # through the SAME codec, so this uses the real guarded adapter with
    # recording, the real `image_utils` guards, `requests_mock` as the
    # transport, and stubs only `client.chat.completions.create`.
    from inference.core.interfaces.workflows_image_codec import (
        ServerImageCodec,
        bind_image_codec,
    )
    from inference.core.workflows.core_steps.models.foundation.openai.v1 import (
        execute_gpt_4v_request,
    )
    from inference.core.workflows.execution_engine.entities.base import (
        WorkflowImageData,
    )

    class _RecordingGuardedCodec(ServerImageCodec):
        def __init__(self):
            self.calls = []

        def fetch_url(self, value, cv_imread_flags=cv2.IMREAD_COLOR):
            self.calls.append(("fetch_url", value))
            return super().fetch_url(value, cv_imread_flags=cv_imread_flags)

        def load_image(self, value, disable_preproc_auto_orient=False):
            self.calls.append(("load_image", value))
            return super().load_image(
                value, disable_preproc_auto_orient=disable_preproc_auto_orient
            )

    url = f"https://{ALLOWED_HOST}/image.png"
    requests_mock.get(url, content=_png_bytes())
    recorder = _RecordingGuardedCodec()
    init_parameters = {CODEC_INIT_PARAMETER: recorder}
    assert bind_image_codec(init_parameters) is recorder
    assert get_image_codec() is recorder

    # Stage 1 - Path A: the engine deserializes the URL input through the
    # bound codec and hands the SAME WorkflowImageData out as its output.
    engine = ExecutionEngine.init(
        workflow_definition=PASSTHROUGH_WORKFLOW, init_parameters=init_parameters
    )
    image = engine.run(runtime_parameters={"image": url})[0]["image"]
    assert isinstance(image, WorkflowImageData)
    assert image.numpy_image.shape == (16, 24, 3)  # cached pixels, no reload
    assert recorder.calls == [("fetch_url", url)]
    # GETs only: the process-global usage collector may POST through the same
    # mocked transport (order-dependent).
    assert _get_urls(requests_mock) == [url]

    # Stage 2 - Path B: the downstream block re-loads THAT image from its
    # inference-format dict through the process registry - the same object.
    payload = image.to_inference_format()
    assert payload == {"type": "url", "value": url}
    client = MagicMock()
    client.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content="a green rectangle"))
    ]
    result = execute_gpt_4v_request(
        client=client,
        image=payload,
        prompt="describe",
        lmm_config=MagicMock(
            gpt_model_version="gpt-4o", gpt_image_detail="auto", max_tokens=16
        ),
    )

    assert result == {
        "content": "a green rectangle",
        "image": {"width": 24, "height": 16},
    }
    assert recorder.calls == [("fetch_url", url), ("load_image", payload)]
    assert _get_urls(requests_mock) == [url, url]
    client.chat.completions.create.assert_called_once()


@mock.patch.object(image_utils, "VALIDATE_IMAGE_URL_REDIRECTS", False)
@mock.patch.object(image_utils, "ALLOW_URL_TO_NON_GLOBAL_ADDRESSES", True)
@mock.patch.object(image_utils, "ALLOW_URL_INPUT", True)
@mock.patch.object(image_utils, "ALLOW_NON_HTTPS_URL_INPUT", False)
@mock.patch.object(image_utils, "ALLOW_URL_INPUT_WITHOUT_FQDN", False)
@mock.patch.object(image_utils, "BLACKLISTED_DESTINATIONS_FOR_URL_INPUT", None)
@mock.patch.object(image_utils, "WHITELISTED_DESTINATIONS_FOR_URL_INPUT", None)
def test_reference_born_image_outside_any_engine_uses_the_process_codec(
    monkeypatch,
) -> None:
    # Narrower than the test above, and deliberately so: this covers the
    # supported reference-only construction API -
    # `WorkflowImageData(parent_metadata=..., image_reference=...)` with no
    # cached pixels - so its first `numpy_image` read fetches through the
    # registry (base.py:573). No engine, no Path A. The production
    # constructors (e.g. `inference/core/models/inference_models_adapters.py`,
    # `modal/modal_app.py`) pass `numpy_image` and therefore serve cached
    # pixels (base.py:549) without loading.
    # The recorder overrides just `fetch_url`, which is the only method this
    # path can reach; the transport is blocked below address validation so a
    # regression to the server loader fails fast. DNS is faked and the proxy
    # env is cleared (CR-1 / Ruling R10-C, Task 10.6's pattern): with a
    # reverted/missing `base.py` repoint the server loader calls
    # `socket.getaddrinfo` (inference/core/utils/url_input.py:121) BEFORE the
    # blocker, so an unguarded RED would otherwise depend on real DNS.
    import socket

    import urllib3.connectionpool as connectionpool

    from inference.core.interfaces.workflows_image_codec import bind_image_codec
    from inference.core.workflows.execution_engine.entities.base import (
        ImageParentMetadata,
        WorkflowImageData,
    )

    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "http_proxy",
        "https_proxy",
        "ALL_PROXY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NO_PROXY", "*")
    monkeypatch.setenv("no_proxy", "*")

    def _fake_getaddrinfo(host, port, *args, **kwargs):
        return [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                ("93.184.216.34", port),
            )
        ]

    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo)

    class _RecordingCodec(WorkflowsLocalImageCodec):
        def __init__(self):
            self.calls = []

        def fetch_url(self, value, cv_imread_flags=cv2.IMREAD_COLOR):
            self.calls.append(("fetch_url", value))
            return np.zeros((16, 24, 3), dtype=np.uint8)

    def _blocked(*args, **kwargs):
        raise AssertionError("transport reached")

    recorder = _RecordingCodec()
    assert bind_image_codec({CODEC_INIT_PARAMETER: recorder}) is recorder
    reference_image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="p"),
        image_reference="https://cdn.example.com/other.jpg",
    )
    with mock.patch.object(
        connectionpool.HTTPConnectionPool, "_new_conn", _blocked
    ), mock.patch.object(connectionpool.HTTPSConnectionPool, "_new_conn", _blocked):
        assert reference_image.numpy_image.shape == (16, 24, 3)
    assert recorder.calls == [("fetch_url", "https://cdn.example.com/other.jpg")]
