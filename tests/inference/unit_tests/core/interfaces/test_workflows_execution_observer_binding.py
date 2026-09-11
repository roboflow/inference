"""Every server composition root must bind the execution observer.

The engine's default is the workflows-local `NullExecutionObserver`, which
records nothing - so a root that forgets the binding keeps working, keeps its
tests green, and silently stops billing.

Two independent guards. The structural one *discovers* engine construction
sites across the production tree rather than trusting a hard-coded list, and
replays each call's own `init_parameters` bindings, so neither a new root in a
new file, nor a stray matching string in an unused dictionary, nor a
dictionary that was overwritten before the call can fool it. The runtime ones
drive each of the four roots and assert the observer that actually reaches
`ExecutionEngine.init`.
"""

import ast
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from inference.core.interfaces.workflows_execution_observer import (
    UsageTrackingExecutionObserver,
)
from inference.core.workflows.core_steps.loader import REGISTERED_INITIALIZERS
from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference.core.workflows.prototypes.observer import (
    ExecutionObserver,
    NullExecutionObserver,
)

# tests/inference/unit_tests/core/interfaces/<this file> -> five levels up is the repo root
REPO_ROOT = Path(__file__).resolve().parents[5]
SEARCH_ROOTS = ("inference", "inference_cli")
# The engine itself constructs engines (nested workflows, tests of the engine);
# only *server* construction sites are composition roots.
EXCLUDED_SUBTREE = REPO_ROOT / "inference" / "core" / "workflows"
BINDING_KEY = "workflows_core.execution_observer"

# Every production site that builds an execution engine, and how many times.
# A new entry here without a binding fails; a new site not listed here fails.
EXPECTED_ROOTS = {
    "inference/core/interfaces/http/http_api.py": 2,
    "inference/core/interfaces/stream/inference_pipeline.py": 1,
    "inference_cli/lib/workflows/local_image_adapter.py": 1,
}


def _python_files():
    for root in SEARCH_ROOTS:
        for path in sorted((REPO_ROOT / root).rglob("*.py")):
            if EXCLUDED_SUBTREE in path.parents:
                continue
            yield path


def _is_engine_init(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "init"
        and getattr(node.func.value, "id", None) == "ExecutionEngine"
    )


def _parents(tree: ast.AST) -> dict:
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _enclosing_scope(node: ast.AST, parents: dict) -> ast.AST:
    current = parents.get(node)
    while current is not None and not isinstance(
        current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)
    ):
        current = parents.get(current)
    return current


def _dict_literal_keys(node: ast.Dict) -> set:
    return {
        key.value
        for key in node.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }


def _literal_dict_in(value: ast.AST):
    """The dict literal a binding assignment builds, if it builds one.

    Either the literal itself, or a literal handed to a wrapper call as its
    first positional argument - the shape Phase 9 gives the HTTP roots,
    `install_workflows_platform_bindings({...})`. Anything else (a name, a
    call without a literal, a comprehension) is opaque: the helper then knows
    nothing about the keys, and says so by starting from an empty set.
    """
    if isinstance(value, ast.Dict):
        return value
    if (
        isinstance(value, ast.Call)
        and value.args
        and isinstance(value.args[0], ast.Dict)
    ):
        return value.args[0]
    return None


def _keys_bound_to_name(
    scope: ast.AST, parents: dict, name: str, before_lineno: int
) -> set:
    """String keys `name` holds when line `before_lineno` runs.

    Replays the assignments to `name` in this scope, in line order, up to the
    engine construction: a rebinding (`name = {...}`, `name = wrapper({...})`,
    `name = anything_else`) replaces whatever the name held, and a subscript
    assignment (`name["k"] = ...`) adds a key. Three restrictions matter.
    Without the scope check, an unrelated nested function that builds its own
    dictionary under the same variable name satisfies the assertion for an
    outer call that binds nothing. Without the line check, an assignment made
    after the engine was constructed counts. Without the replay order, a
    dictionary that was overwritten before the call (`params = {...binding...};
    params = {}`) still reports the binding it lost.
    """
    assignments = sorted(
        (
            node
            for node in ast.walk(scope)
            if isinstance(node, ast.Assign)
            and _enclosing_scope(node, parents) is scope
            and node.lineno < before_lineno
        ),
        key=lambda node: node.lineno,
    )
    keys = set()
    for node in assignments:
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == name:
                literal = _literal_dict_in(node.value)
                keys = _dict_literal_keys(literal) if literal is not None else set()
            elif (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == name
                and isinstance(target.slice, ast.Constant)
                and isinstance(target.slice.value, str)
            ):
                keys.add(target.slice.value)
    return keys


def _init_parameters_keys(call: ast.Call, scope: ast.AST, parents: dict) -> set:
    argument = next(
        (kw.value for kw in call.keywords if kw.arg == "init_parameters"), None
    )
    if argument is None:
        return set()
    if isinstance(argument, ast.Name):
        return _keys_bound_to_name(scope, parents, argument.id, call.lineno)
    literal = _literal_dict_in(argument)
    return _dict_literal_keys(literal) if literal is not None else set()


def _discovered_engine_init_calls():
    """(relative path, call node, enclosing scope, parent map) per production site."""
    found = []
    for path in _python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        parents = _parents(tree)
        for node in ast.walk(tree):
            if _is_engine_init(node):
                found.append(
                    (
                        str(path.relative_to(REPO_ROOT)),
                        node,
                        _enclosing_scope(node, parents),
                        parents,
                    )
                )
    return found


def _keys_for_source(source: str) -> set:
    """Run the structural helper over a synthetic root."""
    tree = ast.parse(source)
    parents = _parents(tree)
    call = next(node for node in ast.walk(tree) if _is_engine_init(node))
    return _init_parameters_keys(call, _enclosing_scope(call, parents), parents)


def test_the_set_of_composition_roots_is_exactly_the_expected_one() -> None:
    counts = {}
    for relative, _, _, _ in _discovered_engine_init_calls():
        counts[relative] = counts.get(relative, 0) + 1
    assert counts == EXPECTED_ROOTS


def test_every_discovered_root_binds_the_observer_in_its_own_parameters() -> None:
    for relative, call, scope, parents in _discovered_engine_init_calls():
        keys = _init_parameters_keys(call, scope, parents)
        assert BINDING_KEY in keys, (relative, call.lineno, sorted(keys))


def test_the_structural_helper_rejects_a_binding_from_another_scope() -> None:
    """A nested function's own dictionary must not satisfy an outer call."""
    source = (
        "def root():\n"
        "    params = {}\n"
        "    ExecutionEngine.init(workflow_definition={}, init_parameters=params)\n"
        "\n"
        "    def unrelated():\n"
        '        params = {"workflows_core.execution_observer": None}\n'
        "        return params\n"
    )
    assert _keys_for_source(source) == set()


def test_the_structural_helper_ignores_an_overwritten_dictionary() -> None:
    """`params = {binding}; params = {}` binds nothing - the last rebinding wins."""
    source = (
        "def root():\n"
        '    params = {"workflows_core.execution_observer": None}\n'
        "    params = {}\n"
        "    ExecutionEngine.init(workflow_definition={}, init_parameters=params)\n"
    )
    assert _keys_for_source(source) == set()
    # ...and a key set after the rebinding, before the call, does count.
    source_with_subscript = (
        "def root():\n"
        '    params = {"workflows_core.execution_observer": None}\n'
        "    params = {}\n"
        '    params["workflows_core.api_key"] = "k"\n'
        "    ExecutionEngine.init(workflow_definition={}, init_parameters=params)\n"
    )
    assert _keys_for_source(source_with_subscript) == {"workflows_core.api_key"}


def test_the_structural_helper_reads_a_literal_handed_to_a_wrapper_call() -> None:
    """Phase 9 wraps the HTTP literals: `install_workflows_platform_bindings({...})`."""
    source = (
        "def root():\n"
        "    params = install_workflows_platform_bindings(\n"
        '        {"workflows_core.api_key": "k", "workflows_core.execution_observer": None}\n'
        "    )\n"
        "    ExecutionEngine.init(workflow_definition={}, init_parameters=params)\n"
    )
    assert _keys_for_source(source) == {
        "workflows_core.api_key",
        "workflows_core.execution_observer",
    }
    # A wrapper that is handed a *name* is opaque: nothing can be claimed.
    opaque = (
        "def root():\n"
        '    params = {"workflows_core.execution_observer": None}\n'
        "    params = install_workflows_platform_bindings(params)\n"
        "    ExecutionEngine.init(workflow_definition={}, init_parameters=params)\n"
    )
    assert _keys_for_source(opaque) == set()


def test_the_registered_default_is_the_null_observer() -> None:
    # Blocks that declare the parameter and get no host binding resolve to a
    # no-op, so a workflows-only process runs without a billing stack.
    assert isinstance(
        REGISTERED_INITIALIZERS["execution_observer"], NullExecutionObserver
    )


def test_the_bound_observer_and_the_default_share_the_protocol() -> None:
    assert isinstance(UsageTrackingExecutionObserver(), ExecutionObserver)
    assert isinstance(REGISTERED_INITIALIZERS["execution_observer"], ExecutionObserver)


# --------------------------------------------------------------------------
# Runtime: what actually reaches ExecutionEngine.init at each of the four roots
# --------------------------------------------------------------------------


@pytest.fixture
def captured_engine_init(monkeypatch):
    """Capture `init_parameters` and return a MagicMock engine."""
    calls = []

    def fake_init(**kwargs):
        calls.append(kwargs)
        return MagicMock()

    monkeypatch.setattr(ExecutionEngine, "init", fake_init)
    return calls


def _bound_observer(kwargs: dict):
    return kwargs["init_parameters"][BINDING_KEY]


class _DummyInstrumentator:
    def __init__(self, app, model_manager, endpoint="/metrics"):
        self.app = app
        self.model_manager = model_manager
        self.endpoint = endpoint

    def set_stream_manager_client(self, stream_manager_client) -> None:
        self.stream_manager_client = stream_manager_client


def _http_client(monkeypatch):
    from starlette.testclient import TestClient

    import inference.core.interfaces.http.http_api as http_api

    monkeypatch.setattr(http_api, "InferenceInstrumentator", _DummyInstrumentator)
    model_manager = MagicMock()
    model_manager.pingback = None
    model_manager.num_errors = 0
    return TestClient(http_api.HttpInterface(model_manager=model_manager).app)


_TRIVIAL_WORKFLOW = {
    "version": "1.0",
    "inputs": [{"type": "WorkflowParameter", "name": "value"}],
    "steps": [],
    "outputs": [],
}


def test_http_run_root_binds_the_observer(monkeypatch, captured_engine_init) -> None:
    client = _http_client(monkeypatch)
    client.post(
        "/workflows/run",
        json={
            "api_key": "binding-key",
            "specification": _TRIVIAL_WORKFLOW,
            "inputs": {"value": 1},
        },
    )
    assert captured_engine_init, "the route never reached ExecutionEngine.init"
    assert isinstance(
        _bound_observer(captured_engine_init[-1]), UsageTrackingExecutionObserver
    )


def test_http_validate_root_binds_the_observer(
    monkeypatch, captured_engine_init
) -> None:
    client = _http_client(monkeypatch)
    client.post("/workflows/validate?api_key=binding-key", json=_TRIVIAL_WORKFLOW)
    assert captured_engine_init, "the route never reached ExecutionEngine.init"
    assert isinstance(
        _bound_observer(captured_engine_init[-1]), UsageTrackingExecutionObserver
    )


def test_pipeline_root_binds_the_observer(monkeypatch, captured_engine_init) -> None:
    from inference.core.interfaces.stream.inference_pipeline import InferencePipeline

    monkeypatch.setattr(
        InferencePipeline, "init_with_custom_logic", MagicMock(return_value=MagicMock())
    )
    InferencePipeline.init_with_workflow(
        video_reference="video.mp4",
        workflow_specification={"version": "1.0"},
        model_manager=MagicMock(),
    )
    assert captured_engine_init, "init_with_workflow never reached ExecutionEngine.init"
    assert isinstance(
        _bound_observer(captured_engine_init[-1]), UsageTrackingExecutionObserver
    )


def test_cli_root_binds_the_observer(
    monkeypatch, tmp_path, captured_engine_init
) -> None:
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np

    from inference_cli.lib.workflows import local_image_adapter

    monkeypatch.setattr(
        local_image_adapter.cv2, "imread", lambda _: np.zeros((2, 2, 3))
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        local_image_adapter._run_workflow_for_single_image_with_inference(
            model_manager=MagicMock(),
            image_path=str(tmp_path / "image.jpg"),
            workflow_specification={"version": "1.0"},
            workflow_id=None,
            image_input_name="image",
            workflow_parameters=None,
            api_key="binding-key",
            thread_pool_executor=pool,
            max_concurrent_workflows_steps=1,
        )
    assert captured_engine_init, "the CLI adapter never reached ExecutionEngine.init"
    assert isinstance(
        _bound_observer(captured_engine_init[-1]), UsageTrackingExecutionObserver
    )
