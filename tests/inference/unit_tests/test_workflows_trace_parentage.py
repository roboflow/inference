"""Span parentage across the engine's thread pool, before and after the port.

The tracer provider is process-global and `inference.core.telemetry` caches the
tracer in a module global, so every scenario runs in a child interpreter that
installs an in-memory exporter before importing anything from `inference`.

Four scenarios, one child each. `bound` and `unbound` are the port itself:
with the host bound the tree is exactly what HEAD produced; with no host bound
the engine must emit *no* spans of its own, which is what proves the tracing
moved into the observer rather than merely surviving. `failing` is the
exception path this phase relocates (`executor/core.py:465`'s `detach_context`
moves into the observer's `step_scope`): a step that raises inside its own
span, then another run on the *same single-worker executor*, checking that
every span of the failed run closed with an ERROR status under the right
parent, that the caller's current span is intact afterwards, that the worker
thread has nothing attached, and that the second run's tree is clean.
`remote` is a custom-Python block executed through the real `ModalExecutor`
with only the HTTP session faked: the existing client-side behaviour is that
the step span wraps the remote call, the executor opens no span of its own,
and no trace context is sent to the sandbox - this phase must preserve all
three, not improve them.

The child's environment is **built from scratch** - a short passthrough list
plus explicit pins - and the child runs in an **empty working directory**,
because neither inheriting nor popping is enough:

* `inference_models/_offline.py:166` writes
  `_ROBOFLOW_INFERENCE_OFFLINE_MODE_AT_PROCESS_START` into `os.environ` at
  import time and `_decide_offline_mode` prefers that latch over
  `OFFLINE_MODE`, so `{**os.environ, "OFFLINE_MODE": "False"}` inherits the
  parent's decision and `telemetry._otel_enabled()` then disables every span.
* `inference/core/env.py:21` runs `load_dotenv(os.getcwd() + "/.env")`, which
  fills in a *missing* key from a `.env` in the working directory. Setting
  `WORKFLOWS_PLUGINS=""` to defeat that is not an option:
  `blocks_loader.get_plugin_modules` splits the string on commas and
  `import_module("")` raises `ValueError: Empty module name` (executed). So the
  key is left ABSENT (`get_plugin_modules` returns `[]` for an absent key) and
  the child's cwd is an empty temporary directory, which has no `.env`.
* `inference/core/env.py:1172-1179` re-inserts the enterprise plugin into
  `WORKFLOWS_PLUGINS` whenever `LOAD_ENTERPRISE_BLOCKS` is true, so that flag
  is pinned false. (After Phase 9, `env.py` also inserts the Roboflow-platform
  plugin unconditionally; it is server code that imports cleanly here and runs
  no step, so it does not change the span tree.)
* The custom-Python execution mode is pinned per scenario (`local`, or `modal`
  with fake Modal credentials so `MODAL_AVAILABLE` is true), and
  `DEFAULT_WORKFLOWS_STEP_ERROR_HANDLER` can name a handler the engine does not
  register - it registers only `"legacy"`.
* With the cwd empty, `PYTHONPATH` must carry the repo root explicitly, or the
  venv's editable install resolves `inference` from a different checkout.

`scripts/workflows_isolation_probe.py:379-406` is the precedent; this goes
further by building the environment rather than filtering it.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

PROBE_TIMEOUT_SECONDS = 300

# tests/inference/unit_tests/<this file> -> three levels up is the repo root
REPO_ROOT = Path(__file__).resolve().parents[3]

_PROBE_TEMPLATE = r"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

_EXPORTER = InMemorySpanExporter()
_PROVIDER = TracerProvider()
_PROVIDER.add_span_processor(SimpleSpanProcessor(_EXPORTER))
trace.set_tracer_provider(_PROVIDER)

from inference.core.telemetry import start_span
from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference.core.workflows.execution_engine.v1.dynamic_blocks import (
    block_scaffolding,
    modal_executor,
)

SCENARIO = "__SCENARIO__"

# The block opens its own span, then fails on request - inside that span, the
# way a model call that raises would.
BLOCK_CODE = '''
def run(self, value) -> BlockResult:
    from inference.core.telemetry import start_span

    with start_span("model.infer"):
        if value == "boom":
            raise RuntimeError("boom")
    return {"result": True}
'''

SPEC = {
    "version": "1.0",
    "inputs": [{"type": "WorkflowParameter", "name": "value"}],
    "dynamic_blocks_definitions": [
        {
            "type": "DynamicBlockDefinition",
            "manifest": {
                "type": "ManifestDescription",
                "block_type": "SpanProbe",
                "inputs": {
                    "value": {
                        "type": "DynamicInputDefinition",
                        "selector_types": ["input_parameter"],
                    }
                },
                "outputs": {"result": {"type": "DynamicOutputDefinition", "kind": []}},
            },
            "code": {"type": "PythonCode", "run_function_code": BLOCK_CODE},
        }
    ],
    "steps": [{"type": "SpanProbe", "name": "probe", "value": "$inputs.value"}],
    "outputs": [
        {"type": "JsonField", "name": "result", "selector": "$steps.probe.result"}
    ],
}


class _StubWorkspaceResolver:
    def resolve_workspace(self, api_key):
        return "test-workspace"


class _FakeResponse:
    status_code = 200
    text = ""

    def json(self):
        return {
            "success": True,
            "result": json.dumps({"result": True}),
            "execution_time_seconds": 0.1,
        }


class _FakeSession:
    # The transport, one level below `_post_execute`, so the headers the
    # client would send are observable.

    def __init__(self):
        self.headers_sent = []

    def post(self, url, data=None, timeout=None, headers=None):
        self.headers_sent.append(dict(headers or {}))
        return _FakeResponse()


def _spans_by_run():
    # Spans grouped per trace, each group in start order, traces in start order.
    spans = _EXPORTER.get_finished_spans()
    by_id = {s.context.span_id: s.name for s in spans}
    runs = {}
    for span in spans:
        runs.setdefault(span.context.trace_id, []).append(span)
    ordered = sorted(runs.values(), key=lambda group: min(s.start_time for s in group))
    return [
        [
            {
                "name": s.name,
                "parent": by_id.get(s.parent.span_id) if s.parent else None,
                "status": s.status.status_code.name,
            }
            for s in sorted(group, key=lambda s: s.start_time)
        ]
        for group in ordered
    ]


init_parameters = {"workflows_core.api_key": "probe-key"}
if SCENARIO != "unbound":
    from inference.core.interfaces.workflows_execution_observer import (
        UsageTrackingExecutionObserver,
    )

    init_parameters["workflows_core.execution_observer"] = (
        UsageTrackingExecutionObserver()
    )

facts = {}
if SCENARIO in ("bound", "unbound"):
    engine = ExecutionEngine.init(workflow_definition=SPEC, init_parameters=init_parameters)
    with start_span("http.request"):
        engine.run(runtime_parameters={"value": 1})
elif SCENARIO == "failing":
    # One worker, so the second run provably reuses the thread the failing
    # step ran on.
    with ThreadPoolExecutor(max_workers=1) as executor:
        engine = ExecutionEngine.init(
            workflow_definition=SPEC, init_parameters=init_parameters, executor=executor
        )
        with start_span("http.request"):
            try:
                engine.run(runtime_parameters={"value": "boom"})
            except Exception as error:  # the step failure, wrapped by the engine
                facts["raised"] = type(error).__name__
            # Still inside the request span: the run's own spans must have closed.
            facts["caller_current_after_failure"] = trace.get_current_span().name
        # Nothing may remain attached in the worker thread's own context.
        facts["worker_has_current_span"] = executor.submit(
            lambda: trace.get_current_span().get_span_context().is_valid
        ).result()
        with start_span("http.request"):
            facts["second_run"] = engine.run(runtime_parameters={"value": 1})
elif SCENARIO == "remote":
    session = _FakeSession()
    real_executor = modal_executor.ModalExecutor("test-workspace")
    patches = [
        mock.patch.object(modal_executor, "validate_code_in_modal", lambda *a, **k: True),
        mock.patch.object(modal_executor, "get_modal_executor", lambda workspace_id=None: real_executor),
        mock.patch.object(modal_executor.ModalExecutor, "_get_endpoint_url", return_value="https://example.invalid"),
        mock.patch.object(modal_executor.ModalExecutor, "_get_session", return_value=session),
    ]
    if hasattr(block_scaffolding, "get_roboflow_workspace"):  # before Phase 9
        patches.append(
            mock.patch.object(block_scaffolding, "get_roboflow_workspace", return_value="test-workspace")
        )
    else:  # after Phase 9: the injected resolver answers
        init_parameters["workflows_core.workspace_resolver"] = _StubWorkspaceResolver()
    for patch in patches:
        patch.start()
    engine = ExecutionEngine.init(workflow_definition=SPEC, init_parameters=init_parameters)
    with start_span("http.request"):
        facts["result"] = engine.run(runtime_parameters={"value": 1})
    facts["requests_sent"] = len(session.headers_sent)
    facts["trace_headers_sent"] = sorted(
        key for headers in session.headers_sent for key in headers if key.lower() in ("traceparent", "tracestate")
    )
else:
    raise SystemExit(f"unknown scenario {SCENARIO}")

print("RUNS:" + json.dumps(_spans_by_run()))
print("FACTS:" + json.dumps(facts, default=str))
"""


# Only these are inherited; everything else the child needs is set below.
# Anything not listed here is deliberately absent, so a developer's shell or a
# CI job cannot change what this test measures. `WORKFLOWS_PLUGINS` is absent
# on purpose (see the module docstring).
PASSTHROUGH_ENV_KEYS = ("PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "VIRTUAL_ENV")


def _child_env(scenario: str) -> dict:
    env = {key: os.environ[key] for key in PASSTHROUGH_ENV_KEYS if key in os.environ}
    env.update(
        {
            # The repo root explicitly: the child's cwd is an empty directory.
            "PYTHONPATH": f"{REPO_ROOT}{os.pathsep}{REPO_ROOT / 'inference_models'}",
            # The process latch beats OFFLINE_MODE in `_decide_offline_mode`
            # (`inference_models/_offline.py:69-72`), so it has to say the same
            # thing rather than be absent.
            "_ROBOFLOW_INFERENCE_OFFLINE_MODE_AT_PROCESS_START": "False",
            "OFFLINE_MODE": "False",  # telemetry helpers are no-ops offline
            # `env.py:1172-1179` re-inserts the enterprise plugin into
            # WORKFLOWS_PLUGINS whenever this is true.
            "LOAD_ENTERPRISE_BLOCKS": "False",
            # The engine registers only "legacy" (`v1/core.py:59-61`).
            "DEFAULT_WORKFLOWS_STEP_ERROR_HANDLER": "legacy",
            "WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE": (
                "modal" if scenario == "remote" else "local"
            ),
            "ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS": "True",
            "ENABLE_TENSOR_DATA_REPRESENTATION": "False",
            "DISABLE_VERSION_CHECK": "True",
            "PYTHONOPTIMIZE": "0",
        }
    )
    if scenario == "remote":
        # Fake credentials make `modal_executor.MODAL_AVAILABLE` true; the HTTP
        # session is replaced inside the child, so nothing is ever sent.
        env.update(
            {
                "MODAL_TOKEN_ID": "probe-token",
                "MODAL_TOKEN_SECRET": "probe-secret",
                "WEBEXEC_TRANSPORT": "http",
            }
        )
    return env


def _run(scenario: str) -> tuple:
    """(spans grouped per run, facts) reported by the child for `scenario`."""
    probe = _PROBE_TEMPLATE.replace("__SCENARIO__", scenario)
    with tempfile.TemporaryDirectory() as empty_cwd:  # no `.env` here
        proc = subprocess.run(
            [sys.executable, "-c", probe],
            env=_child_env(scenario),
            cwd=empty_cwd,
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_SECONDS,
        )
    assert proc.returncode == 0, proc.stderr
    lines = {
        line.split(":", 1)[0]: json.loads(line.split(":", 1)[1])
        for line in proc.stdout.splitlines()
        if line.startswith(("RUNS:", "FACTS:"))
    }
    assert set(lines) == {"RUNS", "FACTS"}, proc.stdout
    return lines["RUNS"], lines["FACTS"]


def _span(name: str, parent, status: str = "UNSET") -> dict:
    return {"name": name, "parent": parent, "status": status}


# The tree HEAD produces for one successful local block, in start order.
_LOCAL_RUN = [
    _span("http.request", None),
    _span("workflow.run", "http.request"),
    _span("workflow.step", "workflow.run"),
    _span("model.infer", "workflow.step"),
]


def test_model_call_inside_a_block_is_a_child_of_the_step_span() -> None:
    # given / when
    runs, _ = _run("bound")

    # then - the exact tree HEAD produced, preserved through the observer
    assert runs == [_LOCAL_RUN]


def test_without_a_host_the_engine_emits_no_spans_of_its_own() -> None:
    # given / when
    runs, _ = _run("unbound")

    # then - the block's own span parents straight to the caller's
    assert runs == [[_span("http.request", None), _span("model.infer", "http.request")]]


def test_a_failing_step_closes_its_spans_and_leaves_nothing_behind() -> None:
    """The exception path: `step_scope` must detach on failure, and the
    observer's `workflow.run` span must close, so a second run on the same
    worker thread starts clean."""
    # given / when
    runs, facts = _run("failing")

    # then - the failed run: every span under its right parent, ERROR status
    # recorded by the span that raised and by the two that wrapped it
    assert runs == [
        [
            _span("http.request", None),
            _span("workflow.run", "http.request", "ERROR"),
            _span("workflow.step", "workflow.run", "ERROR"),
            _span("model.infer", "workflow.step", "ERROR"),
        ],
        _LOCAL_RUN,
    ]
    # the engine re-raised the user-code error under its usual wrapper
    assert facts["raised"] == "DynamicBlockCodeError"
    # the caller's own span was current again once the run had raised
    assert facts["caller_current_after_failure"] == "http.request"
    # nothing stayed attached in the worker thread that ran the failed step
    assert facts["worker_has_current_span"] is False
    assert facts["second_run"] == [{"result": True}]


def test_a_remote_block_keeps_the_existing_client_side_tracing() -> None:
    """Through the real `ModalExecutor` with only the HTTP session faked: the
    step span wraps the remote call, the executor opens no span of its own, and
    no trace context is sent to the sandbox - exactly as before the port."""
    # given / when
    runs, facts = _run("remote")

    # then
    assert runs == [
        [
            _span("http.request", None),
            _span("workflow.run", "http.request"),
            _span("workflow.step", "workflow.run"),
        ]
    ]
    assert facts["result"] == [{"result": True}]
    assert facts["requests_sent"] == 1
    assert facts["trace_headers_sent"] == []
