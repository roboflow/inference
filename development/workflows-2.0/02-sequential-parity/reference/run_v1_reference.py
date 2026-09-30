"""Run the reference cases on the real V1 engine and print observations as JSON.

Run from anywhere with the repository's Python environment::

    python development/workflows-2.0/02-sequential-parity/reference/run_v1_reference.py \\
        --case control.two_gates_intersect

This script is always its own process. Before importing V1 it:

* puts the checkout's ``workflows/`` sources first on ``sys.path``, and its
  ``inference_models`` package and root (for ``inference_sdk``) last;
* sets ``WORKFLOWS_PLUGINS=v1_reference_fixtures`` so ordinary V1 plugin
  discovery loads the fixture blocks next to this file;
* disables font downloads and rejects every network connection.

Nothing else is replaced: the V1 core catalogue, compiler, executor and core
blocks run unchanged. Each compiled block instance gets a transparent ``run``
wrapper that records arguments and results, then calls the original method.
Two derived fields make comparison with V2's observer possible:
``arguments_after_call`` (only when a block changed its arguments in place)
and ``result_resolved`` (a block's future results, read after the run).
The exit code is 1 when a live call count or error differs from its pin.
"""

import copy
import json
import os
import platform
import sys
from collections import Counter
from concurrent.futures import Future
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import click

REFERENCE_DIR = Path(__file__).resolve().parent
FIXTURE_PLUGIN = "v1_reference_fixtures"
NETWORK_ATTEMPTS: List[str] = []


def _find_repository_root(start: Path) -> Path:
    for candidate in start.parents:
        if (candidate / "workflows" / "roboflow_workflows").is_dir():
            return candidate

    raise RuntimeError(f"No checkout with workflows/roboflow_workflows above {start}")


def _deny_network(event: str, arguments: tuple) -> None:
    if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto"}:
        NETWORK_ATTEMPTS.append(event)
        raise RuntimeError(f"The V1 reference runner is offline; rejected {event}")


REPOSITORY_ROOT = _find_repository_root(REFERENCE_DIR)
# workflows/ gives the V1 sources, REFERENCE_DIR the fixture plugin module and
# its parent the `reference` catalogue package.
sys.path[0:0] = [
    str(REPOSITORY_ROOT / "workflows"),
    str(REFERENCE_DIR),
    str(REFERENCE_DIR.parent),
]
# The V1 core catalogue imports the checkout's `inference_models` and
# `inference_sdk`. The `inference_models` project directory must come before
# the root: otherwise the root's `inference_models/` folder is found first as
# an empty namespace package and hides the real one.
sys.path.extend([str(REPOSITORY_ROOT / "inference_models"), str(REPOSITORY_ROOT)])
sys.dont_write_bytecode = True
sys.addaudithook(_deny_network)
os.environ["WORKFLOWS_PLUGINS"] = FIXTURE_PLUGIN

from roboflow_workflows.configuration import (  # noqa: E402
    EngineConfiguration,
    FontsConfiguration,
    WorkflowsConfiguration,
    configure_process,
)

configure_process(
    WorkflowsConfiguration(
        engine=EngineConfiguration(
            allow_custom_python_execution=True,
            custom_python_execution_mode="local",
        ),
        fonts=FontsConfiguration(allow_download=False),
    )
)

import roboflow_workflows  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from reference.catalogue import REFERENCE_CASES, ReferenceCase, Run  # noqa: E402
from roboflow_workflows.execution_engine.core import ExecutionEngine  # noqa: E402
from roboflow_workflows.execution_engine.entities.base import Batch  # noqa: E402
from roboflow_workflows.execution_engine.v1.core import (  # noqa: E402
    EXECUTION_ENGINE_V1_VERSION,
)
from v1_reference_fixtures import AuditLog  # noqa: E402


def _plain(value: Any, *, resolve_futures: bool) -> Any:
    if isinstance(value, Future):
        # Never resolve a future the engine still owns (block results);
        # a future left in the workflow output is resolved for display.
        observed = {"done_when_observed": value.done()}
        if resolve_futures:
            observed["result"] = _plain(value.result(), resolve_futures=True)
        return {"future": observed}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Batch):
        return {
            "batch": [_plain(item, resolve_futures=resolve_futures) for item in value],
            "indices": [list(index) for index in value.indices],
        }
    if isinstance(value, BaseModel):
        return _plain(value.model_dump(), resolve_futures=resolve_futures)
    if isinstance(value, dict):
        return {
            str(key): _plain(item, resolve_futures=resolve_futures)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_plain(item, resolve_futures=resolve_futures) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    return {"object": type(value).__name__, "repr": repr(value)}


def _contains_future(value: Any) -> bool:
    if isinstance(value, Future):
        return True
    if isinstance(value, dict):
        return any(_contains_future(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_future(item) for item in value)

    return False


def _resolved(value: Any) -> Any:
    if isinstance(value, Future):
        return _resolved(value.result())
    if isinstance(value, dict):
        return {key: _resolved(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_resolved(item) for item in value]

    return value


def _describe_error(error: Exception, *, phase: str) -> Dict[str, Any]:
    described = {"phase": phase, "type": type(error).__name__, "message": str(error)}
    block_id = getattr(error, "block_id", None)
    if block_id is not None:
        described["block_id"] = block_id

    return described


class _CaseRecorder:
    """Collects invocations, block instances and hook calls for one case.

    ``session`` and ``run`` hold the position of the run being executed; the
    ``run`` wrappers tag every invocation with them.
    """

    def __init__(self):
        self.invocations: List[Dict[str, Any]] = []
        self.resolver_calls: List[List[Optional[str]]] = []
        self.step_errors: List[Dict[str, str]] = []
        self.session = 0
        self.run = 0
        self._instances: List[Any] = []
        self._future_results: List[Tuple[Dict[str, Any], Any]] = []

    def wrap(self, step_name: str, instance: Any) -> Callable[..., Any]:
        original = instance.run
        ordinal = self._instance_ordinal(instance)

        def recorded_run(**arguments: Any) -> Any:
            entry = {
                "session": self.session,
                "run": self.run,
                "step": step_name,
                "instance": ordinal,
                "arguments": _plain(arguments, resolve_futures=False),
            }
            self.invocations.append(entry)
            try:
                result = original(**arguments)
            except Exception as error:
                entry["error"] = _describe_error(error, phase="block")
                raise
            entry["result"] = _plain(result, resolve_futures=False)
            arguments_after_call = _plain(arguments, resolve_futures=False)
            if arguments_after_call != entry["arguments"]:
                entry["arguments_after_call"] = arguments_after_call
            if _contains_future(result):
                self._future_results.append((entry, result))

            return result

        return recorded_run

    def resolve_future_results(self) -> None:
        # Called after a run; the futures are complete by then.
        for entry, result in self._future_results:
            entry["result_resolved"] = _plain(_resolved(result), resolve_futures=False)
        self._future_results.clear()

    def resolver(self, saved_workflows: Dict[str, dict]) -> Callable[..., dict]:
        def resolve(workspace_id, workflow_id, workflow_version_id, init_parameters):
            self.resolver_calls.append([workspace_id, workflow_id, workflow_version_id])

            return copy.deepcopy(saved_workflows[workflow_id])

        return resolve

    def step_error_handler(self, step_name: str, error: Exception) -> None:
        self.step_errors.append(
            {"step": step_name, "type": type(error).__name__, "message": str(error)}
        )

    def _instance_ordinal(self, instance: Any) -> int:
        # Identity, not equality: shows whether V1 shares or recreates instances.
        for ordinal, known in enumerate(self._instances):
            if known is instance:
                return ordinal
        self._instances.append(instance)

        return len(self._instances) - 1


def _compiled_steps(engine: ExecutionEngine) -> Dict[str, Dict[str, Any]]:
    compiled = engine._engine._compiled_workflow
    described = {}
    for name, compiled_step in compiled.steps.items():
        node = compiled.execution_graph.nodes[f"$steps.{name}"][
            "node_compilation_output"
        ]
        described[name] = {
            "type": compiled_step.manifest.type,
            "data_depth": len(node.data_lineage),
            "execution_depth": node.step_execution_dimensionality,
            "control_depths": list(node.control_flow_lineage_dims),
        }

    return described


def _init_parameters(case: ReferenceCase, *, recorder: _CaseRecorder) -> dict:
    init_parameters = {}
    if case.saved_workflows is not None:
        init_parameters["workflows_core.inner_workflow_spec_resolver"] = (
            recorder.resolver(case.saved_workflows)
        )
    for resource in case.explicit_resources:
        init_parameters[f"{FIXTURE_PLUGIN}.{resource}"] = AuditLog(origin="caller")

    return init_parameters


def _observe_session(
    case: ReferenceCase,
    numbered_runs: List[Tuple[int, Run]],
    *,
    recorder: _CaseRecorder,
) -> Dict[str, Any]:
    engine_options = {}
    if case.record_step_errors:
        engine_options["step_error_handler"] = recorder.step_error_handler
    session = {"session": recorder.session, "compile_error": None, "runs": []}
    try:
        engine = ExecutionEngine.init(
            workflow_definition=copy.deepcopy(case.workflow),
            init_parameters=_init_parameters(case, recorder=recorder),
            max_concurrent_steps=1,
            **engine_options,
        )
    except Exception as error:
        engine = None
        session["compile_error"] = _describe_error(error, phase="compile")
    else:
        session["compiled_steps"] = _compiled_steps(engine)
        session["resource_bindings"] = {}
        for name, compiled_step in engine._engine._compiled_workflow.steps.items():
            instance = compiled_step.step
            instance.run = recorder.wrap(name, instance)
            if isinstance(getattr(instance, "audit", None), AuditLog):
                resource = AuditLog.created.index(instance.audit)
                session["resource_bindings"][name] = resource

    for number, run in numbered_runs:
        recorder.run = number
        observed = _observe_run(
            run,
            engine=engine,
            compile_error=session["compile_error"],
            recorder=recorder,
        )
        session["runs"].append(observed)

    return session


def _observe_run(
    run: Run,
    *,
    engine: Optional[ExecutionEngine],
    compile_error: Optional[Dict[str, Any]],
    recorder: _CaseRecorder,
) -> Dict[str, Any]:
    inputs = copy.deepcopy(run.inputs)
    observed = {
        "run": recorder.run,
        "inputs": copy.deepcopy(run.inputs),
        "resolve_output_futures": run.resolve_output_futures,
        "rows": None,
        "error": compile_error,
    }
    calls_before = len(recorder.invocations)
    if engine is not None:
        try:
            rows = engine.run(
                runtime_parameters=inputs,
                resolve_output_futures=run.resolve_output_futures,
            )
            observed["rows"] = _plain(rows, resolve_futures=True)
        except Exception as error:
            observed["error"] = _describe_error(error, phase="run")
        recorder.resolve_future_results()
    # V1 writes coerced values and defaults back into the caller's mapping.
    observed["inputs_after_run"] = _plain(inputs, resolve_futures=False)

    counts = Counter(entry["step"] for entry in recorder.invocations[calls_before:])
    error_type = None if observed["error"] is None else observed["error"]["type"]
    observed.update(
        call_counts=dict(sorted(counts.items())),
        pinned_call_counts=dict(sorted(run.v1_calls.items())),
        pinned_error=run.v1_error,
    )
    observed["pins_ok"] = (
        observed["call_counts"] == observed["pinned_call_counts"]
        and error_type == run.v1_error
    )

    return observed


def observe_case(case: ReferenceCase) -> Dict[str, Any]:
    """Execute every run of one case on the V1 engine.

    Args:
        case: Reference case from the catalogue.

    Returns:
        The case definition together with live rows, errors, invocations,
        resolver and error-hook calls, resources and pin checks.
    """
    AuditLog.created.clear()
    recorder = _CaseRecorder()
    sessions = []
    for session_number in sorted({run.session for run in case.runs}):
        recorder.session = session_number
        numbered_runs = [
            (number, run)
            for number, run in enumerate(case.runs)
            if run.session == session_number
        ]
        sessions.append(_observe_session(case, numbered_runs, recorder=recorder))

    resources = [
        {
            "resource": ordinal,
            "origin": log.origin,
            "events": _plain(log.events, resolve_futures=False),
        }
        for ordinal, log in enumerate(AuditLog.created)
    ]
    observation = case.as_dict()
    observation.update(
        sessions=sessions,
        invocations=recorder.invocations,
        resolver_calls=recorder.resolver_calls,
        step_error_hook_calls=recorder.step_errors,
        resources=resources,
        pins_ok=all(run["pins_ok"] for session in sessions for run in session["runs"]),
    )

    return observation


def observe_cases(cases: List[ReferenceCase]) -> Dict[str, Any]:
    """Observe the given cases on V1 and describe the engine that ran them.

    Args:
        cases: Cases to execute, in order.

    Returns:
        JSON-compatible report with engine identity and per-case observations.
    """
    observations = [observe_case(case) for case in cases]
    engine_source = Path(roboflow_workflows.__file__).resolve().parent
    report = {
        "schema_version": 1,
        "engine": {
            "family": "workflows-v1",
            "version": str(EXECUTION_ENGINE_V1_VERSION),
            "source": str(engine_source.relative_to(REPOSITORY_ROOT)),
            "python": platform.python_version(),
            "max_concurrent_steps": 1,
            "plugins": [FIXTURE_PLUGIN],
        },
        "network_attempts": NETWORK_ATTEMPTS,
        "cases": observations,
        "summary": {
            "cases": len(observations),
            "pins_ok": sum(1 for case in observations if case["pins_ok"]),
            "pin_failures": [
                case["case_id"] for case in observations if not case["pins_ok"]
            ],
        },
    }

    return report


@click.command()
@click.option(
    "--case",
    "selected",
    type=click.Choice(
        [case.case_id for case in REFERENCE_CASES],
    ),
    multiple=True,
    help="Case to run; repeat for several. Default: every case.",
)
@click.option(
    "--list",
    "list_only",
    is_flag=True,
    default=False,
    help="Print the selected catalogue entries as JSON without running them.",
)
def main(selected: Tuple[str, ...], list_only: bool) -> None:
    """Print V1 reference observations (or the catalogue) as JSON on stdout."""
    cases = [
        case for case in REFERENCE_CASES if not selected or case.case_id in selected
    ]
    if list_only:
        catalogue = {"schema_version": 1, "cases": [case.as_dict() for case in cases]}
        click.echo(json.dumps(catalogue, indent=2))
        return

    report = observe_cases(cases)
    click.echo(json.dumps(report, indent=2))
    if report["summary"]["pin_failures"] or report["network_attempts"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
