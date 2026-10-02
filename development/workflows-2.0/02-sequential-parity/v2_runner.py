"""Run a reference case on the V2 engine and record what happened.

For each case::

    plan = compile_workflow(v2_definition, catalogue=..., options=..., reference_resolver=...)
    for each V1 session:
        session = plan.create_session(resources, observer=recorder, error_handler=...)
        for each run of that session:
            result = session.run(inputs)      # rows = result.rows()

The recorder is an ``ExecutionObserver``: it only listens. Every call, its full
logical index, arguments and result come from the engine's own callbacks.
"""

import copy
from concurrent.futures import Future
from typing import Any, Dict, List, Optional

from blocks import FIXTURE_NAMESPACE, AuditLog, create_fixture_catalogue
from parity import LOCAL_CODE_CASES
from pydantic import BaseModel
from reference.catalogue import ReferenceCase, Run
from roboflow_workflows.execution_engine.v2.compilation import (
    WorkflowReference,
    compile_workflow,
)
from roboflow_workflows.execution_engine.v2.data import Batch
from roboflow_workflows.execution_engine.v2.declaration import Select
from roboflow_workflows.execution_engine.v2.errors import ResolvedParameterError
from roboflow_workflows.execution_engine.v2.plan import (
    CompiledWorkflow,
    CompileOptions,
    ExecutionObserver,
    ExecutionSession,
)
from translation import translate_saved_workflows, translate_workflow


def step_name(path: tuple) -> str:
    """Join a V2 step path the way V1 names inlined steps (``child__echo``).

    Args:
        path: V2 step path tuple.

    Returns:
        The ``__``-joined name.
    """
    name = "__".join(path)

    return name


def plain(value: Any) -> Any:
    """Encode engine values as JSON data, in the V1 runner's encoding.

    Args:
        value: Argument, result or row value.

    Returns:
        JSON-compatible data. A ``Batch`` becomes ``{"batch", "indices"}`` with
        full logical indices, a control result ``{"select": [...]}`` and a
        future ``{"future": {"done_when_observed": ...}}`` (never resolved here).
    """
    if isinstance(value, Future):
        return {"future": {"done_when_observed": value.done()}}
    if isinstance(value, Select):
        return {"select": list(value.targets)}
    if isinstance(value, Batch):
        return {
            "batch": [plain(item) for item in value],
            "indices": [list(index) for index in value.indices],
        }
    if isinstance(value, BaseModel):
        return plain(value.model_dump())
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    return {"object": type(value).__name__, "repr": repr(value)}


def describe_error(error: BaseException, *, phase: str) -> Dict[str, Any]:
    """Describe an exception for the report.

    Args:
        error: Raised exception.
        phase: ``compile``, ``session`` or ``run``.

    Returns:
        Phase, class name, message, step path and cause class when present.
    """
    described = {"phase": phase, "type": type(error).__name__, "message": str(error)}
    step_path = getattr(error, "step_path", None)
    if step_path:
        described["step"] = step_name(step_path)
    if error.__cause__ is not None:
        described["cause_type"] = type(error.__cause__).__name__

    return described


class InvocationRecorder(ExecutionObserver):
    """Observer collecting every call, skip and error-hook call it is told about.

    ``session`` and ``run`` label the entries; set them before each run.
    """

    def __init__(self):
        self.invocations: List[Dict[str, Any]] = []
        self.skips: List[Dict[str, Any]] = []
        self.error_hook_calls: List[Dict[str, Any]] = []
        self.session = 0
        self.run = 0
        self.instances: Dict[str, int] = {}
        self._instance_objects: List[Any] = []

    def remember_instance(self, name: str, instance: Any) -> None:
        # Identity, not equality: shows whether sessions share instances.
        for ordinal, known in enumerate(self._instance_objects):
            if known is instance:
                self.instances[name] = ordinal
                return
        self._instance_objects.append(instance)
        self.instances[name] = len(self._instance_objects) - 1

    def on_invocation(self, *, step, index, arguments, result) -> None:
        self.invocations.append(
            {
                "session": self.session,
                "run": self.run,
                "step": step_name(step),
                "index": None if index is None else list(index),
                "instance": self.instances.get(step_name(step)),
                "arguments": plain(dict(arguments)),
                "result": plain(result),
            }
        )

    def on_error(self, *, error) -> None:
        # A block that raised never reaches on_invocation; count its call here.
        # Resolved-parameter validation fails before the block is called.
        if error.__cause__ is None or isinstance(
            error.__cause__, ResolvedParameterError
        ):
            return
        self.invocations.append(
            {
                "session": self.session,
                "run": self.run,
                "step": step_name(error.step_path),
                "index": None if error.index is None else list(error.index),
                "instance": self.instances.get(step_name(error.step_path)),
                "raised": type(error.__cause__).__name__,
            }
        )

    def on_invocation_skipped(self, *, step, index, reason) -> None:
        self.skips.append(
            {
                "session": self.session,
                "run": self.run,
                "step": step_name(step),
                "index": list(index),
                "reason": str(reason),
            }
        )

    def error_handler(self, error: BaseException) -> None:
        self.error_hook_calls.append(
            {
                "step": step_name(getattr(error, "step_path", ())),
                "type": type(error).__name__,
                "cause_type": (
                    type(error.__cause__).__name__ if error.__cause__ else None
                ),
            }
        )


def _resolver(saved_workflows: Dict[str, dict], calls: List[list]):
    def resolve(reference: WorkflowReference) -> dict:
        calls.append(
            [reference.workspace_id, reference.workflow_id, reference.version_id]
        )

        return copy.deepcopy(saved_workflows[reference.workflow_id])

    return resolve


def observe_v2_case(case: ReferenceCase) -> Dict[str, Any]:
    """Compile and run one reference case on V2.

    Args:
        case: Reference case; its V1 definition is translated first.

    Returns:
        V2 definition, compile error or plan description, sessions with rows,
        errors and call counts, every invocation with its index, skips,
        resolver and error-hook calls, and the audit resources.
    """
    definition = translate_workflow(case.workflow)
    saved_workflows = translate_saved_workflows(case.saved_workflows)
    recorder = InvocationRecorder()
    resolver_calls: List[list] = []
    AuditLog.created.clear()
    observation: Dict[str, Any] = {
        "definition": definition,
        "saved_workflows": saved_workflows,
        "compile_error": None,
        "warnings": [],
        "sessions": [],
    }

    try:
        plan = compile_workflow(
            definition,
            catalogue=create_fixture_catalogue(),
            options=CompileOptions(allow_local_code=case.case_id in LOCAL_CODE_CASES),
            reference_resolver=(
                _resolver(saved_workflows, resolver_calls) if saved_workflows else None
            ),
        )
    except Exception as error:
        plan = None
        observation["compile_error"] = describe_error(error, phase="compile")
    else:
        observation["plan_steps"] = [step_name(step.path) for step in plan.steps]
        observation["expand_outputs"] = {
            step_name(step.path): sorted(
                name
                for name, output in step.outputs.items()
                if output.transform == "expand"
            )
            for step in plan.steps
        }
        observation["warnings"] = list(plan.warnings)

    for session_number in sorted({run.session for run in case.runs}):
        numbered_runs = [
            (number, run)
            for number, run in enumerate(case.runs)
            if run.session == session_number
        ]
        recorder.session = session_number
        session = _observe_session(
            case,
            plan,
            numbered_runs,
            recorder=recorder,
            compile_error=observation["compile_error"],
        )
        observation["sessions"].append(session)

    observation.update(
        invocations=recorder.invocations,
        skips=recorder.skips,
        resolver_calls=resolver_calls,
        error_hook_calls=recorder.error_hook_calls,
        resources=[
            {"resource": ordinal, "origin": log.origin, "events": plain(log.events)}
            for ordinal, log in enumerate(AuditLog.created)
        ],
    )

    return observation


def _observe_session(
    case: ReferenceCase,
    plan: Optional[CompiledWorkflow],
    numbered_runs: List[tuple],
    *,
    recorder: InvocationRecorder,
    compile_error: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    session_record: Dict[str, Any] = {
        "session": recorder.session,
        "session_error": None,
        "runs": [],
    }
    session = None
    error = compile_error
    if plan is not None:
        resources = {
            f"{FIXTURE_NAMESPACE}.{name}": AuditLog(origin="caller")
            for name in case.explicit_resources
        }
        try:
            session = plan.create_session(
                resources,
                observer=recorder,
                error_handler=(
                    recorder.error_handler if case.record_step_errors else None
                ),
            )
        except Exception as raised:
            error = describe_error(raised, phase="session")
            session_record["session_error"] = error
        else:
            session_record["resource_bindings"] = {}
            for path, instance in session.instances.items():
                name = step_name(path)
                recorder.remember_instance(name, instance)
                if isinstance(getattr(instance, "audit", None), AuditLog):
                    resource = AuditLog.created.index(instance.audit)
                    session_record["resource_bindings"][name] = resource

    for number, run in numbered_runs:
        recorder.run = number
        session_record["runs"].append(
            _observe_run(run, number, session=session, error=error, recorder=recorder)
        )

    return session_record


def _observe_run(
    run: Run,
    number: int,
    *,
    session: Optional[ExecutionSession],
    error: Optional[Dict[str, Any]],
    recorder: InvocationRecorder,
) -> Dict[str, Any]:
    inputs = copy.deepcopy(run.inputs)
    observed: Dict[str, Any] = {
        "run": number,
        "inputs": copy.deepcopy(run.inputs),
        "rows": None,
        "error": error,
    }
    calls_before = len(recorder.invocations)
    if session is not None:
        try:
            result = session.run(inputs)
            observed["rows"] = plain(result.rows())
        except Exception as raised:
            observed["error"] = describe_error(raised, phase="run")
    observed["inputs_after_run"] = plain(inputs)

    counts: Dict[str, int] = {}
    for entry in recorder.invocations[calls_before:]:
        counts[entry["step"]] = counts.get(entry["step"], 0) + 1
    observed["call_counts"] = dict(sorted(counts.items()))

    return observed
