"""Boundary examples: parameter validation, nested inputs and execution context.

Each function fills an ``ExampleReport`` from real ``compile_workflow`` ->
``create_session`` -> ``run`` executions of a definition in ``workflows/``.
"""

import copy
from typing import Any, Dict, List, Tuple

from boundary_blocks import SETTINGS_DECODER_CALLS
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.context import (
    NoExecutionContextError,
    get_execution_context,
)
from roboflow_workflows.execution_engine.v2.plan import ExecutionObserver
from v2_runner import plain


class ArgumentObjects(ExecutionObserver):
    """Keep the argument objects of every call, to check identity."""

    def __init__(self):
        self.calls: List[Tuple[str, Any, Dict[str, Any]]] = []

    def on_invocation(self, *, step, index, arguments, result) -> None:
        self.calls.append(("__".join(step), index, dict(arguments)))

    def arguments_of(self, step: str) -> List[Dict[str, Any]]:
        return [arguments for name, _, arguments in self.calls if name == step]


def _failure(error: BaseException) -> Dict[str, Any]:
    cause = error.__cause__
    described = {
        "type": type(error).__name__,
        "index": list(error.index) if getattr(error, "index", None) else None,
        "cause": type(cause).__name__ if cause else None,
        "field_path": list(
            getattr(error, "field_path", None) or getattr(cause, "field_path", ()) or ()
        ),
        "message": str(error),
        "cause_message": str(cause) if cause else None,
    }

    return described


def validated_parameters(report, *, catalogue, load) -> None:
    """Literal and selected values meet the same Params rules; identity is kept."""
    definition = load("validated_parameters")
    plan = compile_workflow(definition, catalogue=catalogue)
    base = {"scores": [0.2, 0.7, 0.9], "threshold": 0.5}

    observer = ArgumentObjects()
    session = plan.create_session(observer=observer)
    options = {}
    rows = session.run({**base, "options": options}).rows()
    report.check(
        "threshold 0.5 selected", [r["passed"] for r in rows], [False, True, True]
    )
    report.check(
        "every call received the caller's options dict itself (not a copy)",
        [
            arguments["options"] is options
            for arguments in observer.arguments_of("check")
        ],
        [True, True, True],
    )
    report.check("the block counted its calls in that dict", options, {"calls": 3})

    rows = session.run({**base, "margin": -0.5, "options": {}}).rows()
    report.check(
        "margin's ge=0 sits on the literal branch only: selected -0.5 is accepted",
        [r["passed"] for r in rows],
        [True, True, True],
    )

    failures = {
        "selected threshold 1.5 breaks the shared le=1": {"threshold": 1.5},
        "selected bounds break the model validator": {"low": 0.8, "high": 0.2},
        "selected blank label breaks the field validator": {"label": "  "},
    }
    expected = {
        "selected threshold 1.5 breaks the shared le=1": ["threshold"],
        "selected bounds break the model validator": [],
        "selected blank label breaks the field validator": ["label"],
    }
    report.details["runtime_failures"] = {}
    for description, change in failures.items():
        untouched = {}
        try:
            session.run({**base, **change, "options": untouched})
            failure = None
        except Exception as error:
            failure = _failure(error)
        report.details["runtime_failures"][description] = failure
        report.check(
            f"{description}: StepExecutionError at index [0], cause"
            " ResolvedParameterError, field path, block never called",
            (
                failure
                and [failure[k] for k in ("type", "index", "cause", "field_path")],
                untouched,
            ),
            (
                [
                    "StepExecutionError",
                    [0],
                    "ResolvedParameterError",
                    expected[description],
                ],
                {},
            ),
        )

    report.details["compile_failures"] = {}
    for field, literal in (("threshold", 2), ("margin", -1)):
        changed = copy.deepcopy(definition)
        changed["steps"][0][field] = literal
        try:
            compile_workflow(changed, catalogue=catalogue)
            failure = None
        except Exception as error:
            failure = _failure(error)
        report.details["compile_failures"][f"{field}={literal}"] = failure
        report.check(
            f"literal {field}={literal} fails at compile time naming the field",
            failure and [failure["type"], failure["field_path"]],
            ["ParamsValidationError", [field]],
        )


def nested_inputs(report, *, catalogue, load) -> None:
    """Child inputs: literals/defaults decoded once per run; selections by identity.

    V1 never decodes child literals or defaults, never checks the kind of a
    value entering a typed child input and cannot forward a constant child
    input to an output; V2 does all three deliberately (D021). Both reject a
    child workflow without steps.
    """
    plan = compile_workflow(load("nested_inputs"), catalogue=catalogue)
    observer = ArgumentObjects()
    session = plan.create_session(observer=observer)
    payload = {"gain": 7}
    inputs = {"payload": payload}

    SETTINGS_DECODER_CALLS.clear()
    first = session.run(inputs)
    first_decodes = sorted(SETTINGS_DECODER_CALLS)
    rows = first.rows()
    report.details["rows"] = plain(rows)
    report.check(
        "[D021] each literal/default child input decoded once in the run (V1: never)",
        first_decodes,
        ["gain=2", "gain=5"],
    )
    row = rows[0]
    report.check(
        "child steps read the default, literal and selected settings",
        [row[key] for key in ("default_a", "default_b", "literal_a", "selected_a")],
        [2, 2, 5, 7],
    )
    report.check(
        "[D021] a child output may forward its input directly, default and literal"
        " included (V1 rejects constant forwarding)",
        [
            row[key]
            for key in ("default_forwarded", "literal_forwarded", "selected_forwarded")
        ],
        [{"gain": 2}, {"gain": 5}, {"gain": 7}],
    )
    read_a = observer.arguments_of("default_child__read_a")
    read_b = observer.arguments_of("default_child__read_b")
    report.check(
        "both child consumers share one decoded object within the run",
        read_a[0]["settings"] is read_b[0]["settings"],
        True,
    )
    report.check(
        "the selected caller payload reaches the child by identity, undecoded",
        observer.arguments_of("selected_child__read_a")[0]["settings"] is payload,
        True,
    )
    report.check(
        "the caller's input mapping is unchanged",
        (inputs, inputs["payload"] is payload),
        ({"payload": {"gain": 7}}, True),
    )

    SETTINGS_DECODER_CALLS.clear()
    session.run(inputs)
    second_read_a = observer.arguments_of("default_child__read_a")[1]["settings"]
    report.check(
        "the next run decodes again into a fresh object",
        (sorted(SETTINGS_DECODER_CALLS), second_read_a is read_a[0]["settings"]),
        (["gain=2", "gain=5"], False),
    )

    try:
        compile_workflow(load("nested_without_steps"), catalogue=catalogue)
        failure = None
    except Exception as error:
        failure = _failure(error)
    report.details["without_steps"] = failure
    report.check(
        "a child workflow without steps is rejected at compile time (as in V1)",
        failure and failure["type"],
        "NestedWorkflowError",
    )

    wildcard = compile_workflow(load("nested_wildcard_into_typed"), catalogue=catalogue)
    observer = ArgumentObjects()
    wildcard_session = wildcard.create_session(observer=observer)
    report.check(
        "a wildcard step output may feed a typed child input; a valid mapping passes",
        wildcard_session.run({"payload": {"gain": 3}}).rows(),
        [{"gain": 3}],
    )
    failures = {}
    for payload in ("gain=4", 12):
        try:
            wildcard_session.run({"payload": payload})
            failures[repr(payload)] = None
        except Exception as error:
            failures[repr(payload)] = _failure(error)
    report.details["wildcard_into_typed"] = failures
    report.check(
        "[D021] a wrong payload is rejected at the child input, without decoding and"
        " before the child runs (V1: passes it through)",
        (
            [failure and failure["type"] for failure in failures.values()],
            all(
                "$steps.typed_child: $inputs.settings" in failure["message"]
                for failure in failures.values()
                if failure
            ),
            len(observer.arguments_of("typed_child__read_a")),
        ),
        (["WorkflowInputError", "WorkflowInputError"], True, 1),
    )


def execution_context(report, *, catalogue, load) -> None:
    """Blocks read their context in the constructor and in run; errors clean up."""
    plan = compile_workflow(load("execution_context"), catalogue=catalogue)
    session = plan.create_session()
    probe = session.instances[("probe",)]
    report.check(
        "constructor context: step, session, no run yet, no indices",
        probe.constructed_in,
        {
            "step_selector": "$steps.probe",
            "session_id": session.session_id,
            "run_id": None,
            "indices": [],
        },
    )
    result = session.run({"items": [5, 7]})
    report.check(
        "call contexts carry this run's id and each call's index",
        [row["seen"] for row in result.rows()],
        [
            {
                "step_selector": "$steps.probe",
                "session_id": session.session_id,
                "run_id": result.run_id,
                "indices": [[position]],
            }
            for position in (0, 1)
        ],
    )

    try:
        session.run({"items": [5, 7], "fail_on": 7})
        failure = None
    except Exception as error:
        failure = _failure(error)
    report.details["failure"] = failure
    report.check(
        "a failing call reports its index and cause",
        failure and [failure["type"], failure["index"], failure["cause"]],
        ["StepExecutionError", [1], "RuntimeError"],
    )
    try:
        get_execution_context()
        leaked = True
    except NoExecutionContextError:
        leaked = False
    report.check("no context is left active after the failure", leaked, False)
    report.check(
        "the same session still runs afterwards",
        len(session.run({"items": [1]}).rows()),
        1,
    )
