"""Dynamic (inline Python) block examples: shared state and representation.

``dynamic_shared_state``: every dynamic step of a session, nested copies
included, sees one ``globals`` mapping; ``init`` state stays per step. A new
session starts with an empty mapping unless the caller supplies one. (V1 used
one process-wide ``globals`` by default.)

``dynamic_representation``: a block declaring ``tensor_native`` needs a
capable representation policy. The default policy refuses it before the
block's imports or ``init`` run; ``CpuArrayPolicy`` below converts lists to
NumPy arrays for the code and back to lists for the workflow.
"""

from typing import Any, Dict, List

import numpy as np
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.dynamic_blocks import RepresentationPolicy
from roboflow_workflows.execution_engine.v2.plan import CompileOptions

LOCAL_CODE = CompileOptions(allow_local_code=True)
SHARED_STATE = "dynamic_workflows_blocks.shared_state"
REPRESENTATION_POLICY = "dynamic_workflows_blocks.representation_policy"


class CpuArrayPolicy(RepresentationPolicy):
    """Accept tensor_native blocks; hand them NumPy arrays, return plain lists.

    Args:
        events: List receiving one entry per hook call, in order.
    """

    def __init__(self, events: List[str]):
        self.events = events

    def check_compatibility(self, manifest: Any) -> None:
        self.events.append(f"check_compatibility {manifest.tensor_compatibility.value}")

    def prepare_inputs(
        self, inputs: Dict[str, Any], *, manifest: Any
    ) -> Dict[str, Any]:
        self.events.append("prepare_inputs")

        return {
            name: np.asarray(value, dtype=float) if isinstance(value, list) else value
            for name, value in inputs.items()
        }

    def prepare_outputs(self, result: Any, *, manifest: Any) -> Any:
        self.events.append("prepare_outputs")

        return {
            name: value.tolist() if isinstance(value, np.ndarray) else value
            for name, value in result.items()
        }


def _tallies(rows: List[Dict[str, Any]]) -> List[List[Any]]:
    row = rows[0]
    results = [row["first"], row["second"], row["child"]]

    return [[item["step"], item["total"], item["step_calls"]] for item in results]


def dynamic_shared_state(report, *, catalogue, load) -> None:
    """One shared mapping per session across direct and nested dynamic steps."""
    plan = compile_workflow(
        load("dynamic_shared_state"), catalogue=catalogue, options=LOCAL_CODE
    )
    steps = ["$steps.first", "$steps.second", "$steps.child/inner"]

    session = plan.create_session()
    first = session.run({"x": 1})
    report.check(
        "one run: the total grows across first, second and the nested step",
        _tallies(first.rows()),
        [[steps[0], 1, 1], [steps[1], 2, 1], [steps[2], 3, 1]],
    )
    report.check(
        "the code's globals is self.shared_state; the context names this run",
        [(r["same_mapping"], r["run"]) for r in first.rows()[0].values()],
        [(True, first.run_id)] * 3,
    )
    report.check(
        "second run of the session: shared total continues, per-step init state counts 2",
        _tallies(session.run({"x": 10}).rows()),
        [[steps[0], 13, 2], [steps[1], 23, 2], [steps[2], 33, 2]],
    )
    report.check(
        "a new session starts with its own empty mapping (V1: one process-wide dict)",
        _tallies(plan.create_session().run({"x": 1}).rows()),
        [[steps[0], 1, 1], [steps[1], 2, 1], [steps[2], 3, 1]],
    )

    shared: Dict[str, Any] = {}
    sessions = [plan.create_session({SHARED_STATE: shared}) for _ in range(2)]
    tallies = [_tallies(item.run({"x": 1}).rows()) for item in sessions]
    report.check(
        "a caller-supplied mapping is shared by two sessions; step state is not",
        (tallies, shared),
        (
            [
                [[steps[0], 1, 1], [steps[1], 2, 1], [steps[2], 3, 1]],
                [[steps[0], 4, 1], [steps[1], 5, 1], [steps[2], 6, 1]],
            ],
            {"total": 6},
        ),
    )


def dynamic_representation(report, *, catalogue, load) -> None:
    """Representation policy order, default refusal and a CPU array adapter."""
    plan = compile_workflow(
        load("dynamic_representation"), catalogue=catalogue, options=LOCAL_CODE
    )

    untouched: Dict[str, Any] = {}
    try:
        plan.create_session({SHARED_STATE: untouched})
        refusal = None
    except Exception as error:
        refusal = [type(error).__name__, type(error.__cause__).__name__, str(error)]
    report.details["default_policy_refusal"] = refusal
    report.check(
        "the default policy refuses tensor_native before imports and init run",
        (refusal and refusal[:2], untouched),
        (["ResourceError", "RepresentationError"], {}),
    )

    events: List[str] = []
    shared = {"events": events}
    policy = CpuArrayPolicy(events)
    session = plan.create_session({SHARED_STATE: shared, REPRESENTATION_POLICY: policy})
    report.check(
        "with CpuArrayPolicy the session is created: compatibility checked, then init",
        (list(events), shared.get("init_ran")),
        (["check_compatibility tensor_native"], True),
    )
    events.clear()
    vector = [1, 2, 4]
    rows = session.run({"vector": vector}).rows()
    report.check(
        "per call: prepare_inputs, then the code (receiving an ndarray), then prepare_outputs",
        events,
        ["prepare_inputs", "code received ndarray", "prepare_outputs"],
    )
    report.check(
        "rows hold plain lists and floats; the caller's list is unchanged",
        (rows, vector),
        ([{"scaled": [0.25, 0.5, 1.0], "mean": 7 / 3}], [1, 2, 4]),
    )
