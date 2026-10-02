"""What V2 must do for each reference case, beyond reproducing V1.

``match`` runs are compared with the live V1 observation. Every other run
(a V1 defect, quirk or decided boundary difference) has an explicit V2
expectation here, written from the decisions, never copied from a V2 run.

Error classes differ between the engines. They are compared through the
category map below; the class names themselves are reported unchanged.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# V1 and V2 error classes that describe the same problem.
ERROR_CATEGORIES = {
    # V1
    "RuntimeInputError": "input",
    "ExecutionGraphStructureError": "cycle",
    "InnerWorkflowCompositionCycleError": "cycle",
    "ControlFlowDefinitionError": "lineage",
    "InnerWorkflowParameterBindingsMissingRequiredError": "nested_composition",
    "InvalidReferenceTargetError": "unknown_reference",
    "StepExecutionError": "step_execution",
    # V2
    "WorkflowInputError": "input",
    "CycleError": "cycle",
    "LineageError": "lineage",
    "NestedWorkflowError": "nested_composition",
    "SelectorError": "unknown_reference",
}

# Cases whose definitions contain inline Python blocks. V2 runs submitted code
# only when the caller opts in.
LOCAL_CODE_CASES = (
    "dynamic.stateful_inline_python_block",
    "dynamic.child_workflow_inline_python_block",
)


@dataclass(frozen=True)
class V2Expectation:
    """Asserted V2 result of one run that deliberately differs from V1.

    Attributes:
        rows: Expected ``RunResult.rows()``.
        calls: Expected invocation count per step (``__``-joined paths);
            omitted steps must not run.
        error_category: Expected error category, if the run must fail.
        indices: Expected invocation indices of selected steps.
    """

    rows: Optional[List[Dict[str, Any]]]
    calls: Dict[str, int]
    error_category: Optional[str] = None
    indices: Optional[Dict[str, List[List[int]]]] = None


V2_EXPECTATIONS: Dict[Tuple[str, int], V2Expectation] = {
    # OBS-V1-01: an empty mask denies; control decisions are a conjunction.
    ("control.two_gates_single_item_empty_mask", 0): V2Expectation(
        rows=[{"unchanged": 1}],
        calls={"above_zero": 1, "above_one": 1},
    ),
    ("control.two_gates_empty_mask_beside_nonempty", 0): V2Expectation(
        rows=[{"unchanged": value} for value in [0, 2, 3, 4]],
        calls={"above_one": 4, "above_hundred": 4},
    ),
    # OBS-V1-03: the parent gate admits every child of the admitted parent.
    ("control.shallow_gate_over_deeper_data", 0): V2Expectation(
        rows=[{"selected": [None]}, {"selected": [20, 21]}, {"selected": []}],
        calls={"expand": 3, "parent_gate": 3, "selected": 2},
        indices={"selected": [[1, 0], [1, 1]]},
    ),
    # Genuinely empty groups reach reducers; V1 skips them.
    ("lineage.genuine_empty_expansion_reducer_skipped", 0): V2Expectation(
        rows=[
            {"scaled_children": [], "sum": 1, "collapsed": []},
            {"scaled_children": [], "sum": 20, "collapsed": []},
        ],
        calls={"expand": 2, "sum": 1, "collapse": 2},
        indices={"collapse": [[0], [1]]},
    ),
    # OBS-V1-02: a gate on a nested workflow governs every child step.
    ("nested.gate_on_child_workflow", 0): V2Expectation(
        rows=[
            {"message": [None]},
            {"message": [None, "child default"]},
            {"message": []},
        ],
        calls={"expand": 3, "child_gate": 3, "child__echo": 1, "child__notice": 1},
        indices={"child__echo": [[1, 1]], "child__notice": [[1, 1]]},
    ),
    ("nested.gate_on_child_workflow", 1): V2Expectation(
        rows=[{"message": [None]}],
        calls={"expand": 1, "child_gate": 1},
    ),
    ("nested.gate_on_child_workflow", 2): V2Expectation(
        rows=[{"message": ["child default"] * 3}],
        calls={"expand": 1, "child_gate": 3, "child__echo": 3, "child__notice": 3},
        indices={
            "child__echo": [[0, 0], [0, 1], [0, 2]],
            "child__notice": [[0, 0], [0, 1], [0, 2]],
        },
    ),
    ("nested.scalar_gate_on_two_root_child", 0): V2Expectation(
        rows=[{"first": None, "second": None}],
        calls={"gate": 1},
    ),
    # Decision 012: a selector-only field accepts a workflow parameter.
    ("binding.parameter_into_selector_only_field", 0): V2Expectation(
        rows=[{"count": 1}],
        calls={"count": 1},
    ),
    # Decision 012: outputs are ready at the boundary; there is no lazy option.
    ("outputs.future_results", 1): V2Expectation(
        rows=[{"doubled": 6, "consumed": 6}],
        calls={"deferred": 1, "consumer": 1},
    ),
}
