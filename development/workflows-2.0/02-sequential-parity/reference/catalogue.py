"""Named V1 reference cases: definitions, input runs, pins and comparison labels.

Each case is one workflow definition executed by the real V1 engine for one or
more input runs. ``v1_calls`` pins the measured per-step invocation counts of
every run; the V1 runner checks them against the live execution, so a pin can
never replace an observation. ``comparison`` says whether V2 should reproduce
the V1 observation or deliberately differ from it.

This module imports no engine code. A V2 demo can read the same definitions
and inputs in its own process.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from . import definitions as wf

MATCH = "match"
V1_DEFECT = "v1_defect"
V1_ANOMALY = "v1_anomaly"
V1_QUIRK = "v1_quirk"
V2_BOUNDARY = "v2_boundary"


@dataclass(frozen=True)
class Comparison:
    """How a V2 result relates to the V1 observation of a case.

    Attributes:
        kind: ``match`` (V2 reproduces values, errors and call counts),
            ``v1_defect`` (decided V2 correction), ``v1_anomaly`` (V1 behaviour
            awaiting a disposition; V2 must not copy it silently),
            ``v1_quirk`` (disclosed, accepted difference) or ``v2_boundary``
            (V2 boundary semantics decided differently from V1).
        label: Stable anomaly identifier, for example ``OBS-V1-01``.
        v1_behaviour: What V1 does, in one sentence.
        intended_v2: What V2 is expected to do instead.
    """

    kind: str = MATCH
    label: Optional[str] = None
    v1_behaviour: Optional[str] = None
    intended_v2: Optional[str] = None


@dataclass(frozen=True)
class Run:
    """One engine run of a case.

    Attributes:
        inputs: Runtime parameters passed to ``ExecutionEngine.run``.
        v1_calls: Measured V1 invocation count per step; omitted steps ran 0 times.
        v1_error: Expected V1 error class name, if the run or compilation fails.
        session: Runs with the same session share one compiled engine and
            its block instances; a new session compiles a new engine.
        resolve_output_futures: Passed to ``ExecutionEngine.run``.
        comparison: Overrides the case comparison for this run only.
    """

    inputs: Dict[str, Any]
    v1_calls: Dict[str, int]
    v1_error: Optional[str] = None
    session: int = 0
    resolve_output_futures: bool = True
    comparison: Optional[Comparison] = None


@dataclass(frozen=True)
class ReferenceCase:
    """A named workflow scenario with its runs and comparison label.

    Attributes:
        case_id: Stable identifier, ``<group>.<name>``.
        title: One-line description of the feature shown.
        acceptance: Re-delivery acceptance rows (P01-P20) exercised.
        origins: Earlier investigation cases this case preserves.
        workflow: V1 workflow definition.
        runs: Input runs in execution order.
        comparison: Expected V2 relation to the V1 observation.
        saved_workflows: Definitions returned by a local saved-workflow resolver.
        explicit_resources: Init parameters the caller supplies explicitly;
            the runner creates one object per session and passes it by name.
        record_step_errors: Pass a recording ``step_error_handler`` to V1.
    """

    case_id: str
    title: str
    acceptance: Tuple[str, ...]
    origins: Tuple[str, ...]
    workflow: Dict[str, Any]
    runs: Tuple[Run, ...]
    comparison: Comparison = field(default_factory=Comparison)
    saved_workflows: Optional[Dict[str, Dict[str, Any]]] = None
    explicit_resources: Tuple[str, ...] = ()
    record_step_errors: bool = False

    def as_dict(self) -> Dict[str, Any]:
        """Return the case as plain data for ``json.dumps``.

        Returns:
            All fields as nested dictionaries; tuples serialise as JSON lists.
        """
        data = asdict(self)

        return data


OBS_V1_01 = Comparison(
    kind=V1_DEFECT,
    label="OBS-V1-01",
    v1_behaviour=(
        "An empty batch control mask is discarded when intersected with a "
        "nonempty mask, so the target runs as if only the other gate existed."
    ),
    intended_v2=(
        "Multiple required control decisions are a conjunction: a gate that "
        "admits nothing causes zero target calls."
    ),
)

OBS_V1_02 = Comparison(
    kind=V1_DEFECT,
    label="OBS-V1-02",
    v1_behaviour=(
        "Control targeting an inner workflow is redirected to the first inlined "
        "child step only; other independent child roots run ungated."
    ),
    intended_v2=(
        "A gate on a nested workflow governs its whole invocation: every child "
        "root, including output-free side effects, is suppressed."
    ),
)

OBS_V1_03 = Comparison(
    kind=V1_DEFECT,
    label="OBS-V1-03",
    v1_behaviour=(
        "A parent-level gate over deeper, already existing data compiles but "
        "admits no child index, so the target makes zero calls."
    ),
    intended_v2=(
        "Decision 007: an ancestor gate admits, by index prefix, every "
        "descendant of an admitted parent; denied parents' children make no calls."
    ),
)

GENUINE_EMPTY_REDUCER_SKIP = Comparison(
    kind=V1_QUIRK,
    label="V1-QUIRK-EMPTY-EXPANSION",
    v1_behaviour=(
        "When an expansion produces a genuinely empty group, V1 never calls "
        "the reducers of that group; their outputs are null."
    ),
    intended_v2=(
        "Genuine empty groups remain distinct from filtered groups and can "
        "reach a reducer; disclose the difference instead of copying the skip."
    ),
)

SELECTOR_ONLY_PARAMETER = Comparison(
    kind=V2_BOUNDARY,
    label="D012-SELECTOR-ONLY",
    v1_behaviour=(
        "A WorkflowParameter bound to a selector-only field fails at run time "
        "with RuntimeInputError."
    ),
    intended_v2=(
        "Decision 012: a selector-only field accepts any compatible selector, "
        "including a workflow parameter; the value is validated by kind."
    ),
)

READY_BOUNDARY_FUTURES = Comparison(
    kind=V2_BOUNDARY,
    label="D012-READY-BOUNDARY",
    v1_behaviour=(
        "With resolve_output_futures=False, V1 returns lazy future wrappers "
        "in the output rows."
    ),
    intended_v2=(
        "Decision 012: outputs leaving the engine are always ready; a deferred "
        "output option is an open design question for an asynchronous mode."
    ),
)

INPUT_PREPARATION_MUTATION = Comparison(
    kind=V2_BOUNDARY,
    label="D012-INPUT-PREPARATION",
    v1_behaviour=(
        "Input preparation writes defaults, coerced values and broadcast lists "
        "back into the caller's runtime parameter mapping."
    ),
    intended_v2=(
        "Decision 012: the caller's mapping is not rewritten; only payload "
        "mutations declared by blocks stay visible."
    ),
)

REFERENCE_CASES: Tuple[ReferenceCase, ...] = (
    # --- Parameter binding --------------------------------------------------
    ReferenceCase(
        case_id="binding.literal_default_and_selectors",
        title="One parameter bound as literal, default, input selector and step selector",
        acceptance=("P02",),
        origins=("claude-probe:literals_defaults_selectors",),
        workflow=wf.LITERAL_DEFAULT_SELECTORS,
        runs=(
            Run(
                inputs={"values": [1, 2]},
                v1_calls={"literal": 2, "default": 2, "from_param": 2, "from_step": 2},
            ),
            Run(
                inputs={"values": [1, 2], "f": 100},
                v1_calls={"literal": 2, "default": 2, "from_param": 2, "from_step": 2},
            ),
        ),
    ),
    ReferenceCase(
        case_id="binding.compound_list_and_dict",
        title="List and dict parameters mixing batch selectors, parameters and literals",
        acceptance=("P02",),
        origins=("claude-probe:compound_bindings",),
        workflow=wf.COMPOUND_BINDINGS,
        runs=(Run(inputs={"values": [1, 2]}, v1_calls={"c": 2}),),
    ),
    ReferenceCase(
        case_id="binding.default_versus_explicit_null",
        title="Omitted parameter takes its default; explicit null stays null",
        acceptance=("P02", "P07"),
        origins=("new",),
        workflow=wf.DEFAULT_VERSUS_NULL,
        runs=(
            Run(inputs={}, v1_calls={"omitted": 1, "explicit_null": 1, "literal": 1}),
        ),
    ),
    ReferenceCase(
        case_id="binding.parameter_into_selector_only_field",
        title="Workflow parameter bound to a selector-only batch field is rejected",
        acceptance=("P03",),
        origins=("claude-probe:parameter_into_selector_only_field",),
        workflow=wf.PARAMETER_INTO_SELECTOR_ONLY,
        runs=(Run(inputs={"scalar": 5}, v1_calls={}, v1_error="RuntimeInputError"),),
        comparison=SELECTOR_ONLY_PARAMETER,
    ),
    ReferenceCase(
        case_id="binding.in_place_mutation",
        title="A block mutates its input in place; the dependent reader sees the change",
        acceptance=("P17",),
        origins=("authoring:mutation_probe",),
        workflow=wf.IN_PLACE_MUTATION,
        runs=(
            Run(inputs={"payload": {"count": 0}}, v1_calls={"change": 1, "read": 1}),
        ),
    ),
    ReferenceCase(
        case_id="validation.step_dependency_cycle",
        title="Steps depending on each other are rejected before any call",
        acceptance=("P02", "P03"),
        origins=("claude-probe:step_cycle",),
        workflow=wf.STEP_CYCLE,
        runs=(Run(inputs={}, v1_calls={}, v1_error="ExecutionGraphStructureError"),),
    ),
    # --- Scalars and batches ------------------------------------------------
    ReferenceCase(
        case_id="batching.scalar_batch_auto_cast",
        title="Scalar step runs once; a scalar fed to a batch parameter becomes Batch([x])",
        acceptance=("P04",),
        origins=("claude-probe:batch_casting",),
        workflow=wf.SCALAR_BATCH_CASTING,
        runs=(
            Run(
                inputs={"values": [1, 2, 3], "scalar": 5},
                v1_calls={"scalar_only": 1, "cast": 1, "batch": 1},
            ),
        ),
    ),
    ReferenceCase(
        case_id="batching.runtime_input_broadcast",
        title="A scalar or singleton batch input broadcasts to the other batch length",
        acceptance=("P04", "P13"),
        origins=(
            "claude-probe:input_broadcast",
            "v1-semantics:test_workflow_batch_input_scalar_and_singleton_broadcast",
        ),
        workflow=wf.RUNTIME_INPUT_BROADCAST,
        runs=(
            Run(inputs={"a": [1, 2, 3], "b": 10}, v1_calls={"s": 3}),
            Run(inputs={"a": [1, 2, 3], "b": [10]}, v1_calls={"s": 3}),
        ),
    ),
    ReferenceCase(
        case_id="batching.runtime_input_length_mismatch",
        title="Batch inputs of incompatible or empty outer length are rejected",
        acceptance=("P04", "P13"),
        origins=(
            "v1-semantics:test_workflow_batch_input_rejects_empty_or_incompatible_outer_length",
        ),
        workflow=wf.RUNTIME_INPUT_LENGTH_MISMATCH,
        runs=(
            Run(
                inputs={"labels": ["x", "y"], "values": []},
                v1_calls={},
                v1_error="RuntimeInputError",
            ),
            Run(
                inputs={"labels": ["x", "y"], "values": ["a", "b", "c"]},
                v1_calls={},
                v1_error="RuntimeInputError",
            ),
        ),
    ),
    ReferenceCase(
        case_id="batching.mixed_scalar_or_batch",
        title="A scalar-or-batch parameter receives a Batch or a plain scalar",
        acceptance=("P04",),
        origins=("new",),
        workflow=wf.MIXED_SCALAR_OR_BATCH,
        runs=(
            Run(
                inputs={"values": [1, 2, 3], "scalar": 5},
                v1_calls={"from_batch": 1, "from_scalar": 1},
            ),
        ),
    ),
    ReferenceCase(
        case_id="batching.input_free_step",
        title="A literal-only step runs once and feeds every batch element",
        acceptance=("P04", "P07"),
        origins=("new",),
        workflow=wf.INPUT_FREE_STEP,
        runs=(Run(inputs={"values": [1, 2]}, v1_calls={"constant": 1, "per_item": 2}),),
    ),
    # --- Control flow -------------------------------------------------------
    ReferenceCase(
        case_id="control.batch_gate_literal_sink",
        title="Per-index gate drives a literal-only, output-free sink",
        acceptance=("P06", "P07"),
        origins=("oracle:control_only_partial", "oracle:control_only_zero"),
        workflow=wf.CONTROL_ONLY_SINK,
        runs=(
            Run(inputs={"items": [0, 1, 2, 0]}, v1_calls={"gate": 4, "notice": 2}),
            Run(inputs={"items": [0, 0]}, v1_calls={"gate": 2}),
        ),
    ),
    ReferenceCase(
        case_id="control.ungated_literal_sink",
        title="Same outputs without the gate, but the sink runs once for the whole run",
        acceptance=("P06", "P07"),
        origins=("oracle:same_values_without_gate",),
        workflow=wf.UNGATED_SINK,
        runs=(Run(inputs={"items": [0, 1, 2, 0]}, v1_calls={"notice": 1}),),
    ),
    ReferenceCase(
        case_id="control.two_gates_intersect",
        title="Two same-level gates on one target admit the intersection",
        acceptance=("P06",),
        origins=("oracle:two_control_gates_converge",),
        workflow=wf.TWO_GATES,
        runs=(
            Run(
                inputs={"items": [0, 1, 2]},
                v1_calls={"above_zero": 3, "above_one": 3, "notice": 1},
            ),
        ),
    ),
    ReferenceCase(
        case_id="control.two_gates_single_item_empty_mask",
        title="One of two gates rejects the only item, yet the target still runs",
        acceptance=("P06",),
        origins=("oracle:two_control_gates_one_mask_empty",),
        workflow=wf.TWO_GATES,
        runs=(
            Run(
                inputs={"items": [1]},
                v1_calls={"above_zero": 1, "above_one": 1, "notice": 1},
            ),
        ),
        comparison=OBS_V1_01,
    ),
    ReferenceCase(
        case_id="control.two_gates_empty_mask_beside_nonempty",
        title="Gates > 1 and > 100 over [0, 2, 3, 4]: V1 ignores the empty mask",
        acceptance=("P06",),
        origins=(
            "v1-semantics:test_observed_v1_empty_batch_mask_is_ignored_beside_nonempty_mask",
        ),
        workflow=wf.TWO_GATES_ONE_ALWAYS_EMPTY,
        runs=(
            Run(
                inputs={"items": [0, 2, 3, 4]},
                v1_calls={"above_one": 4, "above_hundred": 4, "notice": 3},
            ),
        ),
        comparison=OBS_V1_01,
    ),
    ReferenceCase(
        case_id="control.parent_and_child_gates_literal_sink",
        title="Parent and child gates jointly select child invocations of a literal sink",
        acceptance=("P06", "P07"),
        origins=("oracle:two_control_depths_without_data",),
        workflow=wf.PARENT_AND_CHILD_GATES,
        runs=(
            Run(
                inputs={"items": [1, 2, 3]},
                v1_calls={"expand": 3, "parent_gate": 3, "child_gate": 6, "notice": 4},
            ),
        ),
    ),
    ReferenceCase(
        case_id="control.shallow_gate_over_deeper_data",
        title="Parent-level gate over already expanded child data",
        acceptance=("P06",),
        origins=("oracle:shallow_control_deep_data",),
        workflow=wf.SHALLOW_GATE_DEEP_DATA,
        runs=(
            Run(inputs={"items": [1, 2, 0]}, v1_calls={"expand": 3, "parent_gate": 3}),
        ),
        comparison=OBS_V1_03,
    ),
    ReferenceCase(
        case_id="control.deep_gate_over_shallow_data_rejected",
        title="A child-level gate on parent-level data is a compile error",
        acceptance=("P03", "P06"),
        origins=("oracle:deep_control_shallow_data",),
        workflow=wf.DEEP_GATE_SHALLOW_DATA,
        runs=(
            Run(
                inputs={"items": [1, 2, 0]},
                v1_calls={},
                v1_error="ControlFlowDefinitionError",
            ),
        ),
    ),
    ReferenceCase(
        case_id="control.switch_case_branch_recovery",
        title="SwitchCase routes; an empty-accepting merge recovers the surviving branch",
        acceptance=("P06", "P08"),
        origins=("oracle:alternative_branch_data_convergence",),
        workflow=wf.SWITCH_CASE_RECOVERY,
        runs=(
            Run(
                inputs={"items": [1, 2, 3]},
                v1_calls={"route": 3, "left": 1, "right": 1, "merge": 3},
            ),
        ),
    ),
    ReferenceCase(
        case_id="control.gate_propagates_downstream",
        title="Filtered indices propagate to descendants; ungated steps are unaffected",
        acceptance=("P06", "P08"),
        origins=("claude-probe:conditional_execution",),
        workflow=wf.GATE_PROPAGATES,
        runs=(
            Run(
                inputs={"values": [10, 20, 30]},
                v1_calls={"gate": 3, "gated": 2, "after": 2, "ungated": 3},
            ),
        ),
    ),
    ReferenceCase(
        case_id="control.scalar_gate_and_wildcard_output",
        title="A scalar gate switches its branch on or off; $steps.x.* selects all outputs",
        acceptance=("P06", "P13"),
        origins=("claude-probe:scalar_flow_control_and_wildcard",),
        workflow=wf.SCALAR_GATE_WILDCARD,
        runs=(
            Run(inputs={"p": 1}, v1_calls={"gate": 1, "echo": 1}),
            Run(inputs={"p": 20}, v1_calls={"gate": 1, "gated": 1, "echo": 1}),
        ),
    ),
    ReferenceCase(
        case_id="control.fan_out_to_two_targets",
        title="One gate governs two targets: a sink and a data step",
        acceptance=("P06", "P07"),
        origins=("new",),
        workflow=wf.FAN_OUT,
        runs=(
            Run(
                inputs={"items": [0, 1, 2]},
                v1_calls={"gate": 3, "notice": 2, "echo": 2},
            ),
        ),
    ),
    # --- Lineage ------------------------------------------------------------
    ReferenceCase(
        case_id="lineage.expand_scale_reduce",
        title="Expand, per-child step, parent+children join and DimensionCollapse",
        acceptance=("P04", "P05"),
        origins=("claude-probe:expand_and_reduce",),
        workflow=wf.EXPAND_SCALE_REDUCE,
        runs=(
            Run(
                inputs={"values": [1, 2]},
                v1_calls={"expand": 2, "scale_child": 4, "sum": 1, "collapse": 2},
            ),
        ),
    ),
    ReferenceCase(
        case_id="lineage.filter_expand_filter_reduce",
        title="Filter parents, ragged expand, filter children, reduce with sparse indices",
        acceptance=("P05", "P06"),
        origins=("oracle:filtered_expansion_then_reduction",),
        workflow=wf.FILTER_EXPAND_FILTER_REDUCE,
        runs=(
            Run(
                inputs={"items": [0, 1, 2, 3]},
                v1_calls={
                    "parent_gate": 4,
                    "expand": 3,
                    "child_gate": 6,
                    "selected": 4,
                    "reduce": 3,
                },
            ),
        ),
    ),
    ReferenceCase(
        case_id="lineage.filtered_children_reach_parent_join",
        title="Partially filtered children reach the parent join with original indices",
        acceptance=("P05", "P08"),
        origins=("claude-probe:filtered_children_reach_reducer",),
        workflow=wf.FILTERED_CHILDREN_REACH_JOIN,
        runs=(
            Run(
                inputs={"values": [1, 2]},
                v1_calls={"expand": 2, "gate": 4, "scale_child": 2, "sum": 1},
            ),
        ),
    ),
    ReferenceCase(
        case_id="lineage.all_children_filtered_reducer_runs",
        title="A group whose children are all filtered still reaches the reducers",
        acceptance=("P05", "P08"),
        origins=("claude-probe:reducer_edge_groups.all_filtered_for_first_parent",),
        workflow=wf.ALL_CHILDREN_FILTERED,
        runs=(
            Run(
                inputs={"values": [1, 20]},
                v1_calls={
                    "expand": 2,
                    "gate": 4,
                    "scale_child": 2,
                    "sum": 1,
                    "collapse": 2,
                },
            ),
        ),
    ),
    ReferenceCase(
        case_id="lineage.genuine_empty_expansion_reducer_skipped",
        title="Genuinely empty expansions: V1 never calls the reducers",
        acceptance=("P05",),
        origins=("claude-probe:reducer_edge_groups.genuinely_empty_groups",),
        workflow=wf.GENUINE_EMPTY_EXPANSION,
        runs=(Run(inputs={"values": [1, 20]}, v1_calls={"expand": 2}),),
        comparison=GENUINE_EMPTY_REDUCER_SKIP,
    ),
    # --- Nested workflows ---------------------------------------------------
    ReferenceCase(
        case_id="nested.bound_child_on_expanded_children",
        title="Child bound to expanded data runs per child with its default parameter",
        acceptance=("P09",),
        origins=("claude-probe:nested_workflow",),
        workflow=wf.CHILD_ON_EXPANDED_CHILDREN,
        runs=(
            Run(
                inputs={"values": [1, 2]},
                v1_calls={"expand": 2, "child__inner_scale": 4, "after_child": 4},
            ),
        ),
    ),
    ReferenceCase(
        case_id="nested.child_step_error_context",
        title="A failing child step reports its inlined identity to the caller hook",
        acceptance=("P16",),
        origins=("claude-probe:nested_error",),
        workflow=wf.CHILD_STEP_ERROR,
        runs=(
            Run(
                inputs={"values": [1]},
                v1_calls={"child__inner_scale": 1},
                v1_error="StepExecutionError",
            ),
        ),
        record_step_errors=True,
    ),
    ReferenceCase(
        case_id="nested.missing_required_binding",
        title="Child input without binding or default is a compile error",
        acceptance=("P09", "P16"),
        origins=("claude-probe:nested_missing_binding",),
        workflow=wf.CHILD_MISSING_REQUIRED_BINDING,
        runs=(
            Run(
                inputs={"values": [1]},
                v1_calls={},
                v1_error="InnerWorkflowParameterBindingsMissingRequiredError",
            ),
        ),
    ),
    ReferenceCase(
        case_id="nested.missing_child_output",
        title="Selecting an undeclared child output is a compile error",
        acceptance=("P09", "P16"),
        origins=("oracle:nested_missing_output",),
        workflow=wf.MISSING_CHILD_OUTPUT,
        runs=(Run(inputs={}, v1_calls={}, v1_error="InvalidReferenceTargetError"),),
    ),
    ReferenceCase(
        case_id="nested.repeated_embedded_child",
        title="The same embedded child used twice gets two independent step sets",
        acceptance=("P09",),
        origins=("oracle:nested_repeated_definition",),
        workflow=wf.REPEATED_EMBEDDED_CHILD,
        runs=(
            Run(
                inputs={},
                v1_calls={
                    "first__echo": 1,
                    "first__notice": 1,
                    "second__echo": 1,
                    "second__notice": 1,
                },
            ),
        ),
    ),
    ReferenceCase(
        case_id="nested.repeated_saved_reference",
        title="Two uses of one saved child: resolver called once, both uses execute",
        acceptance=("P10",),
        origins=("oracle:nested_repeated_saved_reference",),
        workflow=wf.REPEATED_SAVED_REFERENCE,
        saved_workflows={"saved_child": wf.ECHO_AND_NOTICE_CHILD},
        runs=(
            Run(
                inputs={},
                v1_calls={
                    "first__echo": 1,
                    "first__notice": 1,
                    "second__echo": 1,
                    "second__notice": 1,
                },
            ),
        ),
    ),
    ReferenceCase(
        case_id="nested.saved_reference_cycle",
        title="A saved child that references itself is a compile error",
        acceptance=("P10",),
        origins=(
            "oracle:nested_reference_cycle",
            "claude-probe:nested_reference_cycle",
        ),
        workflow=wf.REPEATED_SAVED_REFERENCE,
        saved_workflows={"saved_child": wf.SELF_REFERENCING_CHILD},
        runs=(
            Run(
                inputs={},
                v1_calls={},
                v1_error="InnerWorkflowCompositionCycleError",
            ),
        ),
    ),
    ReferenceCase(
        case_id="nested.gate_on_child_workflow",
        title="Child-level gate targeting a nested workflow with two independent roots",
        acceptance=("P09",),
        origins=(
            "oracle:nested_conditional_at_child_depth",
            "oracle:nested_conditional_zero_selected",
            "oracle:nested_conditional_three_selected",
        ),
        workflow=wf.GATE_ON_CHILD,
        runs=(
            Run(
                inputs={"items": [1, 2, 0]},
                v1_calls={
                    "expand": 3,
                    "child_gate": 3,
                    "child__echo": 1,
                    "child__notice": 1,
                },
            ),
            Run(
                inputs={"items": [1]},
                v1_calls={"expand": 1, "child_gate": 1, "child__notice": 1},
            ),
            Run(
                inputs={"items": [3]},
                v1_calls={
                    "expand": 1,
                    "child_gate": 3,
                    "child__echo": 3,
                    "child__notice": 1,
                },
            ),
        ),
        comparison=OBS_V1_02,
    ),
    ReferenceCase(
        case_id="nested.scalar_gate_on_two_root_child",
        title="A false scalar gate on a child still lets its second root run",
        acceptance=("P09",),
        origins=(
            "v1-semantics:test_observed_parent_control_targets_only_first_inlined_child_step",
        ),
        workflow=wf.SCALAR_GATE_ON_TWO_ROOT_CHILD,
        runs=(Run(inputs={"value": 0}, v1_calls={"gate": 1, "inner__second": 1}),),
        comparison=OBS_V1_02,
    ),
    ReferenceCase(
        case_id="nested.grouped_reused_child_with_inner_gate",
        title="Reused child with an internal gate keeps the parent's depth-2 grouping",
        acceptance=("P05", "P09"),
        origins=(
            "v1-semantics:test_nested_control_preserves_parent_grouping_and_repeated_child_identity",
        ),
        workflow=wf.GROUPED_REUSED_CHILD,
        runs=(
            Run(
                inputs={"values": [[0, 1], [1, 0]]},
                v1_calls={
                    "left__gate": 4,
                    "left__echo": 2,
                    "right__gate": 4,
                    "right__echo": 2,
                },
            ),
        ),
    ),
    ReferenceCase(
        case_id="nested.child_within_child",
        title="Depth-two composition bound to an upstream parent step, leaf default kept",
        acceptance=("P09", "P10"),
        origins=("new",),
        workflow=wf.CHILD_WITHIN_CHILD,
        runs=(
            Run(
                inputs={"values": [1, 2]},
                v1_calls={"double": 2, "middle__leaf__inner_scale": 2},
            ),
        ),
    ),
    # --- Lifecycle, resources, futures and dynamic blocks -------------------
    ReferenceCase(
        case_id="lifecycle.step_state_across_runs_and_sessions",
        title="One block instance per step keeps state across runs; a new engine starts fresh",
        acceptance=("P11", "P12"),
        origins=("claude-probe:stateful_instance",),
        workflow=wf.STATEFUL_COUNTER,
        runs=(
            Run(inputs={"values": [1, 2]}, v1_calls={"count": 2}),
            Run(inputs={"values": [3]}, v1_calls={"count": 1}),
            Run(inputs={"values": [4]}, v1_calls={"count": 1}, session=1),
        ),
    ),
    ReferenceCase(
        case_id="lifecycle.plugin_initializer_resource_per_step",
        title="Without an explicit resource, the plugin initializer creates one per step",
        acceptance=("P12",),
        origins=("authoring:authoring_probe",),
        workflow=wf.TWO_COUNTERS,
        runs=(Run(inputs={"values": [1, 2]}, v1_calls={"first": 2, "second": 2}),),
    ),
    ReferenceCase(
        case_id="lifecycle.explicit_resource_shared_by_steps",
        title="An explicit namespaced resource is passed by identity to every step",
        acceptance=("P12",),
        origins=("authoring:authoring_probe",),
        workflow=wf.TWO_COUNTERS,
        explicit_resources=("audit",),
        runs=(Run(inputs={"values": [1, 2]}, v1_calls={"first": 2, "second": 2}),),
    ),
    ReferenceCase(
        case_id="outputs.future_results",
        title="Future-valued outputs are resolved for consumers and, by default, outputs",
        acceptance=("P14",),
        origins=("new",),
        workflow=wf.FUTURE_RESULTS,
        runs=(
            Run(inputs={"values": [1, 2]}, v1_calls={"deferred": 2, "consumer": 2}),
            Run(
                inputs={"values": [3]},
                v1_calls={"deferred": 1, "consumer": 1},
                resolve_output_futures=False,
                comparison=READY_BOUNDARY_FUTURES,
            ),
        ),
    ),
    ReferenceCase(
        case_id="dynamic.stateful_inline_python_block",
        title="Inline Python block with init state, compiled from the definition",
        acceptance=("P11", "P15"),
        origins=("new",),
        workflow=wf.DYNAMIC_STATEFUL_BLOCK,
        runs=(
            Run(inputs={"values": [1, 2]}, v1_calls={"offset": 2}),
            Run(inputs={"values": [3]}, v1_calls={"offset": 1}),
        ),
    ),
    ReferenceCase(
        case_id="dynamic.child_workflow_inline_python_block",
        title="A child workflow's own dynamic block runs inside the parent",
        acceptance=("P10", "P15"),
        origins=(
            "existing-test:test_inlined_dynamic_block_matches_inner_workflow_child_definitions",
        ),
        workflow=wf.PARENT_OF_DYNAMIC_CHILD,
        runs=(
            Run(
                inputs={"root_msg": "dynamic-inner-value"}, v1_calls={"nested__pick": 1}
            ),
            Run(inputs={}, v1_calls={"nested__pick": 1}),
        ),
    ),
)


def case_ids() -> List[str]:
    """List the stable identifiers of all reference cases.

    Returns:
        Case identifiers in catalogue order.
    """
    identifiers = [case.case_id for case in REFERENCE_CASES]

    return identifiers


def get_case(case_id: str) -> ReferenceCase:
    """Return one reference case by identifier.

    Args:
        case_id: Stable case identifier, for example ``control.two_gates_intersect``.

    Returns:
        The matching case.

    Raises:
        KeyError: If no case has this identifier.
    """
    for case in REFERENCE_CASES:
        if case.case_id == case_id:
            return case

    raise KeyError(f"Unknown reference case {case_id!r}; known: {case_ids()}")
