"""V1 workflow definitions used by the reference cases.

Everything here is plain JSON data; nothing imports the engine. Block types
``reference/*@v1`` come from the ``v1_reference_fixtures`` plugin. Other types
are real V1 core blocks: ContinueIf, SwitchCase, DimensionCollapse and the
inner workflow composition step.
"""

import copy
from typing import Any, Dict, List, Optional

CONTINUE_IF = "roboflow_core/continue_if@v1"
SWITCH_CASE = "roboflow_core/switch_case@v1"
DIMENSION_COLLAPSE = "roboflow_core/dimension_collapse@v1"
INNER_WORKFLOW = "roboflow_core/inner_workflow@v1"


def workflow(
    steps: List[dict],
    outputs: Dict[str, str],
    *,
    inputs: Optional[List[dict]] = None,
    dynamic_blocks: Optional[List[dict]] = None,
) -> dict:
    """Build a V1 workflow definition.

    Args:
        steps: Step definitions in declaration order.
        outputs: Workflow output name mapped to its selector.
        inputs: Input declarations; defaults to one batch input named ``items``.
        dynamic_blocks: Optional ``dynamic_blocks_definitions`` entries.

    Returns:
        A ``version: "1.0"`` workflow definition.
    """
    definition = {
        "version": "1.0",
        "inputs": [batch_input("items")] if inputs is None else inputs,
        "steps": steps,
        "outputs": [
            {"type": "JsonField", "name": name, "selector": selector}
            for name, selector in outputs.items()
        ],
    }
    if dynamic_blocks is not None:
        definition["dynamic_blocks_definitions"] = dynamic_blocks

    return definition


def batch_input(name: str, *, depth: int = 1, kind: Optional[List[str]] = None) -> dict:
    """Declare a ``WorkflowBatchInput``.

    Args:
        name: Input name.
        depth: Declared nesting depth of the batch.
        kind: Optional kind names.

    Returns:
        The input declaration.
    """
    declaration = {"type": "WorkflowBatchInput", "name": name}
    if depth != 1:
        declaration["dimensionality"] = depth
    if kind is not None:
        declaration["kind"] = kind

    return declaration


def parameter(name: str, *, default: Any = None) -> dict:
    """Declare a ``WorkflowParameter``; ``default=None`` means no default.

    Args:
        name: Parameter name.
        default: Default value used when the caller omits the parameter.

    Returns:
        The input declaration.
    """
    declaration = {"type": "WorkflowParameter", "name": name}
    if default is not None:
        declaration["default_value"] = default

    return declaration


def gate(
    name: str,
    selector: str,
    threshold: float,
    targets: List[str],
    *,
    comparator: str = "(Number) >",
) -> dict:
    """Build a ContinueIf step that admits values passing one comparison.

    Args:
        name: Step name.
        selector: Value compared by the condition.
        threshold: Right-hand side of the comparison.
        targets: Names of steps governed by this gate.
        comparator: V1 comparator type.

    Returns:
        The step definition.
    """
    return {
        "type": CONTINUE_IF,
        "name": name,
        "evaluation_parameters": {"value": selector},
        "condition_statement": {
            "type": "StatementGroup",
            "statements": [
                {
                    "type": "BinaryStatement",
                    "left_operand": {"type": "DynamicOperand", "operand_name": "value"},
                    "comparator": {"type": comparator},
                    "right_operand": {"type": "StaticOperand", "value": threshold},
                }
            ],
        },
        "next_steps": [f"$steps.{target}" for target in targets],
    }


def step(block: str, name: str, **fields: Any) -> dict:
    """Build a step of a ``reference/<block>@v1`` fixture block.

    Args:
        block: Fixture block short name, for example ``echo``.
        name: Step name.
        **fields: Step parameters, literals or selectors.

    Returns:
        The step definition.
    """
    return {"type": f"reference/{block}@v1", "name": name, **fields}


def inner(name: str, **fields: Any) -> dict:
    """Build an inner-workflow composition step.

    Args:
        name: Step name.
        **fields: ``workflow_definition`` or a saved reference, plus bindings.

    Returns:
        The step definition.
    """
    return {"type": INNER_WORKFLOW, "name": name, **fields}


# --- Parameter binding -----------------------------------------------------

LITERAL_DEFAULT_SELECTORS = workflow(
    [
        step("scale", "literal", value="$inputs.values", factor=3),
        step("scale", "default", value="$inputs.values"),
        step("scale", "from_param", value="$inputs.values", factor="$inputs.f"),
        step(
            "scale", "from_step", value="$inputs.values", factor="$steps.literal.scaled"
        ),
    ],
    {
        "literal": "$steps.literal.scaled",
        "default": "$steps.default.scaled",
        "from_param": "$steps.from_param.scaled",
        "from_step": "$steps.from_step.scaled",
    },
    inputs=[batch_input("values", kind=["float"]), parameter("f", default=10)],
)

COMPOUND_BINDINGS = workflow(
    [
        step(
            "compound",
            "c",
            params={"a": "$inputs.values", "b": 5, "c": "$inputs.f", "d": "text"},
            items=["$inputs.f", 9, "$inputs.values"],
        )
    ],
    {"echo": "$steps.c.echo"},
    inputs=[batch_input("values", kind=["float"]), parameter("f", default=7)],
)

DEFAULT_VERSUS_NULL = workflow(
    [
        step("sink", "omitted"),
        step("sink", "explicit_null", payload=None),
        step("sink", "literal", payload="literal"),
    ],
    {},
    inputs=[],
)

PARAMETER_INTO_SELECTOR_ONLY = workflow(
    [step("counter", "count", value="$inputs.scalar")],
    {"count": "$steps.count.count"},
    inputs=[parameter("scalar")],
)

IN_PLACE_MUTATION = workflow(
    [
        step("increment", "change", value="$inputs.payload"),
        step("read_count", "read", value="$steps.change.value"),
    ],
    {"seen": "$steps.read.seen"},
    inputs=[{"type": "WorkflowParameter", "name": "payload", "kind": ["dictionary"]}],
)

STEP_CYCLE = workflow(
    [
        step("scale", "a", value="$steps.b.scaled"),
        step("scale", "b", value="$steps.a.scaled"),
    ],
    {"a": "$steps.a.scaled"},
    inputs=[],
)

# --- Scalars and batches ---------------------------------------------------

SCALAR_BATCH_CASTING = workflow(
    [
        step("scale", "scalar_only", value="$inputs.scalar", factor=2),
        step("batch_scale", "cast", value="$inputs.scalar", factor=2),
        step("batch_scale", "batch", value="$inputs.values", factor="$inputs.scalar"),
    ],
    {
        "scalar_only": "$steps.scalar_only.scaled",
        "cast": "$steps.cast.scaled",
        "batch": "$steps.batch.scaled",
    },
    inputs=[batch_input("values", kind=["float"]), parameter("scalar")],
)

RUNTIME_INPUT_BROADCAST = workflow(
    [step("scale", "s", value="$inputs.a", factor="$inputs.b")],
    {"s": "$steps.s.scaled"},
    inputs=[batch_input("a", kind=["float"]), batch_input("b", kind=["float"])],
)

RUNTIME_INPUT_LENGTH_MISMATCH = workflow(
    [step("echo", "values", value="$inputs.values")],
    {"result": "$steps.values.value"},
    inputs=[batch_input("labels"), batch_input("values")],
)

MIXED_SCALAR_OR_BATCH = workflow(
    [
        step("mixed_scale", "from_batch", value="$inputs.values", factor=10),
        step("mixed_scale", "from_scalar", value="$inputs.scalar", factor=10),
    ],
    {
        "from_batch": "$steps.from_batch.scaled",
        "from_scalar": "$steps.from_scalar.scaled",
    },
    inputs=[batch_input("values", kind=["float"]), parameter("scalar")],
)

INPUT_FREE_STEP = workflow(
    [
        step("scale", "constant", value=4, factor=2),
        step(
            "scale", "per_item", value="$inputs.values", factor="$steps.constant.scaled"
        ),
    ],
    {"constant": "$steps.constant.scaled", "per_item": "$steps.per_item.scaled"},
    inputs=[batch_input("values", kind=["float"])],
)

# --- Control flow ----------------------------------------------------------

CONTROL_ONLY_SINK = workflow(
    [
        gate("gate", "$inputs.items", 0, ["notice"]),
        step("sink", "notice", payload="constant alert"),
    ],
    {"unchanged": "$inputs.items"},
)

UNGATED_SINK = workflow(
    [step("sink", "notice", payload="constant alert")],
    {"unchanged": "$inputs.items"},
)

TWO_GATES = workflow(
    [
        gate("above_zero", "$inputs.items", 0, ["notice"]),
        gate("above_one", "$inputs.items", 1, ["notice"]),
        step("sink", "notice", payload="both gates"),
    ],
    {"unchanged": "$inputs.items"},
)

TWO_GATES_ONE_ALWAYS_EMPTY = workflow(
    [
        gate("above_one", "$inputs.items", 1, ["notice"]),
        gate("above_hundred", "$inputs.items", 100, ["notice"]),
        step("sink", "notice", payload="both gates"),
    ],
    {"unchanged": "$inputs.items"},
)

PARENT_AND_CHILD_GATES = workflow(
    [
        step("ragged_expand", "expand", value="$inputs.items"),
        gate("parent_gate", "$inputs.items", 1, ["notice"]),
        gate("child_gate", "$steps.expand.value", 20, ["notice"]),
        step("sink", "notice", payload="selected child"),
    ],
    {"expanded": "$steps.expand.value"},
)

SHALLOW_GATE_DEEP_DATA = workflow(
    [
        step("ragged_expand", "expand", value="$inputs.items"),
        gate("parent_gate", "$inputs.items", 1, ["selected"]),
        step("echo", "selected", value="$steps.expand.value"),
    ],
    {"selected": "$steps.selected.value"},
)

DEEP_GATE_SHALLOW_DATA = workflow(
    [
        step("ragged_expand", "expand", value="$inputs.items"),
        gate("child_gate", "$steps.expand.value", 20, ["selected"]),
        step("echo", "selected", value="$inputs.items"),
    ],
    {"selected": "$steps.selected.value"},
)

SWITCH_CASE_RECOVERY = workflow(
    [
        {
            "type": SWITCH_CASE,
            "name": "route",
            "value": "$inputs.items",
            "cases": {"1": "$steps.left", "2": "$steps.right"},
        },
        step("echo", "left", value="$inputs.items"),
        step("echo", "right", value="$inputs.items"),
        step("merge", "merge", values=["$steps.left.value", "$steps.right.value"]),
    ],
    {"merged": "$steps.merge.value"},
)

GATE_PROPAGATES = workflow(
    [
        gate("gate", "$inputs.values", 15, ["gated"]),
        step("scale", "gated", value="$inputs.values", factor=2),
        step("scale", "after", value="$steps.gated.scaled", factor=2),
        step("counter", "ungated", value="$inputs.values"),
    ],
    {
        "gated": "$steps.gated.scaled",
        "after": "$steps.after.scaled",
        "ungated": "$steps.ungated.count",
    },
    inputs=[batch_input("values", kind=["float"])],
)

SCALAR_GATE_WILDCARD = workflow(
    [
        gate("gate", "$inputs.p", 15, ["gated"]),
        step("scale", "gated", value="$inputs.p", factor=2),
        step("compound", "echo", params={"p": "$inputs.p"}),
    ],
    {"gated": "$steps.gated.scaled", "everything": "$steps.echo.*"},
    inputs=[parameter("p")],
)

FAN_OUT = workflow(
    [
        gate("gate", "$inputs.items", 0, ["notice", "echo"]),
        step("sink", "notice", payload="fan-out"),
        step("echo", "echo", value="$inputs.items"),
    ],
    {"echo": "$steps.echo.value"},
)

# --- Lineage: expansion, filtering and reduction ---------------------------

EXPAND_SCALE_REDUCE = workflow(
    [
        step("offset_expand", "expand", value="$inputs.values", offsets=[1, 2]),
        step("scale", "scale_child", value="$steps.expand.child", factor=10),
        step(
            "sum_children",
            "sum",
            parent="$inputs.values",
            children="$steps.scale_child.scaled",
        ),
        {"type": DIMENSION_COLLAPSE, "name": "collapse", "data": "$steps.expand.child"},
    ],
    {
        "children": "$steps.expand.child",
        "scaled_children": "$steps.scale_child.scaled",
        "sum": "$steps.sum.total",
        "collapsed": "$steps.collapse.output",
    },
    inputs=[batch_input("values", kind=["float"])],
)

FILTER_EXPAND_FILTER_REDUCE = workflow(
    [
        gate("parent_gate", "$inputs.items", 0, ["expand"]),
        step("ragged_expand", "expand", value="$inputs.items"),
        gate("child_gate", "$steps.expand.value", 20, ["selected"]),
        step("echo", "selected", value="$steps.expand.value"),
        {"type": DIMENSION_COLLAPSE, "name": "reduce", "data": "$steps.selected.value"},
    ],
    {
        "expanded": "$steps.expand.value",
        "selected": "$steps.selected.value",
        "reduced": "$steps.reduce.output",
    },
)


def _filtered_children_workflow(offsets: List[int], *, collapse: bool) -> dict:
    steps = [
        step("offset_expand", "expand", value="$inputs.values", offsets=offsets),
        gate("gate", "$steps.expand.child", 15, ["scale_child"]),
        step("scale", "scale_child", value="$steps.expand.child", factor=10),
        step(
            "sum_children",
            "sum",
            parent="$inputs.values",
            children="$steps.scale_child.scaled",
        ),
    ]
    outputs = {
        "scaled_children": "$steps.scale_child.scaled",
        "sum": "$steps.sum.total",
    }
    if collapse:
        steps.append(
            {
                "type": DIMENSION_COLLAPSE,
                "name": "collapse",
                "data": "$steps.scale_child.scaled",
            }
        )
        outputs["collapsed"] = "$steps.collapse.output"

    filtered = workflow(steps, outputs, inputs=[batch_input("values", kind=["float"])])

    return filtered


FILTERED_CHILDREN_REACH_JOIN = _filtered_children_workflow([1, 20], collapse=False)
ALL_CHILDREN_FILTERED = _filtered_children_workflow([1, 2], collapse=True)
GENUINE_EMPTY_EXPANSION = _filtered_children_workflow([], collapse=True)

# --- Nested workflows ------------------------------------------------------


def _scaling_child(*, fail: bool = False) -> dict:
    inner_step = (
        step("fail", "inner_scale", value="$inputs.x")
        if fail
        else step("scale", "inner_scale", value="$inputs.x", factor="$inputs.k")
    )
    output = "$steps.inner_scale.never" if fail else "$steps.inner_scale.scaled"
    child = workflow(
        [inner_step],
        {"y": output},
        inputs=[batch_input("x", kind=["float"]), parameter("k", default=100)],
    )

    return child


CHILD_ON_EXPANDED_CHILDREN = workflow(
    [
        step("offset_expand", "expand", value="$inputs.values", offsets=[1, 2]),
        inner(
            "child",
            workflow_definition=_scaling_child(),
            parameter_bindings={"x": "$steps.expand.child"},
        ),
        step("scale", "after_child", value="$steps.child.y", factor=1),
    ],
    {"child_y": "$steps.child.y", "after_child": "$steps.after_child.scaled"},
    inputs=[batch_input("values", kind=["float"])],
)

CHILD_STEP_ERROR = workflow(
    [
        inner(
            "child",
            workflow_definition=_scaling_child(fail=True),
            parameter_bindings={"x": "$inputs.values"},
        )
    ],
    {"y": "$steps.child.y"},
    inputs=[batch_input("values", kind=["float"])],
)

CHILD_MISSING_REQUIRED_BINDING = workflow(
    [
        inner(
            "child",
            workflow_definition=_scaling_child(),
            parameter_bindings={"k": 3},
        )
    ],
    {"y": "$steps.child.y"},
    inputs=[batch_input("values", kind=["float"])],
)

# Two independent roots: the echo exports a value, the sink only acts.
ECHO_AND_NOTICE_CHILD = workflow(
    [
        step("echo", "echo", value="$inputs.message"),
        step("sink", "notice", payload="nested alert"),
    ],
    {"message": "$steps.echo.value"},
    inputs=[parameter("message", default="child default")],
)

GATE_ON_CHILD = workflow(
    [
        step("ragged_expand", "expand", value="$inputs.items"),
        gate("child_gate", "$steps.expand.value", 20, ["child"]),
        inner("child", workflow_definition=ECHO_AND_NOTICE_CHILD),
    ],
    {"message": "$steps.child.message"},
)

REPEATED_EMBEDDED_CHILD = workflow(
    [
        inner("first", workflow_definition=ECHO_AND_NOTICE_CHILD),
        inner("second", workflow_definition=ECHO_AND_NOTICE_CHILD),
    ],
    {"first": "$steps.first.message", "second": "$steps.second.message"},
    inputs=[],
)

MISSING_CHILD_OUTPUT = copy.deepcopy(REPEATED_EMBEDDED_CHILD)
MISSING_CHILD_OUTPUT["outputs"][0]["selector"] = "$steps.first.missing"

REPEATED_SAVED_REFERENCE = workflow(
    [
        inner("first", workflow_workspace_id="local", workflow_id="saved_child"),
        inner("second", workflow_workspace_id="local", workflow_id="saved_child"),
    ],
    {"first": "$steps.first.message", "second": "$steps.second.message"},
    inputs=[],
)

SELF_REFERENCING_CHILD = workflow(
    [inner("again", workflow_workspace_id="local", workflow_id="saved_child")],
    {},
    inputs=[],
)

TWO_ROOT_CHILD = workflow(
    [step("echo", "first", value="first"), step("echo", "second", value="second")],
    {"first": "$steps.first.value", "second": "$steps.second.value"},
    inputs=[],
)

SCALAR_GATE_ON_TWO_ROOT_CHILD = workflow(
    [
        gate("gate", "$inputs.value", 0, ["inner"]),
        inner("inner", workflow_definition=TWO_ROOT_CHILD),
    ],
    {"first": "$steps.inner.first", "second": "$steps.inner.second"},
    inputs=[parameter("value")],
)

GATED_LABEL_CHILD = workflow(
    [
        gate("gate", "$inputs.values", 0, ["echo"]),
        step("echo", "echo", value="$inputs.label"),
    ],
    {"result": "$steps.echo.value"},
    inputs=[batch_input("values"), parameter("label", default="default")],
)

GROUPED_REUSED_CHILD = workflow(
    [
        inner(
            name,
            workflow_definition=GATED_LABEL_CHILD,
            parameter_bindings={"values": "$inputs.values", "label": name},
        )
        for name in ["left", "right"]
    ],
    {"left": "$steps.left.result", "right": "$steps.right.result"},
    inputs=[batch_input("values", depth=2)],
)

LEAF_CHILD = _scaling_child()

MIDDLE_CHILD = workflow(
    [
        inner(
            "leaf",
            workflow_definition=LEAF_CHILD,
            parameter_bindings={"x": "$inputs.x"},
        )
    ],
    {"y": "$steps.leaf.y"},
    inputs=[batch_input("x", kind=["float"])],
)

CHILD_WITHIN_CHILD = workflow(
    [
        step("scale", "double", value="$inputs.values", factor=2),
        inner(
            "middle",
            workflow_definition=MIDDLE_CHILD,
            parameter_bindings={"x": "$steps.double.scaled"},
        ),
    ],
    {"y": "$steps.middle.y"},
    inputs=[batch_input("values", kind=["float"])],
)

# --- Lifecycle, resources and futures ---------------------------------------

STATEFUL_COUNTER = workflow(
    [step("counter", "count", value="$inputs.values")],
    {"count": "$steps.count.count"},
    inputs=[batch_input("values", kind=["float"])],
)

TWO_COUNTERS = workflow(
    [
        step("counter", "first", value="$inputs.values"),
        step("counter", "second", value="$inputs.values"),
    ],
    {"first": "$steps.first.count", "second": "$steps.second.count"},
    inputs=[batch_input("values", kind=["float"])],
)

FUTURE_RESULTS = workflow(
    [
        step("deferred", "deferred", value="$inputs.values"),
        step("scale", "consumer", value="$steps.deferred.doubled", factor=1),
    ],
    {"doubled": "$steps.deferred.doubled", "consumed": "$steps.consumer.scaled"},
    inputs=[batch_input("values", kind=["float"])],
)

# --- Dynamic (inline Python) blocks -----------------------------------------

_COUNTING_OFFSET_INIT = """
def init() -> dict:
    return {"calls": 0}
"""

_COUNTING_OFFSET_RUN = """
def run(self, value: float) -> BlockResult:
    self._init_results["calls"] += 1
    return {"total": value + 100, "calls": self._init_results["calls"]}
"""

_ECHO_RUN = """
def run(self, value: str) -> BlockResult:
    return {"output": value}
"""


def _dynamic_block(
    block_type: str, *, inputs: dict, outputs: List[str], code: dict
) -> dict:
    return {
        "type": "DynamicBlockDefinition",
        "manifest": {
            "type": "ManifestDescription",
            "block_type": block_type,
            "inputs": inputs,
            "outputs": {
                name: {"type": "DynamicOutputDefinition", "kind": []}
                for name in outputs
            },
        },
        "code": {"type": "PythonCode", **code},
    }


COUNTING_OFFSET_BLOCK = _dynamic_block(
    "CountingOffset",
    inputs={
        "value": {
            "type": "DynamicInputDefinition",
            "selector_types": ["input_parameter", "step_output"],
        }
    },
    outputs=["total", "calls"],
    code={
        "run_function_code": _COUNTING_OFFSET_RUN,
        "init_function_code": _COUNTING_OFFSET_INIT,
        "init_function_name": "init",
    },
)

DYNAMIC_STATEFUL_BLOCK = workflow(
    [{"type": "CountingOffset", "name": "offset", "value": "$inputs.values"}],
    {"total": "$steps.offset.total", "calls": "$steps.offset.calls"},
    inputs=[batch_input("values", kind=["float"])],
    dynamic_blocks=[COUNTING_OFFSET_BLOCK],
)

INNER_ECHO_BLOCK = _dynamic_block(
    "InnerScalarEcho",
    inputs={
        "value": {
            "type": "DynamicInputDefinition",
            "selector_types": ["input_parameter"],
        }
    },
    outputs=["output"],
    code={"run_function_code": _ECHO_RUN},
)

CHILD_WITH_DYNAMIC_BLOCK = workflow(
    [{"type": "InnerScalarEcho", "name": "pick", "value": "$inputs.child_msg"}],
    {"echo": "$steps.pick.output"},
    inputs=[parameter("child_msg", default="default-child")],
    dynamic_blocks=[INNER_ECHO_BLOCK],
)

PARENT_OF_DYNAMIC_CHILD = workflow(
    [
        inner(
            "nested",
            workflow_definition=CHILD_WITH_DYNAMIC_BLOCK,
            parameter_bindings={"child_msg": "$inputs.root_msg"},
        )
    ],
    {"final": "$steps.nested.echo"},
    inputs=[parameter("root_msg", default="unused-root")],
)
