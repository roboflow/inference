"""Small class-owned blocks and helpers shared by the compiler tests.

Each block is an ordinary V2 ``Block`` subclass. The compiler reads only their
declarations; no test here constructs a block or calls ``run``.
"""

from typing import Any, Dict, List, Mapping, Optional

from pydantic import Field
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.declaration import (
    Block,
    BlockParams,
    Group,
    Output,
    Ref,
    Select,
    StepRef,
    Stop,
)
from roboflow_workflows.execution_engine.v2.kinds import (
    BOOLEAN_KIND,
    DICTIONARY_KIND,
    FLOAT_KIND,
    INTEGER_KIND,
    STRING_KIND,
)
from roboflow_workflows.execution_engine.v2.plan import ChildInputPort


class Scale(Block):
    """Ordinary block: literal, default or selector in the same field."""

    type = "test/scale@v1"
    aliases = ("Scale",)
    outputs = {"scaled": Output(FLOAT_KIND, source="value")}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)
        factor: float | Ref(FLOAT_KIND) = Field(default=2.0, ge=0)

    def run(self, *, value, factor) -> dict:
        return {"scaled": value * factor}


class Echo(Block):
    """Passes one value on; the output declares the value as its source."""

    type = "test/echo@v1"
    outputs = {"value": Output(source="value")}

    class Params(BlockParams):
        value: Any | Ref()

    def run(self, *, value) -> dict:
        return {"value": value}


class Expand(Block):
    """Expands each value into children; two outputs share one child axis."""

    type = "test/expand@v1"
    outputs = {
        "child": Output(FLOAT_KIND, expand="child"),
        "label": Output(STRING_KIND, expand="child"),
        "count": Output(INTEGER_KIND),
    }

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)

    def run(self, *, value) -> dict:
        return {}


class StableExpand(Block):
    """Expands into children with stable identities."""

    type = "test/stable_expand@v1"
    outputs = {"tile": Output(expand="tiles", stationary=True)}

    class Params(BlockParams):
        value: Ref()

    def run(self, *, value) -> dict:
        return {}


class Source(Block):
    """Input-free source creating its own root axis."""

    type = "test/source@v1"
    outputs = {"items": Output(FLOAT_KIND, expand="generated")}

    def run(self) -> dict:
        return {}


class Constant(Block):
    """Input-free scalar source."""

    type = "test/constant@v1"
    outputs = {"value": Output(FLOAT_KIND)}

    class Params(BlockParams):
        value: float = 1.0

    def run(self, *, value) -> dict:
        return {"value": value}


class SumChildren(Block):
    """Parent value plus a group of children; parent- and child-level outputs."""

    type = "test/sum_children@v1"
    outputs = {
        "total": Output(FLOAT_KIND, source="parent"),
        "shifted": Output(FLOAT_KIND, preserve="children"),
    }

    class Params(BlockParams):
        parent: float | Ref(FLOAT_KIND) = 0.0
        children: Group(FLOAT_KIND)

    def run(self, *, parent, children) -> dict:
        return {}


class Collapse(Block):
    """Only a group: runs once per parent of the grouped axis."""

    type = "test/collapse@v1"
    outputs = {"output": Output()}

    class Params(BlockParams):
        data: Group()

    def run(self, *, data) -> dict:
        return {"output": list(data)}


class NamedGroups(Block):
    """A dict of groups under one parent reference."""

    type = "test/named_groups@v1"
    outputs = {
        "summary": Output(source="reference"),
        "kept": Output(preserve="groups"),
    }

    class Params(BlockParams):
        reference: Ref()
        groups: Dict[str, Group()]

    def run(self, *, reference, groups) -> dict:
        return {}


class Consensus(Block):
    """A list of batch-delivered selectors."""

    type = "test/consensus@v1"
    outputs = {"merged": Output()}

    class Params(BlockParams):
        predictions: List[Ref(batch="always")] = Field(min_length=1)
        required: int | Ref(INTEGER_KIND) = 1

    def run(self, *, predictions, required) -> list:
        return []


class Csv(Block):
    """A dict whose selector leaves are scalar-or-batch."""

    type = "test/csv@v1"
    outputs = {"csv": Output(STRING_KIND)}

    class Params(BlockParams):
        columns: Dict[str, str | float | Ref(batch="if_varying")]

    def run(self, *, columns) -> Any:
        return {}


class BatchScale(Block):
    """Vectorized block; ``factor`` is passed once per call."""

    type = "test/batch_scale@v1"
    outputs = {"scaled": Output(FLOAT_KIND)}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND, batch="always")
        factor: float | Ref(FLOAT_KIND) = 2.0

    def run(self, *, value, factor) -> list:
        return []


class Compound(Block):
    """List and dict parameters mixing literals and selectors."""

    type = "test/compound@v1"
    outputs = {"echo": Output()}

    class Params(BlockParams):
        params: Dict[str, Any | Ref()] = {}
        items: List[Any | Ref()] = []
        labels: List[str] = []

    def run(self, *, params, items, labels) -> dict:
        return {}


class Sink(Block):
    """Output-free side-effect block with a literal-or-selector payload."""

    type = "test/sink@v1"

    class Params(BlockParams):
        payload: Optional[str] | Ref() = "default"

    def run(self, *, payload) -> dict:
        return {}


class Gate(Block):
    """Control block: continue to the targets when the value is truthy."""

    type = "test/gate@v1"

    class Params(BlockParams):
        value: Ref()
        next_steps: List[StepRef]

    def run(self, *, value, next_steps):
        return Select(next_steps) if value else Stop()


class Switch(Block):
    """Control block routing each value to one case target."""

    type = "test/switch@v1"

    class Params(BlockParams):
        value: Ref()
        cases: Dict[str, StepRef]
        default: List[StepRef] = []

    def run(self, *, value, cases, default):
        return Select(cases.get(str(value), default))


class Merge(Block):
    """Empty-accepting join of alternative branches."""

    type = "test/merge@v1"
    accepts_empty = True
    outputs = {"value": Output()}

    class Params(BlockParams):
        values: List[Ref()]

    def run(self, *, values) -> dict:
        return {"value": next((item for item in values if item is not None), None)}


class Increment(Block):
    """Mutates its dictionary input in place and passes it on."""

    type = "test/increment@v1"
    mutates = ("value",)
    outputs = {"value": Output(DICTIONARY_KIND, source="value")}

    class Params(BlockParams):
        value: Ref(DICTIONARY_KIND)

    def run(self, *, value) -> dict:
        value["count"] = value.get("count", 0) + 1
        return {"value": value}


class Reader(Block):
    """Reads a dictionary."""

    type = "test/reader@v1"
    outputs = {"seen": Output(INTEGER_KIND)}

    class Params(BlockParams):
        value: Ref(DICTIONARY_KIND)

    def run(self, *, value) -> dict:
        return {"seen": value.get("count", 0)}


class ParseFields(Block):
    """Configured outputs: one output per literal field name."""

    type = "test/parse_fields@v1"
    output_fields = ("expected_fields",)

    class Params(BlockParams):
        raw: str | Ref(STRING_KIND)
        expected_fields: List[str]

    @classmethod
    def describe_outputs(cls, params: BlockParams) -> Mapping[str, Output]:
        outputs = {name: Output() for name in params.expected_fields}
        outputs["error_status"] = Output(BOOLEAN_KIND)

        return outputs

    def run(self, *, raw, expected_fields) -> dict:
        return {}


ALL_BLOCKS = (
    Scale,
    Echo,
    Expand,
    StableExpand,
    Source,
    Constant,
    SumChildren,
    Collapse,
    NamedGroups,
    Consensus,
    Csv,
    BatchScale,
    Compound,
    Sink,
    Gate,
    Switch,
    Merge,
    Increment,
    Reader,
    ParseFields,
)

CATALOGUE = Catalogue(ALL_BLOCKS, namespace="test")


def workflow(
    steps: List[dict],
    outputs: Optional[Dict[str, str]] = None,
    *,
    inputs: Optional[List[dict]] = None,
    dynamic_blocks: Optional[List[dict]] = None,
) -> dict:
    """Build a V2 definition; inputs default to one float batch ``values``.

    Args:
        steps: Step definitions.
        outputs: Output name to selector.
        inputs: Input declarations.
        dynamic_blocks: ``dynamic_blocks_definitions`` entries.

    Returns:
        The definition.
    """
    definition = {
        "version": "2.0",
        "inputs": (
            [{"type": "WorkflowBatchInput", "name": "values", "kind": ["float"]}]
            if inputs is None
            else inputs
        ),
        "steps": steps,
        "outputs": [
            {"type": "JsonField", "name": name, "selector": selector}
            for name, selector in (outputs or {}).items()
        ],
    }
    if dynamic_blocks is not None:
        definition["dynamic_blocks_definitions"] = dynamic_blocks

    return definition


def step(block: str, name: str, **params: Any) -> dict:
    """Build a step of a ``test/<block>@v1`` fixture block.

    Args:
        block: Fixture short name, e.g. ``"scale"``.
        name: Step name.
        **params: Step parameters.

    Returns:
        The step definition.
    """
    return {"type": f"test/{block}@v1", "name": name, **params}


def gate(name: str, value: str, targets: List[str]) -> dict:
    """Build a ``test/gate@v1`` step targeting ``targets``.

    Args:
        name: Step name.
        value: Selector of the decision value.
        targets: Governed step names.

    Returns:
        The step definition.
    """
    return step(
        "gate", name, value=value, next_steps=[f"$steps.{target}" for target in targets]
    )


def nested(name: str, **fields: Any) -> dict:
    """Build a nested workflow step.

    Args:
        name: Step name.
        **fields: ``workflow_definition`` or a saved reference, plus bindings.

    Returns:
        The step definition.
    """
    return {"type": "roboflow_core/inner_workflow@v1", "name": name, **fields}


def batch_input(name: str, *, depth: int = 1, kind: Optional[List[str]] = None) -> dict:
    """Declare a ``WorkflowBatchInput``.

    Args:
        name: Input name.
        depth: Dimensionality.
        kind: Kind names.

    Returns:
        The input declaration.
    """
    declaration: Dict[str, Any] = {"type": "WorkflowBatchInput", "name": name}
    if depth != 1:
        declaration["dimensionality"] = depth
    if kind is not None:
        declaration["kind"] = kind

    return declaration


def parameter(
    name: str, *, default: Any = None, kind: Optional[List[str]] = None
) -> dict:
    """Declare a ``WorkflowParameter``; ``default=None`` means none.

    Args:
        name: Parameter name.
        default: ``default_value``.
        kind: Kind names.

    Returns:
        The input declaration.
    """
    declaration: Dict[str, Any] = {"type": "WorkflowParameter", "name": name}
    if default is not None:
        declaration["default_value"] = default
    if kind is not None:
        declaration["kind"] = kind

    return declaration


def axes_of(layout: Any) -> List[str]:
    """Return a layout's axis ids as a list.

    Args:
        layout: An ``EntryLayout``.

    Returns:
        Axis ids, outermost first.
    """
    return list(layout.axis_ids)


def origin_of(plan: Any, source: Any) -> Any:
    """Follow child input boundaries from ``source`` to the origin they read.

    Args:
        plan: A ``CompiledWorkflow``.
        source: A binding or output source.

    Returns:
        The ``InputPort``, ``StepPort`` or ``Constant`` holding the value.
    """
    while isinstance(source, ChildInputPort):
        source = plan.child_input(source).source

    return source
