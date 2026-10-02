"""Model-free V1 fixture plugin used by the reference scenarios.

The runner loads this module through the ordinary ``WORKFLOWS_PLUGINS``
discovery path, so the real V1 loader, compiler and executor handle these
blocks exactly as they would handle any third-party plugin. The blocks are
deliberately tiny: each one isolates a single engine feature, and the runner
records every ``run`` call made by the engine.
"""

import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, List, Literal, Optional, Type, Union

from pydantic import Field
from roboflow_workflows.execution_engine.entities.base import Batch, OutputDefinition
from roboflow_workflows.execution_engine.entities.types import (
    DICTIONARY_KIND,
    FLOAT_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    Selector,
)
from roboflow_workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

NUMBER_KINDS = [FLOAT_KIND, INTEGER_KIND]


class AuditLog:
    """Shared side-effect record injected into blocks as an init parameter.

    Args:
        origin: Who created the log: the plugin initializer or the caller.
    """

    created: List["AuditLog"] = []

    def __init__(self, origin: str):
        self.origin = origin
        self.events: List[Dict[str, Any]] = []
        AuditLog.created.append(self)


def _new_plugin_audit_log() -> AuditLog:
    # V1 calls a callable initializer once for every step that needs it.
    return AuditLog(origin="plugin_initializer")


REGISTERED_INITIALIZERS = {"audit": _new_plugin_audit_log}


class EchoManifest(WorkflowBlockManifest):
    type: Literal["reference/echo@v1"] = Field(description="Returns its value.")
    value: Union[Selector(), int, float, str] = Field(
        description="Literal or selected value to return unchanged."
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="value")]


class Echo(WorkflowBlock):
    """Return the received value as the ``value`` output."""

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return EchoManifest

    def run(self, value: Any) -> BlockResult:
        return {"value": value}


class SinkManifest(WorkflowBlockManifest):
    type: Literal["reference/sink@v1"] = Field(description="Output-free side effect.")
    payload: Optional[str] = Field(
        default="notice", description="Literal payload; explicit null is kept."
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return []


class Sink(WorkflowBlock):
    """Side-effect step without outputs; only the invocation log reveals it."""

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return SinkManifest

    def run(self, payload: Optional[str]) -> BlockResult:
        return {}


class RaggedExpandManifest(WorkflowBlockManifest):
    type: Literal["reference/ragged_expand@v1"] = Field(
        description="Creates value-many children."
    )
    value: Selector() = Field(description="Integer parent value and child count.")

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="value")]

    @classmethod
    def get_output_dimensionality_offset(cls) -> int:
        return 1


class RaggedExpand(WorkflowBlock):
    """Expand ``n`` into ``[n*10, ..., n*10 + n - 1]``; ``0`` gives no children."""

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return RaggedExpandManifest

    def run(self, value: int) -> BlockResult:
        return [{"value": value * 10 + index} for index in range(value)]


class OffsetExpandManifest(WorkflowBlockManifest):
    type: Literal["reference/offset_expand@v1"] = Field(
        description="Creates one child per offset."
    )
    value: Selector(kind=NUMBER_KINDS) = Field(description="Parent number.")
    offsets: Union[List[int], Selector(kind=[LIST_OF_VALUES_KIND])] = Field(
        default_factory=lambda: [1, 2], description="Offsets added to the parent."
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="child", kind=[FLOAT_KIND])]

    @classmethod
    def get_output_dimensionality_offset(cls) -> int:
        return 1


class OffsetExpand(WorkflowBlock):
    """Expand a number into ``value + offset`` children."""

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return OffsetExpandManifest

    def run(self, value: float, offsets: List[int]) -> BlockResult:
        return [{"child": value + offset} for offset in offsets]


class MergeManifest(WorkflowBlockManifest):
    type: Literal["reference/merge@v1"] = Field(description="Joins branch values.")
    values: List[Selector()] = Field(
        description="Branch outputs; missing branches arrive as null."
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="value")]

    @classmethod
    def accepts_empty_values(cls) -> bool:
        return True


class Merge(WorkflowBlock):
    """Return all branch values, keeping null slots for missing branches."""

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return MergeManifest

    def run(self, values: List[Any]) -> BlockResult:
        return {"value": values}


class ScaleManifest(WorkflowBlockManifest):
    type: Literal["reference/scale@v1"] = Field(description="Multiplies a number.")
    value: Union[float, int, Selector(kind=NUMBER_KINDS)] = Field(
        description="Number to scale."
    )
    factor: Union[float, Selector(kind=NUMBER_KINDS)] = Field(
        default=2.0, description="Multiplier; literal, default or selector."
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="scaled", kind=[FLOAT_KIND])]


class Scale(WorkflowBlock):
    """Scalar block: V1 calls it once per batch element."""

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return ScaleManifest

    def run(self, value: float, factor: float) -> BlockResult:
        return {"scaled": value * factor}


class BatchScaleManifest(ScaleManifest):
    type: Literal["reference/batch_scale@v1"] = Field(
        description="Multiplies a batch of numbers."
    )

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["value"]


class BatchScale(WorkflowBlock):
    """Batch-only ``value``: V1 passes a ``Batch`` and casts scalars into one."""

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BatchScaleManifest

    def run(self, value: Batch[float], factor: float) -> BlockResult:
        return [{"scaled": item * factor} for item in value]


class MixedScaleManifest(ScaleManifest):
    type: Literal["reference/mixed_scale@v1"] = Field(
        description="Multiplies a scalar or a batch."
    )

    @classmethod
    def get_parameters_accepting_batches_and_scalars(cls) -> List[str]:
        return ["value"]


class MixedScale(WorkflowBlock):
    """Scalar-or-batch ``value``: the call shape follows the bound selector."""

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return MixedScaleManifest

    def run(self, value: Union[Batch[float], float], factor: float) -> BlockResult:
        if isinstance(value, Batch):
            return [{"scaled": item * factor} for item in value]

        return {"scaled": value * factor}


class SumChildrenManifest(WorkflowBlockManifest):
    type: Literal["reference/sum_children@v1"] = Field(
        description="Adds each parent to its surviving children."
    )
    parent: Selector(kind=NUMBER_KINDS) = Field(description="Parent-level number.")
    children: Selector(kind=NUMBER_KINDS) = Field(
        description="Child-level numbers, one level deeper than the parent."
    )

    @classmethod
    def get_input_dimensionality_offsets(cls) -> Dict[str, int]:
        return {"children": 1}

    @classmethod
    def get_dimensionality_reference_property(cls) -> Optional[str]:
        return "parent"

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["parent", "children"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="total", kind=[FLOAT_KIND])]


class SumChildren(WorkflowBlock):
    """Mixed parent/child input: one result per parent group."""

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return SumChildrenManifest

    def run(self, parent: Batch[float], children: Batch[Batch[float]]) -> BlockResult:
        return [
            {"total": parent_value + sum(item for item in group if item is not None)}
            for parent_value, group in zip(parent, children)
        ]


class CompoundManifest(WorkflowBlockManifest):
    type: Literal["reference/compound@v1"] = Field(
        description="Echoes compound parameters."
    )
    params: Dict[str, Union[Selector(), float, int, str]] = Field(
        default_factory=dict, description="Mapping mixing literals and selectors."
    )
    items: List[Union[Selector(), float, int]] = Field(
        default_factory=list, description="List mixing literals and selectors."
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="echo")]


class Compound(WorkflowBlock):
    """Return the resolved compound parameters."""

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return CompoundManifest

    def run(self, params: Dict[str, Any], items: List[Any]) -> BlockResult:
        return {"echo": {"params": dict(params), "items": list(items)}}


class CounterManifest(WorkflowBlockManifest):
    type: Literal["reference/counter@v1"] = Field(
        description="Counts its own invocations."
    )
    value: Selector(kind=NUMBER_KINDS) = Field(
        description="Selector-only input; literals and parameters are rejected."
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="count", kind=[INTEGER_KIND])]


class Counter(WorkflowBlock):
    """Stateful block with an injected ``audit`` resource.

    Args:
        audit: Resource resolved by V1 from explicit init parameters or the
            plugin initializer.
    """

    def __init__(self, audit: AuditLog):
        self.audit = audit
        self.count = 0
        audit.events.append({"event": "construct"})

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["audit"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return CounterManifest

    def run(self, value: float) -> BlockResult:
        self.count += 1
        self.audit.events.append({"event": "run", "value": value, "count": self.count})

        return {"count": self.count}


class FailManifest(WorkflowBlockManifest):
    type: Literal["reference/fail@v1"] = Field(description="Always raises.")
    value: Selector() = Field(description="Value named in the error message.")

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="never")]


class Fail(WorkflowBlock):
    """Raise a ``RuntimeError`` naming the received value."""

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return FailManifest

    def run(self, value: Any) -> BlockResult:
        raise RuntimeError(f"reference failure on {value!r}")


class IncrementManifest(WorkflowBlockManifest):
    type: Literal["reference/increment@v1"] = Field(
        description="Mutates its input mapping."
    )
    value: Union[Selector(kind=[DICTIONARY_KIND]), Dict[str, Any]] = Field(
        description="Mapping whose ``count`` is incremented in place."
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="value", kind=[DICTIONARY_KIND])]


class Increment(WorkflowBlock):
    """Increment ``value["count"]`` in place and return the same object."""

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return IncrementManifest

    def run(self, value: Dict[str, Any]) -> BlockResult:
        value["count"] += 1

        return {"value": value}


class ReadCountManifest(WorkflowBlockManifest):
    type: Literal["reference/read_count@v1"] = Field(
        description="Reads a mapping's count."
    )
    value: Union[Selector(kind=[DICTIONARY_KIND]), Dict[str, Any]] = Field(
        description="Mapping to read."
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="seen", kind=[INTEGER_KIND])]


class ReadCount(WorkflowBlock):
    """Return ``value["count"]`` as observed when this step runs."""

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return ReadCountManifest

    def run(self, value: Dict[str, Any]) -> BlockResult:
        return {"seen": value["count"]}


class DeferredManifest(WorkflowBlockManifest):
    type: Literal["reference/deferred@v1"] = Field(
        description="Returns a future instead of a value."
    )
    value: Union[Selector(kind=NUMBER_KINDS), float, int] = Field(
        description="Number doubled on a worker thread."
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="doubled", kind=[FLOAT_KIND])]


def _double_later(value: float) -> float:
    # The delay makes the future genuinely pending when run() returns.
    time.sleep(0.05)

    return value * 2


class Deferred(WorkflowBlock):
    """Return a pending ``Future`` produced by a block-owned worker thread."""

    def __init__(self):
        self._pool = ThreadPoolExecutor(max_workers=1)

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return DeferredManifest

    def run(self, value: float) -> BlockResult:
        doubled: Future = self._pool.submit(_double_later, value)

        return {"doubled": doubled}


def load_blocks() -> List[Type[WorkflowBlock]]:
    """Return the fixture classes for V1 plugin discovery.

    Returns:
        Block classes registered under the ``v1_reference_fixtures`` source.
    """
    return [
        Echo,
        Sink,
        RaggedExpand,
        OffsetExpand,
        Merge,
        Scale,
        BatchScale,
        MixedScale,
        SumChildren,
        Compound,
        Counter,
        Fail,
        Increment,
        ReadCount,
        Deferred,
    ]
