"""``v2/window@v1``: collect successive arrivals into a time axis.

Each upstream pulse contributes one row: the value of every ``collect`` and
``hold`` input. A window of ``size`` rows emits one pulse; the operator then
slides by ``step`` rows (``step == size``: tumbling, smaller: overlapping)::

    rows      r0 r1 r2 r3 r4 r5            size=3, step=2
    windows   [r0 r1 r2]
                    [r2 r3 r4]             r2 is shared, payloads not copied
                          [r4 r5]          tail: emitted once if partial='emit'

Every collect field gains one trailing ``T`` axis after its own axes, so a
leaf at ``parent`` becomes ``(parent..., j)`` for window position ``j``. A
row where the field is filtered or missing keeps a filtered hole at ``j``;
positions are never compacted, so equal ``j`` means the same row in every
field. A hold field is the value of the window's last row (reference at
window close). A row whose collect fields are all filtered contributes
nothing and is skipped.

The collected axes are declared stationary: their producer guarantees that a
position identifies the same logical entity or region in every row. The
engine trusts that declaration and cannot prove region correspondence; it
only fails on contradictory evidence: a parent position whose known
``SampleContext`` source identity differs between rows of one window. Equal
counts or shapes are never taken as evidence.
"""

from collections import deque
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Deque,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from pydantic import Field, model_validator
from roboflow_workflows.execution_engine.v2.data import (
    AXIS_KIND_TIME,
    Axis,
    EntryMetadata,
    Index,
)
from roboflow_workflows.execution_engine.v2.errors import (
    OperatorError,
    OperatorInputError,
    WorkflowCompileError,
)
from roboflow_workflows.execution_engine.v2.execution.entries import (
    Entry,
    common_or_none,
)
from roboflow_workflows.execution_engine.v2.operators.contract import (
    Arrival,
    Operator,
    OperatorInput,
    OperatorParams,
    OperatorPort,
    OperatorPulse,
    TerminationReason,
    operator_step_path,
)

if TYPE_CHECKING:
    from roboflow_workflows.execution_engine.v2.data import SampleContext
    from roboflow_workflows.execution_engine.v2.plan import PulseKey


_UNSTABLE = (
    "axis {axis.id!r} ({axis.kind}) is not stationary. A window appends T per "
    "parent position, so the producer must guarantee that each position keeps "
    "identifying the same logical entity or region in every arrival (fixed "
    "configured crops are one example); the engine cannot verify that regions "
    "correspond. Collapse data-dependent or reordered positions to one value "
    "per parent before collecting"
)


class WindowParams(OperatorParams):
    """Parameters of ``v2/window@v1``."""

    size: int = Field(
        ge=1, strict=True, description="Rows in one full window.", examples=[3]
    )
    step: Optional[int] = Field(
        default=None,
        ge=1,
        strict=True,
        description="Rows the window slides after emitting; equals size when omitted.",
    )
    partial: Literal["drop", "emit"] = Field(
        default="drop",
        description="At termination, discard or emit rows not yet in a full window.",
    )

    @model_validator(mode="after")
    def _step_within_size(self) -> "WindowParams":
        if self.step is not None and self.step > self.size:
            raise ValueError(
                f"step {self.step} is larger than size {self.size}; rows would be "
                "skipped between windows"
            )

        return self


@dataclass(frozen=True, eq=False)
class _Row:
    """The values one upstream pulse contributed."""

    pulse: "PulseKey"
    values: Mapping[str, Entry]


class Window(Operator):
    """Collect ``size`` successive rows into a trailing time axis.

    Retains at most ``size`` rows. Emits the collect fields with an appended
    ``T`` axis and the hold fields of the last row. The new root has no
    temporal context and a sample context only when every row shares one;
    each collected position keeps its own contexts.
    """

    type = "v2/window@v1"
    Params = WindowParams
    input_roles = ("collect", "hold")

    @classmethod
    def plan_ports(
        cls, name: str, params: WindowParams, inputs: Sequence[OperatorInput]
    ) -> Mapping[str, OperatorPort]:
        """Append ``T`` to every collect layout; keep every hold layout.

        Args:
            name: Declared operator name.
            params: Validated parameters.
            inputs: ``collect`` and ``hold`` inputs.

        Returns:
            Port name to planned port, in input order.

        This is the one owner of window eligibility; the compiler adds the
        input's selector and the offending axis's origin to these errors.

        Raises:
            WorkflowCompileError: Without a collect input.
            OperatorInputError: For a collect axis that is already a time axis
                or is not stationary, or a hold layout that is not a parent
                prefix of every collect layout; ``field_path`` is
                ``(role, input)``.
        """
        location = f"$operators.{name} ({cls.type})"
        collected = [item for item in inputs if item.role == "collect"]
        if not collected:
            raise WorkflowCompileError(
                f"{location}: a window needs at least one collect input",
                step_path=operator_step_path(name),
            )

        def fail(item: OperatorInput, message: str, axis_id: Optional[str] = None):
            return OperatorInputError(
                f"{location} {item.role} {item.name!r}: {message}",
                step_path=operator_step_path(name),
                role=item.role,
                input=item.name,
                axis_id=axis_id,
            )

        time_axis = _time_axis(name)
        ports: Dict[str, OperatorPort] = {}
        for item in collected:
            for axis in item.layout.axes:
                if axis.kind == AXIS_KIND_TIME:
                    raise fail(
                        item,
                        f"axis {axis.id!r} is already a time axis; reduce it to one "
                        "value per parent before collecting again",
                        axis.id,
                    )
                if not axis.stationary:
                    raise fail(item, _UNSTABLE.format(axis=axis), axis.id)
            ports[item.name] = OperatorPort(
                *item.kinds, layout=item.layout.append_axis(time_axis)
            )

        for item in inputs:
            if item.role != "hold":
                continue
            for other in collected:
                if other.layout.axes[: item.layout.depth] != item.layout.axes:
                    raise fail(
                        item,
                        f"axes {list(item.layout.axis_ids)} are not a parent prefix "
                        f"of collect {other.name!r} axes {list(other.layout.axis_ids)}",
                    )
            ports[item.name] = OperatorPort(*item.kinds, layout=item.layout)

        return ports

    def __init__(
        self, *, name: str, params: WindowParams, inputs: Sequence[OperatorInput]
    ):
        super().__init__(name=name, params=params, inputs=inputs)
        self._collect = [item for item in self.inputs if item.role == "collect"]
        self._hold = [item.name for item in self.inputs if item.role == "hold"]
        self._step = params.step if params.step is not None else params.size
        self._time_axis = _time_axis(name)
        self._rows: Deque[_Row] = deque()
        self._fresh = 0

    def push(self, arrivals: Sequence[Arrival]) -> List[OperatorPulse]:
        """Add the row of one upstream pulse; emit when the window is full.

        Args:
            arrivals: One arrival per input, all of the same upstream pulse.

        Returns:
            The full window, when this row completed one.

        Raises:
            OperatorError: When a stationary parent position contradicts its
                known source identity within the window.
        """
        self.counters.filtered += sum(
            arrival.entry.is_effectively_filtered() for arrival in arrivals
        )
        values = {arrival.input: arrival.entry for arrival in arrivals}
        if all(values[item.name].is_effectively_filtered() for item in self._collect):
            self.counters.dropped += 1
            return []

        self._rows.append(_Row(pulse=arrivals[0].pulse, values=values))
        self._fresh += 1
        self.counters.peak_retained = max(self.counters.peak_retained, len(self._rows))
        if len(self._rows) < self.params.size:
            return []

        window = self._emit()
        for _ in range(self._step):
            self._rows.popleft()
        self._fresh = 0

        return [window]

    def finish(self, reason: TerminationReason) -> List[OperatorPulse]:
        """Apply the partial policy to rows not yet in a full window, then release.

        Rows that were already emitted in an overlapping window never form a
        tail on their own.

        Args:
            reason: ``eof`` or ``stop``.

        Returns:
            The partial window, at most once, when ``partial='emit'``.
        """
        emitted = []
        if self._fresh:
            if self.params.partial == "emit":
                emitted.append(self._emit())
            else:
                self.counters.partial_dropped += 1
        self._release()

        return emitted

    def close(self) -> None:
        """Release every retained row."""
        self._release()

    def _release(self) -> None:
        self._rows.clear()
        self._fresh = 0

    def _emit(self) -> OperatorPulse:
        rows = list(self._rows)
        ports = {item.name: self._collected(item, rows) for item in self._collect}
        for name in self._hold:
            ports[name] = rows[-1].values[name]
        causes = dict.fromkeys(row.pulse for row in rows)
        pulse = OperatorPulse(ports=ports, causes=tuple(causes))

        return pulse

    def _collected(self, item: OperatorInput, rows: Sequence[_Row]) -> Entry:
        """One field's rows as ``(parent..., j)``; holes stay filtered at ``j``."""
        entries = [row.values[item.name] for row in rows]
        depth = item.layout.depth
        known = _known_nodes(entries, depth=depth)
        self._check_identities(item.name, rows, known=known)

        children: Dict[Index, Tuple[Index, ...]] = {}
        filtered: Set[Index] = set()
        values: Dict[Index, object] = {}
        sample = {(): _common_sample(entries, ())}
        temporal = {(): None}
        for level in range(depth):
            for node in known[level]:
                if not any(_is_open(entry, node) for entry in entries):
                    filtered.add(node)
                    continue
                children[node] = tuple(
                    child for child in known[level + 1] if child[:-1] == node
                )
                if node:
                    sample[node] = _common_sample(entries, node)
        for parent in known[depth]:
            if parent:
                sample[parent] = _common_sample(entries, parent)
            children[parent] = tuple(parent + (j,) for j in range(len(entries)))
            for j, entry in enumerate(entries):
                leaf = parent + (j,)
                if not entry.has_value(parent):
                    filtered.add(leaf)
                    continue
                values[leaf] = entry.values[parent]
                sample[leaf] = entry.metadata.sample_at(parent)
                temporal[leaf] = entry.metadata.temporal_at(parent)

        collected = Entry(
            layout=item.layout.append_axis(self._time_axis),
            metadata=EntryMetadata(sample=sample, temporal=temporal),
            children=children,
            values=values,
            filtered=frozenset(filtered),
        )

        return collected

    def _check_identities(
        self, name: str, rows: Sequence[_Row], *, known: Sequence[Sequence[Index]]
    ) -> None:
        """Fail when a stationary parent position changes its source identity.

        Only rows that know the position with a sample context count; absence
        neither proves nor erases an identity.
        """
        for node in (node for nodes in known for node in nodes):
            first: Optional[_Row] = None
            for row in rows:
                identity = _identity(row.values[name], node)
                if identity is None:
                    continue
                if first is None:
                    first = row
                    continue
                expected = _identity(first.values[name], node)
                if identity != expected:
                    raise OperatorError(
                        f"collect {name!r} position {list(node)} is "
                        f"{_render(expected)} in pulse {_render_pulse(first.pulse)} "
                        f"but {_render(identity)} in pulse "
                        f"{_render_pulse(row.pulse)}; a stationary axis promises "
                        "the same parent at a position across arrivals",
                        operator=self.name,
                        input=name,
                    )


def _time_axis(operator_name: str) -> Axis:
    axis = Axis(f"operators.{operator_name}:t", AXIS_KIND_TIME)

    return axis


def _known_nodes(entries: Sequence[Entry], *, depth: int) -> List[List[Index]]:
    """Every node known in some row, per level, in logical order."""
    known = []
    for level in range(depth + 1):
        nodes: Set[Index] = set()
        for entry in entries:
            nodes.update(entry.nodes_at(level))
        known.append(sorted(nodes))

    return known


def _is_open(entry: Entry, node: Index) -> bool:
    """Whether ``entry`` knows the children of ``node`` (not filtered there)."""
    return node in entry.children and not entry.is_filtered(node)


def _knows(entry: Entry, node: Index) -> bool:
    """Whether ``node`` exists unfiltered in ``entry``; only then its context counts."""
    exists = not node or node in entry.children.get(node[:-1], ())

    return exists and not entry.is_filtered(node)


def _identity(entry: Entry, node: Index) -> Optional[Tuple[str, str]]:
    """``(source_id, source_type)`` known at ``node``, or ``None``."""
    context = entry.metadata.sample_at(node) if _knows(entry, node) else None
    if context is None:
        return None

    return context.source_id, context.source_type


def _common_sample(entries: Sequence[Entry], node: Index) -> Optional["SampleContext"]:
    contexts = [
        entry.metadata.sample_at(node) for entry in entries if _knows(entry, node)
    ]
    common = common_or_none(contexts)

    return common


def _render(identity: Tuple[str, str]) -> str:
    return f"{identity[0]!r} ({identity[1]})"


def _render_pulse(pulse: "PulseKey") -> str:
    return f"{pulse.source}#{pulse.sequence}"
