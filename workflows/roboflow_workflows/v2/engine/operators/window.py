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

History per position (``align='position'``). Rows are kept per *aligned
position*: a node at the deepest axis prefix shared by every collect input,
for example one camera of a sparse stream set. A position absent from a pulse
adds no row and keeps its rows; a pulse emits only the positions it brought
whose window is ready. Each position slides by one row::

    pulses    p0      p1      p2      p3            size=2
    cam0      a0      a1      -       a3
    cam1      b0      -       -       b3
    emitted   -       cam0    -       cam0 [a1 a3]
                      [a0 a1]         cam1 [b0 b3]

Internally both alignments use the same track: the rows retained for one
aligned position. ``align='pulse'`` has exactly one track, at the root.

``fill`` decides what a track emits before its first full window (``grow``:
the rows so far; ``pad``: ``size`` positions with filtered holes in place of
the missing rows; holes carry no payload and no context). ``order`` puts the
oldest or the newest row at ``j == 0``. ``clear_when`` names a hold input; a
``True`` there discards the rows of every track under that hold node before
the pulse's own row is added, so a root-level hold clears every track.

Under ``align='position'`` a position's rows outlive any one window, so its
known source identity does too: a contradiction fails at any later pulse,
across slides and clears. The window keeps payloads by reference; retaining
borrowed images needs an owned copy upstream.
"""

import json
from collections import deque
from dataclasses import dataclass, field
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
from roboflow_workflows.v2.data.batch import (
    AXIS_KIND_SAMPLE,
    AXIS_KIND_TIME,
    Axis,
    EntryLayout,
    EntryMetadata,
    Index,
)
from roboflow_workflows.v2.engine.errors import (
    OperatorError,
    OperatorInputError,
    WorkflowCompileError,
)
from roboflow_workflows.v2.engine.execution.entries import (
    Entry,
    common_or_none,
)
from roboflow_workflows.v2.engine.operators.contract import (
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
    from roboflow_workflows.v2.data.batch import SampleContext
    from roboflow_workflows.v2.engine.plan import PulseKey


_Identity = Tuple[str, str]  # (source_id, source_type)

_UNSTABLE = (
    "axis {axis.id!r} ({axis.kind}) is not stationary. A window appends T per "
    "parent position, so the producer must guarantee that each position keeps "
    "identifying the same logical entity or region in every arrival (fixed "
    "configured crops are one example); the engine cannot verify that regions "
    "correspond. Collapse data-dependent or reordered positions to one value "
    "per parent before collecting"
)
_UNSTABLE_INPUT = (
    "axis {axis.id!r} ({axis.kind}) is not stationary; it comes from a workflow "
    "input. A window appends T per position, so position i must be the same "
    "camera or stream in every call; typed inputs such as WorkflowBatchInput make "
    "no such promise. If your callers keep positions stable, declare each batch "
    'input without "type", with the same stationary axis, e.g. {example}'
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
        description=(
            "Rows the window slides after emitting. When omitted: size under "
            "align='pulse', 1 under align='position' (the only value it accepts)."
        ),
    )
    partial: Literal["drop", "emit"] = Field(
        default="drop",
        description=(
            "At termination, discard or emit rows not yet in a full window. "
            "A plan without sources (passive history) requires 'drop'."
        ),
    )
    align: Literal["pulse", "position"] = Field(
        default="pulse",
        description=(
            "'pulse': one window over whole upstream pulses. 'position': one "
            "window per aligned parent position (for example per camera); a "
            "position absent from a pulse keeps its rows."
        ),
    )
    fill: Literal["wait", "grow", "pad"] = Field(
        default="wait",
        description=(
            "Before the first full window: 'wait' emits nothing, 'grow' emits the "
            "rows so far, 'pad' emits size positions with filtered holes for the "
            "missing rows."
        ),
    )
    order: Literal["oldest_first", "newest_first"] = Field(
        default="oldest_first",
        description="Whether time position 0 holds the oldest or the newest row.",
    )
    clear_when: Optional[str] = Field(
        default=None,
        description=(
            "Name of a hold input. A True value there discards the retained rows "
            "under that hold position before the pulse's own row is added."
        ),
        examples=["reset"],
    )

    @model_validator(mode="after")
    def _step_fits(self) -> "WindowParams":
        if self.step is not None and self.step > self.size:
            raise ValueError(
                f"step {self.step} is larger than size {self.size}; rows would be "
                "skipped between windows"
            )
        if self.align == "position" and self.step not in (None, 1):
            raise ValueError(
                f"step {self.step} is not supported with align='position'; each "
                "position slides by one row, so omit step or set it to 1"
            )

        return self


@dataclass(frozen=True, eq=False)
class _Row:
    """The collect values one upstream pulse contributed to one track."""

    pulse: "PulseKey"
    values: Mapping[str, Entry]


@dataclass(eq=False)
class _Track:
    """Rows retained for one aligned position.

    Args:
        rows: Retained rows, oldest first; at most ``size``.
        fresh: Retained rows not yet emitted in any window.
        filled: Whether a full window was emitted since the track started or
            was last cleared; ``grow`` and ``pad`` only apply before that.
    """

    rows: Deque[_Row] = field(default_factory=deque)
    fresh: int = 0
    filled: bool = False


class Window(Operator):
    """Collect ``size`` successive rows into a trailing time axis.

    Retains at most ``size`` rows per track (one track under
    ``align='pulse'``, one per aligned position under ``align='position'``).
    Emits the collect fields with an appended ``T`` axis and the hold fields
    of the latest contributing pulse. The new root has no temporal context and
    a sample context only when every row shares one; each collected position
    keeps its own contexts.
    """

    type = "v2/window@v1"
    passive_max_emissions = 1  # push emits at most one window, in every align
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
            WorkflowCompileError: Without a collect input, or when
                ``clear_when`` does not name a hold input that the alignment
                can clear.
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
                    raise fail(item, _unstable_message(item, axis=axis), axis.id)
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

        _check_clear_when(params, inputs, location=location, name=name)

        return ports

    def __init__(
        self, *, name: str, params: WindowParams, inputs: Sequence[OperatorInput]
    ):
        super().__init__(name=name, params=params, inputs=inputs)
        self._collect = [item for item in self.inputs if item.role == "collect"]
        self._hold = [item.name for item in self.inputs if item.role == "hold"]
        self._by_position = params.align == "position"
        self._aligned_depth = (
            _shared_prefix_depth([item.layout for item in self._collect])
            if self._by_position
            else 0
        )
        if params.step is not None:
            self._step = params.step
        else:
            self._step = 1 if self._by_position else params.size
        self._time_axis = _time_axis(name)
        self._tracks: Dict[Index, _Track] = {}
        self._hold_values: Mapping[str, Entry] = {}
        self._identities: Dict[Tuple[str, Index], Tuple[_Identity, "PulseKey"]] = {}

    def push(self, arrivals: Sequence[Arrival]) -> List[OperatorPulse]:
        """Add the rows of one upstream pulse; emit the tracks it made ready.

        Args:
            arrivals: One arrival per input, all of the same upstream pulse.

        Returns:
            At most one pulse, holding only the positions this pulse brought
            whose window is ready.

        Raises:
            OperatorError: When a stationary parent position contradicts its
                known source identity.
        """
        self.counters.filtered += sum(
            arrival.entry.is_effectively_filtered() for arrival in arrivals
        )
        values = {arrival.input: arrival.entry for arrival in arrivals}
        self._clear(values)
        rows = self._rows_by_position(values, pulse=arrivals[0].pulse)
        if not rows:
            self.counters.dropped += 1
            return []

        if self._by_position:
            for row in rows.values():
                self._remember_identities(row)
        ready = []
        for position, row in rows.items():
            track = self._tracks.setdefault(position, _Track())
            track.rows.append(row)
            track.fresh += 1
            if self._is_ready(track):
                ready.append(position)
        self._hold_values = {name: values[name] for name in self._hold}
        retained = sum(len(track.rows) for track in self._tracks.values())
        self.counters.peak_retained = max(self.counters.peak_retained, retained)
        if not ready:
            return []

        window = self._emit(ready)
        for position in ready:
            self._advance(self._tracks[position])

        return [window]

    def finish(self, reason: TerminationReason) -> List[OperatorPulse]:
        """Apply the partial policy to rows not yet emitted, then release.

        Rows that were already emitted in an overlapping or growing window
        never form a tail on their own. Each position contributes to at most
        one tail.

        Args:
            reason: ``eof`` or ``stop``.

        Returns:
            One partial window over every position holding unemitted rows,
            when ``partial='emit'``.
        """
        tails = [
            position
            for position in sorted(self._tracks)
            if self._tracks[position].fresh
        ]
        emitted = []
        if tails and self.params.partial == "emit":
            emitted.append(self._emit(tails))
        else:
            self.counters.partial_dropped += len(tails)
        self._release()

        return emitted

    def close(self) -> None:
        """Release every retained row."""
        self._release()

    def _release(self) -> None:
        self._tracks.clear()
        self._hold_values = {}
        self._identities.clear()

    def _clear(self, values: Mapping[str, Entry]) -> None:
        """Discard the tracks under every ``True`` value of ``clear_when``."""
        if self.params.clear_when is None:
            return

        entry = values[self.params.clear_when]
        cleared = [
            node
            for node, value in entry.values.items()
            if value is True and entry.has_value(node)
        ]
        for position in list(self._tracks):
            if any(position[: len(node)] == node for node in cleared):
                self.counters.dropped += self._tracks.pop(position).fresh

    def _rows_by_position(
        self, values: Mapping[str, Entry], *, pulse: "PulseKey"
    ) -> Dict[Index, _Row]:
        """One row per aligned position where some collect field is present."""
        entries = [values[item.name] for item in self._collect]
        positions = sorted(
            {node for entry in entries for node in entry.nodes_at(self._aligned_depth)}
        )
        rows = {}
        for position in positions:
            if all(entry.is_effectively_filtered(position) for entry in entries):
                continue
            rows[position] = _Row(
                pulse=pulse,
                values={
                    item.name: _restricted(values[item.name], position)
                    for item in self._collect
                },
            )

        return rows

    def _is_ready(self, track: _Track) -> bool:
        if len(track.rows) == self.params.size:
            return True

        early = self.params.fill != "wait" and not track.filled

        return early

    def _advance(self, track: _Track) -> None:
        """Mark an emitted track's rows as emitted; slide a full window."""
        track.fresh = 0
        if len(track.rows) < self.params.size:
            return

        track.filled = True
        for _ in range(self._step):
            track.rows.popleft()

    def _emit(self, positions: Sequence[Index]) -> OperatorPulse:
        windows = [self._slots(self._tracks[position]) for position in positions]
        ports = {
            item.name: self._collected(item, windows, positions=positions)
            for item in self._collect
        }
        ports.update(self._hold_values)
        rows = [row for slots in windows for row in slots if row is not None]
        causes = sorted(
            dict.fromkeys(row.pulse for row in rows), key=lambda pulse: pulse.sequence
        )
        pulse = OperatorPulse(ports=ports, causes=tuple(causes))

        return pulse

    def _slots(self, track: _Track) -> List[Optional[_Row]]:
        """A track's rows in ``T`` order; ``None`` is a padding hole."""
        slots: List[Optional[_Row]] = list(track.rows)
        if self.params.fill == "pad":
            slots = [None] * (self.params.size - len(slots)) + slots
        if self.params.order == "newest_first":
            slots.reverse()

        return slots

    def _collected(
        self,
        item: OperatorInput,
        windows: Sequence[Sequence[Optional[_Row]]],
        *,
        positions: Sequence[Index],
    ) -> Entry:
        """One field's windows joined into one entry."""
        parts = [self._collected_window(item, slots) for slots in windows]
        if len(parts) == 1:
            return parts[0]

        joined = _joined(parts, positions=positions)

        return joined

    def _collected_window(
        self, item: OperatorInput, slots: Sequence[Optional[_Row]]
    ) -> Entry:
        """One track's rows as ``(parent..., j)``; holes stay filtered at ``j``."""
        entries = [None if row is None else row.values[item.name] for row in slots]
        present = [entry for entry in entries if entry is not None]
        depth = item.layout.depth
        known = _known_nodes(present, depth=depth)
        if not self._by_position:
            rows = [row for row in slots if row is not None]
            self._check_identities(item.name, rows, known=known)

        children: Dict[Index, Tuple[Index, ...]] = {}
        filtered: Set[Index] = set()
        values: Dict[Index, object] = {}
        sample = {(): _common_sample(present, ())}
        temporal = {(): None}
        for level in range(depth):
            for node in known[level]:
                if not any(_is_open(entry, node) for entry in present):
                    filtered.add(node)
                    continue
                children[node] = tuple(
                    child for child in known[level + 1] if child[:-1] == node
                )
                if node:
                    sample[node] = _common_sample(present, node)
        for parent in known[depth]:
            if parent:
                sample[parent] = _common_sample(present, parent)
            children[parent] = tuple(parent + (j,) for j in range(len(entries)))
            for j, entry in enumerate(entries):
                leaf = parent + (j,)
                if entry is None or not entry.has_value(parent):
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
            first: Optional[Tuple[_Identity, "PulseKey"]] = None
            for row in rows:
                identity = _identity(row.values[name], node)
                if identity is None:
                    continue
                if first is None:
                    first = (identity, row.pulse)
                    continue
                if identity != first[0]:
                    raise self._identity_error(
                        name, node, first=first, found=(identity, row.pulse)
                    )

    def _remember_identities(self, row: _Row) -> None:
        """Check a new row against every identity known for its positions.

        Under ``align='position'`` the first known identity of a node is kept
        until the run ends, so slides and clears never forget it.
        """
        for item in self._collect:
            entry = row.values[item.name]
            for level in range(item.layout.depth + 1):
                for node in entry.nodes_at(level):
                    identity = _identity(entry, node)
                    if identity is None:
                        continue
                    first = self._identities.setdefault(
                        (item.name, node), (identity, row.pulse)
                    )
                    if identity != first[0]:
                        raise self._identity_error(
                            item.name, node, first=first, found=(identity, row.pulse)
                        )

    def _identity_error(
        self,
        name: str,
        node: Index,
        *,
        first: Tuple[_Identity, "PulseKey"],
        found: Tuple[_Identity, "PulseKey"],
    ) -> OperatorError:
        error = OperatorError(
            f"collect {name!r} position {list(node)} is "
            f"{_render(first[0])} in pulse {_render_pulse(first[1])} "
            f"but {_render(found[0])} in pulse {_render_pulse(found[1])}; a "
            "stationary axis promises the same parent at a position across "
            "arrivals",
            operator=self.name,
            input=name,
        )

        return error


def _time_axis(operator_name: str) -> Axis:
    axis = Axis(f"operators.{operator_name}:t", AXIS_KIND_TIME)

    return axis


def _unstable_message(item: OperatorInput, *, axis: Axis) -> str:
    scope, _, _ = axis.id.partition(":")
    from_input = axis.kind == AXIS_KIND_SAMPLE and not scope.startswith(
        ("sources.", "operators.")
    )
    if not from_input:
        return _UNSTABLE.format(axis=axis)

    # Stationarity stays the caller's promise; show the declaration that makes it.
    # Typed inputs share the root axis "inputs"; the example names its own axis.
    declaration = {
        "name": "<input>",
        "kind": [kind.name for kind in item.kinds] or ["*"],
        "axes": [
            {
                "id": "camera" if axis.id == "inputs" else axis.id,
                "kind": AXIS_KIND_SAMPLE,
                "stationary": True,
            }
        ],
    }
    message = _UNSTABLE_INPUT.format(axis=axis, example=json.dumps(declaration))

    return message


def _check_clear_when(
    params: WindowParams,
    inputs: Sequence[OperatorInput],
    *,
    location: str,
    name: str,
) -> None:
    """``clear_when`` names a hold input; under ``align='pulse'`` an ungrouped one.

    A pulse-aligned window keeps whole pulses, so it cannot clear the rows of
    one position only.
    """
    if params.clear_when is None:
        return

    holds = {item.name: item for item in inputs if item.role == "hold"}
    hold = holds.get(params.clear_when)
    if hold is None:
        raise WorkflowCompileError(
            f"{location}: clear_when {params.clear_when!r} is not a hold input; "
            f"declared holds: {sorted(holds)}",
            step_path=operator_step_path(name),
        )
    if params.align == "pulse" and hold.layout.depth:
        raise WorkflowCompileError(
            f"{location}: clear_when {params.clear_when!r} has axes "
            f"{list(hold.layout.axis_ids)}; align='pulse' keeps whole pulses and "
            "can only be cleared by an ungrouped hold. Use align='position' to "
            "clear one position",
            step_path=operator_step_path(name),
        )


def _shared_prefix_depth(layouts: Sequence[EntryLayout]) -> int:
    """Number of leading axes every layout shares."""
    depth = 0
    for axes in zip(*(layout.axes for layout in layouts)):
        if any(axis != axes[0] for axis in axes):
            break
        depth += 1

    return depth


def _restricted(entry: Entry, position: Index) -> Entry:
    """``entry`` reduced to ``position``, its ancestors and its subtree.

    A track then retains only its own position's payloads.
    """
    if not position:
        return entry

    def kept(index: Index) -> bool:
        return index[: len(position)] == position or position[: len(index)] == index

    children = {
        node: tuple(child for child in nodes if kept(child))
        for node, nodes in entry.children.items()
        if kept(node)
    }
    metadata = EntryMetadata(
        sample={i: c for i, c in entry.metadata.sample.items() if kept(i)},
        temporal={i: c for i, c in entry.metadata.temporal.items() if kept(i)},
    )
    restricted = Entry(
        layout=entry.layout,
        metadata=metadata,
        children=children,
        values={index: value for index, value in entry.values.items() if kept(index)},
        filtered=frozenset(index for index in entry.filtered if kept(index)),
    )

    return restricted


def _joined(parts: Sequence[Entry], *, positions: Sequence[Index]) -> Entry:
    """Join disjoint tracks; an ancestor filter applies only to its track."""
    children: Dict[Index, Set[Index]] = {}
    contexts: Dict[Index, List[Optional["SampleContext"]]] = {}
    values: Dict[Index, object] = {}
    filtered: Set[Index] = set()
    temporal: Dict[Index, object] = {}
    for position, part in zip(positions, parts):
        for node, nodes in part.children.items():
            children.setdefault(node, set()).update(nodes)
        for node, context in part.metadata.sample.items():
            contexts.setdefault(node, []).append(context)
        values.update(part.values)
        for node in part.filtered:
            if len(node) < len(position) and position[: len(node)] == node:
                # This field has no known subtree for this track. Keep it
                # filtered at the track, without hiding a sibling's history.
                node = position
                for level in range(len(position)):
                    children.setdefault(position[:level], set()).add(
                        position[: level + 1]
                    )
            filtered.add(node)
        temporal.update(part.metadata.temporal)
    sample = {node: common_or_none(found) for node, found in contexts.items()}

    joined = Entry(
        layout=parts[0].layout,
        metadata=EntryMetadata(sample=sample, temporal=temporal),
        children={node: tuple(sorted(nodes)) for node, nodes in children.items()},
        values=values,
        filtered=frozenset(filtered),
    )

    return joined


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


def _identity(entry: Entry, node: Index) -> Optional[_Identity]:
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


def _render(identity: _Identity) -> str:
    return f"{identity[0]!r} ({identity[1]})"


def _render_pulse(pulse: "PulseKey") -> str:
    return f"{pulse.source}#{pulse.sequence}"
