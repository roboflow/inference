"""Generic data API of the V2 execution engine.

This module defines the small, payload-agnostic data model that flows through
the V2 engine:

* ``Batch``: an immutable-membership grouping container with logical indices.
  Payload objects inside a batch stay plugin-owned and may be mutable.
* ``Axis`` / ``EntryLayout``: the ordered logical axes of one named entry.
* ``SampleContext`` / ``TemporalContext`` / ``EntryMetadata``: indexed source
  and temporal context with longest-prefix inheritance.
* ``InputValue``: what a caller hands to the engine at the workflow boundary.
* ``WorkflowsBuffer``: the engine-owned carrier of named entries.
* ``validate_entry``: the boundary check for one entry's grouping tree.

Nothing in this module knows about images or any other payload type. Plain
lists, dictionaries and ``None`` are ordinary payloads; only ``Batch`` denotes
an engine grouping axis. The engine owns traversal and regrouping; this module
only validates.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from types import MappingProxyType
from typing import (
    Any,
    Generic,
    Iterable,
    Iterator,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

from roboflow_workflows.execution_engine.v2.errors import ContractError

Index = Tuple[int, ...]
"""Logical index path. ``()`` addresses a whole entry."""

AXIS_KIND_SAMPLE = "sample"
AXIS_KIND_STATIC_NESTING = "static_nesting"
AXIS_KIND_DYNAMIC_NESTING = "dynamic_nesting"
AXIS_KIND_TIME = "time"
AXIS_KINDS: Tuple[str, ...] = (
    AXIS_KIND_SAMPLE,
    AXIS_KIND_STATIC_NESTING,
    AXIS_KIND_DYNAMIC_NESTING,
    AXIS_KIND_TIME,
)

SOURCE_TYPE_STATIC = "static"

T = TypeVar("T")


def _validate_index(value: Any, *, what: str) -> Index:
    if not isinstance(value, tuple):
        raise ContractError(
            f"{what} must be a tuple of non-negative integers, "
            f"got {type(value).__name__}: {value!r}"
        )

    for component in value:
        if isinstance(component, bool) or not isinstance(component, int):
            raise ContractError(f"{what} must contain only integers, got {value!r}")
        if component < 0:
            raise ContractError(
                f"{what} must contain only non-negative integers, got {value!r}"
            )

    return value


def _validate_non_empty_string(value: Any, *, what: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{what} must be a non-empty string, got {value!r}")

    return value


def _freeze_metadata_value(value: Any) -> Any:
    """Recursively freeze mapping/list/set containers without copying leaves."""
    if isinstance(value, Mapping):
        frozen_items = {
            key: _freeze_metadata_value(item) for key, item in value.items()
        }
        return MappingProxyType(frozen_items)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_metadata_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(value)

    return value


@dataclass(frozen=True)
class Axis:
    """One logical grouping axis of an entry.

    Identity follows lineage, not matching dimension sizes. Two axes with equal
    IDs describe corresponding positions; two axes with equal lengths do not.

    Args:
        id: Lineage identity of the axis; unique within a layout.
        kind: One of ``sample``, ``static_nesting``, ``dynamic_nesting``, ``time``.
        stationary: Whether logical positions identify the same samples across
            successive arrivals. ``static_nesting`` implies ``True``; sample
            stationarity is declared explicitly by the caller.

    Raises:
        ContractError: On an empty ID, unknown kind, or a ``dynamic_nesting``
            axis declared stationary.
    """

    id: str
    kind: str
    stationary: bool = False

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.id, what="Axis id")
        if self.kind not in AXIS_KINDS:
            raise ContractError(
                f"Axis '{self.id}' has unknown kind {self.kind!r}; "
                f"expected one of {list(AXIS_KINDS)}"
            )
        if not isinstance(self.stationary, bool):
            raise ContractError(
                f"Axis '{self.id}' stationary flag must be a bool, "
                f"got {self.stationary!r}"
            )
        if self.kind == AXIS_KIND_STATIC_NESTING and not self.stationary:
            object.__setattr__(self, "stationary", True)
        if self.kind == AXIS_KIND_DYNAMIC_NESTING and self.stationary:
            raise ContractError(
                f"Axis '{self.id}' is dynamic_nesting and cannot be stationary; "
                "declare static_nesting for stable child identities"
            )


@dataclass(frozen=True)
class EntryLayout:
    """Ordered logical axes of one named entry, outermost first.

    Invariants: unique axis IDs, a ``sample`` axis only in first position, at
    most one ``time`` axis, and every axis before ``time`` stationary. An
    ungrouped entry has no axes.

    Args:
        axes: Axes from outermost to innermost. A list is accepted and stored
            as a tuple.

    Raises:
        ContractError: When an invariant is violated.
    """

    axes: Tuple[Axis, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.axes, Axis) or isinstance(self.axes, str):
            raise ContractError(
                "EntryLayout axes must be a sequence of Axis objects, "
                f"got {self.axes!r}"
            )

        axes = tuple(self.axes)
        object.__setattr__(self, "axes", axes)

        seen_ids = set()
        time_positions = []
        for position, axis in enumerate(axes):
            if not isinstance(axis, Axis):
                raise ContractError(
                    f"EntryLayout axis at position {position} must be an Axis, "
                    f"got {type(axis).__name__}"
                )
            if axis.id in seen_ids:
                raise ContractError(f"EntryLayout has duplicate axis id '{axis.id}'")
            seen_ids.add(axis.id)
            if axis.kind == AXIS_KIND_SAMPLE and position != 0:
                raise ContractError(
                    f"Sample axis '{axis.id}' must be the first axis, "
                    f"found at position {position}"
                )
            if axis.kind == AXIS_KIND_TIME:
                time_positions.append(position)

        if len(time_positions) > 1:
            raise ContractError(
                "EntryLayout may contain at most one time axis, found "
                f"{len(time_positions)}: "
                f"{[axes[position].id for position in time_positions]}"
            )

        if time_positions:
            time_position = time_positions[0]
            for axis in axes[:time_position]:
                if not axis.stationary:
                    raise ContractError(
                        f"Axis '{axis.id}' precedes time axis "
                        f"'{axes[time_position].id}' and must be stationary"
                    )

    @property
    def depth(self) -> int:
        """Number of grouping axes."""
        return len(self.axes)

    @property
    def axis_ids(self) -> Tuple[str, ...]:
        """Axis IDs from outermost to innermost."""
        return tuple(axis.id for axis in self.axes)

    @property
    def has_time(self) -> bool:
        """Whether a time axis is present."""
        return any(axis.kind == AXIS_KIND_TIME for axis in self.axes)

    @property
    def last_axis(self) -> Axis:
        """The innermost axis.

        Raises:
            ContractError: When the layout is ungrouped.
        """
        if not self.axes:
            raise ContractError("Ungrouped EntryLayout has no last axis")

        return self.axes[-1]

    def append_axis(self, axis: Axis) -> "EntryLayout":
        """Return a new layout with ``axis`` appended as the innermost axis.

        Args:
            axis: Axis to append.

        Returns:
            The extended layout; all invariants are re-validated.
        """
        extended_layout = EntryLayout(axes=self.axes + (axis,))

        return extended_layout

    def remove_last_axis(self) -> "EntryLayout":
        """Return a new layout without the innermost axis.

        Returns:
            The invocation prefix layout.

        Raises:
            ContractError: When the layout is ungrouped.
        """
        if not self.axes:
            raise ContractError("Cannot remove an axis from an ungrouped EntryLayout")

        prefix_layout = EntryLayout(axes=self.axes[:-1])

        return prefix_layout


@dataclass(frozen=True)
class Timestamp:
    """A point on a clock.

    Args:
        ticks: Integer tick count.
        time_base: Seconds per tick; an ``int`` is converted to ``Fraction``.
        clock_id: Identity of the clock the ticks refer to.

    Raises:
        ContractError: On non-integer ticks, non-positive time base or an
            empty clock ID.
    """

    ticks: int
    time_base: Fraction
    clock_id: str

    def __post_init__(self) -> None:
        if isinstance(self.ticks, bool) or not isinstance(self.ticks, int):
            raise ContractError(f"Timestamp ticks must be an int, got {self.ticks!r}")
        if isinstance(self.time_base, int) and not isinstance(self.time_base, bool):
            object.__setattr__(self, "time_base", Fraction(self.time_base))
        if not isinstance(self.time_base, Fraction):
            raise ContractError(
                f"Timestamp time_base must be a Fraction, got {self.time_base!r}"
            )
        if self.time_base <= 0:
            raise ContractError(
                f"Timestamp time_base must be positive, got {self.time_base}"
            )
        _validate_non_empty_string(self.clock_id, what="Timestamp clock_id")

    @property
    def seconds(self) -> Fraction:
        """Exact position in seconds."""
        return self.ticks * self.time_base


@dataclass(frozen=True)
class TimeSpan:
    """Half-open interval ``[start, end)`` on one clock.

    Args:
        start: Interval start.
        end: Interval end; must be later than ``start`` on the same clock.

    Raises:
        ContractError: When the endpoints are not timestamps, use different
            clocks, or ``end`` is not after ``start``.
    """

    start: Timestamp
    end: Timestamp

    def __post_init__(self) -> None:
        if not isinstance(self.start, Timestamp) or not isinstance(self.end, Timestamp):
            raise ContractError("TimeSpan endpoints must be Timestamp objects")
        if self.start.clock_id != self.end.clock_id:
            raise ContractError(
                "TimeSpan endpoints must share a clock, got "
                f"'{self.start.clock_id}' and '{self.end.clock_id}'"
            )
        if self.end.seconds <= self.start.seconds:
            raise ContractError(
                "TimeSpan end must be after start, got "
                f"{self.start.seconds}s .. {self.end.seconds}s"
            )


TimeCoverage = Union[Timestamp, TimeSpan]


@dataclass(frozen=True)
class TemporalContext:
    """Temporal description of the items an index path represents.

    Args:
        observed_coverage: When the represented inputs were observed by
            Workflows.
        media_coverage: Position or interval on the media timeline, if any.
        capture_coverage: Physical capture time or interval, if any.

    Raises:
        ContractError: When a coverage is not a ``Timestamp`` or ``TimeSpan``.
    """

    observed_coverage: TimeCoverage
    media_coverage: Optional[TimeCoverage] = None
    capture_coverage: Optional[TimeCoverage] = None

    def __post_init__(self) -> None:
        for name, coverage in (
            ("observed_coverage", self.observed_coverage),
            ("media_coverage", self.media_coverage),
            ("capture_coverage", self.capture_coverage),
        ):
            if coverage is None and name != "observed_coverage":
                continue
            if not isinstance(coverage, (Timestamp, TimeSpan)):
                raise ContractError(
                    f"TemporalContext.{name} must be a Timestamp or TimeSpan, "
                    f"got {type(coverage).__name__}"
                )


@dataclass(frozen=True)
class SampleContext:
    """Source description of the items an index path represents.

    ``source_metadata`` is snapshotted into a read-only mapping. Nested
    dictionaries, lists and sets are frozen recursively; other leaf objects are
    referenced, not copied.

    Args:
        source_id: Identity of the source.
        source_type: Source category; ``static`` by default.
        source_metadata: Source-specific metadata mapping.

    Raises:
        ContractError: On an empty source ID/type or a non-mapping metadata.
    """

    source_id: str
    source_type: str = SOURCE_TYPE_STATIC
    source_metadata: Mapping[str, Any] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.source_id, what="SampleContext source_id")
        _validate_non_empty_string(self.source_type, what="SampleContext source_type")
        if not isinstance(self.source_metadata, Mapping):
            raise ContractError(
                "SampleContext source_metadata must be a mapping, "
                f"got {type(self.source_metadata).__name__}"
            )
        frozen_metadata = _freeze_metadata_value(dict(self.source_metadata))
        object.__setattr__(self, "source_metadata", frozen_metadata)


def _snapshot_context_map(
    value: Any,
    *,
    what: str,
    context_type: type,
) -> Mapping[Index, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{what} must be a mapping, got {type(value).__name__}")

    snapshot = {}
    for key, context in value.items():
        index = _validate_index(key, what=f"{what} key")
        if context is not None and not isinstance(context, context_type):
            raise ContractError(
                f"{what}[{index!r}] must be a {context_type.__name__} or None, "
                f"got {type(context).__name__}"
            )
        snapshot[index] = context

    frozen_snapshot = MappingProxyType(snapshot)

    return frozen_snapshot


def _lookup_longest_prefix(mapping: Mapping[Index, Any], index: Index) -> Any:
    for depth in range(len(index), -1, -1):
        prefix = index[:depth]
        if prefix in mapping:
            return mapping[prefix]

    return None


@dataclass(frozen=True)
class EntryMetadata:
    """Indexed source and temporal context of one entry.

    Both maps are keyed by logical index paths; ``()`` addresses the whole
    entry. Lookup uses the longest defined prefix. A missing key inherits from
    its ancestors; a key explicitly mapped to ``None`` stops inheritance. A
    more specific temporal context replaces the inherited one as a whole.

    Supplied mappings are snapshotted and exposed read-only, so later edits to
    the caller's dictionaries cannot change attached metadata.

    Args:
        sample: Source context by index path.
        temporal: Temporal context by index path.

    Raises:
        ContractError: On malformed keys or values of the wrong type.
    """

    sample: Mapping[Index, Optional[SampleContext]] = field(
        default_factory=dict, hash=False
    )
    temporal: Mapping[Index, Optional[TemporalContext]] = field(
        default_factory=dict, hash=False
    )

    def __post_init__(self) -> None:
        sample_snapshot = _snapshot_context_map(
            self.sample,
            what="EntryMetadata.sample",
            context_type=SampleContext,
        )
        temporal_snapshot = _snapshot_context_map(
            self.temporal,
            what="EntryMetadata.temporal",
            context_type=TemporalContext,
        )
        object.__setattr__(self, "sample", sample_snapshot)
        object.__setattr__(self, "temporal", temporal_snapshot)

    @property
    def is_empty(self) -> bool:
        """Whether neither map has any entry."""
        return not self.sample and not self.temporal

    @property
    def max_depth(self) -> int:
        """Length of the longest index path used by either map."""
        depths = [len(index) for index in self.sample] + [
            len(index) for index in self.temporal
        ]
        max_depth = max(depths, default=0)

        return max_depth

    def sample_at(self, index: Index) -> Optional[SampleContext]:
        """Resolve the source context applicable at ``index``.

        Args:
            index: Full logical index path of an item or group.

        Returns:
            The context at the longest defined prefix, or ``None`` when no
            prefix is defined or the nearest prefix is explicitly ``None``.
        """
        validated_index = _validate_index(index, what="Lookup index")
        resolved_context = _lookup_longest_prefix(self.sample, validated_index)

        return resolved_context

    def temporal_at(self, index: Index) -> Optional[TemporalContext]:
        """Resolve the temporal context applicable at ``index``.

        Args:
            index: Full logical index path of an item or group.

        Returns:
            The context at the longest defined prefix, or ``None`` when no
            prefix is defined or the nearest prefix is explicitly ``None``.
        """
        validated_index = _validate_index(index, what="Lookup index")
        resolved_context = _lookup_longest_prefix(self.temporal, validated_index)

        return resolved_context


class Batch(Generic[T]):
    """Immutable-membership grouping container with logical indices.

    A ``Batch`` denotes exactly one grouping axis. Elements may themselves be
    batches (nested grouping). Membership and indices cannot change after
    construction; the payload objects themselves stay plugin-owned.

    Indices are full logical paths extending ``parent_index``. A group has
    exactly one trailing component per index. The engine may also deliver a
    flat view over a deeper domain, e.g. one vectorized call over indices
    ``(0, 0), (0, 1), (1, 0)``; all indices of one batch then share one depth.
    Blocks constructing a batch directly get local one-component indices by
    default; engine-built batches carry full paths and an attached read-only
    ``layout``/``metadata`` view. A batch knows its ``parent_index`` even when
    it has no children, so an empty nested group still identifies its parent.

    Args:
        content: Payloads or nested batches. Strings, bytes and mappings are
            rejected because they are never implicit groups.
        indices: Full logical index per element, all of one depth greater
            than ``parent_index``. Defaults to ``parent_index + (position,)``.
        layout: Read-only layout view attached by the engine, if any.
        metadata: Read-only metadata view attached by the engine, if any.
        parent_index: Logical index of the group's parent; ``()`` at the root.

    Raises:
        ContractError: On a non-iterable or forbidden content type, index
            count mismatch, malformed, duplicate or non-prefixed indices.
    """

    __slots__ = ("_content", "_indices", "_layout", "_metadata", "_parent_index")

    def __init__(
        self,
        content: Iterable[T],
        *,
        indices: Optional[Sequence[Index]] = None,
        layout: Optional[EntryLayout] = None,
        metadata: Optional[EntryMetadata] = None,
        parent_index: Index = (),
    ) -> None:
        if content is None or isinstance(content, (str, bytes, bytearray, Mapping)):
            raise ContractError(
                "Batch content must be an iterable of payloads, got "
                f"{type(content).__name__}; plain values are never implicit groups"
            )
        try:
            stored_content = tuple(content)
        except TypeError as error:
            raise ContractError(
                f"Batch content must be iterable, got {type(content).__name__}"
            ) from error

        validated_parent = _validate_index(parent_index, what="Batch parent_index")
        if indices is None:
            stored_indices = tuple(
                validated_parent + (position,)
                for position in range(len(stored_content))
            )
        else:
            stored_indices = _validate_batch_indices(
                indices,
                expected_count=len(stored_content),
                parent_index=validated_parent,
            )

        if layout is not None and not isinstance(layout, EntryLayout):
            raise ContractError(
                f"Batch layout must be an EntryLayout, got {type(layout).__name__}"
            )
        if metadata is not None and not isinstance(metadata, EntryMetadata):
            raise ContractError(
                "Batch metadata must be an EntryMetadata, "
                f"got {type(metadata).__name__}"
            )

        self._content = stored_content
        self._indices = stored_indices
        self._layout = layout
        self._metadata = metadata
        self._parent_index = validated_parent

    @classmethod
    def of(
        cls,
        content: Iterable[T],
        *,
        indices: Optional[Sequence[Index]] = None,
    ) -> "Batch[T]":
        """Construct a root-level batch, the usual block return form.

        Args:
            content: Child payloads.
            indices: Optional explicit local indices, e.g. to keep configured
                positions when some children are omitted.

        Returns:
            A batch with local one-component indices.

        Raises:
            ContractError: When an explicit index has more than one component.
        """
        batch = cls(content, indices=indices)
        for index in batch.indices:
            if len(index) != 1:
                raise ContractError(
                    f"Batch.of index {index!r} must have 1 component; "
                    "local indices are one-component tuples such as (0,)"
                )

        return batch

    @classmethod
    def empty(cls, *, parent_index: Index = ()) -> "Batch[T]":
        """Construct an ordinary valid empty group.

        Args:
            parent_index: Logical index of the parent the empty group belongs to.

        Returns:
            A batch with no children that still knows its parent.
        """
        batch = cls((), parent_index=parent_index)

        return batch

    @property
    def content(self) -> Tuple[T, ...]:
        """Elements in positional order."""
        return self._content

    @property
    def indices(self) -> Tuple[Index, ...]:
        """Full logical index of each element in positional order."""
        return self._indices

    @property
    def layout(self) -> Optional[EntryLayout]:
        """Read-only layout view attached by the engine, if any."""
        return self._layout

    @property
    def metadata(self) -> Optional[EntryMetadata]:
        """Read-only metadata view attached by the engine, if any."""
        return self._metadata

    @property
    def parent_index(self) -> Index:
        """Logical index of the parent group; ``()`` at the root."""
        return self._parent_index

    def with_view(
        self,
        *,
        layout: Optional[EntryLayout],
        metadata: Optional[EntryMetadata],
    ) -> "Batch[T]":
        """Return a batch sharing this membership with another attached view.

        Args:
            layout: Layout view to attach.
            metadata: Metadata view to attach.

        Returns:
            ``self`` when the view already matches, otherwise a new batch that
            shares content and indices.
        """
        if layout is self._layout and metadata is self._metadata:
            return self

        viewed_batch = Batch(
            self._content,
            indices=self._indices,
            layout=layout,
            metadata=metadata,
            parent_index=self._parent_index,
        )

        return viewed_batch

    def iter_with_indices(self) -> Iterator[Tuple[Index, T]]:
        """Iterate ``(index, element)`` pairs in positional order.

        Returns:
            Iterator over index/element pairs.
        """
        return iter(zip(self._indices, self._content))

    def __len__(self) -> int:
        return len(self._content)

    def __iter__(self) -> Iterator[T]:
        return iter(self._content)

    def __getitem__(self, position: int) -> T:
        if isinstance(position, bool) or not isinstance(position, int):
            raise TypeError(
                f"Batch positions are integers, got {type(position).__name__}"
            )

        return self._content[position]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Batch):
            return NotImplemented

        return (
            self._parent_index == other._parent_index
            and self._indices == other._indices
            and self._content == other._content
        )

    __hash__ = None

    def __repr__(self) -> str:
        return (
            f"Batch(parent_index={self._parent_index!r}, "
            f"indices={self._indices!r}, content={self._content!r})"
        )


def _validate_batch_indices(
    indices: Any,
    *,
    expected_count: int,
    parent_index: Index,
) -> Tuple[Index, ...]:
    if isinstance(indices, (str, bytes)) or not isinstance(indices, Iterable):
        raise ContractError(
            f"Batch indices must be a sequence of tuples, got {type(indices).__name__}"
        )

    stored_indices = tuple(indices)
    if len(stored_indices) != expected_count:
        raise ContractError(
            f"Batch has {expected_count} elements but {len(stored_indices)} indices"
        )

    minimal_depth = len(parent_index) + 1
    shared_depth = None
    seen = set()
    for index in stored_indices:
        validated_index = _validate_index(index, what="Batch index")
        if len(validated_index) < minimal_depth:
            raise ContractError(
                f"Batch index {validated_index!r} must have at least {minimal_depth} "
                f"component(s) under parent {parent_index!r}"
            )
        if shared_depth is None:
            shared_depth = len(validated_index)
        if len(validated_index) != shared_depth:
            raise ContractError(
                f"Batch indices must share one depth, got {stored_indices[0]!r} "
                f"and {validated_index!r}"
            )
        if validated_index[: len(parent_index)] != parent_index:
            raise ContractError(
                f"Batch index {validated_index!r} does not extend parent index "
                f"{parent_index!r}"
            )
        if validated_index in seen:
            raise ContractError(f"Batch index {validated_index!r} is duplicated")
        seen.add(validated_index)

    return stored_indices


@dataclass(frozen=True)
class InputValue:
    """A workflow input supplied to the engine at the run boundary.

    Args:
        data: Payload or ``Batch`` of the declared input.
        metadata: Indexed context for the entry; empty by default.

    Raises:
        ContractError: When ``metadata`` is not an ``EntryMetadata``.
    """

    data: Any
    metadata: EntryMetadata = field(default_factory=EntryMetadata)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, EntryMetadata):
            raise ContractError(
                "InputValue metadata must be an EntryMetadata, "
                f"got {type(self.metadata).__name__}"
            )


def _collect_positions(
    data: Any,
    *,
    depth: int,
    path: Index,
    positions: set,
) -> None:
    """Validate the grouping tree under ``path`` and collect logical positions."""
    if depth == 0:
        if isinstance(data, Batch):
            raise ContractError(
                f"Value at logical index {path!r} is a Batch but the layout "
                "declares no further grouping axis"
            )
        return

    if not isinstance(data, Batch):
        raise ContractError(
            f"Value at logical index {path!r} must be a Batch (remaining depth "
            f"{depth}), got {type(data).__name__}; plain lists are payloads, "
            "not groups"
        )
    if data.parent_index != path:
        raise ContractError(
            f"Batch at logical index {path!r} reports parent_index "
            f"{data.parent_index!r}; nested groups must carry full logical paths"
        )

    for index, element in data.iter_with_indices():
        if len(index) != len(path) + 1:
            raise ContractError(
                f"Batch at logical index {path!r} holds index {index!r}; an entry "
                "group adds exactly one component per nesting level"
            )
        positions.add(index)
        _collect_positions(
            element,
            depth=depth - 1,
            path=index,
            positions=positions,
        )


def validate_entry(
    data: Any,
    *,
    layout: EntryLayout,
    metadata: EntryMetadata,
) -> None:
    """Validate one named entry at an engine boundary.

    Checks that the grouping depth of ``data`` equals the layout depth, that
    every nested batch carries full logical index paths consistent with its
    parent, and that every metadata index path addresses the whole entry or an
    existing group/item. Empty nested groups are valid.

    Args:
        data: Payload (ungrouped layout) or nested ``Batch`` tree.
        layout: Declared axes of the entry.
        metadata: Indexed context of the entry.

    Raises:
        ContractError: When the tree, its indices or the metadata paths do not
            match the layout.
    """
    if not isinstance(layout, EntryLayout):
        raise ContractError(
            f"Entry layout must be an EntryLayout, got {type(layout).__name__}"
        )
    if not isinstance(metadata, EntryMetadata):
        raise ContractError(
            f"Entry metadata must be an EntryMetadata, got {type(metadata).__name__}"
        )

    positions: set = set()
    _collect_positions(
        data,
        depth=layout.depth,
        path=(),
        positions=positions,
    )

    for map_name, context_map in (
        ("sample", metadata.sample),
        ("temporal", metadata.temporal),
    ):
        for index in context_map:
            if index == ():
                continue
            if len(index) > layout.depth:
                raise ContractError(
                    f"Metadata '{map_name}' index {index!r} is deeper than the "
                    f"layout depth {layout.depth}"
                )
            if index not in positions:
                raise ContractError(
                    f"Metadata '{map_name}' index {index!r} does not address an "
                    "existing group or item of the entry"
                )


def _snapshot_named_mapping(value: Any, *, what: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{what} must be a mapping, got {type(value).__name__}")

    for name in value:
        _validate_non_empty_string(name, what=f"{what} entry name")

    snapshot = MappingProxyType(dict(value))

    return snapshot


@dataclass(frozen=True)
class WorkflowsBuffer:
    """Engine-owned carrier of named entries for one pulse of one lineage.

    ``data``, ``layout`` and ``metadata`` always have exactly the same keys.
    Each entry has its own layout and context. A filtered buffer has all three
    mappings empty while keeping its lineage and pulse identity. Supplied
    mappings are snapshotted and exposed read-only.

    Args:
        lineage_id: Identity of the producing lineage.
        pulse_id: Pulse number within the lineage.
        data: Named payloads or nested ``Batch`` trees.
        layout: Named entry layouts.
        metadata: Named entry metadata.

    Raises:
        ContractError: On mismatched keys, wrong value types or an entry that
            fails ``validate_entry``.
    """

    lineage_id: str
    pulse_id: int
    data: Mapping[str, Any] = field(default_factory=dict, hash=False)
    layout: Mapping[str, EntryLayout] = field(default_factory=dict, hash=False)
    metadata: Mapping[str, EntryMetadata] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.lineage_id, what="WorkflowsBuffer lineage_id")
        if isinstance(self.pulse_id, bool) or not isinstance(self.pulse_id, int):
            raise ContractError(
                f"WorkflowsBuffer pulse_id must be an int, got {self.pulse_id!r}"
            )

        data = _snapshot_named_mapping(self.data, what="WorkflowsBuffer data")
        layout = _snapshot_named_mapping(self.layout, what="WorkflowsBuffer layout")
        metadata = _snapshot_named_mapping(
            self.metadata, what="WorkflowsBuffer metadata"
        )

        data_keys = set(data)
        if data_keys != set(layout) or data_keys != set(metadata):
            raise ContractError(
                "WorkflowsBuffer data, layout and metadata must have equal keys; "
                f"data={sorted(data)}, layout={sorted(layout)}, "
                f"metadata={sorted(metadata)}"
            )

        for name in data:
            entry_layout = layout[name]
            entry_metadata = metadata[name]
            if not isinstance(entry_layout, EntryLayout):
                raise ContractError(
                    f"Entry '{name}': layout must be an EntryLayout, "
                    f"got {type(entry_layout).__name__}"
                )
            if not isinstance(entry_metadata, EntryMetadata):
                raise ContractError(
                    f"Entry '{name}': metadata must be an EntryMetadata, "
                    f"got {type(entry_metadata).__name__}"
                )
            try:
                validate_entry(
                    data[name],
                    layout=entry_layout,
                    metadata=entry_metadata,
                )
            except ContractError as error:
                raise ContractError(f"Entry '{name}': {error}") from error

        object.__setattr__(self, "data", data)
        object.__setattr__(self, "layout", layout)
        object.__setattr__(self, "metadata", metadata)

    @classmethod
    def filtered(cls, *, lineage_id: str, pulse_id: int) -> "WorkflowsBuffer":
        """Construct an explicitly filtered emission.

        Args:
            lineage_id: Identity of the producing lineage.
            pulse_id: Pulse number within the lineage.

        Returns:
            A buffer with empty data, layout and metadata.
        """
        buffer = cls(lineage_id=lineage_id, pulse_id=pulse_id)

        return buffer

    @property
    def is_filtered(self) -> bool:
        """Whether this buffer is an explicitly filtered emission."""
        return not self.data

    @property
    def entry_names(self) -> Tuple[str, ...]:
        """Names of the carried entries in insertion order."""
        return tuple(self.data)
