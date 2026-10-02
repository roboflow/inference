"""Run-time value records of the V2 executor.

Every workflow input and every step output of one run is held as an ``Entry``.
An entry keeps the complete *known* index structure of its value, including
positions that were filtered, so later steps, recovery joins and output rows
still know every original index when all data at some position is gone::

    layout [N, crops]
    children  {(): ((0,), (1,), (2,)),  (0,): ((0, 0), (0, 1)),  (2,): ()}
    values    {(0, 0): a, (0, 1): b}
    filtered  {(1,)}                    # subtree of (1,) filtered, children unknown

    (2,)  is a genuine empty group: known, no children
    (1,)  is filtered: it or an ancestor is in ``filtered``

Entries are private to one run. Payload objects are shared, never copied, so
declared in-place mutations stay visible to later consumers.
"""

from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Tuple

from roboflow_workflows.execution_engine.v2.data import (
    Batch,
    EntryLayout,
    EntryMetadata,
    Index,
)


@dataclass(frozen=True)
class Entry:
    """One value source of a run with its known index structure.

    Args:
        layout: Axes of the value.
        metadata: Indexed source and temporal context.
        children: Known child indices of every known, unfiltered group node
            above the leaf level, in logical order.
        values: Present leaf payloads by full index; ``None`` is a payload.
        filtered: Filtered nodes. Everything under them is filtered.
    """

    layout: EntryLayout
    metadata: EntryMetadata
    children: Mapping[Index, Tuple[Index, ...]]
    values: Mapping[Index, Any]
    filtered: FrozenSet[Index]

    @property
    def depth(self) -> int:
        """Number of axes."""
        return self.layout.depth

    def is_filtered(self, index: Index) -> bool:
        """Return whether ``index`` or one of its ancestors was filtered.

        Args:
            index: Logical index at any depth of the entry.

        Returns:
            ``True`` when the position is filtered.
        """
        filtered = any(
            index[:length] in self.filtered for length in range(len(index) + 1)
        )

        return filtered

    def has_value(self, index: Index) -> bool:
        """Return whether a present leaf exists at ``index``.

        Args:
            index: Full leaf index.

        Returns:
            ``True`` when a payload (possibly ``None``) is stored there.
        """
        return index in self.values and not self.is_filtered(index)

    def nodes_at(self, depth: int) -> List[Index]:
        """Return every known node at ``depth``, filtered nodes included.

        The walk does not descend below filtered nodes, whose children are
        not necessarily known.

        Args:
            depth: Number of index components, at most the entry depth.

        Returns:
            Node indices in logical order.
        """
        level = [()]
        for _ in range(depth):
            deeper: List[Index] = []
            for node in level:
                if node in self.filtered:
                    continue
                deeper.extend(self.children.get(node, ()))
            level = deeper

        return level

    def is_effectively_filtered(self, index: Index = ()) -> bool:
        """Return whether no unfiltered content remains under ``index``.

        A group whose children are all filtered counts as filtered; a genuine
        empty group does not.

        Args:
            index: Node to inspect; the whole entry by default.

        Returns:
            ``True`` when the node is filtered or every child of it is.
        """
        if self.is_filtered(index):
            return True
        if len(index) == self.depth:
            return index not in self.values

        children = self.children.get(index)
        if children is None:
            return True

        effectively_filtered = bool(children) and all(
            self.is_effectively_filtered(child) for child in children
        )

        return effectively_filtered

    def minimal_filtered_paths(self) -> Tuple[Index, ...]:
        """Return filtered nodes that have no filtered ancestor, sorted."""
        minimal = sorted(
            path
            for path in self.filtered
            if not any(path[:length] in self.filtered for length in range(len(path)))
        )

        return tuple(minimal)

    def to_tree(self) -> Any:
        """Build the surviving payload tree with attached layout/metadata views.

        Returns:
            A payload for an ungrouped entry, otherwise a nested ``Batch``
            holding only unfiltered nodes under their original indices.
        """
        metadata = restrict_metadata(self.metadata, existing=self._surviving_nodes())
        tree = self._build_tree((), metadata=metadata)

        return tree

    def _build_tree(self, index: Index, *, metadata: EntryMetadata) -> Any:
        # A method rather than a nested function: a recursive closure refers to
        # itself, and that cycle would keep the payloads alive until the cyclic
        # garbage collector runs.
        if len(index) == self.depth:
            return self.values[index]

        kept = [
            child
            for child in self.children.get(index, ())
            if child not in self.filtered
            and (len(child) < self.depth or child in self.values)
        ]
        group = Batch(
            [self._build_tree(child, metadata=metadata) for child in kept],
            indices=kept,
            layout=self.layout,
            metadata=metadata,
            parent_index=index,
        )

        return group

    def surviving_metadata(self) -> EntryMetadata:
        """Return metadata restricted to nodes present in ``to_tree()``."""
        metadata = restrict_metadata(self.metadata, existing=self._surviving_nodes())

        return metadata

    def _surviving_nodes(self) -> FrozenSet[Index]:
        nodes = {()}
        level = [()]
        for depth in range(self.depth):
            deeper = []
            for node in level:
                for child in self.children.get(node, ()):
                    if child in self.filtered:
                        continue
                    if depth + 1 == self.depth and child not in self.values:
                        continue
                    deeper.append(child)
            nodes.update(deeper)
            level = deeper

        return frozenset(nodes)


def scalar_entry(value: Any, *, metadata: EntryMetadata) -> Entry:
    """Create an ungrouped entry holding one payload.

    Args:
        value: The payload, possibly ``None``.
        metadata: Context of the payload.

    Returns:
        The entry.
    """
    entry = Entry(
        layout=EntryLayout(),
        metadata=metadata,
        children={},
        values={(): value},
        filtered=frozenset(),
    )

    return entry


def entry_from_tree(
    data: Any, *, layout: EntryLayout, metadata: EntryMetadata
) -> Entry:
    """Create an entry from a validated payload or nested ``Batch`` tree.

    Args:
        data: Payload (ungrouped layout) or nested ``Batch`` with full indices.
        layout: Axes of the value.
        metadata: Indexed context.

    Returns:
        The entry; nothing is filtered.
    """
    children: Dict[Index, Tuple[Index, ...]] = {}
    values: Dict[Index, Any] = {}
    _collect_tree(data, (), depth=layout.depth, children=children, values=values)
    entry = Entry(
        layout=layout,
        metadata=metadata,
        children=children,
        values=values,
        filtered=frozenset(),
    )

    return entry


def _collect_tree(
    node: Any,
    index: Index,
    *,
    depth: int,
    children: Dict[Index, Tuple[Index, ...]],
    values: Dict[Index, Any],
) -> None:
    # Module-level, not nested: a recursive closure would form a reference
    # cycle holding ``values`` (see ``Entry._build_tree``).
    if len(index) == depth:
        values[index] = node
        return

    children[index] = tuple(node.indices)
    for child_index, child in node.iter_with_indices():
        _collect_tree(child, child_index, depth=depth, children=children, values=values)


def restrict_metadata(
    metadata: EntryMetadata, *, existing: Iterable[Index]
) -> EntryMetadata:
    """Drop metadata keys that address nodes absent from a tree.

    Args:
        metadata: Metadata to restrict.
        existing: Indices of the nodes that exist.

    Returns:
        ``metadata`` itself when every key survives, otherwise a copy.
    """
    existing = set(existing) | {()}
    sample = {key: value for key, value in metadata.sample.items() if key in existing}
    temporal = {
        key: value for key, value in metadata.temporal.items() if key in existing
    }
    if len(sample) == len(metadata.sample) and len(temporal) == len(metadata.temporal):
        return metadata

    restricted = EntryMetadata(sample=sample, temporal=temporal)

    return restricted


def common_or_none(contexts: Iterable[Any]) -> Optional[Any]:
    """Return the one context shared by all contributors, else ``None``.

    Args:
        contexts: Contributing contexts; ``None`` is a context too.

    Returns:
        The shared context, or ``None`` when contributors differ or none exist.
    """
    collected = list(contexts)
    if not collected:
        return None

    first = collected[0]
    common = first if all(item == first for item in collected[1:]) else None

    return common
