"""Data contract of a recording: groups, entries, kinds and layouts.

A recording stores one ``RecordedGroupSchema`` per recorded output group. It is
the compatibility contract between capture and replay: two schemas are
compatible exactly when they are equal. The capture definition digest is
provenance and is not part of the schema.

::

    RecordingSchema
      groups: name -> RecordedGroupSchema
                       name, anchor_domain
                       entries: RecordedEntrySchema per selected port
                                key     "<field>" or "<field>/<port>" (wildcard)
                                field, port, kinds, layout
"""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Tuple

from roboflow_workflows.execution_engine.v2.data import Axis, EntryLayout
from roboflow_workflows.execution_engine.v2.errors import ContractError
from roboflow_workflows.execution_engine.v2.recording.errors import (
    RecordingSchemaError,
)

FORMAT_VERSION = 1
RECORDING_STATUSES = ("recording", "complete", "stopped", "failed", "cancelled")
FINAL_STATUSES = ("complete", "stopped", "failed", "cancelled")
REPLAYABLE_STATUSES = ("complete", "stopped")


@dataclass(frozen=True)
class RecordedEntrySchema:
    """One recorded entry: a selected port of an output group field.

    Args:
        key: Entry key of ``GroupResult``: the field name, or
            ``"<field>/<port>"`` for one port of a wildcard field.
        field: Name of the group field.
        port: Port name inside a wildcard field; ``None`` otherwise.
        kinds: Declared kind names of the port.
        layout: Plan-scoped layout of the port.

    Raises:
        RecordingSchemaError: When the key does not match field and port, or
            a value has the wrong type.
    """

    key: str
    field: str
    port: Optional[str]
    kinds: Tuple[str, ...]
    layout: EntryLayout

    def __post_init__(self) -> None:
        object.__setattr__(self, "kinds", tuple(self.kinds))
        for name in ("key", "field"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise RecordingSchemaError(
                    f"Recorded entry {name} must be a non-empty string, "
                    f"got {getattr(self, name)!r}"
                )
        expected_key = self.field if self.port is None else f"{self.field}/{self.port}"
        if self.key != expected_key:
            raise RecordingSchemaError(
                f"Recorded entry key {self.key!r} must be {expected_key!r}"
            )
        if not all(isinstance(kind, str) and kind for kind in self.kinds):
            raise RecordingSchemaError(
                f"Recorded entry {self.key!r} kinds must be non-empty strings"
            )
        if not isinstance(self.layout, EntryLayout):
            raise RecordingSchemaError(
                f"Recorded entry {self.key!r} layout must be an EntryLayout"
            )

    def to_json(self) -> Dict[str, Any]:
        """Return the manifest form of this entry.

        Returns:
            JSON-friendly mapping with ``key``, ``field``, ``port``, ``kinds``
            and ``layout``.
        """
        serialized = {
            "key": self.key,
            "field": self.field,
            "port": self.port,
            "kinds": list(self.kinds),
            "layout": layout_to_json(self.layout),
        }

        return serialized

    @classmethod
    def from_json(cls, value: Any) -> "RecordedEntrySchema":
        """Rebuild an entry from ``to_json`` output.

        Args:
            value: Manifest form of the entry.

        Returns:
            The entry schema.

        Raises:
            RecordingSchemaError: When the value is malformed.
        """
        body = _require_keys(
            value, keys=("key", "field", "port", "kinds", "layout"), what="entry"
        )
        if not isinstance(body["kinds"], list):
            raise RecordingSchemaError("Recorded entry kinds must be a list")

        entry = cls(
            key=body["key"],
            field=body["field"],
            port=body["port"],
            kinds=tuple(body["kinds"]),
            layout=layout_from_json(body["layout"]),
        )

        return entry


@dataclass(frozen=True)
class RecordedGroupSchema:
    """Recorded contract of one output group.

    Equality is compatibility: names, entry keys, ports, kinds and axis
    structure (id, kind, stationarity) must all match.

    Args:
        name: Output group name.
        anchor_domain: Source or operator the group was anchored to.
        entries: Recorded entries in selected-port order.

    Raises:
        RecordingSchemaError: On an empty name, no entries, or duplicate keys.
    """

    name: str
    anchor_domain: str
    entries: Tuple[RecordedEntrySchema, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "entries", tuple(self.entries))
        for label in ("name", "anchor_domain"):
            if not isinstance(getattr(self, label), str) or not getattr(self, label):
                raise RecordingSchemaError(
                    f"Recorded group {label} must be a non-empty string"
                )
        if not self.entries:
            raise RecordingSchemaError(f"Recorded group {self.name!r} has no entries")
        if not all(isinstance(entry, RecordedEntrySchema) for entry in self.entries):
            raise RecordingSchemaError(
                f"Recorded group {self.name!r} entries must be RecordedEntrySchema"
            )

        keys = [entry.key for entry in self.entries]
        if len(set(keys)) != len(keys):
            raise RecordingSchemaError(
                f"Recorded group {self.name!r} has duplicate entry keys {keys}"
            )

    @property
    def keys(self) -> Tuple[str, ...]:
        """Entry keys in order."""
        return tuple(entry.key for entry in self.entries)

    @property
    def fields(self) -> Tuple[str, ...]:
        """Field names in order; a wildcard field appears once."""
        return tuple(dict.fromkeys(entry.field for entry in self.entries))

    def entry(self, key: str) -> RecordedEntrySchema:
        """Look up one entry by key.

        Args:
            key: Entry key.

        Returns:
            The entry schema.

        Raises:
            RecordingSchemaError: When the group has no such entry.
        """
        for entry in self.entries:
            if entry.key == key:
                return entry

        raise RecordingSchemaError(
            f"Recorded group {self.name!r} has no entry {key!r}; "
            f"entries: {list(self.keys)}"
        )

    def differences(self, other: "RecordedGroupSchema") -> List[str]:
        """List field-level differences from another schema.

        Args:
            other: Schema to compare with, usually the one a reader expects.

        Returns:
            One human-readable line per difference; empty when compatible.
        """
        differences = []
        for label in ("name", "anchor_domain"):
            if getattr(self, label) != getattr(other, label):
                differences.append(
                    f"{label}: recorded {getattr(self, label)!r}, "
                    f"expected {getattr(other, label)!r}"
                )

        mine = {entry.key: entry for entry in self.entries}
        theirs = {entry.key: entry for entry in other.entries}
        for key in other.keys:
            if key not in mine:
                differences.append(f"entry {key!r}: expected but not recorded")
        for key in self.keys:
            if key not in theirs:
                differences.append(f"entry {key!r}: recorded but not expected")
        for key in other.keys:
            if key not in mine:
                continue
            for label in ("field", "port", "kinds", "layout"):
                if getattr(mine[key], label) != getattr(theirs[key], label):
                    differences.append(
                        f"entry {key!r} {label}: recorded "
                        f"{_describe(getattr(mine[key], label))}, expected "
                        f"{_describe(getattr(theirs[key], label))}"
                    )
        if not differences and self.keys != other.keys:
            differences.append(
                f"entry order: recorded {list(self.keys)}, expected {list(other.keys)}"
            )

        return differences

    def require_compatible(self, expected: "RecordedGroupSchema") -> None:
        """Raise unless this recorded schema equals the expected one.

        Args:
            expected: Schema the reader was compiled against.

        Raises:
            RecordingSchemaError: Listing every field-level difference.
        """
        if self == expected:
            return

        differences = self.differences(expected)
        raise RecordingSchemaError(
            "recorded schema is incompatible: " + "; ".join(differences),
            group=self.name,
        )

    def to_json(self) -> Dict[str, Any]:
        """Return the manifest form of this group schema.

        Returns:
            JSON-friendly mapping with ``name``, ``anchor_domain`` and
            ``entries``.
        """
        serialized = {
            "name": self.name,
            "anchor_domain": self.anchor_domain,
            "entries": [entry.to_json() for entry in self.entries],
        }

        return serialized

    @classmethod
    def from_json(cls, value: Any) -> "RecordedGroupSchema":
        """Rebuild a group schema from ``to_json`` output.

        Args:
            value: Manifest form of the group schema.

        Returns:
            The group schema.

        Raises:
            RecordingSchemaError: When the value is malformed.
        """
        body = _require_keys(
            value, keys=("name", "anchor_domain", "entries"), what="group"
        )
        if not isinstance(body["entries"], list):
            raise RecordingSchemaError("Recorded group entries must be a list")

        group = cls(
            name=body["name"],
            anchor_domain=body["anchor_domain"],
            entries=tuple(
                RecordedEntrySchema.from_json(entry) for entry in body["entries"]
            ),
        )

        return group


@dataclass(frozen=True)
class RecordingSchema:
    """Schemas of every recorded group, in declaration order.

    Args:
        groups: Group schemas by group name. A sequence of group schemas is
            also accepted.

    Raises:
        RecordingSchemaError: On no groups or a key that differs from its
            schema's name.
    """

    groups: Mapping[str, RecordedGroupSchema]

    def __post_init__(self) -> None:
        groups = self.groups
        if not isinstance(groups, Mapping):
            groups = {group.name: group for group in groups}
        if not groups:
            raise RecordingSchemaError("A recording needs at least one group")
        for name, group in groups.items():
            if not isinstance(group, RecordedGroupSchema) or group.name != name:
                raise RecordingSchemaError(
                    f"Recording group {name!r} must map to its RecordedGroupSchema"
                )

        object.__setattr__(self, "groups", MappingProxyType(dict(groups)))

    def group(self, name: str) -> RecordedGroupSchema:
        """Look up one group schema.

        Args:
            name: Group name.

        Returns:
            The group schema.

        Raises:
            RecordingSchemaError: When the recording has no such group.
        """
        if name not in self.groups:
            raise RecordingSchemaError(
                f"no recorded group {name!r}; recorded groups: {list(self.groups)}"
            )

        return self.groups[name]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RecordingSchema):
            return NotImplemented

        return list(self.groups.items()) == list(other.groups.items())

    __hash__ = None


def layout_to_json(layout: EntryLayout) -> List[Dict[str, Any]]:
    """Return the manifest form of a layout.

    Args:
        layout: Layout to describe.

    Returns:
        One ``{"id", "kind", "stationary"}`` mapping per axis, outermost first.
    """
    serialized = [
        {"id": axis.id, "kind": axis.kind, "stationary": axis.stationary}
        for axis in layout.axes
    ]

    return serialized


def layout_from_json(value: Any) -> EntryLayout:
    """Rebuild a layout from ``layout_to_json`` output.

    Args:
        value: Manifest form of the layout.

    Returns:
        The layout; all layout invariants are re-validated.

    Raises:
        RecordingSchemaError: When the value is malformed or violates a
            layout invariant.
    """
    if not isinstance(value, list):
        raise RecordingSchemaError("A recorded layout must be a list of axes")

    try:
        axes = []
        for axis in value:
            body = _require_keys(axis, keys=("id", "kind", "stationary"), what="axis")
            axes.append(
                Axis(id=body["id"], kind=body["kind"], stationary=body["stationary"])
            )
        layout = EntryLayout(axes=tuple(axes))
    except ContractError as error:
        raise RecordingSchemaError(f"Invalid recorded layout: {error}") from error

    return layout


def _require_keys(value: Any, *, keys: Tuple[str, ...], what: str) -> Mapping:
    if not isinstance(value, Mapping) or set(value) != set(keys):
        raise RecordingSchemaError(
            f"A recorded {what} must be an object with exactly the keys {list(keys)}"
        )

    return value


def _describe(value: Any) -> str:
    if isinstance(value, EntryLayout):
        description = repr(
            [
                f"{axis.id}:{axis.kind}{'*' if axis.stationary else ''}"
                for axis in value.axes
            ]
        )
        return description

    return repr(list(value)) if isinstance(value, tuple) else repr(value)
