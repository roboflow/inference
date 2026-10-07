"""Event declarations of V2 blocks and workflow signals.

A block declares the events it may emit next to its outputs::

    class ZoneCounter(Block):
        type = "demo/zone_counter@v1"
        outputs = {"count": Output(INTEGER_KIND)}
        events = {
            "threshold_crossed": Event(
                {"count": INTEGER_KIND, "frame": IMAGE_KIND},
                description="The per-source count reached the threshold.",
            )
        }

An ``Event`` names its payload fields and their kinds, as ``Output`` does for
step outputs. Workflow handlers subscribe with ``$steps.<step>.events.<name>``
and bind a subset of the fields with ``$event.<field>``. Workflow ``signals``
use the same ``Event`` shape for external ingress.

The module only imports kinds and errors, so the declaration, the compiler and
the runtime can all import it without cycles.
"""

import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional, Tuple

from roboflow_workflows.execution_engine.v2.errors import ContractError
from roboflow_workflows.execution_engine.v2.kinds import (
    WILDCARD_KIND_NAME,
    Kind,
    normalize_kinds,
)

EVENT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_\-]*")


class EventPayloadError(ContractError):
    """An emitted event payload does not match its declaration.

    Args:
        message: Human-readable explanation.
        event: Name of the event.
    """

    def __init__(self, message: str, *, event: str):
        super().__init__(message)
        self.event = event


@dataclass(frozen=True)
class Event:
    """Declared payload of one event.

    Args:
        fields: Field name to one ``Kind`` or an iterable of kinds. An empty
            iterable means the wildcard kind. An event without fields is a
            pure notification.
        description: Human-readable meaning of the event.

    Raises:
        ContractError: On a field name that is not a selector segment, or on
            kinds that are not ``Kind`` objects.
    """

    fields: Mapping[str, Tuple[Kind, ...]] = field(
        default_factory=lambda: MappingProxyType({})
    )
    description: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.fields, Mapping):
            raise ContractError(
                f"Event fields must be a mapping of field name to kinds, got "
                f"{type(self.fields).__name__}"
            )
        if not isinstance(self.description, str):
            raise ContractError("Event description must be a string")

        normalized = {}
        for name, kinds in self.fields.items():
            if not isinstance(name, str) or not EVENT_NAME.fullmatch(name):
                raise ContractError(
                    f"Event field name {name!r} must be letters, digits, '_' or "
                    "'-', starting with a letter or '_'"
                )
            declared = (kinds,) if isinstance(kinds, Kind) else kinds
            if not isinstance(declared, Iterable) or isinstance(declared, str):
                raise ContractError(
                    f"Event field '{name}' must declare a Kind or a list of kinds, "
                    f"got {kinds!r}"
                )
            normalized[name] = normalize_kinds(
                declared, context=f"Event field '{name}'"
            )
        object.__setattr__(self, "fields", MappingProxyType(normalized))

    def kind_names(self, field_name: str) -> Tuple[str, ...]:
        """Kind names of one field.

        Args:
            field_name: Declared field name.

        Returns:
            Kind names in declaration order.
        """
        names = tuple(kind.name for kind in self.fields[field_name])

        return names

    def check_payload(self, name: str, values: Mapping[str, Any]) -> None:
        """Validate one emitted payload against this declaration.

        Every declared field must be present and no other field is allowed.
        Each value must pass the ``validate`` hook of at least one of its
        kinds; kinds without a hook accept every value.

        Args:
            name: Event name, used in messages.
            values: Emitted field values.

        Raises:
            EventPayloadError: On missing or unknown fields or a value that no
                declared kind accepts.
        """
        missing = [item for item in self.fields if item not in values]
        unknown = [item for item in values if item not in self.fields]
        if missing or unknown:
            raise EventPayloadError(
                f"Event '{name}' declares fields {sorted(self.fields)}; "
                f"missing {sorted(missing)}, unknown {sorted(unknown)}",
                event=name,
            )
        for field_name, value in values.items():
            kinds = self.fields[field_name]
            if not any(_accepts(kind, value) for kind in kinds):
                raise EventPayloadError(
                    f"Event '{name}' field '{field_name}' got "
                    f"{type(value).__name__}, which none of the kinds "
                    f"{list(self.kind_names(field_name))} accepts",
                    event=name,
                )

    def describe(self) -> Mapping[str, Any]:
        """Describe the event as JSON-friendly data.

        Returns:
            ``{"fields": {name: [kind names]}, "description": ...}``.
        """
        description = {
            "fields": {name: list(self.kind_names(name)) for name in self.fields},
            "description": self.description,
        }

        return description


def _accepts(kind: Kind, value: Any) -> bool:
    if kind.name == WILDCARD_KIND_NAME or kind.validate is None:
        return True
    accepted = bool(kind.validate(value))

    return accepted


def validate_events(
    declared: Any, *, owner: str, fail: Optional[Any] = None
) -> Mapping[str, Event]:
    """Validate an ``events`` declaration of a block or a workflow.

    Args:
        declared: Mapping of event name to ``Event``.
        owner: Who declares the events, used in messages.
        fail: Optional ``message -> Exception`` factory; defaults to
            ``ContractError``.

    Returns:
        Read-only mapping of event name to ``Event``.

    Raises:
        ContractError: Or the exception built by ``fail``, on a declaration
            that is not a mapping, a bad event name or a value that is not an
            ``Event``.
    """
    make_error = fail if fail is not None else ContractError
    if not isinstance(declared, Mapping):
        raise make_error(
            f"{owner} events must be a mapping of name to Event, got "
            f"{type(declared).__name__}"
        )

    events = {}
    for name, event in declared.items():
        if not isinstance(name, str) or not EVENT_NAME.fullmatch(name):
            raise make_error(
                f"{owner} event name {name!r} must be letters, digits, '_' or "
                "'-', starting with a letter or '_'"
            )
        if not isinstance(event, Event):
            raise make_error(
                f"{owner} event '{name}' must be an Event, got "
                f"{type(event).__name__}"
            )
        events[name] = event

    return MappingProxyType(events)
