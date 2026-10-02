"""Payload kinds of the V2 execution engine.

A kind names a type of payload and owns every payload-specific hook the engine
may call at its boundaries:

* ``validate(payload) -> bool``: checks a payload entering or leaving a block.
* ``deserialize(value) -> payload``: turns a caller-supplied workflow input
  value (for example JSON) into the payload blocks expect.
* ``serialize(payload) -> value``: turns a payload into a JSON-friendly value
  when the caller asks for serialized output rows.
* ``convert_output(payload, options) -> payload``: adapts a payload for one
  declared workflow output, using that output's options (for example a
  coordinate system for detections).

The engine never inspects payloads itself. Image, detection and other media
kinds live in block catalogues, not here. This module only provides the
wildcard and a few plain Python kinds whose names match V1.

Compatibility follows V1: a producer and a consumer are compatible when their
kind lists share a name or either side contains the wildcard.
"""

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Optional, Tuple

from roboflow_workflows.execution_engine.v2.errors import ContractError

WILDCARD_KIND_NAME = "*"

PayloadValidator = Callable[[Any], bool]
PayloadCodec = Callable[[Any], Any]
OutputConverter = Callable[[Any, Mapping[str, Any]], Any]


@dataclass(frozen=True)
class Kind:
    """A named payload type and its optional boundary hooks.

    Args:
        name: Unique kind name, e.g. ``"image"`` or ``"float"``.
        description: Human-readable meaning of the payload.
        validate: Returns ``True`` for an acceptable payload. ``None`` accepts
            every payload.
        deserialize: Converts a caller-supplied input value into a payload.
            ``None`` passes values through unchanged.
        serialize: Converts a payload into a JSON-friendly value. ``None``
            passes payloads through unchanged.
        convert_output: Adapts a payload for a workflow output, given that
            output's options. ``None`` leaves payloads unchanged.

    Raises:
        ContractError: On an empty name or a hook that is not callable.
    """

    name: str
    description: str = ""
    validate: Optional[PayloadValidator] = None
    deserialize: Optional[PayloadCodec] = None
    serialize: Optional[PayloadCodec] = None
    convert_output: Optional[OutputConverter] = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ContractError(
                f"Kind name must be a non-empty string, got {self.name!r}"
            )

        for hook_name in ("validate", "deserialize", "serialize", "convert_output"):
            hook = getattr(self, hook_name)
            if hook is not None and not callable(hook):
                raise ContractError(
                    f"Kind '{self.name}' {hook_name} hook must be callable, "
                    f"got {type(hook).__name__}"
                )

    def check(self, payload: Any) -> None:
        """Raise when ``payload`` is not a valid payload of this kind.

        Args:
            payload: Value to check.

        Raises:
            ContractError: When the validator rejects the payload, raises, or
                returns something other than a bool. A raising validator is
                preserved as the cause.
        """
        if self.validate is None:
            return

        try:
            accepted = self.validate(payload)
        except Exception as error:
            raise ContractError(
                f"Validator of kind '{self.name}' failed on "
                f"{type(payload).__name__}: {error}"
            ) from error
        if not isinstance(accepted, bool):
            raise ContractError(
                f"Validator of kind '{self.name}' must return a bool, "
                f"got {type(accepted).__name__}"
            )
        if not accepted:
            raise ContractError(
                f"Payload of type {type(payload).__name__} is not a valid "
                f"'{self.name}'"
            )

    def to_payload(self, value: Any) -> Any:
        """Deserialize a caller-supplied value into a payload.

        Args:
            value: Value supplied at the workflow boundary.

        Returns:
            The payload; ``value`` itself when no deserializer is declared.
        """
        if self.deserialize is None:
            return value

        payload = self.deserialize(value)

        return payload

    def to_serialized(self, payload: Any) -> Any:
        """Serialize a payload for output rows.

        Args:
            payload: Payload produced by the workflow.

        Returns:
            The serialized value; ``payload`` itself when no serializer exists.
        """
        if self.serialize is None:
            return payload

        serialized = self.serialize(payload)

        return serialized

    def to_output(self, payload: Any, *, options: Mapping[str, Any]) -> Any:
        """Adapt a payload for one declared workflow output.

        Args:
            payload: Payload produced by the workflow.
            options: Options of the workflow output declaration.

        Returns:
            The converted payload; ``payload`` itself when no converter exists.
        """
        if self.convert_output is None:
            return payload

        converted = self.convert_output(payload, options)

        return converted


def _is_boolean(payload: Any) -> bool:
    return isinstance(payload, bool)


def _is_integer(payload: Any) -> bool:
    return isinstance(payload, int) and not isinstance(payload, bool)


def _is_float(payload: Any) -> bool:
    return isinstance(payload, (int, float)) and not isinstance(payload, bool)


def _is_string(payload: Any) -> bool:
    return isinstance(payload, str)


def _is_dictionary(payload: Any) -> bool:
    return isinstance(payload, Mapping)


def _is_list_of_values(payload: Any) -> bool:
    return isinstance(payload, (list, tuple))


WILDCARD_KIND = Kind(name=WILDCARD_KIND_NAME, description="Any payload.")
BOOLEAN_KIND = Kind(name="boolean", description="Python bool.", validate=_is_boolean)
INTEGER_KIND = Kind(name="integer", description="Python int.", validate=_is_integer)
FLOAT_KIND = Kind(name="float", description="Python int or float.", validate=_is_float)
STRING_KIND = Kind(name="string", description="Python str.", validate=_is_string)
DICTIONARY_KIND = Kind(
    name="dictionary", description="Mapping payload.", validate=_is_dictionary
)
LIST_OF_VALUES_KIND = Kind(
    name="list_of_values",
    description="List or tuple payload; a payload, not a logical group.",
    validate=_is_list_of_values,
)

BUILTIN_KINDS: Tuple[Kind, ...] = (
    WILDCARD_KIND,
    BOOLEAN_KIND,
    INTEGER_KIND,
    FLOAT_KIND,
    STRING_KIND,
    DICTIONARY_KIND,
    LIST_OF_VALUES_KIND,
)


def normalize_kinds(kinds: Iterable[Any], *, context: str) -> Tuple[Kind, ...]:
    """Validate a declared kind list; an empty list means the wildcard.

    Args:
        kinds: ``Kind`` objects as written in a declaration.
        context: Where the kinds are declared, used in error messages.

    Returns:
        Tuple of unique kinds in declaration order, or ``(WILDCARD_KIND,)``.

    Raises:
        ContractError: When an element is not a ``Kind`` or names repeat.
    """
    normalized = []
    seen_names = set()
    for kind in kinds:
        if not isinstance(kind, Kind):
            raise ContractError(
                f"{context} must list Kind objects, got {type(kind).__name__}: "
                f"{kind!r}. Import the kind object instead of writing its name."
            )
        if kind.name in seen_names:
            raise ContractError(f"{context} lists kind '{kind.name}' twice")
        seen_names.add(kind.name)
        normalized.append(kind)

    if not normalized:
        return (WILDCARD_KIND,)

    return tuple(normalized)


def kinds_compatible(produced: Iterable[str], accepted: Iterable[str]) -> bool:
    """Return whether produced and accepted kind names are compatible.

    Args:
        produced: Kind names a producer declares.
        accepted: Kind names a consumer accepts.

    Returns:
        ``True`` when the names overlap or either side contains the wildcard.
    """
    produced_names = set(produced)
    accepted_names = set(accepted)
    if WILDCARD_KIND_NAME in produced_names or WILDCARD_KIND_NAME in accepted_names:
        return True

    compatible = bool(produced_names & accepted_names)

    return compatible
