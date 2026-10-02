"""Selector grammar, selector annotations and their analysis.

Block authors use ``Ref``, ``Group`` and ``StepRef`` (re-exported by
``declaration``). Each is an ``Annotated[str, <grammar>, SelectorMarker, ...]``
alternative of a ``Params`` field or of one list element / dict value.

A *position* is the whole field or one container leaf. Its annotation is a
union of alternatives (at most one selector alternative) plus optional shared
metadata, e.g. ``float | Ref(FLOAT_KIND) = Field(ge=0, le=1)``. Pydantic
keeps shared field constraints as outer ``Annotated`` metadata, or merges them
into the selector alternative's own metadata when it is the only one; both
spellings are recognised here.

Names follow V1 selector segments: letters, digits, ``_`` and ``-``, including
a leading digit (``$steps.parse.2026``). Complete strings are matched with
``fullmatch``.
"""

import re
import types
from dataclasses import dataclass
from typing import (
    Annotated,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    get_args,
    get_origin,
)

from pydantic import BaseModel, Field, StringConstraints
from pydantic.fields import FieldInfo
from roboflow_workflows.execution_engine.v2.data import Batch
from roboflow_workflows.execution_engine.v2.errors import (
    ContractError,
    DeclarationError,
    SelectorError,
)
from roboflow_workflows.execution_engine.v2.kinds import Kind, normalize_kinds

BatchMode = Literal["never", "always", "if_varying"]
SelectorRole = Literal["item", "group", "step"]
ContainerKind = Literal["list", "dict"]

BATCH_MODES: Tuple[str, ...] = ("never", "always", "if_varying")

SELECTOR_SEGMENT = r"[A-Za-z0-9_\-]+"
DATA_SELECTOR = (
    rf"\$(?:inputs\.{SELECTOR_SEGMENT}"
    rf"|steps\.{SELECTOR_SEGMENT}\.(?:{SELECTOR_SEGMENT}|\*)"
    rf"|sources\.{SELECTOR_SEGMENT}\.{SELECTOR_SEGMENT}"
    rf"|operators\.{SELECTOR_SEGMENT}\.{SELECTOR_SEGMENT})"
)
STEP_SELECTOR = rf"\$steps\.{SELECTOR_SEGMENT}"
DATA_SELECTOR_PATTERN = f"^{DATA_SELECTOR}$"
STEP_SELECTOR_PATTERN = f"^{STEP_SELECTOR}$"
SELECTOR_PREFIXES: Tuple[str, ...] = (
    "$inputs.",
    "$steps.",
    "$sources.",
    "$operators.",
)

_SEGMENT = re.compile(SELECTOR_SEGMENT)
_DATA = re.compile(DATA_SELECTOR)
_STEP = re.compile(STEP_SELECTOR)


def is_selector_segment(value: Any) -> bool:
    """Return whether ``value`` is a complete selector name segment.

    Args:
        value: Candidate step, input or output name.

    Returns:
        ``True`` for a non-empty string of letters, digits, ``_`` and ``-``.
    """
    matched = isinstance(value, str) and _SEGMENT.fullmatch(value) is not None

    return matched


@dataclass(frozen=True)
class ParsedSelector:
    """Structure of one selector string.

    Args:
        target: ``"input"``, ``"step_output"``, ``"source_output"``,
            ``"operator_output"`` or ``"step"``.
        name: Workflow input, step, source or operator name.
        output: Step, source or operator output name (``"*"`` for all
            outputs of a step); ``None`` otherwise.
    """

    target: Literal["input", "step_output", "source_output", "operator_output", "step"]
    name: str
    output: Optional[str] = None


def parse_selector(text: Any) -> ParsedSelector:
    """Parse a complete data or step selector.

    Args:
        text: ``$inputs.<name>``, ``$steps.<step>.<output>``,
            ``$steps.<step>.*``, ``$sources.<source>.<output>``,
            ``$operators.<operator>.<output>`` or ``$steps.<step>``.

    Returns:
        The parsed selector.

    Raises:
        SelectorError: When ``text`` follows neither grammar completely.
    """
    if isinstance(text, str) and _DATA.fullmatch(text):
        parts = text[1:].split(".")
        if parts[0] == "inputs":
            parsed = ParsedSelector(target="input", name=parts[1])
            return parsed
        if parts[0] == "sources":
            parsed = ParsedSelector(
                target="source_output", name=parts[1], output=parts[2]
            )
            return parsed
        if parts[0] == "operators":
            parsed = ParsedSelector(
                target="operator_output", name=parts[1], output=parts[2]
            )
            return parsed

        parsed = ParsedSelector(target="step_output", name=parts[1], output=parts[2])
        return parsed

    if isinstance(text, str) and _STEP.fullmatch(text):
        parsed = ParsedSelector(target="step", name=text[len("$steps.") :])
        return parsed

    raise SelectorError(
        f"Malformed selector {text!r}. Use $inputs.<name>, $steps.<step>.<output>, "
        "$steps.<step>.*, $sources.<source>.<output>, "
        "$operators.<operator>.<output> or, for control targets, "
        "$steps.<step>."
    )


@dataclass(frozen=True)
class SelectorMarker:
    """Declaration of one selector position.

    Args:
        role: ``item`` (one element per invocation), ``group`` (the trailing
            group of children per invocation) or ``step`` (control target).
        kinds: Accepted kinds of the selected payload (of each child for
            ``group``); the wildcard when none were declared.
        batch: ``never`` delivers one value per invocation; ``always``
            delivers a ``Batch`` over the step's invocations to one call;
            ``if_varying`` does so only when the bound value varies.
        temporal: For a ``group``: the consumed final axis must be the time
            axis (a T-oriented collapse). ``False`` accepts any final axis.
    """

    role: SelectorRole
    kinds: Tuple[Kind, ...]
    batch: BatchMode = "never"
    temporal: bool = False

    @property
    def kind_names(self) -> Tuple[str, ...]:
        """Accepted kind names."""
        return tuple(kind.name for kind in self.kinds)

    def matches(self, value: Any) -> bool:
        """Return whether ``value`` is a complete selector for this role.

        Args:
            value: Candidate value found at this position.

        Returns:
            ``True`` for a string following this role's selector grammar.
        """
        if not isinstance(value, str):
            return False

        pattern = _STEP if self.role == "step" else _DATA
        matched = pattern.fullmatch(value) is not None

        return matched

    def check_payload(self, value: Any) -> None:
        """Raise unless ``value`` is an acceptable selected payload.

        An item must satisfy one of the kinds; a group must be a ``Batch``
        whose children each satisfy one of them.

        Args:
            value: Resolved value delivered for this position.

        Raises:
            ContractError: On a wrong kind or a group that is not a ``Batch``.
        """
        if self.role != "group":
            _check_kinds(self.kinds, value)
            return

        if not isinstance(value, Batch):
            raise ContractError(f"a group must be a Batch, got {type(value).__name__}")
        for child_index, child in value.iter_with_indices():
            try:
                _check_kinds(self.kinds, child)
            except ContractError as error:
                raise ContractError(f"child {list(child_index)}: {error}") from error

    def describe(self) -> Dict[str, Any]:
        """Return the JSON-friendly metadata published in schemas."""
        description = {
            "role": self.role,
            "kinds": list(self.kind_names),
            "batch": self.batch,
        }
        if self.role == "group":
            description["temporal"] = self.temporal

        return description


def _check_kinds(kinds: Tuple[Kind, ...], value: Any) -> None:
    errors = []
    for kind in kinds:
        try:
            kind.check(value)
        except ContractError as error:
            errors.append(str(error))
            continue
        return

    raise ContractError(" and ".join(errors))


def _marker_annotation(marker: SelectorMarker, *, pattern: str) -> Any:
    annotation = Annotated[
        str,
        StringConstraints(pattern=pattern),
        marker,
        Field(json_schema_extra={"selector": marker.describe()}),
    ]

    return annotation


def _make_marker(
    *, role: SelectorRole, kinds: Iterable[Any], batch: Any, temporal: Any = False
) -> SelectorMarker:
    if not isinstance(temporal, bool):
        raise DeclarationError(f"temporal must be a bool, got {temporal!r}")
    if batch not in BATCH_MODES:
        raise DeclarationError(
            f"batch must be one of {list(BATCH_MODES)}, got {batch!r}"
        )

    try:
        normalized_kinds = normalize_kinds(kinds, context=f"{role} selector")
    except ContractError as error:
        raise DeclarationError(str(error)) from error

    marker = SelectorMarker(
        role=role, kinds=normalized_kinds, batch=batch, temporal=temporal
    )

    return marker


def Ref(*kinds: Kind, batch: BatchMode = "never") -> Any:  # noqa: N802
    """Declare a selector to one element per invocation.

    Combine with literal types to accept both: ``float | Ref(FLOAT_KIND)``.
    Usable as a whole field or as a list element / dict value.

    Args:
        *kinds: Accepted kinds; none means the wildcard.
        batch: ``never``, ``always`` or ``if_varying`` (see ``SelectorMarker``).

    Returns:
        An annotation usable in a ``BlockParams`` field.

    Raises:
        DeclarationError: On an unknown batch mode or invalid kinds.
    """
    marker = _make_marker(role="item", kinds=kinds, batch=batch)
    annotation = _marker_annotation(marker, pattern=DATA_SELECTOR_PATTERN)

    return annotation


def Group(  # noqa: N802
    *kinds: Kind, batch: BatchMode = "never", temporal: bool = False
) -> Any:
    """Declare a selector to the trailing group of children per invocation.

    The step runs once per parent; the block receives a ``Batch`` of that
    parent's children with their full logical indices. Usable as a whole
    field or as a list element / dict value.

    An ordinary (S-oriented) group collapses whatever axis is last, sample
    nesting or time alike. ``temporal=True`` declares a T-oriented group: the
    compiler then requires the consumed last axis to be the time axis, so
    ``[N, T]`` is accepted and ``[N, T, C]`` is rejected.

    Args:
        *kinds: Accepted kinds of the children; none means the wildcard.
        batch: ``always`` or ``if_varying`` deliver a ``Batch`` of groups.
        temporal: Require the consumed final axis to be time.

    Returns:
        An annotation usable in a ``BlockParams`` field.

    Raises:
        DeclarationError: On an unknown batch mode, invalid kinds or a
            non-bool ``temporal``.
    """
    marker = _make_marker(role="group", kinds=kinds, batch=batch, temporal=temporal)
    annotation = _marker_annotation(marker, pattern=DATA_SELECTOR_PATTERN)

    return annotation


StepRef = _marker_annotation(
    SelectorMarker(role="step", kinds=normalize_kinds((), context="step")),
    pattern=STEP_SELECTOR_PATTERN,
)
"""Annotation of a control target, e.g. ``next_steps: list[StepRef]``."""


@dataclass(frozen=True)
class Arm:
    """The selector alternative of one position, split into its parts.

    Args:
        marker: The selector declaration.
        constraints: Metadata on the alternative other than the marker and the
            selector string grammar, e.g. an outer
            ``Field(ge=0)`` that Pydantic merged into a selector-only field.
            They constrain the selected payload, not the selector string.
    """

    marker: SelectorMarker
    constraints: Tuple[Any, ...]


def selector_arm(annotation: Any) -> Optional[Arm]:
    """Split a ``Ref``/``Group``/``StepRef`` annotation into its parts.

    Args:
        annotation: Candidate alternative.

    Returns:
        The arm, or ``None`` when the annotation is not a selector alternative.
    """
    if get_origin(annotation) is not Annotated:
        return None

    metadata = get_args(annotation)[1:]
    markers = [extra for extra in metadata if isinstance(extra, SelectorMarker)]
    if not markers:
        return None

    constraints = tuple(
        extra
        for extra in metadata
        if not isinstance(extra, SelectorMarker)
        and not _is_grammar(extra)
        and not isinstance(extra, FieldInfo)
    )
    arm = Arm(marker=markers[0], constraints=constraints)

    return arm


def _is_grammar(extra: Any) -> bool:
    patterns = (DATA_SELECTOR_PATTERN, STEP_SELECTOR_PATTERN)
    is_grammar = isinstance(extra, StringConstraints) and extra.pattern in patterns

    return is_grammar


def split_shared(annotation: Any) -> Tuple[Any, Tuple[Any, ...]]:
    """Separate outer metadata shared by every alternative of a position.

    Args:
        annotation: Position annotation.

    Returns:
        The annotation without outer metadata, and that metadata. A selector
        alternative keeps its own metadata (see ``selector_arm``).
    """
    if get_origin(annotation) is not Annotated or selector_arm(annotation):
        return annotation, ()

    base, *shared = get_args(annotation)

    return base, tuple(shared)


def union_members(annotation: Any) -> List[Any]:
    """Return the alternatives of a union, or the annotation itself."""
    if get_origin(annotation) in (Union, types.UnionType):
        members = list(get_args(annotation))
        return members

    return [annotation]


def container_of(annotation: Any) -> Tuple[Optional[ContainerKind], Any]:
    """Return the container kind and element annotation of an alternative.

    Args:
        annotation: One alternative, possibly with ``Annotated`` metadata.

    Returns:
        ``("list", element)``, ``("dict", value)`` or ``(None, None)``.
    """
    base, _ = split_shared(annotation)
    origin = get_origin(base)
    arguments = get_args(base)
    if origin is list and arguments:
        return "list", arguments[0]
    if origin is dict and len(arguments) == 2:
        return "dict", arguments[1]

    return None, None


def field_annotation(info: Any) -> Any:
    """Return the full annotation of a Pydantic field, including its metadata."""
    annotation = info.rebuild_annotation()

    return annotation


@dataclass(frozen=True)
class FieldStructure:
    """Selector positions of one field annotation.

    Args:
        whole: Marker when the whole field may be a selector.
        leaves: Marker for list elements or dict values.
        container: ``"list"`` or ``"dict"`` when ``leaves`` is set.
        literal_allowed: Whether the whole field accepts a non-selector value.
    """

    whole: Optional[SelectorMarker]
    leaves: Optional[SelectorMarker]
    container: Optional[ContainerKind]
    literal_allowed: bool


Fail = Callable[[str], DeclarationError]


def analyze_annotation(
    annotation: Any, *, field_name: str, fail: Fail
) -> FieldStructure:
    """Find the selector positions of one ``Params`` field.

    Args:
        annotation: Full field annotation, including ``Annotated`` metadata.
        field_name: Field name, for error messages.
        fail: Builds the ``DeclarationError`` to raise.

    Returns:
        The field's selector structure.

    Raises:
        DeclarationError: On ambiguous or too deeply nested selectors, or on
            leaves mixing roles with the whole field.
    """
    base, _ = split_shared(annotation)
    members = union_members(base)
    whole = _single_marker(members, field_name=field_name, fail=fail, where="field")

    leaves: Optional[SelectorMarker] = None
    container: Optional[ContainerKind] = None
    literal_members = [member for member in members if selector_arm(member) is None]
    for member in literal_members:
        member_container, element = container_of(member)
        element_base, _ = split_shared(element) if member_container else (None, ())
        element_members = union_members(element_base) if member_container else []
        element_marker = _single_marker(
            element_members, field_name=field_name, fail=fail, where="container element"
        )
        if element_marker is None:
            _reject_nested_markers(member, field_name=field_name, fail=fail)
            continue

        if leaves is not None:
            raise fail(
                f"field {field_name!r} has selectors inside two different containers; "
                "selector positions must be unambiguous"
            )
        for element_member in element_members:
            if selector_arm(element_member) is None:
                _reject_nested_markers(element_member, field_name=field_name, fail=fail)
        leaves, container = element_marker, member_container

    if whole is not None and leaves is not None and whole.role != leaves.role:
        raise fail(
            f"field {field_name!r} mixes selector roles {whole.role!r} and {leaves.role!r}"
        )
    for marker in (whole, leaves):
        if marker is not None and marker.role == "step" and marker.batch != "never":
            raise fail(f"field {field_name!r}: StepRef cannot request batch delivery")

    structure = FieldStructure(
        whole=whole,
        leaves=leaves,
        container=container,
        literal_allowed=bool(literal_members),
    )

    return structure


def contains_marker(annotation: Any, _seen: Optional[set] = None) -> bool:
    """Return whether any selector marker occurs inside ``annotation``.

    Args:
        annotation: Type annotation, possibly a Pydantic model.

    Returns:
        ``True`` when a marker is found at any depth.
    """
    seen = _seen if _seen is not None else set()
    if id(annotation) in seen:
        return False
    seen.add(id(annotation))

    if selector_arm(annotation) is not None:
        return True
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return any(
            contains_marker(field_annotation(info), seen)
            for info in annotation.model_fields.values()
        )

    contains = any(contains_marker(argument, seen) for argument in get_args(annotation))

    return contains


def _single_marker(
    members: List[Any], *, field_name: str, fail: Fail, where: str
) -> Optional[SelectorMarker]:
    arms = [arm for arm in map(selector_arm, members) if arm is not None]
    if len(arms) > 1:
        raise fail(
            f"field {field_name!r} has {len(arms)} selector alternatives for one "
            f"{where}; declare one Ref/Group/StepRef with several kinds instead"
        )
    if not arms:
        return None

    return arms[0].marker


def _reject_nested_markers(annotation: Any, *, field_name: str, fail: Fail) -> None:
    if contains_marker(annotation):
        raise fail(
            f"field {field_name!r} nests a selector deeper than one list or dict "
            "level; selectors may be the whole field or a direct list/dict element"
        )
