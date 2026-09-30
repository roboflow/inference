"""Validation of step parameters against the class-owned ``Params`` model.

The author writes one ``Params`` model. Two cached Pydantic subclasses of it,
built with public Pydantic APIs, apply the parameter policy (decision 018):

* The *definition* model validates the step as written. At a selector
  position, a string that looks like a selector skips the position's payload
  constraints; they apply to the selected value later. Literals and defaults
  are parsed, constrained and normalized here, once.
* The *resolved* model checks one logical invocation after selectors were
  resolved. A selector alternative becomes a kind check that returns the value
  itself and is tried before any literal alternative, so a selected payload is
  never narrowed to a literal type or copied. Constraints shared by the whole
  position (outer ``Field`` metadata) and the author's field/model validators
  then run on the resolved values. Fields without selectors are not
  revalidated, so literal normalization is not replayed.

Only fields containing selectors are redefined in the subclasses; aliases,
defaults, descriptions and validators are inherited from the author's model.
Runtime validators must be pure checks: a validator that replaces a selected
payload is rejected.
"""

import copy
from typing import (
    Annotated,
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PlainValidator,
    TypeAdapter,
    ValidationError,
    WrapValidator,
    create_model,
)
from pydantic_core import PydanticCustomError
from roboflow_workflows.execution_engine.v2._selectors import (
    SELECTOR_PREFIXES,
    Arm,
    SelectorMarker,
    container_of,
    contains_marker,
    field_annotation,
    selector_arm,
    split_shared,
    union_members,
)
from roboflow_workflows.execution_engine.v2.errors import (
    ContractError,
    FieldPath,
    ResolvedParameterError,
)

SelectedLeaf = Tuple[FieldPath, SelectorMarker]
"""Field path of one selector in a step definition and its declaration."""


class ParamsValidator:
    """Definition and resolved-invocation validation for one ``Params`` model.

    Args:
        params_model: The author's ``Params`` model.
    """

    def __init__(self, params_model: type):
        self._model = params_model
        selector_fields = {
            name: info
            for name, info in params_model.model_fields.items()
            if contains_marker(field_annotation(info))
        }
        self.definition_model = _subclass(
            params_model, selector_fields, runtime=False, config=None
        )
        self.resolved_model = _subclass(
            params_model,
            selector_fields,
            runtime=True,
            config=ConfigDict(frozen=False, validate_assignment=True),
        )
        self._position_adapters = _position_adapters(selector_fields)

    def validate_definition(self, values: Mapping[str, Any]) -> BaseModel:
        """Validate step parameters as written, keeping selectors as strings.

        Args:
            values: Parameters of the step.

        Returns:
            An instance of the definition model (a subclass of ``Params``).

        Raises:
            ValidationError: When a literal or the model is invalid.
            TypeError: When a declared constraint cannot apply to a literal.
        """
        params = self.definition_model.model_validate(values)

        return params

    def check_resolved(
        self,
        params: BaseModel,
        arguments: Mapping[str, Any],
        selected: Sequence[SelectedLeaf],
    ) -> None:
        """Check one logical invocation whose selectors were resolved.

        Args:
            params: Definition-validated parameters.
            arguments: One value per ``Params`` field; selected leaves hold
                the resolved payloads (``None`` when unavailable).
            selected: Field paths and markers of the data selectors.

        Raises:
            ResolvedParameterError: On a wrong kind, a violated constraint or
                validator, or a validator that replaced a selected payload.
                Any other ``Exception`` raised by a validator is located at
                the field being assigned (its model validators included) and
                kept as ``__cause__``.
        """
        for path, marker in selected:
            value = _value_at(arguments, path)
            if value is None:
                continue
            try:
                marker.check_payload(value)
            except ContractError as error:
                raise ResolvedParameterError(
                    f"parameter {_render(path)} is not a valid "
                    f"{list(marker.kind_names)}: {error}",
                    field_path=path,
                ) from error

        field_names = list(dict.fromkeys(path[0] for path, _ in selected))
        if not field_names:
            return

        instance = self.resolved_model.model_construct(
            _fields_set=set(params.model_fields_set),
            **{name: arguments[name] for name in self._model.model_fields},
        )
        for name in field_names:
            try:
                setattr(instance, name, arguments[name])
            except Exception as error:
                # Pydantic lets validator exceptions other than ValueError and
                # AssertionError propagate unwrapped.
                raise _resolved_error(error, field_name=name, arguments=arguments)

        for path, _ in selected:
            if _value_at(instance.__dict__, path) is not _value_at(arguments, path):
                raise ResolvedParameterError(
                    f"a validator of parameter {_render(path)} replaced the selected "
                    "payload; runtime validators must return the value they check",
                    field_path=path,
                )

    def check_value(
        self, field_name: str, value: Any, *, position: Tuple[Any, ...]
    ) -> None:
        """Check one selected value against its position alone.

        Applies the position's kinds and shared constraints, without the
        author's field or model validators.

        Args:
            field_name: Field holding the selector.
            value: Resolved value.
            position: ``()`` for the whole field or ``(index_or_key,)``.

        Raises:
            ResolvedParameterError: On a violation.
        """
        adapter = self._position_adapters.get((field_name, bool(position)))
        if adapter is None:
            return

        path = (field_name,) + tuple(position)
        try:
            adapter.validate_python(value)
        except Exception as error:
            raise ResolvedParameterError(
                f"parameter {_render(path)} received {value!r}: {_first_message(error)}",
                field_path=path,
            ) from error


def clean_errors(
    error: ValidationError, values: Mapping[str, Any]
) -> List[Tuple[FieldPath, str]]:
    """Turn Pydantic errors into (field path, message) pairs.

    Pydantic locations name union branches (``int``, ``constrained-str``,
    ``function-wrap[...]``); only the parts that address the parameters as
    written are kept, and the messages of one path are combined. A missing
    field keeps its whole location: the input path it should be written at
    (for an ``AliasPath``, the nested keys).

    Args:
        error: The validation error.
        values: The validated input, keyed like the error locations.

    Returns:
        One entry per distinct field path, in first-seen order.
    """
    messages: Dict[FieldPath, List[str]] = {}
    for item in error.errors():
        if item["type"] == "missing":
            path = tuple(item["loc"])
        else:
            path = _written_path(item["loc"], values)
        message = item["msg"]
        bucket = messages.setdefault(path, [])
        if message not in bucket:
            bucket.append(message)

    cleaned = [(path, " or ".join(texts)) for path, texts in messages.items()]

    return cleaned


def _written_path(location: Sequence[Any], values: Any) -> FieldPath:
    path: List[Any] = []
    current = values
    for part in location:
        if isinstance(current, Mapping) and part in current:
            path.append(part)
            current = current[part]
        elif (
            isinstance(current, list) and isinstance(part, int) and part < len(current)
        ):
            path.append(part)
            current = current[part]

    written = tuple(path) if path else tuple(location[:1])

    return written  # () for a model-level error


def _resolved_error(
    error: Exception, *, field_name: str, arguments: Mapping[str, Any]
) -> ResolvedParameterError:
    if isinstance(error, ValidationError):
        # A field error is located at that field; a model validator at ().
        path, message = clean_errors(error, arguments)[0]
    elif isinstance(error, TypeError):
        path, message = (field_name,), str(error)
    else:
        path, message = (field_name,), f"{type(error).__name__}: {error}"
    subject = f"parameter {_render(path)} is" if path else "parameters are"
    resolved_error = ResolvedParameterError(
        f"{subject} invalid after resolution: {message}", field_path=path
    )
    resolved_error.__cause__ = error

    return resolved_error


def _first_message(error: Exception) -> str:
    if isinstance(error, ValidationError) and error.errors():
        return error.errors()[0]["msg"]
    if isinstance(error, TypeError):
        return str(error)

    return f"{type(error).__name__}: {error}"


def _render(path: FieldPath) -> str:
    rendered = ".".join(str(part) for part in path)

    return rendered


def _value_at(values: Mapping[str, Any], path: FieldPath) -> Any:
    current: Any = values
    for part in path:
        current = current[part]

    return current


def _subclass(
    model: type,
    fields: Mapping[str, Any],
    *,
    runtime: bool,
    config: Optional[ConfigDict],
) -> type:
    definitions = {}
    for name, info in fields.items():
        redefined = copy.copy(info)
        redefined.metadata = []  # already part of the projected annotation
        annotation = _project_position(field_annotation(info), runtime=runtime)
        definitions[name] = (annotation, redefined)

    subclass = create_model(
        model.__name__,
        __base__=model,
        __cls_kwargs__=config,
        __module__=model.__module__,
        **definitions,
    )

    return subclass


def _position_adapters(
    fields: Mapping[str, Any],
) -> Dict[Tuple[str, bool], TypeAdapter]:
    adapters: Dict[Tuple[str, bool], TypeAdapter] = {}
    for name, info in fields.items():
        annotation = field_annotation(info)
        adapters[(name, False)] = TypeAdapter(
            _project_position(annotation, runtime=True)
        )
        base, _ = split_shared(annotation)
        for member in union_members(base):
            _, element = container_of(member)
            if element is not None and contains_marker(element):
                adapters[(name, True)] = TypeAdapter(
                    _project_position(element, runtime=True)
                )

    return adapters


def _project_position(annotation: Any, *, runtime: bool) -> Any:
    """Rewrite one selector position for definition or resolved validation."""
    base, shared = split_shared(annotation)
    members = [member for member in union_members(base) if member is not type(None)]
    nullable = len(members) < len(union_members(base))
    if runtime:
        members.sort(key=_runtime_preference)
    members = [_project_member(member, runtime=runtime) for member in members]

    parts: List[Any] = [members[0] if len(members) == 1 else Union[tuple(members)]]
    if runtime and len(members) > 1:
        parts.append(Field(union_mode="left_to_right"))  # before shared constraints
    # Shared constraints apply to non-null values, as for Pydantic ``Optional[X]``.
    parts.extend(shared)
    projected = _annotated(parts)
    if nullable:
        projected = Optional[projected]

    wrap = _skip_unavailable if runtime else _defer_selector_constraints
    positioned = _annotated([projected, WrapValidator(wrap)])

    return positioned


def _project_member(member: Any, *, runtime: bool) -> Any:
    arm = selector_arm(member)
    if arm is not None:
        projected_arm = _resolved_arm(arm) if runtime else _definition_arm(arm)
        return projected_arm

    container, element = container_of(member)
    if container is None or not contains_marker(element):
        return member

    base, shared = split_shared(member)
    projected_element = _project_position(element, runtime=runtime)
    rebuilt = (
        List[projected_element]
        if container == "list"
        else Dict[base.__args__[0], projected_element]
    )
    projected_member = _annotated([rebuilt, *shared])

    return projected_member


def _runtime_preference(member: Any) -> int:
    # Selector arms first (they return the payload itself), then containers
    # holding selector leaves, then literal-only alternatives.
    if selector_arm(member) is not None:
        return 0
    if contains_marker(member):
        return 1

    return 2


def _definition_arm(arm: Arm) -> Any:
    # Only a complete selector passes. Payload constraints of the arm never
    # apply to the selector string itself.
    annotation = _annotated([Any, PlainValidator(_selector_check(arm.marker))])

    return annotation


def _selector_check(marker: SelectorMarker) -> Callable[[Any], Any]:
    expected = (
        "$steps.<step>"
        if marker.role == "step"
        else "$inputs.<name> or $steps.<step>.<output>"
    )

    def check(value: Any) -> Any:
        if not marker.matches(value):
            raise PydanticCustomError(
                "selector",
                "Input should be a selector like {expected}",
                {"expected": expected},
            )
        return value

    return check


def _resolved_arm(arm: Arm) -> Any:
    if arm.marker.role == "step":
        return _definition_arm(arm)  # control targets stay selector strings

    annotation = _annotated(
        [Any, PlainValidator(_payload_check(arm.marker)), *arm.constraints]
    )

    return annotation


def _payload_check(marker: SelectorMarker) -> Callable[[Any], Any]:
    def check(value: Any) -> Any:
        marker.check_payload(value)
        return value

    return check


def _defer_selector_constraints(value: Any, handler: Callable[[Any], Any]) -> Any:
    # Selector grammar is checked by the declaration after model validation.
    if isinstance(value, str) and value.startswith(SELECTOR_PREFIXES):
        return value

    return handler(value)


def _skip_unavailable(value: Any, handler: Callable[[Any], Any]) -> Any:
    # An unavailable selected leaf arrives as None; there is nothing to check.
    if value is None:
        return None

    return handler(value)


def _annotated(parts: Sequence[Any]) -> Any:
    """Build ``Annotated[parts[0], *parts[1:]]``, or ``parts[0]`` alone."""
    if len(parts) == 1:
        return parts[0]

    annotated = Annotated[tuple(parts)]

    return annotated
