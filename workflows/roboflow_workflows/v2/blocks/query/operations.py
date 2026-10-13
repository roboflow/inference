"""Compile and run UQL operation chains.

Each operation literal maps to one function in ``_LEAF_OPERATIONS``, except
three that need more than their input value:

    SequenceApply      a nested chain, applied per element
    DetectionsFilter   a nested condition, evaluated per detection
    DetectionsRename   runtime parameters named by class_map / strict

Generic operations below use plain Python and follow the current V1 operation
bodies, including their edge cases. Detection, classification, image and frame
operations live in ``detection_ops``, ``detections`` and ``media``.
"""

import json
import random
import re
from collections import Counter
from datetime import datetime
from typing import AbstractSet, Any, Callable, Dict, Mapping, Optional, Sequence, Union

from pydantic import BaseModel
from roboflow_workflows.v2.blocks.query import detection_ops, media
from roboflow_workflows.v2.blocks.query.detections import (
    detection_row,
    detections_property,
    is_detections,
    row_count,
)
from roboflow_workflows.v2.blocks.query.errors import (
    OperationFailure,
    QueryDefinitionError,
    QueryRuntimeError,
    describe_value,
    preview_value,
)
from roboflow_workflows.v2.blocks.query.schema import (
    DEFAULT_OPERAND_NAME,
    OPERATION_LIST,
    DetectionsFilter,
    DetectionsRename,
    SequenceApply,
    validate_definition,
)

MAX_JOINED_STRING_LENGTH = 1_000_000

# Operations whose output is the same detections carrier as their input.
DETECTIONS_PRESERVING_OPERATIONS = frozenset(
    {
        "DetectionsFilter",
        "DetectionsSelection",
        "SortDetections",
        "DetectionsOffset",
        "DetectionsShift",
        "DetectionsRename",
        "PickDetectionsByParentClass",
    }
)

# (value, runtime parameters) -> new value
ChainFunction = Callable[[Any, Mapping[str, Any]], Any]


class CompiledOperations:
    """Validated operation chain, ready to apply to runtime values."""

    def __init__(self, chain: ChainFunction) -> None:
        self._chain = chain

    def apply(self, value: Any, parameters: Optional[Mapping[str, Any]] = None) -> Any:
        """Apply the chain to ``value``.

        Args:
            value: Input of the first operation.
            parameters: Runtime named values, forwarded to nested operations.

        Returns:
            Output of the last operation, or ``value`` for an empty chain.

        Raises:
            QueryRuntimeError: If an operation cannot process its input.
        """
        result = self._chain(value, parameters or {})

        return result


def compile_operations(
    operations: Sequence[Union[BaseModel, dict]],
    *,
    path: str = "$",
    declared_names: Optional[AbstractSet[str]] = None,
) -> CompiledOperations:
    """Validate a literal operation list once and compile it.

    Args:
        operations: A list of operation JSON objects or validated models.
        path: JSON path of the list, used in error messages.
        declared_names: Runtime parameter names the caller will pass to
            ``apply``. When given, a nested filter operand or a string
            ``class_map``/``strict`` naming anything else is a definition error.

    Returns:
        Compiled chain; call ``apply`` per runtime value.

    Raises:
        QueryDefinitionError: If ``operations`` is not a list, is invalid or
            names an undeclared parameter.
    """
    if not isinstance(operations, (list, tuple)):
        raise QueryDefinitionError(
            f"operations must be a list of operation objects, got "
            f"{type(operations).__name__}",
            path=path,
        )
    parsed = validate_definition(OPERATION_LIST, list(operations), path=path)
    chain = compile_operation_chain(parsed, path=path, declared_names=declared_names)
    compiled = CompiledOperations(chain)

    return compiled


def compile_operation_chain(
    operations: Sequence[BaseModel],
    *,
    path: str,
    declared_names: Optional[AbstractSet[str]] = None,
) -> ChainFunction:
    """Compile validated operation models into one chain function.

    Args:
        operations: Validated operation models.
        path: JSON path of the operation list.
        declared_names: Declared runtime parameter names, or ``None`` to defer
            name checks to run time.

    Returns:
        Function ``(value, parameters) -> value`` applying operations in order.
    """
    steps = [
        _compile_operation(
            operation, path=f"{path}[{index}]", declared_names=declared_names
        )
        for index, operation in enumerate(operations)
    ]

    def apply_chain(value: Any, parameters: Mapping[str, Any]) -> Any:
        for step in steps:
            value = step(value, parameters)

        return value

    return apply_chain


def _compile_operation(
    operation: BaseModel, *, path: str, declared_names: Optional[AbstractSet[str]]
) -> ChainFunction:
    if isinstance(operation, SequenceApply):
        apply_to_element = compile_operation_chain(
            operation.operations,
            path=f"{path}.operations",
            declared_names=declared_names,
        )

        def function(value: Any, parameters: Mapping[str, Any]) -> Any:
            return [apply_to_element(element, parameters) for element in value]

    elif isinstance(operation, DetectionsFilter):
        # Imported here: conditions compile operation chains through this module.
        from roboflow_workflows.v2.blocks.query.conditions import compile_statement

        keep_row = compile_statement(
            operation.filter_operation,
            path=f"{path}.filter_operation",
            declared_names=(
                None
                if declared_names is None
                else {*declared_names, DEFAULT_OPERAND_NAME}
            ),
        )

        def function(value: Any, parameters: Mapping[str, Any]) -> Any:
            return detection_ops.filter_detections(value, keep_row, parameters)

    elif isinstance(operation, DetectionsRename):
        _check_parameter_names(operation, path=path, declared_names=declared_names)

        def function(value: Any, parameters: Mapping[str, Any]) -> Any:
            return detection_ops.rename_detections(operation, value, parameters)

    else:
        leaf = _LEAF_OPERATIONS[operation.type]

        def function(value: Any, parameters: Mapping[str, Any]) -> Any:
            return leaf(operation, value)

    def run_step(value: Any, parameters: Mapping[str, Any]) -> Any:
        try:
            result = function(value, parameters)
        except QueryRuntimeError:
            raise
        except Exception as error:
            parameter = error.parameter if isinstance(error, OperationFailure) else None
            raise QueryRuntimeError(
                f"cannot apply to {describe_value(value)}: {error}",
                path=path,
                operation=operation.type,
                parameter=parameter,
            ) from error

        return result

    return run_step


def _check_parameter_names(
    operation: DetectionsRename,
    *,
    path: str,
    declared_names: Optional[AbstractSet[str]],
) -> None:
    if declared_names is None:
        return

    for parameter in ("class_map", "strict"):
        name = getattr(operation, parameter)
        if isinstance(name, str) and name not in declared_names:
            raise QueryDefinitionError(
                f"runtime parameter `{name}` is not declared "
                f"(declared: {sorted(declared_names)})",
                path=f"{path}.{parameter}",
                operation=operation.type,
                parameter=parameter,
            )


def _require(value: Any, expected: type, name: str) -> Any:
    if not isinstance(value, expected):
        raise OperationFailure(f"expected {name}")

    return value


def _lookup(table: dict, key: Any) -> Any:
    try:
        return table[key]
    except KeyError:
        raise OperationFailure(
            f"`{preview_value(key)}` is not a key of the lookup table "
            f"(keys: {list(table)})",
            parameter="lookup_table",
        ) from None


def _string_sub_sequence(operation, value):
    # The default end=-1 drops the last character, as in V1.
    text = _require(value, str, "a string")

    return text[operation.start : operation.end]


def _to_number(operation, value):
    # V1 always parses through float, so int("3.7") style input becomes 3.
    number = float(value)
    if operation.cast_to == "int":
        number = int(number)

    return number


def _divide(operation, value):
    try:
        return value / operation.other
    except ZeroDivisionError:
        raise OperationFailure("division by zero", parameter="other") from None


def _numeric_aggregate(operation, value):
    if len(value) == 0:
        return operation.neutral_value
    aggregate = {"min": min, "max": max, "sum": sum}[operation.function]

    return aggregate(value)


def _sequence_aggregate(operation, value):
    if len(value) == 0:
        raise OperationFailure("an empty sequence cannot be aggregated")
    if operation.mode == "first":
        return value[0]
    if operation.mode == "last":
        return value[-1]
    # Counter keeps first-encounter order among equal counts.
    by_count = [element for element, _ in Counter(value).most_common()]

    return by_count[0] if operation.mode == "most_common" else by_count[-1]


def _count_elements(operation, value):
    counts: Dict[str, int] = {}
    for element in value:
        key = preview_value(element)
        counts[key] = counts.get(key, 0) + 1

    return counts


def _join(operation, value):
    parts = [str(element) for element in value]
    length = sum(len(part) for part in parts)
    length += len(operation.separator) * max(len(parts) - 1, 0)
    if length > MAX_JOINED_STRING_LENGTH:
        raise OperationFailure(
            f"joined string would have {length} characters, above the "
            f"{MAX_JOINED_STRING_LENGTH} character limit"
        )

    return operation.separator.join(parts)


def _random_number(operation, value):
    if operation.min_value == operation.max_value:
        return operation.min_value
    # V1 offsets from min_value even when it exceeds max_value.
    span = abs(operation.max_value - operation.min_value)

    return operation.min_value + random.random() * span


def _dictionary_to_json(operation, value):
    _require(value, dict, "a dictionary")

    return json.dumps(value)


def _timestamp_to_iso(operation, value):
    _require(value, datetime, "a datetime timestamp")

    return value.isoformat()


_LEAF_OPERATIONS: Dict[str, Callable[[BaseModel, Any], Any]] = {
    "StringToLowerCase": lambda op, value: _require(value, str, "a string").lower(),
    "StringToUpperCase": lambda op, value: _require(value, str, "a string").upper(),
    "StringSubSequence": _string_sub_sequence,
    "StringMatches": lambda op, value: bool(
        re.search(op.regex, _require(value, str, "a string"))
    ),
    "ToString": lambda op, value: str(value),
    "ToNumber": _to_number,
    "NumberRound": lambda op, value: round(value, op.decimal_digits),
    "ToBoolean": lambda op, value: bool(value),
    "Multiply": lambda op, value: value * op.other,
    "Divide": _divide,
    "LookupTable": lambda op, value: _lookup(op.lookup_table, value),
    "SequenceMap": lambda op, value: [_lookup(op.lookup_table, v) for v in value],
    "NumericSequenceAggregate": _numeric_aggregate,
    "SequenceAggregate": _sequence_aggregate,
    "SequenceElementsCount": _count_elements,
    "SequenceJoin": _join,
    "SequenceLength": lambda op, value: sequence_length(value),
    "RandomNumber": _random_number,
    "ConvertDictionaryToJSON": _dictionary_to_json,
    "TimestampToISOFormat": _timestamp_to_iso,
    "DetectionsPropertyExtract": lambda op, value: detections_property(
        value, op.property_name
    ),
    "ExtractDetectionProperty": lambda op, value: detection_row(value).property(
        op.property_name
    ),
    "DetectionsSelection": detection_ops.select_detection,
    "SortDetections": detection_ops.sort_detections,
    "DetectionsOffset": detection_ops.offset_detections,
    "DetectionsShift": detection_ops.shift_detections,
    "PickDetectionsByParentClass": detection_ops.pick_detections_by_parent_class,
    "DetectionsToDictionary": detection_ops.detections_to_dictionary,
    "ClassificationPropertyExtract": media.classification_property,
    "ExtractImageProperty": media.image_property,
    "ExtractFrameMetadata": media.frame_metadata,
    "ConvertImageToJPEG": media.image_to_jpeg,
    "ConvertImageToBase64": media.image_to_base64,
}


def sequence_length(value: Any) -> int:
    """``len`` of a value; a keypoint prediction counts its skeletons, not 2."""
    length = row_count(value) if is_detections(value) else len(value)

    return length
