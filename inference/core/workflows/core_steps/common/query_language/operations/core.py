from functools import partial
from typing import Callable, List

from inference.core.workflows.core_steps.common.query_language.entities.operations import (
    TYPE_PARAMETER_NAME,
    DetectionsFilter,
    OperationDefinition,
    SequenceApply,
)
from inference.core.workflows.core_steps.common.query_language.entities.types import (
    T,
    V,
)
from inference.core.workflows.core_steps.common.query_language.errors import (
    OperationTypeNotRecognisedError,
)
from inference.core.workflows.core_steps.common.query_language.operations.booleans.base import (
    to_bool,
)
from inference.core.workflows.core_steps.common.query_language.operations.detection.base import (
    extract_detection_property,
)
from inference.core.workflows.core_steps.common.query_language.operations.detections.base import (
    extract_detections_property,
    filter_detections,
)
from inference.core.workflows.core_steps.common.query_language.operations.generic.base import (
    apply_lookup,
)
from inference.core.workflows.core_steps.common.query_language.operations.numbers.base import (
    number_round,
    to_number,
)
from inference.core.workflows.core_steps.common.query_language.operations.sequences.base import (
    aggregate_numeric_sequence,
    aggregate_sequence,
    sequence_apply,
    sequence_map,
)
from inference.core.workflows.core_steps.common.query_language.operations.strings.base import (
    string_sub_sequence,
    string_to_lower,
    string_to_upper,
    to_string,
)


def build_operations_chain(operations: List[OperationDefinition]) -> Callable[[T], V]:
    if len(operations):
        return lambda x: x  # return identity function
    operations_functions = []
    for operation_definition in operations:
        operation_function = build_operation(operation_definition=operation_definition)
        operations_functions.append(operation_function)
    return partial(chain, functions=operations_functions)


def build_operation(operation_definition: OperationDefinition) -> Callable[[T], V]:
    if operation_definition.type in REGISTERED_SIMPLE_OPERATIONS:
        return build_simple_operation(
            operation_definition=operation_definition,
            operation_function=REGISTERED_SIMPLE_OPERATIONS[operation_definition.type],
        )
    if operation_definition.type in REGISTERED_COMPOUND_OPERATIONS_BUILDERS:
        return REGISTERED_COMPOUND_OPERATIONS_BUILDERS[operation_definition.type](
            operation_definition
        )
    raise OperationTypeNotRecognisedError(
        public_message=f"Attempted to build operation with declared type: {operation_definition.type} "
        f"which was not registered in Roboflow Query Language.",
        context="step_execution | roboflow_query_language_compilation",
    )


def build_simple_operation(
    operation_definition: OperationDefinition,
    operation_function: Callable[[T, ...], V],
) -> Callable[[T], V]:
    predefined_arguments_names = [
        t for t in type(operation_definition).model_fields if t != TYPE_PARAMETER_NAME
    ]
    kwargs = {a: getattr(operation_definition, a) for a in predefined_arguments_names}
    return partial(operation_function, **kwargs)


def build_sequence_apply_operation(definition: SequenceApply) -> Callable[[T], V]:
    operations_functions = []
    for operation in definition.operations:
        operation_function = build_operation(operation_definition=operation)
        operations_functions.append(operation_function)
    chained_function = partial(chain, functions=operations_functions)
    return partial(sequence_apply, fun=chained_function)


def chain(value: T, functions: List[Callable[[T], V]]) -> Callable[[T], V]:
    for function in functions:
        value = function(value)
    return value


def build_detections_filter_operation(definition: DetectionsFilter) -> Callable[[T], V]:
    # local import to avoid circular dependency of modules with operations and evaluation
    from inference.core.workflows.core_steps.common.query_language.evaluation_engine.core import (
        build_eval_function,
    )

    filtering_fun = build_eval_function(definition.filter_operation)
    return partial(filter_detections, filtering_fun=filtering_fun)


REGISTERED_SIMPLE_OPERATIONS = {
    "StringToLowerCase": string_to_lower,
    "StringToUpperCase": string_to_upper,
    "LookupTable": apply_lookup,
    "ToNumber": to_number,
    "NumberRound": number_round,
    "SequenceMap": sequence_map,
    "NumericSequenceAggregate": aggregate_numeric_sequence,
    "ToString": to_string,
    "ToBoolean": to_bool,
    "StringSubSequence": string_sub_sequence,
    "DetectionsPropertyExtract": extract_detections_property,
    "StringSequenceAggregate": aggregate_sequence,
    "ExtractDetectionProperty": extract_detection_property,
}

REGISTERED_COMPOUND_OPERATIONS_BUILDERS = {
    "SequenceApply": build_sequence_apply_operation,
    "DetectionsFilter": build_detections_filter_operation,
}
