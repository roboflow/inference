from functools import partial
from typing import Any, Callable, Dict, List, Optional

from inference.core.workflows.core_steps.common.query_language.entities.operations import (
    TYPE_PARAMETER_NAME,
    DetectionsFilter,
    OperationDefinition,
    OperationsChain,
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
from inference.core.workflows.core_steps.common.query_language.operations.classification_results.base import (
    extract_classification_property,
)
from inference.core.workflows.core_steps.common.query_language.operations.detection.base import (
    extract_detection_property,
)
from inference.core.workflows.core_steps.common.query_language.operations.detections.base import (
    detections_to_dictionary,
    extract_detections_property,
    filter_detections,
    offset_detections,
    pick_detections_by_parent_class,
    rename_detections,
    select_detections,
    shift_detections,
    sort_detections,
)
from inference.core.workflows.core_steps.common.query_language.operations.dictionaries.base import (
    dictionary_to_json,
)
from inference.core.workflows.core_steps.common.query_language.operations.generic.base import (
    apply_lookup,
    generate_random_number,
)
from inference.core.workflows.core_steps.common.query_language.operations.images.base import (
    encode_image_to_base64,
    encode_image_to_jpeg,
    extract_image_property,
)
from inference.core.workflows.core_steps.common.query_language.operations.numbers.base import (
    divide,
    multiply,
    number_round,
    to_number,
)
from inference.core.workflows.core_steps.common.query_language.operations.sequences.base import (
    aggregate_numeric_sequence,
    aggregate_sequence,
    get_sequence_elements_count,
    get_sequence_length,
    sequence_apply,
    sequence_map,
)
from inference.core.workflows.core_steps.common.query_language.operations.strings.base import (
    string_matches,
    string_sub_sequence,
    string_to_lower,
    string_to_upper,
    to_string,
)


def execute_operations(
    value: T, operations: List[dict], global_parameters: Optional[Dict[str, Any]] = None
) -> V:
    operations_parsed = OperationsChain.model_validate({"operations": operations})
    ops_chain = build_operations_chain(operations_parsed.operations)
    if global_parameters is None:
        global_parameters = {}
    return ops_chain(value, global_parameters=global_parameters)


def identity(value: Any, **kwargs) -> Any:
    return value


def build_operations_chain(
    operations: List[OperationDefinition], execution_context: str = "<root>"
) -> Callable[[T, Dict[str, Any]], V]:
    if not len(operations):
        return identity  # return identity function
    operations_functions = []
    for operation_id, operation_definition in enumerate(operations):
        operation_context = f"{execution_context}[{operation_id}]"
        operation_function = build_operation(
            operation_definition=operation_definition,
            execution_context=operation_context,
        )
        operations_functions.append(operation_function)
    return partial(chain, functions=operations_functions)


def build_operation(
    operation_definition: OperationDefinition,
    execution_context: str,
) -> Callable[[T], V]:
    if operation_definition.type in REGISTERED_SIMPLE_OPERATIONS:
        return build_simple_operation(
            operation_definition=operation_definition,
            operation_function=REGISTERED_SIMPLE_OPERATIONS[operation_definition.type],
            execution_context=execution_context,
        )
    if operation_definition.type in REGISTERED_COMPOUND_OPERATIONS_BUILDERS:
        return REGISTERED_COMPOUND_OPERATIONS_BUILDERS[operation_definition.type](
            operation_definition, execution_context
        )
    raise OperationTypeNotRecognisedError(
        public_message=f"Attempted to build operation with declared type: {operation_definition.type} "
        f"which was not registered in Roboflow Query Language.",
        context="step_execution | roboflow_query_language_compilation",
    )


def build_simple_operation(
    operation_definition: OperationDefinition,
    operation_function: Callable[[T], V],
    execution_context: str,
) -> Callable[[T], V]:
    predefined_arguments_names = [
        t for t in type(operation_definition).model_fields if t != TYPE_PARAMETER_NAME
    ]
    kwargs = {a: getattr(operation_definition, a) for a in predefined_arguments_names}
    kwargs["execution_context"] = execution_context
    return partial(operation_function, **kwargs)


def build_sequence_apply_operation(
    definition: SequenceApply, execution_context: str
) -> Callable[[T], V]:
    operations_functions = []
    for operation in definition.operations:
        operation_function = build_operation(
            operation_definition=operation, execution_context=execution_context
        )
        operations_functions.append(operation_function)
    chained_function = partial(chain, functions=operations_functions)
    return partial(sequence_apply, fun=chained_function)


def chain(
    value: T, global_parameters: Dict[str, Any], functions: List[Callable[[T], V]]
) -> Callable[[T, Dict[str, Any]], V]:
    for function in functions:
        value = function(value, global_parameters=global_parameters)
    return value


def build_detections_filter_operation(
    definition: DetectionsFilter,
    execution_context: str,
) -> Callable[[T], V]:
    # local import to avoid circular dependency of modules with operations and evaluation
    from inference.core.workflows.core_steps.common.query_language.evaluation_engine.core import (
        build_eval_function,
    )

    filtering_fun = build_eval_function(
        definition=definition.filter_operation,
        execution_context=execution_context,
    )
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
    "SequenceAggregate": aggregate_sequence,
    "ExtractDetectionProperty": extract_detection_property,
    "DetectionsOffset": offset_detections,
    "DetectionsShift": shift_detections,
    "RandomNumber": generate_random_number,
    "StringMatches": string_matches,
    "SequenceLength": get_sequence_length,
    "SequenceElementsCount": get_sequence_elements_count,
    "ExtractImageProperty": extract_image_property,
    "Multiply": multiply,
    "Divide": divide,
    "DetectionsSelection": select_detections,
    "SortDetections": sort_detections,
    "ClassificationPropertyExtract": extract_classification_property,
    "DetectionsRename": rename_detections,
    "ConvertImageToJPEG": encode_image_to_jpeg,
    "ConvertImageToBase64": encode_image_to_base64,
    "DetectionsToDictionary": detections_to_dictionary,
    "ConvertDictionaryToJSON": dictionary_to_json,
    "PickDetectionsByParentClass": pick_detections_by_parent_class,
}

REGISTERED_COMPOUND_OPERATIONS_BUILDERS = {
    "SequenceApply": build_sequence_apply_operation,
    "DetectionsFilter": build_detections_filter_operation,
}
