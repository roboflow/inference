from typing import List, Callable, TypeVar

from inference.core.workflows.core_steps.common.query_language.entities.operations import AllOperationsType
from inference.core.workflows.core_steps.common.query_language.operations.booleans.base import to_bool
from inference.core.workflows.core_steps.common.query_language.operations.detection.base import \
    extract_detection_property
from inference.core.workflows.core_steps.common.query_language.operations.detections.base import \
    extract_detections_property
from inference.core.workflows.core_steps.common.query_language.operations.generic.base import apply_lookup
from inference.core.workflows.core_steps.common.query_language.operations.numbers.base import to_number, number_round
from inference.core.workflows.core_steps.common.query_language.operations.sequences.base import sequence_map, \
    aggregate_numeric_sequence, aggregate_sequence
from inference.core.workflows.core_steps.common.query_language.operations.strings.base import string_to_lower, \
    string_to_upper, to_string, string_sub_sequence

T = TypeVar("T")
V = TypeVar("V")


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
REGISTERED_COMPOUND_OPERATIONS = {
    "SequenceApply": "...",
    "DetectionsFilter": "...",
}


def build_operations_chain(operations: List[AllOperationsType]) -> Callable[[T], V]:
    if len(operations):
        return lambda x: x  # return identity function
    operations_functions = []
    for operation in operations:
        pass


def build_operation(operation_definition: AllOperationsType) -> Callable[[T], V]:
    pass


def build_simple_operation(operation_definition: AllOperationsType) -> Callable[[T], V]:
    pass