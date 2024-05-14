from collections import Counter
from typing import Any, List

from inference.core.workflows.core_steps.common.query_language.entities.enums import (
    SequenceAggregationFunction,
    SequenceAggregationMode,
)
from inference.core.workflows.core_steps.common.query_language.errors import (
    InvalidInputTypeError,
)


def sequence_map(value: Any, lookup_table: dict) -> List[Any]:
    try:
        return [lookup_table[v] for v in value]
    except (TypeError, ValueError) as e:
        raise InvalidInputTypeError(
            public_message=f"While executing sequence_map(...), encountered "
            f"value of type {type(value)} which is not a sequence to be iterated",
            context="step_execution | roboflow_query_language_evaluation",
            inner_error=e,
        )
    except KeyError as e:
        raise InvalidInputTypeError(
            public_message=f"While executing operation sequence_map(...), encountered "
            f"value `{e}` which cannot be found in lookup "
            f"table with keys: {list(lookup_table.keys())}",
            context="step_execution | roboflow_query_language_evaluation",
            inner_error=e,
        )


def sequence_apply(value: Any, fun: callable) -> List[Any]:
    try:
        return [fun(v) for v in value]
    except (TypeError, ValueError) as e:
        raise InvalidInputTypeError(
            public_message=f"While executing sequence_apply(...), encountered "
            f"value of type {type(value)} which is not a sequence to be iterated",
            context="step_execution | roboflow_query_language_evaluation",
            inner_error=e,
        )


AGGREGATION_FUNCTIONS = {
    SequenceAggregationFunction.MIN: min,
    SequenceAggregationFunction.MAX: max,
}


def aggregate_numeric_sequence(
    value: Any, function: SequenceAggregationFunction
) -> Any:
    try:
        return AGGREGATION_FUNCTIONS[function](value)
    except (TypeError, ValueError) as e:
        raise InvalidInputTypeError(
            public_message=f"While executing aggregate_numeric_sequence(...), encountered "
            f"value of type {type(value)} which is not suited to execute operation. Details: {e}",
            context="step_execution | roboflow_query_language_evaluation",
            inner_error=e,
        )
    except KeyError as e:
        raise InvalidInputTypeError(
            public_message=f"While executing aggregate_numeric_sequence(...), "
            f"requested aggregation function {function.value} which is not supported.",
            context="step_execution | roboflow_query_language_evaluation",
            inner_error=e,
        )


def aggregate_sequence(value: Any, mode: SequenceAggregationMode) -> Any:
    try:
        if len(value) < 1:
            raise InvalidInputTypeError(
                public_message=f"While executing aggregate_sequence(...), encountered "
                f"value empty sequence which cannot be aggregated",
                context="step_execution | roboflow_query_language_evaluation",
            )
        if mode in {SequenceAggregationMode.FIRST, SequenceAggregationMode.LAST}:
            index = 0 if mode is SequenceAggregationMode.FIRST else -1
            return value[index]
        ctr_ordered = [e[0] for e in Counter(value).most_common()]
        index = 0 if mode is SequenceAggregationMode.MOST_COMMON else -1
        return ctr_ordered[index]
    except (TypeError, ValueError) as e:
        raise InvalidInputTypeError(
            public_message=f"While executing aggregate_sequence(...), encountered "
            f"value of type {type(value)} which is not suited to execute operation. Details: {e}",
            context="step_execution | roboflow_query_language_evaluation",
            inner_error=e,
        )


def get_sequence_length(value: Any) -> int:
    try:
        return len(value)
    except TypeError as e:
        raise InvalidInputTypeError(
            public_message=f"While executing get_sequence_length(...), encountered "
                           f"value of type {type(value)} which is not suited to execute operation. Details: {e}",
            context="step_execution | roboflow_query_language_evaluation",
            inner_error=e,
        )
