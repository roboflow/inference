from collections import Counter
from typing import Any, List

from inference.core.workflows.core_steps.common.query_language.entities.enums import (
    SequenceAggregationFunction,
    SequenceAggregationMode,
)
from inference.core.workflows.core_steps.common.query_language.errors import (
    InvalidInputTypeError,
)


def sequence_map(
    value: Any, lookup_table: dict, execution_context: str, **kwargs
) -> List[Any]:
    try:
        return [lookup_table[v] for v in value]
    except (TypeError, ValueError) as e:
        raise InvalidInputTypeError(
            public_message=f"While executing sequence_map(...) in context {execution_context}, encountered "
            f"value of type {type(value)} which is not a sequence to be iterated",
            context=f"step_execution | roboflow_query_language_evaluation | {execution_context}",
            inner_error=e,
        )
    except KeyError as e:
        raise InvalidInputTypeError(
            public_message=f"While executing operation sequence_map(...) in context {execution_context}, encountered "
            f"value `{e}` which cannot be found in lookup "
            f"table with keys: {list(lookup_table.keys())}",
            context=f"step_execution | roboflow_query_language_evaluation | {execution_context}",
            inner_error=e,
        )


def sequence_apply(
    value: Any, fun: callable, execution_context: str, **kwargs
) -> List[Any]:
    try:
        return [fun(v) for v in value]
    except (TypeError, ValueError) as e:
        raise InvalidInputTypeError(
            public_message=f"While executing sequence_apply(...) in context {execution_context}, encountered "
            f"value of type {type(value)} which is not a sequence to be iterated",
            context=f"step_execution | roboflow_query_language_evaluation | {execution_context}",
            inner_error=e,
        )


AGGREGATION_FUNCTIONS = {
    SequenceAggregationFunction.MIN: min,
    SequenceAggregationFunction.MAX: max,
}


def aggregate_numeric_sequence(
    value: Any,
    function: SequenceAggregationFunction,
    neutral_value: Any,
    execution_context: str,
    **kwargs,
) -> Any:
    try:
        if len(value) == 0:
            return neutral_value
        return AGGREGATION_FUNCTIONS[function](value)
    except (TypeError, ValueError) as e:
        raise InvalidInputTypeError(
            public_message=f"While executing aggregate_numeric_sequence(...) in context {execution_context}, "
            f"encountered value of type {type(value)} which is not suited to execute operation. "
            f"Details: {e}",
            context=f"step_execution | roboflow_query_language_evaluation | {execution_context}",
            inner_error=e,
        )
    except KeyError as e:
        raise InvalidInputTypeError(
            public_message=f"While executing aggregate_numeric_sequence(...) in context {execution_context}, "
            f"requested aggregation function {function.value} which is not supported.",
            context=f"step_execution | roboflow_query_language_evaluation | {execution_context}",
            inner_error=e,
        )


def aggregate_sequence(
    value: Any, mode: SequenceAggregationMode, execution_context: str, **kwargs
) -> Any:
    try:
        if len(value) < 1:
            raise InvalidInputTypeError(
                public_message=f"While executing aggregate_sequence(...) in context {execution_context}, encountered "
                f"value empty sequence which cannot be aggregated",
                context=f"step_execution | roboflow_query_language_evaluation | {execution_context}",
            )
        if mode in {SequenceAggregationMode.FIRST, SequenceAggregationMode.LAST}:
            index = 0 if mode is SequenceAggregationMode.FIRST else -1
            return value[index]
        ctr_ordered = [e[0] for e in Counter(value).most_common()]
        index = 0 if mode is SequenceAggregationMode.MOST_COMMON else -1
        return ctr_ordered[index]
    except (TypeError, ValueError) as e:
        raise InvalidInputTypeError(
            public_message=f"While executing aggregate_sequence(...) in context {execution_context}, encountered "
            f"value of type {type(value)} which is not suited to execute operation. Details: {e}",
            context=f"step_execution | roboflow_query_language_evaluation | {execution_context}",
            inner_error=e,
        )


def get_sequence_length(value: Any, execution_context: str, **kwargs) -> int:
    try:
        return len(value)
    except TypeError as e:
        raise InvalidInputTypeError(
            public_message=f"While executing get_sequence_length(...) in context {execution_context}, encountered "
            f"value of type {type(value)} which is not suited to execute operation. Details: {e}",
            context=f"step_execution | roboflow_query_language_evaluation | {execution_context}",
            inner_error=e,
        )
