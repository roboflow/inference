import random
from typing import Any

from inference.core.workflows.core_steps.common.query_language.errors import (
    InvalidInputTypeError,
)
from inference.core.workflows.core_steps.common.query_language.operations.utils import (
    safe_stringify,
)


def apply_lookup(
    value: Any, lookup_table: dict, execution_context: str, **kwargs
) -> Any:
    try:
        return lookup_table[value]
    except TypeError as e:
        raise InvalidInputTypeError(
            public_message=f"While executing operation apply_lookup(...) in context {execution_context}, "
            f"passed value of type {type(value)} which does not let it be searched in lookup table.",
            context=f"step_execution | roboflow_query_language_evaluation | {execution_context}",
            inner_error=e,
        )
    except KeyError as e:
        value_as_str = safe_stringify(value=value)
        raise InvalidInputTypeError(
            public_message=f"While executing operation apply_lookup(...) in context {execution_context}, encountered "
            f"value `{value_as_str}` of type {type(value)} which cannot be found in lookup "
            f"table with keys: {list(lookup_table.keys())}",
            context=f"step_execution | roboflow_query_language_evaluation | {execution_context}",
            inner_error=e,
        )


def generate_random_number(
    value: Any, min_value: float, max_value: float, **kwargs
) -> float:
    if min_value == max_value:
        return min_value
    raw_random = random.random()
    diff = abs(max_value - min_value)
    return min_value + (raw_random * diff)
