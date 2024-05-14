from typing import Any

from inference.core.workflows.core_steps.common.query_language.errors import (
    InvalidInputTypeError,
)
from inference.core.workflows.core_steps.common.query_language.operations.utils import (
    safe_stringify,
)


def apply_lookup(value: Any, lookup_table: dict) -> Any:
    try:
        return lookup_table[value]
    except TypeError as e:
        raise InvalidInputTypeError(
            public_message=f"While executing operation apply_lookup(...), passed value of type "
            f"{type(value)} which does not let it be searched in lookup table.",
            context="step_execution | roboflow_query_language_evaluation",
            inner_error=e,
        )
    except KeyError as e:
        value_as_str = safe_stringify(value=value)
        raise InvalidInputTypeError(
            public_message=f"While executing operation apply_lookup(...), encountered "
            f"value `{value_as_str}` of type {type(value)} which cannot be found in lookup "
            f"table with keys: {list(lookup_table.keys())}",
            context="step_execution | roboflow_query_language_evaluation",
            inner_error=e,
        )
