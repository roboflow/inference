"""Environment variable parsing used by `ModelConfig.init`.

Copy of the two helpers from `inference.core.utils.environment`. `str2bool`
raises the historical `InvalidEnvironmentVariableError` class, which
`inference.core.exceptions` re-exports from the stream package.
"""

import os
from typing import Any, Callable, Optional, Type, TypeVar, Union

from inference.core.interfaces.stream.exceptions import InvalidEnvironmentVariableError

T = TypeVar("T")


def safe_env_to_type(
    variable_name: str,
    default_value: Optional[T] = None,
    type_constructor: Optional[Union[Type[T], Callable[[str], T]]] = None,
) -> Optional[T]:
    """
    Converts env variable to specified type, but only if variable is set - otherwise default is returned.
    If `type_constructor` is not given - value of type str will be returned.
    """
    if variable_name not in os.environ:
        return default_value
    variable_value = os.environ[variable_name]
    if type_constructor is None:
        return variable_value
    return type_constructor(variable_value)


def str2bool(value: Any) -> bool:
    """
    Converts an environment variable to a boolean value.

    Args:
        value (str or bool): The environment variable value to be converted.

    Returns:
        bool: The converted boolean value.

    Raises:
        InvalidEnvironmentVariableError: If the value is not 'true', 'false', or a boolean.
    """
    if isinstance(value, bool):
        return value
    if not issubclass(type(value), str):
        raise InvalidEnvironmentVariableError(
            f"Expected a boolean environment variable (true or false) but got '{value}'"
        )
    if value.lower() == "true":
        return True
    elif value.lower() == "false":
        return False
    else:
        raise InvalidEnvironmentVariableError(
            f"Expected a boolean environment variable (true or false) but got '{value}'"
        )
