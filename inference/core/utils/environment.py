import os
from typing import Any, Callable, List, Optional, Type, TypeVar, Union

from inference.core.exceptions import InvalidEnvironmentVariableError

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


def safe_split_value(value: Optional[str], delimiter: str = ",") -> Optional[List[str]]:
    """
    Splits a separated environment variable into a list.

    Args:
        value (str): The environment variable value to be split.
        delimiter(str): Delimiter to be used

    Returns:
        list or None: The split values as a list, or None if the input is None.
    """
    if value is None:
        return None
    else:
        return value.split(delimiter)
