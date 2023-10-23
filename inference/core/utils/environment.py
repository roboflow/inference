from typing import Any, List, Optional

from inference.core.exceptions import InvalidEnvironmentVariableError


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
