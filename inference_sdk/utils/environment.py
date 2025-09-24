from typing import Union


def str2bool(value: Union[str, bool]) -> bool:
    """Convert a string or boolean value to a boolean.

    Args:
        value (Union[str, bool]): The value to convert. Can be either a string ('true'/'false')
            or a boolean value.

    Returns:
        bool: The boolean value. Returns True for 'true' (case-insensitive) or True input,
            False for 'false' (case-insensitive) or False input.

    Raises:
        ValueError: If the input string is not 'true' or 'false' (case-insensitive).
    """
    if isinstance(value, bool):
        return value
    if value.lower() == "true":
        return True
    elif value.lower() == "false":
        return False
    else:
        raise ValueError(
            f"Expected a boolean environment variable (true or false) but got '{value}'"
        )
