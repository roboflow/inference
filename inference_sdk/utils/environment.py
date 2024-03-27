from typing import Union


def str2bool(value: Union[str, bool]) -> bool:
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
