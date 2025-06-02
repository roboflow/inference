from typing import Any, List

from inference.v1.errors import InvalidEnvVariable


def parse_comma_separated_values(values: str) -> List[str]:
    return [v.strip() for v in values.split(",")]


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if not issubclass(type(value), str):
        raise InvalidEnvVariable(
            f"Expected a boolean environment variable (true or false) but got '{value}'"
        )
    if value.lower() == "true":
        return True
    elif value.lower() == "false":
        return False
    else:
        raise InvalidEnvVariable(
            f"Expected a boolean environment variable (true or false) but got '{value}'"
        )
