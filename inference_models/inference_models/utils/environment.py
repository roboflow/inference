from typing import Any, List

from inference_models.errors import InvalidEnvVariable


def parse_comma_separated_values(values: str) -> List[str]:
    if not values:
        return []
    return [v.strip() for v in values.split(",") if v.strip()]


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if not issubclass(type(value), str):
        raise InvalidEnvVariable(
            message=f"Expected a boolean environment variable (true or false) but got '{value}'",
            help_url="https://todo",
        )
    if value.lower() == "true":
        return True
    elif value.lower() == "false":
        return False
    else:
        raise InvalidEnvVariable(
            message=f"Expected a boolean environment variable (true or false) but got '{value}'",
            help_url="https://todo",
        )
