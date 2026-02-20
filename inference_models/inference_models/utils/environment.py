import os
from typing import Any, List, Optional

from inference_models.errors import InvalidEnvVariable


def get_boolean_from_env(variable_name: str, default: Optional[bool] = None) -> bool:
    value = os.getenv(variable_name)
    if value is None:
        if default is None:
            raise InvalidEnvVariable(
                message=f"Environment variable {variable_name} is required",
                help_url="https://inference-models.roboflow.com/errors/runtime-environment/#invalidenvvariable",
            )
        return default
    return str2bool(value, variable_name=variable_name)


def get_integer_from_env(variable_name: str, default: Optional[int] = None) -> int:
    value = os.getenv(variable_name)
    if value is None:
        if default is None:
            raise InvalidEnvVariable(
                message=f"Environment variable {variable_name} is required",
                help_url="https://inference-models.roboflow.com/errors/runtime-environment/#invalidenvvariable",
            )
        return default
    try:
        return int(value)
    except ValueError:
        raise InvalidEnvVariable(
            message=f"Expected a environment variable `{variable_name}` to be integer but got '{value}'",
            help_url="https://inference-models.roboflow.com/errors/runtime-environment/#invalidenvvariable",
        )


def get_float_from_env(variable_name: str, default: Optional[float] = None) -> float:
    value = os.getenv(variable_name)
    if value is None:
        if default is None:
            raise InvalidEnvVariable(
                message=f"Environment variable {variable_name} is required",
                help_url="https://inference-models.roboflow.com/errors/runtime-environment/#invalidenvvariable",
            )
        return default
    try:
        return float(value)
    except ValueError:
        raise InvalidEnvVariable(
            message=f"Expected a environment variable `{variable_name}` to be integer but got '{value}'",
            help_url="https://inference-models.roboflow.com/errors/runtime-environment/#invalidenvvariable",
        )


def get_comma_separated_list_of_integers_from_env(
    variable_name: str, default: Optional[List[int]] = None
) -> List[int]:
    value = os.getenv(variable_name)
    if value is None:
        if default is None:
            raise InvalidEnvVariable(
                message=f"Environment variable {variable_name} is required",
                help_url="https://inference-models.roboflow.com/errors/runtime-environment/#invalidenvvariable",
            )
        return default
    try:
        return [int(v) for v in parse_comma_separated_values(value)]
    except ValueError:
        raise InvalidEnvVariable(
            message=f"Expected a environment variable `{variable_name}` to be comma separated list of integers but got '{value}'",
            help_url="https://inference-models.roboflow.com/errors/runtime-environment/#invalidenvvariable",
        )


def parse_comma_separated_values(values: str) -> List[str]:
    if not values:
        return []
    return [v.strip() for v in values.split(",") if v.strip()]


def str2bool(value: Any, variable_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if not issubclass(type(value), str):
        raise InvalidEnvVariable(
            message=f"Expected a environment variable `{variable_name}` to be (true or false) but got '{value}'",
            help_url="https://inference-models.roboflow.com/errors/runtime-environment/#invalidenvvariable",
        )
    if value.lower() == "true":
        return True
    elif value.lower() == "false":
        return False
    else:
        raise InvalidEnvVariable(
            message=f"Expected a environment variable `{variable_name}` to be (true or false) but got '{value}'",
            help_url="https://inference-models.roboflow.com/errors/runtime-environment/#invalidenvvariable",
        )
