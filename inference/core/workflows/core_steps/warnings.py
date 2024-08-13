import os
import warnings
from typing import Any

from inference.core.workflows.errors import WorkflowEnvironmentConfigurationError


class RoboflowCoreBlocksIncompatibilityWarning(Warning):
    pass


def setup_warnings() -> None:
    disable_warnings = str2bool(
        os.getenv("DISABLE_ROBOFLOW_CORE_BLOCKS_INCOMPATIBILITY_WARNINGS", "False")
    )
    if disable_warnings:
        warnings.simplefilter("ignore", RoboflowCoreBlocksIncompatibilityWarning)


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
        raise WorkflowEnvironmentConfigurationError(
            public_message="Expected a boolean environment variable (true or false)",
            context="plugin_setup | env_variable_parsing",
        )
    if value.lower() == "true":
        return True
    elif value.lower() == "false":
        return False
    else:
        raise WorkflowEnvironmentConfigurationError(
            public_message="Expected a boolean environment variable (true or false).",
            context="plugin_setup | env_variable_parsing",
        )
