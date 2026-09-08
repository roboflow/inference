"""Workflows-local copies of pure helpers previously imported from `inference.core.utils`.

The bodies are copied verbatim from the server implementations, with the single
intentional difference that the workflows-local exception / warning classes are
raised and emitted instead of the `inference.core.*` ones.
"""

import functools
import os
import warnings
from typing import Any, Union

import numpy as np

from inference.core.workflows.errors import WorkflowsInvalidEnvironmentValueError
from inference.core.workflows.warnings import WorkflowsExperimentalFeatureWarning


def str2bool(value: Any) -> bool:
    """
    Converts an environment variable to a boolean value.

    Args:
        value (str or bool): The environment variable value to be converted.

    Returns:
        bool: The converted boolean value.

    Raises:
        WorkflowsInvalidEnvironmentValueError: If the value is not 'true', 'false', or a boolean.
    """
    if isinstance(value, bool):
        return value
    if not issubclass(type(value), str):
        raise WorkflowsInvalidEnvironmentValueError(
            f"Expected a boolean environment variable (true or false) but got '{value}'"
        )
    if value.lower() == "true":
        return True
    elif value.lower() == "false":
        return False
    else:
        raise WorkflowsInvalidEnvironmentValueError(
            f"Expected a boolean environment variable (true or false) but got '{value}'"
        )


def ensure_parent_dir_exists(path: str) -> None:
    absolute_path = os.path.abspath(path)
    parent_dir = os.path.dirname(absolute_path)
    os.makedirs(parent_dir, exist_ok=True)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> Union[np.number, np.ndarray]:
    """
    Compute the cosine similarity between two vectors.

    Args:
        a (np.ndarray): Vector A.
        b (np.ndarray): Vector B.

    Returns:
        float: Cosine similarity between vectors A and B.
    """
    return np.dot(a, b) / np.sqrt(np.vdot(a, a) * np.vdot(b, b))


def experimental(reason: str):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"{func.__name__} is experimental: {reason}",
                category=WorkflowsExperimentalFeatureWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return wrapper

    return decorator
