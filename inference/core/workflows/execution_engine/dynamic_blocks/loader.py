from typing import Any, Callable, Dict, Union

from inference.core.env import ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS
from inference.core.workflows.execution_engine.dynamic_blocks.entities import (
    BLOCK_SOURCE,
)


def load_dynamic_blocks_initializers() -> Dict[str, Union[Any, Callable[[None], Any]]]:
    return {
        f"{BLOCK_SOURCE}.allow_custom_python_execution": ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS
    }
