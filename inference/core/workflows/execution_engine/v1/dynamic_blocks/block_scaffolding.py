import types
from typing import List, Type

from inference.core.env import ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS
from inference.core.workflows.errors import (
    DynamicBlockError,
    WorkflowEnvironmentConfigurationError,
)
from inference.core.workflows.execution_engine.v1.dynamic_blocks.entities import (
    PythonCode,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

IMPORTS_LINES = [
    "from typing import Any, List, Dict, Set, Optional",
    "import supervision as sv",
    "import numpy as np",
    "import math",
    "import time",
    "import json",
    "import os",
    "import requests",
    "import cv2",
    "import shapely",
    "from inference.core.workflows.execution_engine.entities.base import Batch, WorkflowImageData",
    "from inference.core.workflows.prototypes.block import BlockResult",
]


def assembly_custom_python_block(
    block_type_name: str,
    unique_identifier: str,
    manifest: Type[WorkflowBlockManifest],
    python_code: PythonCode,
) -> Type[WorkflowBlock]:
    code_module = create_dynamic_module(
        block_type_name=block_type_name,
        python_code=python_code,
        module_name=f"dynamic_module_{unique_identifier}",
    )
    if not hasattr(code_module, python_code.run_function_name):
        raise DynamicBlockError(
            public_message=f"Cannot find function: {python_code.run_function_name} in declared code for "
            f"dynamic block: `{block_type_name}`",
            context="workflow_compilation | dynamic_block_compilation | declared_symbols_fetching",
        )
    run_function = getattr(code_module, python_code.run_function_name)

    def run(self, *args, **kwargs) -> BlockResult:
        if not ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS:
            raise WorkflowEnvironmentConfigurationError(
                public_message="Cannot use dynamic blocks with custom Python code in this installation of `workflows`. "
                "This can be changed by setting environmental variable "
                "`ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS=True`",
                context="workflow_execution | step_execution | dynamic_step",
            )
        return run_function(self, *args, **kwargs)

    if python_code.init_function_code is not None and not hasattr(
        code_module, python_code.init_function_name
    ):
        raise DynamicBlockError(
            public_message=f"Cannot find function: {python_code.init_function_name} in declared code for "
            f"dynamic block: `{block_type_name}`",
            context="workflow_compilation | dynamic_block_compilation | declared_symbols_fetching",
        )

    init_function = getattr(code_module, python_code.init_function_name, dict)

    def constructor(self):
        self._init_results = init_function()

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return []

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return manifest

    return type(
        f"DynamicBlock[{unique_identifier}]",
        (WorkflowBlock,),
        {
            "__init__": constructor,
            "get_init_parameters": get_init_parameters,
            "get_manifest": get_manifest,
            "run": run,
        },
    )


def create_dynamic_module(
    block_type_name: str, python_code: PythonCode, module_name: str
) -> types.ModuleType:
    imports = "\n".join(IMPORTS_LINES) + "\n" + "\n".join(python_code.imports) + "\n\n"
    code = python_code.run_function_code
    if python_code.init_function_code:
        code += "\n\n" + python_code.init_function_code
    code = imports + code
    try:
        dynamic_module = types.ModuleType(module_name)
        exec(code, dynamic_module.__dict__)
        return dynamic_module
    except Exception as error:
        raise DynamicBlockError(
            public_message=f"Error of type `{error.__class__.__name__}` encountered while attempting to "
            f"create Python module with code for block: {block_type_name}. Error message: {error}. Full code:\n{code}",
            context="workflow_compilation | dynamic_block_compilation | dynamic_module_creation",
            inner_error=error,
        ) from error
