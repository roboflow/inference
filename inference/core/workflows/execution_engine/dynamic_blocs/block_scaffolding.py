import types
from typing import List, Type

from inference.core.workflows.execution_engine.dynamic_blocs.entities import PythonCode
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

IMPORTS_LINES = [
    "from typing import Any, List, Dict, Set",
    "import supervision as sv",
    "import numpy as np",
    "import math",
    "from inference.core.workflows.entities.base import Batch, WorkflowImageData",
    "from inference.core.workflows.prototypes.block import BlockResult",
]


def assembly_custom_python_block(
    unique_identifier: str,
    manifest: Type[WorkflowBlockManifest],
    python_code: PythonCode,
) -> Type[WorkflowBlock]:
    code_module = create_dynamic_module(
        code=python_code.function_code,
        module_name=f"dynamic_module_{unique_identifier}",
    )
    if not hasattr(code_module, python_code.function_name):
        raise ValueError(
            f"Cannot find function: {python_code.function_name} in declared code."
        )
    run_function = getattr(code_module, python_code.function_name)

    async def run(self, *args, **kwargs) -> BlockResult:
        if not self._allow_custom_python_execution:
            raise RuntimeError(
                "It is not possible to execute CustomPython block in that configuration of `inference`. Set "
                "ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS=true"
            )
        return run_function(*args, **kwargs)

    def constructor(self, allow_custom_python_execution: bool):
        self._allow_custom_python_execution = allow_custom_python_execution

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["allow_custom_python_execution"]

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


def create_dynamic_module(code: str, module_name: str) -> types.ModuleType:
    dynamic_module = types.ModuleType(module_name)
    imports = "\n".join(IMPORTS_LINES) + "\n\n\n\n"
    exec(imports + code, dynamic_module.__dict__)
    return dynamic_module
