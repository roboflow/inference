import types
from typing import List, Literal, Type
from uuid import uuid4

from inference.core.workflows.core_steps.common.dynamic_blocks.entities import (
    ManifestDescription,
)
from inference.core.workflows.core_steps.common.dynamic_blocks.manifest_assembler import (
    assembly_dynamic_block_manifest,
)
from inference.core.workflows.entities.base import OutputDefinition
from inference.core.workflows.entities.types import WILDCARD_KIND
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


class CustomPythonDeclaredManifest(WorkflowBlockManifest):
    name: str
    type: Literal["CustomPython"]
    manifest_description: ManifestDescription
    python_code: str
    function_name: str

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="*", kind=[WILDCARD_KIND])]


def assembly_custom_python_block(
    declared_manifest: CustomPythonDeclaredManifest,
) -> Type[WorkflowBlock]:
    actual_manifest = assembly_dynamic_block_manifest(
        block_name=declared_manifest.name,
        block_type=declared_manifest.type,
        manifest_description=declared_manifest.manifest_description,
    )
    code_module = create_dynamic_module(
        code=declared_manifest.python_code,
        module_name=f"dynamic_module_{uuid4()}",
    )
    if not hasattr(code_module, declared_manifest.function_name):
        raise ValueError(
            f"Cannot find function: {declared_manifest.function_name} in declared code."
        )
    run_function = getattr(code_module, declared_manifest.function_name)

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
        return actual_manifest

    return type(
        f"CustomPythonBlock-{uuid4()}",
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
