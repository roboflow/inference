import traceback
import types
from typing import List, Optional, Type

from inference.core.env import (
    ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS,
    WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE,
    ALLOW_ANONYMOUS_MODAL_EXECUTION,
)
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
    workspace_id: Optional[str] = None,
) -> Type[WorkflowBlock]:
    code_module = create_dynamic_module(
        block_type_name=block_type_name,
        python_code=python_code,
        module_name=f"dynamic_module_{unique_identifier}",
        workspace_id=workspace_id,
    )
    if not hasattr(code_module, python_code.run_function_name):
        raise DynamicBlockError(
            public_message=f"Cannot find function: {python_code.run_function_name} in declared code for "
            f"dynamic block: `{block_type_name}`",
            context="workflow_compilation | dynamic_block_compilation | declared_symbols_fetching",
        )
    run_function = getattr(code_module, python_code.run_function_name)

    def run(self, *args, **kwargs) -> BlockResult:
        # Check if we're using Modal remote execution
        if WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE == "modal":
            # Remote execution via Modal - allowed even if local execution is disabled
            from inference.core.workflows.execution_engine.v1.dynamic_blocks.modal_executor import (
                ModalExecutor,
            )

            # Use the api_key that was stored during initialization
            api_key = getattr(self, "_api_key", None)
            
            # Fall back to "anonymous" for non-authenticated users (if allowed)
            if not api_key:
                if not ALLOW_ANONYMOUS_MODAL_EXECUTION:
                    raise DynamicBlockError(
                        public_message="Modal execution requires an API key when anonymous execution is disabled. "
                        "Please provide an API key or enable anonymous execution by setting "
                        "ALLOW_ANONYMOUS_MODAL_EXECUTION=True",
                        context="workflow_execution | dynamic_block_execution | modal_authentication",
                    )
                api_key = "anonymous"

            executor = ModalExecutor(workspace_id=api_key)
            return executor.execute_remote(
                block_type_name=block_type_name,
                python_code=python_code,
                inputs=kwargs,
                workspace_id=api_key,
            )
        else:
            # Local execution - check if allowed
            if not ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS:
                raise WorkflowEnvironmentConfigurationError(
                    public_message="Cannot use dynamic blocks with custom Python code in this installation of `workflows`. "
                    "This can be changed by setting environmental variable "
                    "`ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS=True`",
                    context="workflow_execution | step_execution | dynamic_step",
                )
            try:
                return run_function(self, *args, **kwargs)
            except Exception as error:
                tb = traceback.extract_tb(error.__traceback__)
                if tb:
                    frame = tb[-1]
                    line_number = frame.lineno - len(
                        _get_python_code_imports(python_code).splitlines()
                    )
                    function_name = frame.name
                    message = f"Error in line {line_number}, in {function_name}: {error.__class__.__name__}: {error}"
                else:
                    message = f"{error.__class__.__name__}: {error}"
                raise Exception(message) from error

    if python_code.init_function_code is not None and not hasattr(
        code_module, python_code.init_function_name
    ):
        raise DynamicBlockError(
            public_message=f"Cannot find function: {python_code.init_function_name} in declared code for "
            f"dynamic block: `{block_type_name}`",
            context="workflow_compilation | dynamic_block_compilation | declared_symbols_fetching",
        )

    init_function = getattr(code_module, python_code.init_function_name, dict)

    def constructor(self, api_key=None, **kwargs):
        # Store the api_key as an instance variable
        self._api_key = api_key
        self._init_results = init_function()

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        # Return an empty list but we'll handle api_key specially in the init parameter resolution
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


def _get_python_code_imports(python_code: PythonCode) -> str:
    return "\n".join(IMPORTS_LINES) + "\n" + "\n".join(python_code.imports) + "\n\n"


def create_dynamic_module(
    block_type_name: str,
    python_code: PythonCode,
    module_name: str,
    workspace_id: Optional[str] = None,
) -> types.ModuleType:
    imports = _get_python_code_imports(python_code)
    code = python_code.run_function_code
    if python_code.init_function_code:
        code += "\n\n" + python_code.init_function_code
    code = imports + code

    # If using Modal and local execution is disabled, validate code remotely
    if WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE == "modal":
        # Validate code in Modal sandbox for security
        from inference.core.workflows.execution_engine.v1.dynamic_blocks.modal_executor import (
            validate_code_in_modal,
        )

        # Pass workspace_id directly - validate_code_in_modal will handle None case
        # This will raise if validation fails (including if anonymous is not allowed)
        validate_code_in_modal(python_code, workspace_id)

        # Create a stub module for local reference
        dynamic_module = types.ModuleType(module_name)
        # Add placeholder function
        setattr(
            dynamic_module, python_code.run_function_name, lambda *args, **kwargs: None
        )
        if python_code.init_function_code:
            setattr(dynamic_module, python_code.init_function_name, lambda: {})
        return dynamic_module
    else:
        # Local validation and module creation
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
