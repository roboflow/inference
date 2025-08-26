"""
Modal executor for Custom Python Blocks in Workflows.

This module handles the execution of untrusted user code in Modal sandboxes
with proper serialization and security restrictions.
"""

import hashlib
import json
import os
import tempfile
from typing import Any, Dict, List, Optional

import modal

from inference.core.env import (
    MODAL_TOKEN_ID,
    MODAL_TOKEN_SECRET,
    MODAL_WORKSPACE_NAME,
)
from inference.core.workflows.errors import DynamicBlockError
from inference.core.workflows.execution_engine.v1.dynamic_blocks.entities import (
    PythonCode,
)
from inference.core.workflows.prototypes.block import BlockResult

# Configure Modal client if credentials are available
if MODAL_TOKEN_ID and MODAL_TOKEN_SECRET:
    os.environ["MODAL_TOKEN_ID"] = MODAL_TOKEN_ID
    os.environ["MODAL_TOKEN_SECRET"] = MODAL_TOKEN_SECRET


class ModalExecutor:
    """Manages execution of Custom Python Blocks in Modal sandboxes."""
    
    def __init__(self, workspace_id: str):
        """Initialize the Modal executor for a specific workspace.
        
        Args:
            workspace_id: The workspace ID to namespace Modal Apps
        """
        self.workspace_id = workspace_id
        self.app_name = f"inference-workspace-{workspace_id}"
        self._function_cache = {}
        
    def _get_code_hash(self, code: str) -> str:
        """Generate MD5 hash for code block identification."""
        return hashlib.md5(code.encode()).hexdigest()
    
    def _get_inference_image(self):
        """Get the Modal Image for inference."""
        from inference.core.version import __version__
        
        # Use the pre-built shared image or create on-the-fly
        image = (
            modal.Image.debian_slim(python_version="3.11")
            .apt_install(
                "libgl1-mesa-glx",
                "libglib2.0-0", 
                "libgomp1",
                "libsm6",
                "libxext6",
                "libxrender-dev",
                "ffmpeg",
                "wget",
            )
            .run_commands("pip install --upgrade pip uv")
            .run_commands(f"uv pip install --system inference=={__version__}")
        )
        return image
    
    def execute_remote(
        self, 
        block_type_name: str,
        python_code: PythonCode,
        inputs: Dict[str, Any],
        workspace_id: Optional[str] = None
    ) -> BlockResult:
        """Execute a Custom Python Block in a Modal sandbox.
        
        Args:
            block_type_name: Name of the block type
            python_code: The Python code to execute
            inputs: Input data for the function
            workspace_id: Optional workspace ID override
            
        Returns:
            BlockResult from the execution
            
        Raises:
            DynamicBlockError: If execution fails
        """
        # Use provided workspace_id or fall back to instance default
        if workspace_id:
            self.workspace_id = workspace_id
            self.app_name = f"inference-workspace-{workspace_id}"
        
        if not self.workspace_id:
            raise DynamicBlockError(
                public_message="Workspace ID is required for Modal execution",
                context="modal_executor | workspace_validation"
            )
        
        try:
            # Create the execution code
            exec_code = self._build_execution_code(python_code)
            
            # Write to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(exec_code)
                temp_file = f.name
            
            # Get or create app
            code_hash = self._get_code_hash(python_code.code)
            function_name = f"block-{code_hash}"
            
            # Check cache
            if function_name in self._function_cache:
                func = self._function_cache[function_name]
            else:
                # Create Modal app and function
                app = modal.App(self.app_name)
                
                # Create function with the code
                @app.function(
                    name=function_name,
                    image=self._get_inference_image(),
                    restrict_modal_access=True,
                    max_inputs=1,
                    timeout=20,
                    region="us-central1",
                )
                def custom_block_executor(serialized_inputs_str: str) -> str:
                    """Execute the custom block with serialized inputs."""
                    # Import inside function
                    import json
                    import sys
                    import traceback
                    
                    # Read the code file that was included in the image
                    with open(temp_file, 'r') as code_file:
                        code = code_file.read()
                    
                    # Create a namespace and execute the code
                    namespace = {"__name__": "__main__"}
                    exec(code, namespace)
                    
                    # Call the execute function
                    if "execute_custom_block" in namespace:
                        try:
                            serialized_inputs = json.loads(serialized_inputs_str)
                            result = namespace["execute_custom_block"](serialized_inputs)
                            return json.dumps(result)
                        except Exception as e:
                            error_result = {
                                "error": str(e),
                                "error_type": type(e).__name__,
                                "traceback": traceback.format_exc()
                            }
                            return json.dumps(error_result)
                    else:
                        return json.dumps({"error": "execute_custom_block function not found"})
                
                # Deploy the function
                with app.run():
                    func = custom_block_executor
                    self._function_cache[function_name] = func
            
            # Serialize inputs
            from inference.core.workflows.execution_engine.v1.dynamic_blocks.serializers import (
                serialize_inputs, deserialize_outputs
            )
            serialized_inputs = serialize_inputs(inputs)
            serialized_str = json.dumps(serialized_inputs)
            
            # Execute remotely
            result_str = func.remote(serialized_str)
            result = json.loads(result_str)
            
            # Check for errors
            if isinstance(result, dict) and "error" in result:
                raise DynamicBlockError(
                    public_message=f"Custom block execution failed: {result['error']}",
                    context="modal_executor | remote_execution"
                )
            
            # Deserialize outputs
            return deserialize_outputs(result)
            
        except Exception as e:
            if isinstance(e, DynamicBlockError):
                raise
            raise DynamicBlockError(
                public_message=f"Failed to execute custom block remotely: {str(e)}",
                context="modal_executor | remote_execution"
            )
        finally:
            # Clean up temp file
            if 'temp_file' in locals() and os.path.exists(temp_file):
                os.unlink(temp_file)

    def _build_execution_code(self, python_code: PythonCode) -> str:
        """Build the complete execution code for the Modal function.
        
        Args:
            python_code: The Python code to execute
            
        Returns:
            Complete Python code as string
        """
        # Build imports
        imports = [
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
            "from inference.core.workflows.execution_engine.v1.dynamic_blocks.serializers import (",
            "    deserialize_inputs,",
            "    serialize_outputs,",
            ")",
        ]
        
        if python_code.imports:
            imports.extend(python_code.imports)
        
        imports_str = "\n".join(imports)
        
        # Build the complete code
        return f"""
{imports_str}

# User code
{python_code.code}

def execute_custom_block(serialized_inputs: Dict[str, Any]) -> Dict[str, Any]:
    \"\"\"Execute the custom block with proper error handling.\"\"\"
    try:
        # Deserialize inputs
        inputs = deserialize_inputs(serialized_inputs)
        
        # Execute the user function
        result = {python_code.run_function_name}(**inputs)
        
        # Serialize outputs
        return serialize_outputs(result)
    except Exception as e:
        import traceback
        return {{
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc()
        }}
"""


def validate_code_in_modal(python_code: PythonCode, workspace_id: str) -> bool:
    """Validate Python code syntax in a Modal sandbox.
    
    Args:
        python_code: The Python code to validate
        workspace_id: The workspace ID for Modal App
        
    Returns:
        True if code is valid, raises otherwise
        
    Raises:
        DynamicBlockError: If code validation fails
    """
    executor = ModalExecutor(workspace_id)
    
    # Create a validation version of the code
    validation_code = PythonCode(
        imports=python_code.imports,
        code=f"""
{python_code.code}

def validate_code():
    # Try to compile the user function
    import ast
    try:
        # This will raise SyntaxError if code is invalid
        compile('''{python_code.code}''', '<string>', 'exec')
        return {{"valid": True}}
    except SyntaxError as e:
        return {{"valid": False, "error": str(e)}}
""",
        run_function_name="validate_code",
        run_function_code="",
        init_function_name=None,
        init_function_code=None,
    )
    
    try:
        result = executor.execute_remote(
            block_type_name="validation",
            python_code=validation_code,
            inputs={},
            workspace_id=workspace_id
        )
        
        if result.get("valid") is False:
            raise DynamicBlockError(
                public_message=f"Code validation failed: {result.get('error', 'Unknown error')}",
                context="modal_executor | code_validation"
            )
        
        return True
        
    except Exception as e:
        raise DynamicBlockError(
            public_message=f"Code validation failed: {str(e)}",
            context="modal_executor | code_validation"
        )
