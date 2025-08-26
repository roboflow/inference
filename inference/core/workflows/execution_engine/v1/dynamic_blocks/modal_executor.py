"""
Modal executor for Custom Python Blocks in Workflows.

This module handles the execution of untrusted user code in Modal sandboxes
with proper serialization and security restrictions.
"""

import hashlib
import json
import os
from typing import Any, Dict, Optional

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
    MODAL_AVAILABLE = True
else:
    MODAL_AVAILABLE = False

# Create the Modal App
app = modal.App("inference-custom-blocks")

def _get_inference_image():
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
        .uv_pip_install(f"inference=={__version__}")
    )
    return image

# Define the parameterized Modal class for execution
@app.cls(
    image=_get_inference_image(),
    restrict_modal_access=True,
    max_inputs=1,
    timeout=20,
    region="us-central1",
)
class CustomBlockExecutor:
    """Parameterized Modal class for executing custom Python blocks."""
    
    workspace_id: str = modal.parameter()
    code_hash: str = modal.parameter()
    
    @modal.method()
    def execute_block(
        self, 
        code_str: str,
        imports: list[str],
        run_function_name: str,
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute the custom block with the given inputs.
        
        Args:
            code_str: The Python code to execute
            imports: List of import statements
            run_function_name: Name of the function to call
            inputs: Dictionary of inputs (already JSON-safe)
            
        Returns:
            Dictionary with results or error information
        """
        import traceback
        import sys
        
        # Build the execution namespace
        namespace = {"__name__": "__main__"}
        
        # Execute imports
        import_code = "\n".join(imports) if imports else ""
        full_imports = f"""
from typing import Any, List, Dict, Set, Optional
import supervision as sv
import numpy as np
import math
import time
import json
import os
import requests
import cv2
import shapely
from inference.core.workflows.execution_engine.entities.base import Batch, WorkflowImageData
from inference.core.workflows.prototypes.block import BlockResult

{import_code}
"""
        
        try:
            # Execute imports
            exec(full_imports, namespace)
            
            # Execute the user code
            exec(code_str, namespace)
            
            # Call the user function
            if run_function_name not in namespace:
                return {
                    "error": f"Function '{run_function_name}' not found in code",
                    "error_type": "NameError"
                }
            
            # Execute the function - inputs are already deserialized
            result = namespace[run_function_name](**inputs)
            
            # Return the result
            return {"success": True, "result": result}
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc()
            }


class ModalExecutor:
    """Manages execution of Custom Python Blocks in Modal sandboxes."""
    
    def __init__(self, workspace_id: Optional[str] = None):
        """Initialize the Modal executor for a specific workspace.
        
        Args:
            workspace_id: The workspace ID to namespace execution, defaults to "anonymous"
        """
        self.workspace_id = workspace_id or "anonymous"
        self._executor_cache = {}
        
    def _get_code_hash(self, code: str) -> str:
        """Generate MD5 hash for code block identification."""
        return hashlib.md5(code.encode()).hexdigest()[:8]  # Use first 8 chars for brevity
    
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
        # Check if Modal is available
        if not MODAL_AVAILABLE:
            raise DynamicBlockError(
                public_message="Modal credentials not configured. Please set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET environment variables.",
                context="modal_executor | credentials_check"
            )
        
        # Use provided workspace_id or fall back to instance default
        workspace = workspace_id if workspace_id else self.workspace_id
        
        try:
            # Serialize complex inputs using existing serializers
            serialized_inputs = self._serialize_inputs(inputs)
            
            # Get or create executor
            code_hash = self._get_code_hash(python_code.code)
            cache_key = f"{workspace}-{code_hash}"
            
            if cache_key not in self._executor_cache:
                # Create a new executor instance
                with app.run():
                    executor = CustomBlockExecutor(
                        workspace_id=workspace,
                        code_hash=code_hash
                    )
                    self._executor_cache[cache_key] = executor
            else:
                executor = self._executor_cache[cache_key]
            
            # Execute remotely - pass already serialized inputs
            result = executor.execute_block.remote(
                code_str=python_code.code,
                imports=python_code.imports or [],
                run_function_name=python_code.run_function_name,
                inputs=serialized_inputs
            )
            
            # Check for errors
            if not result.get("success", False):
                error_msg = result.get("error", "Unknown error")
                error_type = result.get("error_type", "RuntimeError")
                traceback = result.get("traceback", "")
                
                raise DynamicBlockError(
                    public_message=f"{error_type}: {error_msg}",
                    context=f"modal_executor | remote_execution\n{traceback}"
                )
            
            # Deserialize and return the result
            return self._deserialize_outputs(result.get("result", {}))
            
        except Exception as e:
            if isinstance(e, DynamicBlockError):
                raise
            raise DynamicBlockError(
                public_message=f"Failed to execute custom block remotely: {str(e)}",
                context="modal_executor | remote_execution"
            )
    
    def _serialize_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize inputs for Modal transport using existing serializers.
        
        Args:
            inputs: Raw inputs dictionary
            
        Returns:
            Serialized inputs safe for JSON transport
        """
        from inference.core.workflows.core_steps.common.serializers import (
            serialize_wildcard_kind
        )
        
        serialized = {}
        for key, value in inputs.items():
            serialized[key] = serialize_wildcard_kind(value)
        return serialized
    
    def _deserialize_outputs(self, outputs: Any) -> BlockResult:
        """Deserialize outputs from Modal transport.
        
        Since we're using existing serializers which already produce
        JSON-safe output, we mostly just need to ensure the result
        conforms to BlockResult format.
        
        Args:
            outputs: Outputs from Modal function
            
        Returns:
            BlockResult
        """
        # BlockResult is a TypedDict, ensure outputs conform
        if isinstance(outputs, dict):
            return outputs
        else:
            # Wrap non-dict results in BlockResult format
            return {"result": outputs}


def validate_code_in_modal(python_code: PythonCode, workspace_id: Optional[str] = None) -> bool:
    """Validate Python code syntax in a Modal sandbox.
    
    Args:
        python_code: The Python code to validate
        workspace_id: The workspace ID for Modal App
        
    Returns:
        True if code is valid, raises otherwise
        
    Raises:
        DynamicBlockError: If code validation fails
    """
    # Check if Modal is available
    if not MODAL_AVAILABLE:
        raise DynamicBlockError(
            public_message="Modal credentials not configured. Please set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET environment variables.",
            context="modal_executor | credentials_check"
        )
    
    workspace = workspace_id or "anonymous"
    
    # Simple validation code that checks syntax
    validation_code = PythonCode(
        imports=[],
        code=f"""
import ast

def validate_syntax():
    try:
        # Try to compile the user code
        code = '''{python_code.code}'''
        compile(code, '<string>', 'exec')
        # Try to parse as AST to check structure
        ast.parse(code)
        return {{"valid": True}}
    except SyntaxError as e:
        return {{"valid": False, "error": str(e), "line": e.lineno}}
    except Exception as e:
        return {{"valid": False, "error": str(e)}}
""",
        run_function_name="validate_syntax",
        run_function_code="",
        init_function_name=None,
        init_function_code=None,
    )
    
    executor = ModalExecutor(workspace_id=workspace)
    
    try:
        result = executor.execute_remote(
            block_type_name="validation",
            python_code=validation_code,
            inputs={},
            workspace_id=workspace
        )
        
        if result.get("valid") is False:
            error_msg = result.get("error", "Unknown syntax error")
            line_no = result.get("line", None)
            if line_no:
                error_msg = f"Line {line_no}: {error_msg}"
            raise DynamicBlockError(
                public_message=f"Code validation failed: {error_msg}",
                context="modal_executor | code_validation"
            )
        
        return True
        
    except Exception as e:
        if isinstance(e, DynamicBlockError):
            raise
        raise DynamicBlockError(
            public_message=f"Code validation failed: {str(e)}",
            context="modal_executor | code_validation"
        )
