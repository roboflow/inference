"""
Modal executor for Custom Python Blocks in Workflows.

This module handles the execution of untrusted user code in Modal sandboxes
with proper serialization and security restrictions.
"""

import hashlib
import json
import os
from typing import Any, Dict, Optional

from inference.core.env import (
    MODAL_TOKEN_ID,
    MODAL_TOKEN_SECRET,
    MODAL_WORKSPACE_NAME,
)

# Set Modal environment variables before importing
if MODAL_TOKEN_ID and MODAL_TOKEN_SECRET:
    os.environ["MODAL_TOKEN_ID"] = MODAL_TOKEN_ID
    os.environ["MODAL_TOKEN_SECRET"] = MODAL_TOKEN_SECRET
    print(f"Modal credentials set in environment: {MODAL_TOKEN_ID[:4]}...{MODAL_TOKEN_ID[-4:]}")

# Try to import modal, but handle gracefully if not installed
try:
    import modal
    MODAL_INSTALLED = True
except ImportError:
    MODAL_INSTALLED = False
    modal = None

from inference.core.workflows.errors import DynamicBlockError
from inference.core.workflows.execution_engine.v1.dynamic_blocks.entities import (
    PythonCode,
)
from inference.core.workflows.prototypes.block import BlockResult

# Check if Modal is available
if MODAL_INSTALLED and MODAL_TOKEN_ID and MODAL_TOKEN_SECRET:
    MODAL_AVAILABLE = True
else:
    MODAL_AVAILABLE = False
    if MODAL_INSTALLED:
        print("Modal installed but credentials not found")
    else:
        print("Modal not installed")


# Create the Modal App only if Modal is installed
if MODAL_INSTALLED:
    app = modal.App("inference-custom-blocks")
    cls = modal.Cls.from_name("inference-custom-blocks", "CustomBlockExecutor")
else:
    app = None
    cls = None

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
if MODAL_INSTALLED and app:
    @app.cls(
        image=_get_inference_image(),
        restrict_modal_access=True,
        timeout=20,
        enable_memory_snapshot=True,  # Enable memory snapshotting for faster cold starts
        scaledown_window=60,
        cloud="gcp",
        region="us-central1",

    )
    class CustomBlockExecutor:
        """Parameterized Modal class for executing custom Python blocks."""
        
        # Only parameterize by workspace_id - code_hash removed per feedback
        workspace_id: str = modal.parameter()
        
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
                inputs: Dictionary of inputs (uses Modal's built-in pickling)
                
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
                
                # Get the user's function
                user_function = namespace[run_function_name]
                
                # Check if function expects a 'self' parameter
                import inspect
                sig = inspect.signature(user_function)
                params = list(sig.parameters.keys())
                
                # If function expects 'self' as first param, create a simple object to pass
                if params and params[0] == 'self':
                    # Create a simple object to pass as self
                    class BlockSelf:
                        pass
                    
                    block_self = BlockSelf()
                    # Execute with self parameter
                    result = user_function(block_self, **inputs)
                else:
                    # Execute without self parameter
                    result = user_function(**inputs)
                
                # Return the result - Modal handles pickling
                return {"success": True, "result": result}
                
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "traceback": traceback.format_exc()
                }
else:
    CustomBlockExecutor = None


class ModalExecutor:
    """Manages execution of Custom Python Blocks in Modal sandboxes."""
    
    def __init__(self, workspace_id: Optional[str] = None):
        """Initialize the Modal executor for a specific workspace.
        
        Args:
            workspace_id: The workspace ID to namespace execution, defaults to "anonymous"
        """
        self.workspace_id = workspace_id or "anonymous"
        self._executor_cache = {}
        
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
        if not MODAL_INSTALLED:
            raise DynamicBlockError(
                public_message="Modal is not installed. Please install with: pip install modal",
                context="modal_executor | installation_check"
            )
        
        if not MODAL_AVAILABLE:
            raise DynamicBlockError(
                public_message="Modal credentials not configured. Please set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET environment variables.",
                context="modal_executor | credentials_check"
            )
        
        # Use provided workspace_id or fall back to instance default
        workspace = workspace_id if workspace_id else self.workspace_id
        
        try:
            # Get or create executor for this workspace (no code_hash needed)
            cache_key = workspace
            
            if cache_key not in self._executor_cache:
                # Create a new executor instance for this workspace using the deployed app
                if MODAL_INSTALLED and modal:
                    # Look up the deployed class
                    executor = cls(workspace_id=workspace)
                    self._executor_cache[cache_key] = executor
                else:
                    raise DynamicBlockError(
                        public_message="Modal is not properly configured",
                        context="modal_executor | class_lookup"
                    )
            else:
                executor = self._executor_cache[cache_key]
            
            # Execute remotely - pass inputs directly, Modal handles pickling
            result = executor.execute_block.remote(
                code_str=python_code.run_function_code,
                imports=python_code.imports or [],
                run_function_name=python_code.run_function_name,
                inputs=inputs  # No serialization needed - Modal pickles automatically
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
            
            # Get the result and deserialize outputs only
            raw_result = result.get("result", {})
            return self._deserialize_outputs(raw_result)
            
        except Exception as e:
            if isinstance(e, DynamicBlockError):
                raise
            raise DynamicBlockError(
                public_message=f"Failed to execute custom block remotely: {str(e)}",
                context="modal_executor | remote_execution"
            )
    
    def _deserialize_outputs(self, outputs: Any) -> BlockResult:
        """Deserialize outputs from Modal transport.
        
        We need to serialize outputs back to JSON-safe format for workflows.
        
        Args:
            outputs: Outputs from Modal function (already unpickled by Modal)
            
        Returns:
            BlockResult with serialized outputs
        """
        from inference.core.workflows.core_steps.common.serializers import (
            serialize_wildcard_kind
        )
        
        # Handle special numpy cases for better compatibility
        def _prepare_numpy_types(value):
            # Convert numpy shapes to lists for consistency
            if isinstance(value, tuple) and all(isinstance(x, int) for x in value):
                return list(value)
            # Convert numpy arrays and other special types
            return value
        
        # If outputs is a dict, serialize each value
        if isinstance(outputs, dict):
            serialized = {}
            for key, value in outputs.items():
                value = _prepare_numpy_types(value)
                serialized[key] = serialize_wildcard_kind(value)
            return serialized
        else:
            # Wrap non-dict results in BlockResult format
            return {"result": serialize_wildcard_kind(_prepare_numpy_types(outputs))}


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
    if not MODAL_INSTALLED:
        raise DynamicBlockError(
            public_message="Modal is not installed. Please install with: pip install modal",
            context="modal_executor | installation_check"
        )
    
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
        # For validation, we don't need to pass any complex inputs
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
