"""
Modal executor for Custom Python Blocks in Workflows using Web Endpoints.

This module handles the execution of untrusted user code in Modal sandboxes
using web endpoints for better security and no size limitations.
"""

import hashlib
import json
import os
from typing import Any, Dict, Optional
import requests

import numpy as np

from inference.core.env import MODAL_TOKEN_ID, MODAL_TOKEN_SECRET
from inference.core.logger import logger

# Set Modal environment variables before importing
if MODAL_TOKEN_ID and MODAL_TOKEN_SECRET:
    os.environ["MODAL_TOKEN_ID"] = MODAL_TOKEN_ID
    os.environ["MODAL_TOKEN_SECRET"] = MODAL_TOKEN_SECRET

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
        logger.warning("Modal installed but credentials not found")
    else:
        logger.info("Modal not installed")


# Create the Modal App only if Modal is installed - using different name to avoid conflicts
if MODAL_INSTALLED:
    app = modal.App("inference-custom-blocks-web")  # Different name from original
else:
    app = None


def _get_inference_image():
    """Get the Modal Image for inference."""
    try:
        from inference.core.version import __version__
        inference_version = f"inference=={__version__}"
    except ImportError:
        # If we can't import inference (e.g., during deployment), use latest
        inference_version = "inference"

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
        .pip_install(inference_version)
        .pip_install("fastapi[standard]")  # Add FastAPI for web endpoints
    )
    return image


# Define the parameterized Modal class for execution with web endpoint
if MODAL_INSTALLED and app:

    @app.cls(
        image=_get_inference_image(),
        restrict_modal_access=True,  # Still restrict Modal access for security
        timeout=20,
        enable_memory_snapshot=True,  # Enable memory snapshotting for faster cold starts
        scaledown_window=60,
        cloud="gcp",
        region="us-central1"
    )
    class CustomBlockExecutor:
        """Parameterized Modal class for executing custom Python blocks via web endpoint."""

        # Parameterize by workspace_id
        workspace_id: str = modal.parameter()

        @modal.fastapi_endpoint(method="POST")
        def execute_block(self, request: Dict[str, Any]) -> Dict[str, Any]:
            """Execute the custom block with the given inputs via web endpoint.

            Args:
                request: JSON request containing:
                    - code_str: The Python code to execute
                    - imports: List of import statements
                    - run_function_name: Name of the function to call
                    - inputs_json: JSON string of inputs

            Returns:
                Dictionary with results or error information
            """
            import json
            import sys
            import traceback
            import os
            from datetime import datetime  # Import datetime at the top level

            import numpy as np

            # Extract parameters from request
            code_str = request.get("code_str", "")
            imports = request.get("imports", [])
            run_function_name = request.get("run_function_name", "")
            inputs_json = request.get("inputs_json", "{}")

            # Build the execution namespace
            namespace = {"__name__": "__main__"}

            try:
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

from datetime import datetime

from inference.core.workflows.core_steps.common.deserializers import (
    deserialize_image_kind,
)
"""

                # Custom decoder for special types
                def decode_inputs(obj):
                    """Decode special types in inputs."""
                    # datetime is already imported at the top level
                    
                    if isinstance(obj, dict):
                        # Check for special type markers
                        if "_type" in obj:
                            if obj["_type"] == "datetime":
                                return datetime.fromisoformat(obj["value"])
                            elif obj["_type"] == "ndarray":
                                arr = np.array(obj["value"], dtype=obj["dtype"])
                                return arr.reshape(obj["shape"])
                            elif obj["_type"] == "object":
                                return obj["value"]

                        # Check if this is a serialized WorkflowImageData
                        if obj.get("type") == "base64" and "value" in obj:
                            from inference.core.workflows.core_steps.common.deserializers import (
                                deserialize_image_kind,
                            )
                            # Decode nested datetimes first
                            if "video_metadata" in obj and obj["video_metadata"]:
                                obj["video_metadata"] = decode_inputs(
                                    obj["video_metadata"]
                                )
                            return deserialize_image_kind("input", obj)

                        # Recursively process dict values
                        return {k: decode_inputs(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [decode_inputs(item) for item in obj]
                    else:
                        return obj

                # Parse and decode inputs
                parsed_inputs = json.loads(inputs_json)
                inputs = decode_inputs(parsed_inputs)

                # Execute imports
                exec(full_imports, namespace)

                # Execute the user code
                exec(code_str, namespace)

                # Call the user function
                if run_function_name not in namespace:
                    return {
                        "error": f"Function '{run_function_name}' not found in code",
                        "error_type": "NameError",
                    }

                # Get the user's function
                user_function = namespace[run_function_name]

                # Check if function expects a 'self' parameter
                import inspect

                sig = inspect.signature(user_function)
                params = list(sig.parameters.keys())

                # If function expects 'self' as first param, create a simple object to pass
                if params and params[0] == "self":
                    # Create a simple object to pass as self
                    class BlockSelf:
                        pass

                    block_self = BlockSelf()
                    # Execute with self parameter
                    result = user_function(block_self, **inputs)
                else:
                    # Execute without self parameter
                    result = user_function(**inputs)

                # IMPORTANT: Serialize the result before returning to avoid pickle issues
                from inference.core.workflows.core_steps.common.serializers import (
                    serialize_wildcard_kind,
                )

                # Custom JSON encoder to handle datetime and other special types
                class InferenceJSONEncoder(json.JSONEncoder):
                    def default(self, obj):
                        if isinstance(obj, datetime):
                            return {"_type": "datetime", "value": obj.isoformat()}
                        elif isinstance(obj, np.ndarray):
                            return {
                                "_type": "ndarray",
                                "value": obj.tolist(),
                                "dtype": str(obj.dtype),
                                "shape": obj.shape,
                            }
                        elif hasattr(obj, "__dict__"):
                            # Handle any remaining objects by converting to dict
                            return {
                                "_type": "object",
                                "class": obj.__class__.__name__,
                                "value": str(obj),
                            }
                        return super().default(obj)

                # Serialize the result
                if isinstance(result, dict):
                    serialized_result = {}
                    for key, value in result.items():
                        serialized_result[key] = serialize_wildcard_kind(value)
                else:
                    serialized_result = serialize_wildcard_kind(result)

                # Convert to JSON string to ensure everything is pickle-safe
                json_result = json.dumps(serialized_result, cls=InferenceJSONEncoder)

                # Return the serialized result with success flag
                return {"success": True, "result": json_result}

            except Exception as e:
                # On error, return error details
                return {
                    "success": False,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "traceback": traceback.format_exc(),
                }



else:
    CustomBlockExecutor = None


class ModalWebExecutor:
    """Manages execution of Custom Python Blocks in Modal sandboxes via web endpoints."""

    def __init__(self, workspace_id: Optional[str] = None):
        """Initialize the Modal executor for a specific workspace.

        Args:
            workspace_id: The workspace ID to namespace execution, defaults to "anonymous"
        """
        self.workspace_id = workspace_id or "anonymous"
        self._base_url = None

    def _get_endpoint_url(self, workspace_id: str) -> str:
        """Get the web endpoint URL for a workspace.
        
        Args:
            workspace_id: The workspace ID
            
        Returns:
            The endpoint URL with query parameter for workspace_id
        """
        # Get base URL once (it's the same for all workspace_ids)
        if self._base_url is None:
            # First check for environment variable override
            env_url = os.environ.get("MODAL_WEB_ENDPOINT_URL")
            if env_url:
                self._base_url = env_url
            else:
                # Try to get URL dynamically from the deployed app
                try:
                    if MODAL_INSTALLED and modal:
                        # Try to get the Function and its web URL
                        cls = modal.Cls.from_name("inference-custom-blocks-web", "CustomBlockExecutor")
                        # Create a dummy instance to get the method
                        instance = cls(workspace_id="dummy")
                        # Get the execute_block method's web URL
                        if hasattr(instance, 'execute_block') and hasattr(instance.execute_block, 'get_web_url'):
                            web_url = instance.execute_block.get_web_url()
                            if web_url:
                                # Remove any query parameters that might be in the URL
                                self._base_url = web_url.split('?')[0]
                                logger.info(f"Dynamically retrieved Modal web endpoint URL: {self._base_url}")
                except Exception as e:
                    logger.debug(f"Could not dynamically retrieve Modal URL: {e}")
                
                # If we couldn't get it dynamically, construct it based on expected pattern
                if not self._base_url:
                    # URL pattern: https://{workspace}--{app}-{class}-{method_truncated}.modal.run
                    # Note: Modal truncates long labels to 63 chars with a hash suffix
                    workspace = "roboflow"
                    app_name = "inference-custom-blocks-web"
                    class_name = "customblockexecutor"
                    method_name = "execute-block"
                    
                    # The label would be: inference-custom-blocks-web-customblockexecutor-execute-block
                    # This is 62 chars, which might get truncated
                    label = f"{app_name}-{class_name}-{method_name}"
                    if len(label) > 56:  # Modal truncates at 56 chars and adds 7-char hash
                        import hashlib
                        hash_str = hashlib.sha256(label.encode()).hexdigest()[:6]
                        label = f"{label[:56]}-{hash_str}"
                    
                    self._base_url = f"https://{workspace}--{label}.modal.run"
                    logger.warning(f"Using constructed Modal URL (may not be accurate): {self._base_url}")
        
        # Add workspace_id as query parameter
        return f"{self._base_url}?workspace_id={workspace_id}"

    def execute_remote(
        self,
        block_type_name: str,
        python_code: PythonCode,
        inputs: Dict[str, Any],
        workspace_id: Optional[str] = None,
    ) -> BlockResult:
        """Execute a Custom Python Block in a Modal sandbox via web endpoint.

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
                context="modal_executor_web | installation_check",
            )

        if not MODAL_AVAILABLE:
            raise DynamicBlockError(
                public_message="Modal credentials not configured. Please set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET environment variables.",
                context="modal_executor_web | credentials_check",
            )

        # Use provided workspace_id or fall back to instance default
        workspace = workspace_id if workspace_id else self.workspace_id

        try:
            # Get endpoint URL for this workspace
            endpoint_url = self._get_endpoint_url(workspace)

            # Serialize inputs to JSON
            from datetime import datetime

            import numpy as np

            from inference.core.workflows.core_steps.common.serializers import (
                serialize_wildcard_kind,
            )

            # Custom JSON encoder for inputs
            class InputJSONEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, datetime):
                        return {"_type": "datetime", "value": obj.isoformat()}
                    elif isinstance(obj, np.ndarray):
                        return {
                            "_type": "ndarray",
                            "value": obj.tolist(),
                            "dtype": str(obj.dtype),
                            "shape": obj.shape,
                        }
                    elif hasattr(obj, "__dict__"):
                        return {
                            "_type": "object",
                            "class": obj.__class__.__name__,
                            "value": str(obj),
                        }
                    return super().default(obj)

            # Serialize inputs
            serialized_inputs = {}
            for key, value in inputs.items():
                serialized_inputs[key] = serialize_wildcard_kind(value)

            # Convert to JSON string
            inputs_json = json.dumps(serialized_inputs, cls=InputJSONEncoder)

            # Prepare request payload
            request_payload = {
                "code_str": python_code.run_function_code,
                "imports": python_code.imports or [],
                "run_function_name": python_code.run_function_name,
                "inputs_json": inputs_json,
            }

            # Make HTTP request to Modal endpoint
            response = requests.post(
                endpoint_url,
                json=request_payload,
                timeout=30,  # 30 second timeout
                headers={
                    "Content-Type": "application/json",
                }
            )

            # Check HTTP status
            if response.status_code != 200:
                raise DynamicBlockError(
                    public_message=f"Modal endpoint returned status {response.status_code}: {response.text}",
                    context="modal_executor_web | http_request",
                )

            # Parse response
            result = response.json()

            # Check for errors
            if not result.get("success", False):
                error_msg = result.get("error", "Unknown error")
                error_type = result.get("error_type", "RuntimeError")
                traceback = result.get("traceback", "")

                raise DynamicBlockError(
                    public_message=f"{error_type}: {error_msg}",
                    context=f"modal_executor_web | remote_execution\n{traceback}",
                )

            # Get the result and deserialize from JSON
            json_result = result.get("result", "{}")
            return self._deserialize_json_result(json_result)

        except requests.exceptions.RequestException as e:
            raise DynamicBlockError(
                public_message=f"Failed to connect to Modal endpoint: {str(e)}",
                context="modal_executor_web | http_connection",
            )
        except Exception as e:
            if isinstance(e, DynamicBlockError):
                raise
            raise DynamicBlockError(
                public_message=f"Failed to execute custom block remotely: {str(e)}",
                context="modal_executor_web | remote_execution",
            )

    def _deserialize_json_result(self, json_result: str) -> BlockResult:
        """Deserialize JSON result from Modal transport.

        Args:
            json_result: JSON string from Modal function

        Returns:
            BlockResult with deserialized outputs
        """
        from datetime import datetime

        import numpy as np

        from inference.core.workflows.core_steps.common.deserializers import (
            deserialize_image_kind,
        )

        # Custom JSON decoder to handle special types
        def decode_special_types(obj, defer_images=False):
            """Recursively decode special types in JSON objects.

            Args:
                obj: Object to decode
                defer_images: If True, do not deserialize images yet (for first pass)
            """
            if isinstance(obj, dict):
                # Check for special type markers
                if "_type" in obj:
                    if obj["_type"] == "datetime":
                        return datetime.fromisoformat(obj["value"])
                    elif obj["_type"] == "ndarray":
                        arr = np.array(obj["value"], dtype=obj["dtype"])
                        return arr.reshape(obj["shape"])
                    elif obj["_type"] == "object":
                        # For unknown objects, return as a string representation
                        return obj["value"]

                # Check if this is an image that needs deserialization
                if not defer_images and obj.get("type") == "base64" and "value" in obj:
                    # This is a serialized WorkflowImageData
                    # First make sure any nested datetimes are decoded
                    if "video_metadata" in obj and obj["video_metadata"]:
                        obj["video_metadata"] = decode_special_types(
                            obj["video_metadata"], defer_images=False
                        )
                    return deserialize_image_kind("result", obj)

                # Recursively process dict values
                return {
                    k: decode_special_types(v, defer_images) for k, v in obj.items()
                }
            elif isinstance(obj, list):
                # Recursively process list items
                return [decode_special_types(item, defer_images) for item in obj]
            else:
                return obj

        # Parse JSON
        parsed = json.loads(json_result)

        # First pass: decode everything except images (to handle datetimes in metadata)
        decoded = decode_special_types(parsed, defer_images=True)

        # Second pass: now deserialize images with all datetimes already decoded
        decoded = decode_special_types(decoded, defer_images=False)

        # Ensure result is a dict for BlockResult format
        if isinstance(decoded, dict):
            return decoded
        else:
            # Wrap non-dict results in BlockResult format
            return {"result": decoded}


# For backwards compatibility and parallel testing, provide aliases
ModalExecutor = ModalWebExecutor


def validate_code_in_modal(
    python_code: PythonCode, workspace_id: Optional[str] = None
) -> bool:
    """Validate Python code syntax in a Modal sandbox via web endpoint.

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
            context="modal_executor_web | installation_check",
        )

    if not MODAL_AVAILABLE:
        raise DynamicBlockError(
            public_message="Modal credentials not configured. Please set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET environment variables.",
            context="modal_executor_web | credentials_check",
        )

    workspace = workspace_id or "anonymous"
    
    # Construct the full code to validate (same as in create_dynamic_module)
    full_code = python_code.run_function_code
    if python_code.init_function_code:
        full_code += "\n\n" + python_code.init_function_code
    
    # Escape the code for safe embedding in the validation function
    # Use repr() to properly escape quotes and special characters
    escaped_code = repr(full_code)

    # Simple validation code that checks syntax
    validation_code = PythonCode(
        type="PythonCode",
        imports=[],
        run_function_code=f"""
import ast

def validate_syntax():
    try:
        # Try to compile the user code
        code = {escaped_code}
        compile(code, "<string>", "exec")
        # Try to parse as AST to check structure
        ast.parse(code)
        return {{"valid": True}}
    except SyntaxError as e:
        return {{"valid": False, "error": str(e), "line": e.lineno}}
    except Exception as e:
        return {{"valid": False, "error": str(e)}}
""",
        run_function_name="validate_syntax",
        init_function_code=None,
        init_function_name="init",
    )

    executor = ModalWebExecutor(workspace_id=workspace)

    try:
        # For validation, we don't need complex inputs, just pass empty JSON
        result = executor.execute_remote(
            block_type_name="validation",
            python_code=validation_code,
            inputs={},
            workspace_id=workspace,
        )

        if result.get("valid") is False:
            error_msg = result.get("error", "Unknown syntax error")
            line_no = result.get("line", None)
            if line_no:
                error_msg = f"Line {line_no}: {error_msg}"
            raise DynamicBlockError(
                public_message=f"Code validation failed: {error_msg}",
                context="modal_executor_web | code_validation",
            )

        return True

    except Exception as e:
        if isinstance(e, DynamicBlockError):
            raise
        raise DynamicBlockError(
            public_message=f"Code validation failed: {str(e)}",
            context="modal_executor_web | code_validation",
        )
