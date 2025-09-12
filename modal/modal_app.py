"""
Modal app definition for Custom Python Blocks web endpoint.

This module contains the Modal-specific code for executing untrusted user code
in sandboxes. It's separated from the main executor to avoid requiring Modal
as a dependency for the main inference package.
"""

from typing import Any, Dict
import base64

import modal

# Create the Modal App
app = modal.App("webexec")


def get_inference_image():
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


@app.cls(
    image=get_inference_image(),
    restrict_modal_access=True,  # Restrict Modal access for security
    timeout=20,
    enable_memory_snapshot=True,  # Enable memory snapshotting for faster cold starts
    scaledown_window=60,
    cloud="aws",
    region="us-east-1",
    buffer_containers=1
)
class Executor:
    """Parameterized Modal class for executing custom Python blocks via web endpoint."""

    # Parameterize by workspace_id
    workspace_id: str = modal.parameter()

    @modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
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
        import os
        import sys
        import traceback
        from datetime import datetime  # Import datetime at the top level

        import numpy as np

        # Import deserializers at the top level so they're available for decode_inputs
        from inference.core.workflows.core_steps.common.deserializers import (
            deserialize_image_kind,
            deserialize_detections_kind,
            deserialize_video_metadata_kind,
            deserialize_classification_prediction_kind,
        )

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
                        elif obj["_type"] == "bytes":
                            return base64.b64decode(obj["value"])
                        elif obj["_type"] == "ndarray":
                            arr = np.array(obj["value"], dtype=obj["dtype"])
                            return arr.reshape(obj["shape"])
                        elif obj["_type"] == "object":
                            return obj["value"]
                        elif obj["_type"] == "sv_detections":
                            # First decode any nested special types in the dict
                            decoded_obj = {k: decode_inputs(v) for k, v in obj.items() if k != "_type"}
                            return deserialize_detections_kind("input", decoded_obj)
                        elif obj["_type"] == "video_metadata":
                            # First decode any nested special types
                            decoded_obj = {k: decode_inputs(v) for k, v in obj.items() if k != "_type"}
                            return deserialize_video_metadata_kind("input", decoded_obj)
                        elif obj["_type"] == "workflow_image":
                            # First decode any nested special types
                            decoded_obj = {k: decode_inputs(v) for k, v in obj.items() if k != "_type"}
                            return deserialize_image_kind("input", decoded_obj)

                    # For backward compatibility, check if this is a WorkflowImageData without _type marker
                    if obj.get("type") == "base64" and "value" in obj and "_type" not in obj:
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
                        return {
                            "_type": "datetime",
                            "value": obj.isoformat(),
                        }
                    elif isinstance(obj, bytes):
                        return {
                            "_type": "bytes",
                            "value": base64.b64encode(obj).decode("utf-8"),
                        }
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
            result = {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
            }

            # Get the line number and function name from evaluated code
            tb = traceback.extract_tb(e.__traceback__)
            if tb:
                frame = tb[-1]
                result["line_number"] = frame.lineno
                result["function_name"] = frame.name

            return result
