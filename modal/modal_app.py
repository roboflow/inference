"""
Modal app definition for Custom Python Blocks web endpoint.

This module contains the Modal-specific code for executing untrusted user code
in sandboxes. It's separated from the main executor to avoid requiring Modal
as a dependency for the main inference package.
"""

from typing import Any, Dict
import base64
import hashlib

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
    buffer_containers=1,
)
class Executor:
    """Parameterized Modal class for executing custom Python blocks via web endpoint."""

    # Parameterize by workspace_id
    workspace_id: str = modal.parameter()
    
    # Store state for each unique code block within this container
    # Key is the hash of the code, value is the namespace dict for that code
    _code_namespaces: Dict[str, dict] = {}
    
    # Shared globals dict that all custom python blocks can access
    _shared_globals: Dict[str, Any] = {}

    @modal.enter()
    def identify(self):
        print(f"Initializing sandbox for {self.workspace_id}")
        # Initialize the namespaces dict and shared globals
        self._code_namespaces = {}
        self._shared_globals = {}

    def _get_code_hash(self, code_str: str, imports: list) -> str:
        """Compute a stable hash for the code to identify unique blocks."""
        # Combine code and imports to create a unique identifier
        content = code_str + "\n" + "\n".join(imports if imports else [])
        return hashlib.md5(content.encode()).hexdigest()

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
        import traceback
        from datetime import datetime  # Import datetime at the top level

        import numpy as np

        # Import deserializers at the top level so they're available for decode_inputs
        from inference.core.workflows.core_steps.common.deserializers import (
            deserialize_image_kind,
            deserialize_detections_kind,
            deserialize_video_metadata_kind,
        )

        # Extract parameters from request
        code_str = request.get("code_str", "")
        imports = request.get("imports", [])
        run_function_name = request.get("run_function_name", "")
        inputs_json = request.get("inputs_json", "{}")

        # Get the hash of this code to identify it uniquely
        code_hash = self._get_code_hash(code_str, imports)
        
        # Check if we already have a namespace for this code
        if code_hash not in self._code_namespaces:
            # Create a new namespace for this code block
            self._code_namespaces[code_hash] = {
                "__name__": "__main__",
                "globals": self._shared_globals  # Inject the shared globals dict
            }
            
            # Execute imports and code in the namespace to initialize it
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
            try:
                # Execute imports in this namespace
                exec(full_imports, self._code_namespaces[code_hash])
                
                # Execute the user code to define functions in this namespace
                exec(code_str, self._code_namespaces[code_hash])
            except Exception as e:
                # If there's an error in code initialization, remove the namespace
                del self._code_namespaces[code_hash]
                return {
                    "success": False,
                    "error": f"Code initialization failed: {str(e)}",
                    "error_type": type(e).__name__,
                }
        
        # Get the namespace for this code
        namespace = self._code_namespaces[code_hash]

        try:
            # we should import serialize_for_modal_remote_execution and deserialize_for_modal_remote_execution
            # from inference package, but need to have them included in the modal build for that
            # so just copy pasted for now
            from inference.core.workflows.prototypes.block import BlockResult
            from datetime import datetime
            from inference.core.workflows.core_steps.common.deserializers import (
                deserialize_image_kind,
                deserialize_detections_kind,
                deserialize_video_metadata_kind,
            )

            def serialize_for_modal_remote_execution(inputs: Dict[str, Any]) -> str:
                from datetime import datetime
                import numpy as np

                class InputJSONEncoder(json.JSONEncoder):
                    def default(self, obj):
                        if isinstance(obj, datetime):
                            return {"_type": "datetime", "value": obj.isoformat()}
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
                            return {
                                "_type": "object",
                                "class": obj.__class__.__name__,
                                "value": str(obj),
                            }
                        return super().default(obj)

                # Patch inputs with type markers for Modal serialization
                def patch_for_modal_serialization(value):
                    """Serialize value and add _type markers for Modal deserialization."""
                    import supervision as sv
                    from inference.core.workflows.execution_engine.entities.base import (
                        WorkflowImageData,
                        VideoMetadata,
                    )
                    from inference.core.workflows.core_steps.common.serializers import (
                        serialize_video_metadata_kind,
                        serialise_sv_detections,
                        serialise_image,
                    )

                    # Apply standard serialization and add type markers based on original type
                    if isinstance(value, sv.Detections):
                        serialized = serialise_sv_detections(detections=value)
                        serialized["_type"] = "sv_detections"
                    elif isinstance(value, WorkflowImageData):
                        serialized = serialise_image(image=value)
                        serialized["_type"] = "workflow_image"
                    elif isinstance(value, VideoMetadata):
                        serialized = serialize_video_metadata_kind(value)
                        serialized["_type"] = "video_metadata"
                    elif isinstance(value, dict):
                        # Recursively process dict values
                        serialized = {
                            k: patch_for_modal_serialization(v) if k != "_type" else v
                            for k, v in value.items()
                        }
                    elif isinstance(value, list):
                        # Recursively process list items
                        serialized = [
                            patch_for_modal_serialization(item) for item in value
                        ]
                    else:
                        serialized = value

                    return serialized

                serialized_inputs = {}
                for key, value in inputs.items():
                    serialized_inputs[key] = patch_for_modal_serialization(value)

                # Convert to JSON string
                return json.dumps(serialized_inputs, cls=InputJSONEncoder)

            def deserialize_for_modal_remote_execution(json_str: str) -> BlockResult:
                def decode_inputs(obj):
                    """Decode from modal remote execution."""
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
                                decoded_obj = {
                                    k: decode_inputs(v)
                                    for k, v in obj.items()
                                    if k != "_type"
                                }
                                return deserialize_detections_kind("input", decoded_obj)
                            elif obj["_type"] == "video_metadata":
                                # First decode any nested special types
                                decoded_obj = {
                                    k: decode_inputs(v)
                                    for k, v in obj.items()
                                    if k != "_type"
                                }
                                return deserialize_video_metadata_kind(
                                    "input", decoded_obj
                                )
                            elif obj["_type"] == "workflow_image":
                                # First decode any nested special types
                                decoded_obj = {
                                    k: decode_inputs(v)
                                    for k, v in obj.items()
                                    if k != "_type"
                                }
                                return deserialize_image_kind("input", decoded_obj)

                        # TODO: Not sure we actually need this anymore?
                        # For backward compatibility, check if this is a WorkflowImageData without _type marker
                        if (
                            obj.get("type") == "base64"
                            and "value" in obj
                            and "_type" not in obj
                        ):
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
                parsed_inputs = json.loads(json_str)
                inputs = decode_inputs(parsed_inputs)
                return inputs

            inputs = deserialize_for_modal_remote_execution(inputs_json)

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

            json_result = serialize_for_modal_remote_execution(result)

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
