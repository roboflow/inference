"""
E2B Sandbox executor for Custom Python Blocks.
Handles remote execution of user-provided Python code in isolated E2B sandboxes.
"""

import json
import traceback
from typing import Any, Dict, List, Optional, Type
from uuid import uuid4

import numpy as np
from e2b_code_interpreter import Sandbox

from inference import get_version
from inference.core.env import (
    E2B_API_KEY,
    E2B_SANDBOX_IDLE_TIMEOUT,
    E2B_SANDBOX_TIMEOUT,
    E2B_TEMPLATE_ID,
)
from inference.core.workflows.errors import DynamicBlockError
from inference.core.workflows.execution_engine.v1.dynamic_blocks.entities import (
    PythonCode,
)
from inference.core.workflows.prototypes.block import BlockResult

# Import serializers and deserializers for workflow kinds
from inference.core.workflows.core_steps.common.serializers import (
    serialise_sv_detections,
    serialise_image,
    serialize_video_metadata_kind,
)
from inference.core.workflows.core_steps.common.deserializers import (
    deserialize_detections_kind,
    deserialize_image_kind,
    deserialize_video_metadata_kind,
    deserialize_numpy_array,
)
from inference.core.workflows.execution_engine.entities.base import (
    WorkflowImageData,
)


class E2BSandboxExecutor:
    """
    Manages execution of Custom Python Blocks in E2B sandboxes.
    
    Future enhancement: Will support sandbox reuse via Redis-based
    mapping between user code and running sandboxes.
    """
    
    def __init__(self):
        """Initialize the E2B executor."""
        self.api_key = E2B_API_KEY
        self.template_id = self._get_template_id()
        self.timeout = E2B_SANDBOX_TIMEOUT
        self.idle_timeout = E2B_SANDBOX_IDLE_TIMEOUT
        
        # Future: Initialize Redis connection for sandbox mapping
        self._sandbox_cache = {}  # Placeholder for future Redis integration
    
    def _get_template_id(self) -> str:
        """Get the E2B template ID, defaulting to inference version if not set."""
        if E2B_TEMPLATE_ID:
            return E2B_TEMPLATE_ID
        
        # Default to inference version-based template
        inference_version = get_version()
        # Replace dots with dashes for E2B compatibility
        version_str = inference_version.replace('.', '-')
        return f"inference-sandbox-v{version_str}"
    
    def execute_in_sandbox(
        self,
        python_code: PythonCode,
        block_type_name: str,
        inputs: Dict[str, Any],
        manifest_outputs: Dict[str, Any],
    ) -> BlockResult:
        """
        Execute Custom Python Block code in an E2B sandbox.
        
        Args:
            python_code: The Python code definition
            block_type_name: Name of the dynamic block type
            inputs: Input parameters for the block
            manifest_outputs: Output definitions from the block manifest
            
        Returns:
            BlockResult with deserialized outputs
        """
        # For now, always create a new sandbox (future: check cache)
        sandbox_id = self._create_sandbox(python_code, block_type_name)
        
        try:
            # Serialize inputs
            serialized_inputs = self._serialize_inputs(inputs)
            
            # Execute the function in sandbox
            result = self._execute_function(
                sandbox_id=sandbox_id,
                function_name=python_code.run_function_name,
                inputs=serialized_inputs,
                python_code=python_code,
            )
            
            # Deserialize outputs
            deserialized_result = self._deserialize_outputs(result, manifest_outputs)
            
            return deserialized_result
            
        finally:
            # For now, always terminate sandbox after execution
            self._terminate_sandbox(sandbox_id)
    
    def _create_sandbox(self, python_code: PythonCode, block_type_name: str) -> str:
        """
        Create a new E2B sandbox and initialize it with the user's code.
        
        Future: Check cache for existing sandbox before creating new one.
        """
        try:
            # Create sandbox with our custom template
            sandbox = Sandbox(
                api_key=self.api_key,
                template_id=self.template_id,
                timeout=self.timeout,
            )
            
            # Generate unique sandbox ID
            sandbox_id = str(uuid4())
            self._sandbox_cache[sandbox_id] = sandbox
            
            # Initialize sandbox with user's code
            self._initialize_sandbox_code(
                sandbox=sandbox,
                python_code=python_code,
                block_type_name=block_type_name,
            )
            
            return sandbox_id
            
        except Exception as e:
            raise DynamicBlockError(
                public_message=f"Failed to create E2B sandbox for block '{block_type_name}': {str(e)}",
                context="workflow_execution | e2b_sandbox_creation",
            )
    
    def _initialize_sandbox_code(
        self,
        sandbox: Sandbox,
        python_code: PythonCode,
        block_type_name: str,
    ) -> None:
        """Initialize the sandbox with the user's custom code."""
        # Build the complete code including imports and wrapper
        wrapper_code = self._build_wrapper_code(python_code)
        
        try:
            # Execute the code definition in sandbox
            result = sandbox.run_code(wrapper_code)
            
            if result.error:
                raise DynamicBlockError(
                    public_message=f"Failed to initialize code in sandbox for block '{block_type_name}': {result.error}",
                    context="workflow_execution | e2b_code_initialization",
                )
                
            # If there's an init function, execute it
            if python_code.init_function_code:
                init_result = sandbox.run_code(f"_init_results = {python_code.init_function_name}()")
                if init_result.error:
                    raise DynamicBlockError(
                        public_message=f"Failed to execute init function for block '{block_type_name}': {init_result.error}",
                        context="workflow_execution | e2b_init_execution",
                    )
                    
        except Exception as e:
            if isinstance(e, DynamicBlockError):
                raise
            raise DynamicBlockError(
                public_message=f"Error initializing sandbox code for block '{block_type_name}': {str(e)}",
                context="workflow_execution | e2b_code_initialization",
            )
    
    def _build_wrapper_code(self, python_code: PythonCode) -> str:
        """
        Build the complete Python code including imports, user code, and serialization wrapper.
        """
        imports = [
            "import json",
            "import base64",
            "import numpy as np",
            "import supervision as sv",
            "import cv2",
            "import math",
            "import time",
            "import os",
            "import requests",
            "import shapely",
            "from typing import Any, List, Dict, Set, Optional",
            "from inference.core.workflows.execution_engine.entities.base import Batch, WorkflowImageData",
            "from inference.core.workflows.prototypes.block import BlockResult",
            "from inference.core.workflows.core_steps.common.serializers import *",
            "from inference.core.workflows.core_steps.common.deserializers import *",
        ]
        
        # Add user's custom imports
        if python_code.imports:
            imports.extend(python_code.imports)
        
        imports_str = "\n".join(imports)
        
        # Add init function if provided
        init_code = ""
        if python_code.init_function_code:
            init_code = f"\n{python_code.init_function_code}\n"
        
        # Add the user's run function
        run_code = python_code.run_function_code
        
        # Create wrapper function for serialization/deserialization
        wrapper = f"""
{imports_str}

{init_code}

{run_code}

# Import base64 and json for serialization
import base64
import json
import numpy as np

# Wrapper function to handle serialization
def execute_wrapped(serialized_inputs_json: str) -> str:
    \"\"\"
    Wrapper that deserializes inputs, calls user function, and serializes outputs.
    \"\"\"
    try:
        # Deserialize inputs
        serialized_inputs = json.loads(serialized_inputs_json)
        inputs = {{}}
        
        for key, value in serialized_inputs.items():
            if value is None:
                inputs[key] = None
            elif isinstance(value, dict) and '_serialized_type' in value:
                # Handle special serialized types
                serialized_type = value['_serialized_type']
                if serialized_type == 'numpy_array':
                    # Deserialize base64 numpy array
                    data = base64.b64decode(value['data'])
                    inputs[key] = np.frombuffer(data, dtype=value.get('dtype', 'float64')).reshape(value.get('shape', [-1]))
                elif serialized_type == 'detections':
                    # For detections, we pass the raw dict
                    inputs[key] = value['data']
                elif serialized_type == 'image':
                    # For images, pass the dict
                    inputs[key] = value['data']
                elif serialized_type == 'video_metadata':
                    inputs[key] = value['data']
                else:
                    inputs[key] = value['data']
            else:
                inputs[key] = value        
        # Create mock self object with init results if available
        class MockSelf:
            def __init__(self):
                self._init_results = globals().get('_init_results', {})
        
        mock_self = MockSelf()
        
        # Call the user's run function
        result = {python_code.run_function_name}(mock_self, **inputs)
        
        # Serialize outputs
        serialized_result = {{}}
        for key, value in result.items():
            if value is None:
                serialized_result[key] = None
            elif isinstance(value, np.ndarray):
                # Serialize numpy array to base64
                serialized_result[key] = {{
                    '_serialized_type': 'numpy_array',
                    'data': base64.b64encode(value.tobytes()).decode('utf-8'),
                    'dtype': str(value.dtype),
                    'shape': list(value.shape)
                }}
            elif hasattr(value, '__class__') and 'Detections' in str(value.__class__):
                # Serialize supervision Detections
                from inference.core.workflows.core_steps.common.serializers import serialise_sv_detections
                serialized_result[key] = {{
                    '_serialized_type': 'detections',
                    'data': serialise_sv_detections(value)
                }}
            elif hasattr(value, '__class__') and 'WorkflowImageData' in str(value.__class__):
                # Serialize WorkflowImageData
                from inference.core.workflows.core_steps.common.serializers import serialise_image
                serialized_result[key] = {{
                    '_serialized_type': 'image',
                    'data': serialise_image(value)
                }}
            elif hasattr(value, '__class__') and 'VideoMetadata' in str(value.__class__):
                # Serialize VideoMetadata
                from inference.core.workflows.core_steps.common.serializers import serialize_video_metadata_kind
                serialized_result[key] = {{
                    '_serialized_type': 'video_metadata',
                    'data': serialize_video_metadata_kind(value)
                }}
            else:
                # For basic types and lists/dicts, just pass through
                serialized_result[key] = value
        
        return json.dumps(serialized_result)
        
    except Exception as e:
        import traceback
        error_info = {{
            'error': str(e),
            'traceback': traceback.format_exc()
        }}
        return json.dumps({{'_error': error_info}})
"""
        
        return wrapper
    
    def _execute_function(
        self,
        sandbox_id: str,
        function_name: str,
        inputs: Dict[str, Any],
        python_code: PythonCode,
    ) -> Dict[str, Any]:
        """Execute the wrapped function in the sandbox with serialized inputs."""
        sandbox = self._sandbox_cache.get(sandbox_id)
        if not sandbox:
            raise DynamicBlockError(
                public_message=f"Sandbox {sandbox_id} not found in cache",
                context="workflow_execution | e2b_sandbox_execution",
            )
        
        try:
            # Convert inputs to JSON
            inputs_json = json.dumps(inputs)
            
            # Execute the wrapper function with timeout of 20 seconds
            execution_code = f"""
import json
result = execute_wrapped('{inputs_json.replace("'", "\\'")}')
print(result)
"""
            
            result = sandbox.run_code(execution_code, timeout=20)
            
            if result.error:
                raise DynamicBlockError(
                    public_message=f"Error executing function in sandbox: {result.error}",
                    context="workflow_execution | e2b_function_execution",
                )
            
            # Parse the result from stdout
            if result.logs and result.logs.stdout:
                output_lines = result.logs.stdout
                if output_lines:
                    result_json = output_lines[-1] if isinstance(output_lines, list) else output_lines
                    parsed_result = json.loads(result_json)
                    
                    # Check for errors
                    if '_error' in parsed_result:
                        error_info = parsed_result['_error']
                        raise DynamicBlockError(
                            public_message=f"Error in custom code execution: {error_info['error']}",
                            context="workflow_execution | custom_code_error",
                        )
                    
                    return parsed_result
            
            raise DynamicBlockError(
                public_message="No output received from sandbox execution",
                context="workflow_execution | e2b_no_output",
            )
            
        except json.JSONDecodeError as e:
            raise DynamicBlockError(
                public_message=f"Failed to parse sandbox output: {str(e)}",
                context="workflow_execution | e2b_output_parsing",
            )
        except Exception as e:
            if isinstance(e, DynamicBlockError):
                raise
            raise DynamicBlockError(
                public_message=f"Unexpected error during sandbox execution: {str(e)}",
                context="workflow_execution | e2b_execution_error",
            )
    
    def _serialize_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize complex input types for transmission to sandbox."""
        import base64
        serialized = {}
        
        for key, value in inputs.items():
            if value is None:
                serialized[key] = None
            elif isinstance(value, np.ndarray):
                serialized[key] = {
                    '_serialized_type': 'numpy_array',
                    'data': base64.b64encode(value.tobytes()).decode('utf-8'),
                    'dtype': str(value.dtype),
                    'shape': list(value.shape)
                }
            # Handle Batch types
            elif hasattr(value, '__class__') and 'Batch' in str(value.__class__):
                # Serialize each item in the batch
                batch_items = []
                for item in value:
                    if isinstance(item, WorkflowImageData):
                        batch_items.append({
                            '_serialized_type': 'image',
                            'data': serialise_image(item)
                        })
                    elif hasattr(item, 'to_dict'):
                        batch_items.append(item.to_dict())
                    else:
                        batch_items.append(item)
                serialized[key] = batch_items
            # Handle supervision Detections
            elif hasattr(value, '__class__') and 'Detections' in str(value.__class__):
                serialized[key] = {
                    '_serialized_type': 'detections',
                    'data': serialise_sv_detections(value)
                }
            # Handle WorkflowImageData
            elif hasattr(value, '__class__') and 'WorkflowImageData' in str(value.__class__):
                serialized[key] = {
                    '_serialized_type': 'image',
                    'data': serialise_image(value)
                }
            else:
                # For basic types, lists, dicts, pass through
                serialized[key] = value
        
        return serialized
    
    def _deserialize_outputs(
        self,
        outputs: Dict[str, Any],
        manifest_outputs: Dict[str, Any]
    ) -> BlockResult:
        """Deserialize outputs received from sandbox."""
        import base64
        deserialized = {}
        
        for key, value in outputs.items():
            if value is None:
                deserialized[key] = None
            elif isinstance(value, dict) and '_serialized_type' in value:
                serialized_type = value['_serialized_type']
                if serialized_type == 'numpy_array':
                    # Deserialize numpy array from base64
                    data = base64.b64decode(value['data'])
                    dtype = value.get('dtype', 'float64')
                    shape = value.get('shape', [-1])
                    deserialized[key] = np.frombuffer(data, dtype=dtype).reshape(shape)
                elif serialized_type == 'detections':
                    # For detections, use the deserializer
                    deserialized[key] = deserialize_detections_kind(
                        parameter=key,
                        detections=value['data']
                    )
                elif serialized_type == 'image':
                    # For images, use the deserializer
                    deserialized[key] = deserialize_image_kind(
                        parameter=key,
                        image=value['data']
                    )
                elif serialized_type == 'video_metadata':
                    deserialized[key] = deserialize_video_metadata_kind(
                        parameter=key,
                        video_metadata=value['data']
                    )
                else:
                    deserialized[key] = value
            else:
                deserialized[key] = value
        
        return deserialized
    
    def _terminate_sandbox(self, sandbox_id: str) -> None:
        """Terminate a sandbox and remove it from cache."""
        sandbox = self._sandbox_cache.pop(sandbox_id, None)
        if sandbox:
            try:
                sandbox.close()
            except Exception as e:
                # Log but don't raise - sandbox termination is best effort
                print(f"Warning: Failed to terminate sandbox {sandbox_id}: {str(e)}")


# Singleton instance for reuse across executions
_executor_instance = None


def get_e2b_executor() -> E2BSandboxExecutor:
    """Get or create the singleton E2B executor instance."""
    global _executor_instance
    if _executor_instance is None:
        _executor_instance = E2BSandboxExecutor()
    return _executor_instance