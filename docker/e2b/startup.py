#!/usr/bin/env python3
"""
Startup script for E2B Sandbox to configure the environment for inference.
This ensures /app is in the Python path and pre-imports common modules.
"""

import sys
import os

# Ensure /app is in Python path (should already be from PYTHONPATH env var)
if '/app' not in sys.path:
    sys.path.insert(0, '/app')

# Verify the environment is set up correctly
print("E2B Sandbox Environment Initialized")
print(f"Python path includes: {sys.path[:3]}")

# Test that inference is accessible
try:
    import inference
    print(f"✅ Inference available at: {inference.__file__}")
except ImportError as e:
    print(f"⚠️  Inference import issue: {e}")

# Pre-import common modules to reduce cold start time
print("Pre-importing common modules...")

try:
    # Core imports that are always available in dynamic blocks
    import supervision as sv
    import numpy as np
    import math
    import time
    import json
    import requests
    import cv2
    import shapely
    
    # Inference-specific imports
    from inference.core.workflows.execution_engine.entities.base import Batch, WorkflowImageData
    from inference.core.workflows.prototypes.block import BlockResult
    
    # Pre-import serializers and deserializers
    from inference.core.workflows.core_steps.common.serializers import (
        deserialize_numpy_array,
        deserialize_detection_from_inference_format,
    )
    from inference.core.workflows.core_steps.common.deserializers import (
        deserialize_detection_from_dict,
        deserialize_numpy_from_base64_string,
    )
    
    print("✅ All modules pre-imported successfully")
except ImportError as e:
    print(f"⚠️  Some imports failed: {e}")

print("Sandbox ready")
