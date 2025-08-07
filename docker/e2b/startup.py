#!/usr/bin/env python3
"""
Startup script for E2B Sandbox to pre-import inference modules.
This reduces cold start time when executing Custom Python Blocks.
"""

import sys
import os

# Add app directory to Python path
sys.path.insert(0, '/app')

# Pre-import common modules used in Custom Python Blocks
print("Pre-importing inference modules...")

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

print("Inference modules pre-imported successfully")

# Keep the process running for E2B
while True:
    time.sleep(1)
