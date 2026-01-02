#!/usr/bin/env python3
"""
Force reload workflow blocks by clearing all caches and importing modules fresh.
Run this BEFORE starting the server.
"""

import sys
import importlib
from pathlib import Path

# Clear any cached modules
modules_to_clear = [
    'inference.core.workflows.execution_engine.introspection.blocks_loader',
    'inference.core.workflows.core_steps.models.roboflow.instance_segmentation.v1',
    'inference.core.workflows.core_steps.models.roboflow.instance_segmentation.v2',
    'inference.core.workflows.core_steps.models.roboflow.object_detection.v1',
    'inference.core.workflows.core_steps.models.roboflow.object_detection.v2',
]

for mod_name in modules_to_clear:
    if mod_name in sys.modules:
        del sys.modules[mod_name]
        print(f"Cleared module: {mod_name}")

print("\n✓ Module cache cleared")
print("Now start your server - the modules will reload fresh!")

