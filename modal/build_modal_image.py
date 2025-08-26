#!/usr/bin/env python3
"""
Build and push a Modal Image for Custom Python Blocks execution.

This script creates a shared Modal Image with the inference package and
all necessary dependencies for running Custom Python Blocks.

Usage:
    python modal/build_modal_image.py
"""

import modal
import subprocess
import sys
from pathlib import Path

# Get the current inference version
def get_inference_version():
    try:
        # Read from inference/core/version.py
        version_file = Path(__file__).parent.parent / "inference" / "core" / "version.py"
        with open(version_file) as f:
            for line in f:
                if line.startswith("__version__"):
                    return line.split("=")[1].strip().strip('"')
    except Exception:
        return "0.53.0"  # Fallback version

INFERENCE_VERSION = get_inference_version()
IMAGE_NAME = f"inference-custom-blocks-{INFERENCE_VERSION.replace('.', '-')}"

def build_inference_image():
    """Build the Modal Image for Custom Python Blocks."""
    
    print(f"Building Modal Image for inference version {INFERENCE_VERSION}")
    
    # Create the Modal Image with all necessary dependencies
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install(
            "libgl1-mesa-glx",
            "libglib2.0-0",
            "libgomp1",
            "libsm6",
            "libxext6",
            "libxrender-dev",
            "libgomp1",
            "ffmpeg",
            "wget",
        )
        .uv_pip_install(f"inference=={INFERENCE_VERSION}")
        .run_function(pre_import_modules)
    )
    
    return image

def pre_import_modules():
    """Pre-import common modules to reduce cold starts."""
    import sys
    import numpy as np
    import supervision as sv
    import cv2
    import json
    import time
    import math
    import requests
    import shapely
    
    # Import inference modules
    from inference.core.workflows.execution_engine.entities.base import (
        Batch, 
        WorkflowImageData
    )
    from inference.core.workflows.prototypes.block import BlockResult
    
    print("Pre-imported common modules successfully")

if __name__ == "__main__":
    # Build the image
    image = build_inference_image()
    
    # Register the image with a name that can be referenced
    app = modal.App(name=IMAGE_NAME)
    
    # Create a dummy function to force image building
    @app.function(image=image)
    def dummy():
        return "Image built successfully"
    
    # Deploy the app to build and register the image
    print(f"Deploying image as app: {IMAGE_NAME}")
    app.deploy(name=IMAGE_NAME)
    
    print(f"✅ Successfully built and registered Modal Image: {IMAGE_NAME}")
    print(f"   You can now reference this image in other Modal apps")
