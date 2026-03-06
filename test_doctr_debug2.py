"""
Debug script to see what packages are returned by the weights provider
"""
import os
os.environ["ROBOFLOW_API_KEY"] = "zaRavHwbvIXpGerDM3wi"
os.environ["API_BASE_URL"] = "https://staging.api.roboflow.com"

import sys
sys.path.insert(0, "/home/matveipopov/inference/inference_models")

from inference_models.weights_providers.roboflow import RoboflowWeightsProvider

# Check what packages the weights provider returns for doctr/default
print("Getting packages via RoboflowWeightsProvider...")
try:
    wp = RoboflowWeightsProvider()
    candidates = wp.get_model_packages("doctr/default")
    print(f"\nFound {len(candidates)} candidate(s):")
    for i, pkg in enumerate(candidates):
        print(f"\n  Candidate {i+1}:")
        print(f"    package_id: {pkg.package_id}")
        print(f"    model_architecture: {pkg.model_architecture}")
        print(f"    task_type: {pkg.task_type}")
        print(f"    backend_type: {pkg.backend}")
        print(f"    model_features: {pkg.model_features}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
