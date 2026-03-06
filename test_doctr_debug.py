"""
Debug script to see what packages are returned by the API
"""
import os
os.environ["ROBOFLOW_API_KEY"] = "zaRavHwbvIXpGerDM3wi"
os.environ["API_BASE_URL"] = "https://staging.api.roboflow.com"

import sys
sys.path.insert(0, "/home/matveipopov/inference/inference_models")

from inference_models.models.auto_loaders.packages_registry import get_package_candidates

# Check what packages the API returns for doctr/default
print("Getting package candidates for doctr/default...")
try:
    candidates = get_package_candidates("doctr/default")
    print(f"\nFound {len(candidates)} candidate(s):")
    for i, pkg in enumerate(candidates):
        print(f"\n  Candidate {i+1}:")
        print(f"    package_id: {pkg.package_id}")
        print(f"    model_architecture: {pkg.model_architecture}")
        print(f"    task_type: {pkg.task_type}")
        print(f"    backend_type: {pkg.backend_type}")
        print(f"    model_features: {pkg.model_features}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
