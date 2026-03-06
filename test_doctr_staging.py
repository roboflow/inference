"""
Test DocTR on actual staging environment
"""
import os
os.environ["PROJECT"] = "roboflow-staging"
os.environ["API_BASE_URL"] = "https://api.roboflow.one"
os.environ["ROBOFLOW_API_KEY"] = "GYaBMWQ6xDqVFsEJIoan"
os.environ["ROBOFLOW_ENVIRONMENT"] = "staging"

import sys
sys.path.insert(0, "/home/matveipopov/inference/inference_models")

from inference_models.weights_providers.roboflow import get_model_metadata

print("Fetching model metadata for doctr/default from STAGING...")
try:
    metadata = get_model_metadata(model_id="doctr/default", api_key="GYaBMWQ6xDqVFsEJIoan")
    print(f"\nModel ID: {metadata.model_id}")
    print(f"Architecture: {metadata.model_architecture}")
    print(f"Task: {metadata.task_type}")
    print(f"\nPackages ({len(metadata.model_packages)}):")
    for i, pkg in enumerate(metadata.model_packages):
        if hasattr(pkg, 'package_id'):
            print(f"\n  {i+1}. package_id: {pkg.package_id}")
            print(f"     model_features: {pkg.model_features}")
            print(f"     type: {pkg.type}")
        else:
            print(f"\n  {i+1}. (dict) {pkg}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
