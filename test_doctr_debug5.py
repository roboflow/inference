"""
Debug script - check pagination and next_page
"""
import os
os.environ["ROBOFLOW_API_KEY"] = "zaRavHwbvIXpGerDM3wi"
os.environ["API_BASE_URL"] = "https://staging.api.roboflow.com"

import sys
sys.path.insert(0, "/home/matveipopov/inference/inference_models")

from inference_models.weights_providers.roboflow import get_roboflow_model

print("Fetching model metadata for doctr/default with full pagination...")
try:
    metadata = get_roboflow_model(model_id="doctr/default")
    print(f"\nModel ID: {metadata.model_id}")
    print(f"\nPackages ({len(metadata.model_packages)}):")
    for i, pkg in enumerate(metadata.model_packages):
        print(f"\n  {i+1}. package_id: {pkg.package_id}")
        print(f"     model_features: {pkg.model_features}")
        print(f"     backend: {pkg.backend}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
