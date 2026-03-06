"""
Debug script - directly query the registered model by id
"""
import os
os.environ["ROBOFLOW_API_KEY"] = "zaRavHwbvIXpGerDM3wi"
os.environ["API_BASE_URL"] = "https://staging.api.roboflow.com"

import sys
sys.path.insert(0, "/home/matveipopov/inference/inference_models")

from inference_models.weights_providers.roboflow import get_roboflow_model

# Try the specific model ID that was registered
model_ids = [
    "doctr-dbnet-rn50/crnn-vgg16",  # model we registered with
]

for model_id in model_ids:
    print(f"\n{'='*60}")
    print(f"Fetching: {model_id}")
    print('='*60)
    try:
        metadata = get_roboflow_model(model_id=model_id)
        print(f"  Packages ({len(metadata.model_packages)}):")
        for i, pkg in enumerate(metadata.model_packages):
            print(f"\n    {i+1}. package_id: {pkg.package_id}")
            print(f"       model_features: {pkg.model_features}")
    except Exception as e:
        print(f"  Error: {e}")
