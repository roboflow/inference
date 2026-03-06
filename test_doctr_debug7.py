"""
Debug - check if there are two different model IDs
"""
import os
os.environ["ROBOFLOW_API_KEY"] = "zaRavHwbvIXpGerDM3wi"
os.environ["API_BASE_URL"] = "https://staging.api.roboflow.com"

import sys
sys.path.insert(0, "/home/matveipopov/inference/inference_models")

from inference_models.weights_providers.roboflow import get_roboflow_model

# The v1 registration script used different model IDs
# Check various possible model IDs
model_ids = [
    "doctr-dbnet-rn50/crnn-vgg16",  # v2 format
    "doctr_rec/crnn_vgg16_bn",       # v1 format for rec only
    "doctr/dbnet_resnet50/crnn_vgg16_bn",  # possible v1 full format
]

for model_id in model_ids:
    print(f"\n{'='*60}")
    print(f"Fetching: {model_id}")
    print('='*60)
    try:
        metadata = get_roboflow_model(model_id=model_id)
        print(f"  Model ID resolved to: {metadata.model_id}")
        print(f"  Packages ({len(metadata.model_packages)}):")
        for i, pkg in enumerate(metadata.model_packages):
            print(f"\n    {i+1}. package_id: {pkg.package_id}")
            print(f"       model_features: {pkg.model_features}")
    except Exception as e:
        print(f"  Error: {type(e).__name__}: {e}")
