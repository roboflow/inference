"""
Test OLD code (main branch) on staging - should try to load v1 package and fail with size mismatch
"""
import os
os.environ["PROJECT"] = "roboflow-staging"
os.environ["API_BASE_URL"] = "https://api.roboflow.one"
os.environ["ROBOFLOW_API_KEY"] = "GYaBMWQ6xDqVFsEJIoan"
os.environ["ROBOFLOW_ENVIRONMENT"] = "staging"

import sys
sys.path.insert(0, "/home/matveipopov/inference/inference_models")

from inference_models import AutoModel
from inference_models.models.auto_loaders.models_registry import model_implementation_exists, REGISTERED_MODELS
from inference_models.models.auto_loaders.entities import BackendType

# Check what's registered for DocTR
doctr_key = ("doctr", "structured-ocr", BackendType.TORCH)
entry = REGISTERED_MODELS.get(doctr_key)
print(f"DocTR registry entry: {entry}")
if hasattr(entry, 'required_model_features'):
    print(f"  required_model_features: {entry.required_model_features}")
    print(f"  supported_model_features: {entry.supported_model_features}")
else:
    print("  No required_model_features attribute (old code)")

# Test model_implementation_exists with different feature sets
print("\nTesting model_implementation_exists:")
print(f"  No features: {model_implementation_exists('doctr', 'structured-ocr', BackendType.TORCH, model_features=None)}")
print(f"  v2 features: {model_implementation_exists('doctr', 'structured-ocr', BackendType.TORCH, model_features={'doctr_vocab_127'})}")

# Try to load the model with CACHE DISABLED
print("\nTrying to load DocTR model with use_auto_resolution_cache=False...")
try:
    model = AutoModel.from_pretrained("doctr/default", verbose=True, use_auto_resolution_cache=False)
    print(f"\nSUCCESS! Loaded model: {type(model)}")
except Exception as e:
    print(f"\nFAILED to load model: {e}")
