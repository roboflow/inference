"""
Full test on staging - verify both old and new code paths work
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
print(f"  Empty features: {model_implementation_exists('doctr', 'structured-ocr', BackendType.TORCH, model_features=set())}")
print(f"  v2 features (doctr_vocab_127): {model_implementation_exists('doctr', 'structured-ocr', BackendType.TORCH, model_features={'doctr_vocab_127'})}")

# Clear cache
import shutil
cache_dir = os.path.expanduser("~/.cache/inference_models/packages")
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    print("\nCleared model cache")

# Try to load the model
print("\nTrying to load DocTR model...")
try:
    model = AutoModel.from_pretrained("doctr/default", verbose=True)
    print(f"\nSUCCESS! Loaded model: {type(model)}")
    print(f"Model package ID: {getattr(model, '_model_package_id', 'N/A')}")
except Exception as e:
    print(f"\nFAILED to load model: {e}")
