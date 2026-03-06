"""
Test script to verify DocTR v1/v2 package compatibility.

Run this with different code branches:
- OLD code (no required_model_features): should load v1 package
- NEW code (with required_model_features): should load v2 package
"""
import os
os.environ["ROBOFLOW_API_KEY"] = "zaRavHwbvIXpGerDM3wi"
os.environ["API_BASE_URL"] = "https://staging.api.roboflow.com"

import sys
# Add inference_models to path
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
else:
    print("  No required_model_features attribute (old code)")

# Test model_implementation_exists with different feature sets
print("\nTesting model_implementation_exists:")
print(f"  No features: {model_implementation_exists('doctr', 'structured-ocr', BackendType.TORCH, model_features=None)}")
print(f"  Empty features: {model_implementation_exists('doctr', 'structured-ocr', BackendType.TORCH, model_features=set())}")
print(f"  v2 features (doctr_vocab_127): {model_implementation_exists('doctr', 'structured-ocr', BackendType.TORCH, model_features={'doctr_vocab_127'})}")

# Try to load the model
print("\nTrying to load DocTR model...")
try:
    model = AutoModel.from_pretrained("doctr/default")
    print(f"SUCCESS! Loaded model: {type(model)}")
    print(f"Model package ID: {getattr(model, '_model_package_id', 'N/A')}")
except Exception as e:
    print(f"FAILED to load model: {e}")
