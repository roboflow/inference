"""
Debug script with verbose mode
"""
import os
os.environ["ROBOFLOW_API_KEY"] = "zaRavHwbvIXpGerDM3wi"
os.environ["API_BASE_URL"] = "https://staging.api.roboflow.com"

import sys
sys.path.insert(0, "/home/matveipopov/inference/inference_models")

from inference_models import AutoModel

# Try to load the model with verbose output
print("Trying to load DocTR model with verbose=True...")
try:
    model = AutoModel.from_pretrained("doctr/default", verbose=True)
    print(f"SUCCESS! Loaded model: {type(model)}")
except Exception as e:
    print(f"\nFinal error: {e}")
