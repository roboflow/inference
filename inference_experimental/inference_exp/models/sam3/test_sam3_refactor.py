"""
End-to-end test script for the refactored Sam3ImageModel and Sam3Session.

This script verifies that the model can be built, an image can be processed, 
prompts can be added, and a prediction can be made without runtime errors.
"""
import torch
import numpy as np
import os

# Important: This assumes the script is run from a context where the `sam3` and `inference_experimental`
# modules are available in the Python path.

from inference_exp.models.sam3.sam3_image_model import build_sam3_model
from inference_exp.models.sam3.sam3_session import Sam3Session

# --- USER CONFIGURATION ---
# Please provide the correct paths to your BPE vocabulary and model checkpoint.
# The script will not run without a valid BPE file.
# The checkpoint is optional but recommended for a meaningful test.
BPE_PATH = "/home/hansent/sam3/assets/bpe_simple_vocab_16e6.txt.gz"  # <--- CHANGE THIS
CHECKPOINT_PATH = "/home/hansent/weights-sam3/sam3_prod_v12_interactive_5box_image_only.pt" # <--- CHANGE THIS (optional)

def run_test():
    """
    Executes the test case.
    """
    print("--- Starting SAM3 Refactor Test ---")

    if not os.path.exists(BPE_PATH):
        print(f"\nERROR: BPE vocabulary file not found at '{BPE_PATH}'.")
        print("Please update the BPE_PATH variable in this script.")
        return

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"\nWARNING: Model checkpoint not found at '{CHECKPOINT_PATH}'.")
        print("The model will run with random weights, which is fine for a structural test.")
        checkpoint_to_load = None
    else:
        print(f"Loading checkpoint from: {CHECKPOINT_PATH}")
        checkpoint_to_load = CHECKPOINT_PATH

    # 1. Build the model
    print("\n1. Building Sam3ImageModel...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   Using device: {device}")
        sam_model = build_sam3_model(
            bpe_path=BPE_PATH,
            checkpoint_path=checkpoint_to_load,
            device=device,
            eval_mode=True
        )
        print("   Model built successfully.")
    except Exception as e:
        print(f"   ERROR: Failed to build model. {e}")
        return

    # 2. Create a session
    print("\n2. Creating Sam3Session...")
    session = Sam3Session(sam_model)
    print("   Session created.")

    # 3. Create a dummy image and set it
    print("\n3. Creating and setting a dummy image...")
    dummy_image_np = np.random.randint(0, 256, size=(600, 800, 3), dtype=np.uint8)
    session.set_image(dummy_image_np)
    print(f"   Image set. Original size: {session.original_size}, Cached features are present: {session.image_features is not None}")

    # 4. Add a dummy box prompt
    print("\n4. Adding a dummy box prompt...")
    # Box is [xmin, ymin, xmax, ymax]
    session.add_box_prompt(boxes=[[200, 250, 400, 450]])
    print(f"   Box prompt added and converted to cxcywh: {session.prompts['boxes_cxcywh']}")

    # 5. Run prediction
    print("\n5. Running prediction...")
    try:
        predictions = session.predict(output_prob_thresh=0.5)
        print("   Prediction successful!")
    except Exception as e:
        print(f"   ERROR: Prediction failed. {e}")
        return

    # 6. Print results
    print("\n6. Prediction Output:")
    print(f"   Output keys: {predictions.keys()}")
    for key, value in predictions.items():
        if isinstance(value, np.ndarray):
            print(f"   - {key}: shape={value.shape}, dtype={value.dtype}")
    
    print("\n--- Test Completed Successfully ---")


if __name__ == "__main__":
    run_test()

