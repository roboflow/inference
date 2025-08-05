"""
End-to-end test script for the refactored Sam3ImageModel and Sam3Session.

This script verifies that the model can be built, an image can be processed, 
prompts can be added, and a prediction can be made without runtime errors.
"""
import torch
import numpy as np
import os
import traceback
from PIL import Image

from inference_exp.models.sam3.sam3_image_model import build_sam3_model
from inference_exp.models.sam3.sam3_session import Sam3Session

# --- USER CONFIGURATION ---
BPE_PATH = "/home/hansent/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
CHECKPOINT_PATH = "/home/hansent/weights-sam3/sam3_prod_v12_interactive_5box_image_only.pt"

def run_basic_test(sam_model):
    """
    Runs a basic test with a dummy image and box prompt.
    """
    print("\n=== Running Basic Test with Dummy Image ===")
    
    # Create a session
    session = Sam3Session(sam_model)
    print("Session created.")

    # Create a dummy image and set it
    print("\nCreating and setting a dummy image...")
    dummy_image_np = np.random.randint(0, 256, size=(600, 800, 3), dtype=np.uint8)
    session.set_image(dummy_image_np)
    print(f"Image set. Original size: {session.original_size}, Cached features are present: {session.image_features is not None}")

    # Add a dummy box prompt
    print("\nAdding a dummy box prompt...")
    # Box is [xmin, ymin, xmax, ymax]
    session.add_box_prompt(boxes=[[200, 250, 400, 450]])
    print(f"Box prompt added and converted to cxcywh: {session.prompts['boxes_cxcywh']}")

    # Run prediction
    print("\nRunning prediction...")
    try:
        predictions = session.predict(output_prob_thresh=0.5)
        print("Prediction successful!")
    except Exception:
        print("ERROR: Prediction failed.")
        traceback.print_exc()
        return False

    # Print results
    print("\nPrediction Output:")
    print(f"Output keys: {predictions.keys()}")
    for key, value in predictions.items():
        if isinstance(value, np.ndarray):
            print(f"  - {key}: shape={value.shape}, dtype={value.dtype}")
    
    return True

def run_real_image_test(sam_model, image_path, text_prompt):
    """
    Runs a test with a real image and text prompt.
    """
    print(f"\n=== Running Real Image Test ===")
    print(f"Image: {image_path}")
    print(f"Text prompt: '{text_prompt}'")
    
    if not os.path.exists(image_path):
        print(f"ERROR: Image not found at {image_path}")
        return False
    
    # Create a session
    session = Sam3Session(sam_model)
    print("Session created.")
    
    # Load and set the real image
    print("\nLoading and setting the image...")
    image_pil = Image.open(image_path).convert("RGB")
    image_np = np.array(image_pil)
    print(f"Image loaded. Shape: {image_np.shape}")
    
    session.set_image(image_np)
    print(f"Image set. Original size: {session.original_size}")
    
    # Set text prompt
    print(f"\nSetting text prompt: '{text_prompt}'")
    session.set_text_prompt(text_prompt)
    
    # Run prediction
    print("\nRunning prediction...")
    try:
        predictions = session.predict(output_prob_thresh=0.5)
        print("Prediction successful!")
    except Exception:
        print("ERROR: Prediction failed.")
        traceback.print_exc()
        return False
    
    # Print results
    print("\nPrediction Output:")
    print(f"Output keys: {predictions.keys()}")
    
    num_objects = len(predictions["out_probs"])
    print(f"\nFound {num_objects} objects matching '{text_prompt}'")
    
    if num_objects > 0:
        print("\nDetection details:")
        for i in range(min(5, num_objects)):  # Show up to 5 detections
            print(f"  Object {i+1}:")
            print(f"    - Confidence: {predictions['out_probs'][i]:.3f}")
            print(f"    - Box (xywh): {predictions['out_boxes_xywh'][i]}")
            print(f"    - Mask shape: {predictions['out_binary_masks'][i].shape}")
    
    return True

def run_test():
    """
    Executes the test cases.
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

    # Build the model
    print("\nBuilding Sam3ImageModel...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        sam_model = build_sam3_model(
            bpe_path=BPE_PATH,
            checkpoint_path=checkpoint_to_load,
            device=device,
            eval_mode=True
        )
        print("Model built successfully.")
    except Exception:
        print("ERROR: Failed to build model.")
        traceback.print_exc()
        return

    # Run basic test
    basic_test_passed = run_basic_test(sam_model)
    
    if basic_test_passed:
        # Run real image test
        real_image_path = "/home/hansent/images/traffic.jpg"
        text_prompt = "cars"
        run_real_image_test(sam_model, real_image_path, text_prompt)
    
    print("\n--- Test Completed ---")


if __name__ == "__main__":
    run_test()