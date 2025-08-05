"""
Comparison test between the original Sam3ImageInteractiveDemo and the refactored implementation.
This ensures that both implementations produce the same outputs for the same inputs.
"""
import torch
import numpy as np
import os
from PIL import Image

# Import original implementation
from sam3.model_builder import build_sam3_image_model as build_sam3_original
from sam3.model.sam3_demo import Sam3ImageInteractiveDemo

# Import refactored implementation
from inference_exp.models.sam3.sam3_image_model import build_sam3_model
from inference_exp.models.sam3.sam3_session import Sam3Session

# Test configuration
BPE_PATH = "/home/hansent/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
CHECKPOINT_PATH = "/home/hansent/weights-sam3/sam3_prod_v12_interactive_5box_image_only.pt"
TEST_IMAGE_PATH = "/home/hansent/images/traffic.jpg"


def compare_outputs(out1, out2, name, rtol=1e-4, atol=1e-5):
    """Compare two outputs and report differences."""
    print(f"\n--- Comparing {name} ---")
    
    # Compare keys
    keys1 = set(out1.keys())
    keys2 = set(out2.keys())
    
    if keys1 != keys2:
        print(f"WARNING: Different keys!")
        print(f"  Original only: {keys1 - keys2}")
        print(f"  Refactored only: {keys2 - keys1}")
        common_keys = keys1 & keys2
    else:
        print(f"✓ Same keys: {keys1}")
        common_keys = keys1
    
    # Compare values for common keys
    all_close = True
    for key in sorted(common_keys):
        val1 = out1[key]
        val2 = out2[key]
        
        if isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
            if val1.shape != val2.shape:
                print(f"✗ {key}: Different shapes - {val1.shape} vs {val2.shape}")
                all_close = False
            else:
                if val1.dtype == bool and val2.dtype == bool:
                    # For boolean masks, compare exact match
                    match_ratio = np.mean(val1 == val2)
                    if match_ratio < 0.99:  # Allow 1% difference for boolean masks
                        print(f"✗ {key}: Boolean arrays differ - {match_ratio:.2%} match")
                        all_close = False
                    else:
                        print(f"✓ {key}: Boolean arrays match - {match_ratio:.2%} match")
                else:
                    # For float arrays, use tolerance
                    if np.allclose(val1, val2, rtol=rtol, atol=atol):
                        max_diff = np.max(np.abs(val1 - val2))
                        print(f"✓ {key}: Arrays match (max diff: {max_diff:.2e})")
                    else:
                        max_diff = np.max(np.abs(val1 - val2))
                        rel_diff = np.max(np.abs(val1 - val2) / (np.abs(val1) + 1e-10))
                        print(f"✗ {key}: Arrays differ - max abs diff: {max_diff:.2e}, max rel diff: {rel_diff:.2e}")
                        all_close = False
    
    return all_close


def test_text_prompt_comparison():
    """Test with text prompt only."""
    print("\n=== Testing Text Prompt ===")
    
    # Load models
    print("Loading original model...")
    original_model = build_sam3_original(
        bpe_path=BPE_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        device="cuda",
        eval_mode=True
    )
    
    print("Loading refactored model...")
    refactored_model = build_sam3_model(
        bpe_path=BPE_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        device="cuda",
        eval_mode=True
    )
    
    # Load test image
    image_pil = Image.open(TEST_IMAGE_PATH).convert("RGB")
    image_np = np.array(image_pil)
    
    # Test with original implementation
    print("\nRunning original implementation...")
    inference_state = original_model.init_state(TEST_IMAGE_PATH)
    original_out = original_model.add_prompt(
        inference_state,
        frame_idx=0,
        text_str="cars",
        output_prob_thresh=0.5
    )
    
    # Test with refactored implementation
    print("\nRunning refactored implementation...")
    session = Sam3Session(refactored_model)
    session.set_image(image_np)
    session.set_text_prompt("cars")
    refactored_out = session.predict(output_prob_thresh=0.5)
    
    # Compare outputs
    return compare_outputs(original_out, refactored_out, "Text Prompt Results")


def test_box_prompt_comparison():
    """Test with box prompt (visual prompt)."""
    print("\n=== Testing Box Prompt (Visual Prompt) ===")
    
    # Load models
    original_model = build_sam3_original(
        bpe_path=BPE_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        device="cuda",
        eval_mode=True
    )
    
    refactored_model = build_sam3_model(
        bpe_path=BPE_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        device="cuda",
        eval_mode=True
    )
    
    # Load test image
    image_pil = Image.open(TEST_IMAGE_PATH).convert("RGB")
    image_np = np.array(image_pil)
    h, w = image_np.shape[:2]
    
    # Define test box in normalized coordinates [x, y, w, h]
    test_box = [0.4, 0.3, 0.2, 0.3]  # A box in the middle of the image
    
    # Test with original implementation
    print("\nRunning original implementation...")
    inference_state = original_model.init_state(TEST_IMAGE_PATH)
    original_out = original_model.add_prompt(
        inference_state,
        frame_idx=0,
        boxes_xywh=np.array([test_box], dtype=np.float32),
        box_labels=np.array([1], dtype=np.int64),
        output_prob_thresh=0.5
    )
    
    # Test with refactored implementation
    print("\nRunning refactored implementation...")
    session = Sam3Session(refactored_model)
    session.set_image(image_np)
    # Convert normalized box to pixel coordinates for session API
    pixel_box = [
        test_box[0] * w,  # xmin
        test_box[1] * h,  # ymin
        (test_box[0] + test_box[2]) * w,  # xmax
        (test_box[1] + test_box[3]) * h   # ymax
    ]
    session.add_box_prompt([pixel_box])
    refactored_out = session.predict(output_prob_thresh=0.5)
    
    # Compare outputs
    return compare_outputs(original_out, refactored_out, "Box Prompt Results")


def test_combined_prompts_comparison():
    """Test with text + box prompts (refinement)."""
    print("\n=== Testing Combined Prompts (Text + Box Refinement) ===")
    
    # Load models
    original_model = build_sam3_original(
        bpe_path=BPE_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        device="cuda",
        eval_mode=True
    )
    
    refactored_model = build_sam3_model(
        bpe_path=BPE_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        device="cuda",
        eval_mode=True
    )
    
    # Load test image
    image_pil = Image.open(TEST_IMAGE_PATH).convert("RGB")
    image_np = np.array(image_pil)
    h, w = image_np.shape[:2]
    
    # Test with original implementation
    print("\nRunning original implementation...")
    inference_state = original_model.init_state(TEST_IMAGE_PATH)
    
    # First add text prompt
    original_model.add_prompt(
        inference_state,
        frame_idx=0,
        text_str="cars",
        output_prob_thresh=0.5
    )
    
    # Then add box as refinement
    test_box = [0.5, 0.1, 0.15, 0.1]
    original_out = original_model.add_prompt(
        inference_state,
        frame_idx=0,
        boxes_xywh=np.array([test_box], dtype=np.float32),
        box_labels=np.array([1], dtype=np.int64),
        output_prob_thresh=0.5
    )
    
    # Test with refactored implementation
    print("\nRunning refactored implementation...")
    session = Sam3Session(refactored_model)
    session.set_image(image_np)
    session.set_text_prompt("cars")
    
    # First prediction with text
    session.predict(output_prob_thresh=0.5)
    
    # Add box refinement
    pixel_box = [
        test_box[0] * w,
        test_box[1] * h,
        (test_box[0] + test_box[2]) * w,
        (test_box[1] + test_box[3]) * h
    ]
    session.add_box_prompt([pixel_box])
    refactored_out = session.predict(output_prob_thresh=0.5)
    
    # Compare outputs
    return compare_outputs(original_out, refactored_out, "Combined Prompts Results")


def test_multimask_output():
    """Test multimask output mode."""
    print("\n=== Testing Multi-mask Output ===")
    
    # Load models
    original_model = build_sam3_original(
        bpe_path=BPE_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        device="cuda",
        eval_mode=True
    )
    
    refactored_model = build_sam3_model(
        bpe_path=BPE_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        device="cuda",
        eval_mode=True
    )
    
    # Load test image
    image_pil = Image.open(TEST_IMAGE_PATH).convert("RGB")
    image_np = np.array(image_pil)
    
    # Test with original - it uses different API for multimask
    print("\nRunning original implementation...")
    original_model.multimask_output = True
    inference_state = original_model.init_state(TEST_IMAGE_PATH)
    original_out = original_model.add_prompt(
        inference_state,
        frame_idx=0,
        text_str="cars",
        output_prob_thresh=0.5
    )
    
    # Test with refactored implementation
    print("\nRunning refactored implementation...")
    session = Sam3Session(refactored_model)
    session.set_image(image_np)
    session.set_text_prompt("cars")
    refactored_out = session.predict(output_prob_thresh=0.5, multimask_output=True)
    
    # Compare outputs
    return compare_outputs(original_out, refactored_out, "Multi-mask Output Results", rtol=1e-3)


def main():
    """Run all comparison tests."""
    print("=== SAM3 Implementation Comparison Test ===")
    print(f"BPE Path: {BPE_PATH}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Test Image: {TEST_IMAGE_PATH}")
    
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"ERROR: Test image not found at {TEST_IMAGE_PATH}")
        return
    
    # Set up reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run tests
    results = []
    
    try:
        results.append(("Text Prompt", test_text_prompt_comparison()))
    except Exception as e:
        print(f"\n✗ Text Prompt test failed with error: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Text Prompt", False))
    
    try:
        results.append(("Box Prompt", test_box_prompt_comparison()))
    except Exception as e:
        print(f"\n✗ Box Prompt test failed with error: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Box Prompt", False))
    
    try:
        results.append(("Combined Prompts", test_combined_prompts_comparison()))
    except Exception as e:
        print(f"\n✗ Combined Prompts test failed with error: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Combined Prompts", False))
    
    try:
        results.append(("Multi-mask Output", test_multimask_output()))
    except Exception as e:
        print(f"\n✗ Multi-mask Output test failed with error: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Multi-mask Output", False))
    
    # Summary
    print("\n=== Test Summary ===")
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(passed for _, passed in results)
    if all_passed:
        print("\n✓ All tests passed! The refactored implementation matches the original.")
    else:
        print("\n✗ Some tests failed. Please review the differences above.")


if __name__ == "__main__":
    main()