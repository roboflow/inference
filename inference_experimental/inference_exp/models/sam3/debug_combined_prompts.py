"""Debug script for combined prompts (text + box refinement)."""
import torch
import numpy as np
from PIL import Image

from sam3.model_builder import build_sam3_image_model as build_sam3_original
from inference_exp.models.sam3.sam3_image_model import build_sam3_model
from inference_exp.models.sam3.sam3_session import Sam3Session

BPE_PATH = "/home/hansent/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
CHECKPOINT_PATH = "/home/hansent/weights-sam3/sam3_prod_v12_interactive_5box_image_only.pt"
TEST_IMAGE_PATH = "/home/hansent/images/traffic.jpg"


def debug_combined_original():
    """Debug original model with combined prompts."""
    print("=== Original Model - Combined Prompts ===")
    
    model = build_sam3_original(
        bpe_path=BPE_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        device="cuda",
        eval_mode=True
    )
    
    inference_state = model.init_state(TEST_IMAGE_PATH)
    
    # First add text prompt
    print("\n1. Adding text prompt 'cars'")
    out1 = model.add_prompt(
        inference_state,
        frame_idx=0,
        text_str="cars",
        output_prob_thresh=0.5
    )
    print(f"   After text prompt: {len(out1['out_probs'])} detections")
    print(f"   Probabilities: {out1['out_probs']}")
    
    # Check state after first prompt
    print(f"\n   State after text prompt:")
    print(f"   - has visual prompt: {inference_state['per_frame_visual_prompt'][0] is not None}")
    print(f"   - has previous output: {inference_state['previous_stages_out'][0] is not None}")
    print(f"   - text_prompt: '{inference_state['text_prompt']}'")
    
    # Then add box as refinement
    print("\n2. Adding box refinement [0.5, 0.1, 0.15, 0.1]")
    test_box = np.array([[0.5, 0.1, 0.15, 0.1]], dtype=np.float32)
    
    # Hook to see what happens
    original_get_visual = model._get_visual_prompt
    def debug_visual(inf_state, frame_idx, boxes, labels):
        result = original_get_visual(inf_state, frame_idx, boxes, labels)
        boxes_out, labels_out, visual = result
        print(f"   _get_visual_prompt:")
        print(f"   - Input boxes: {boxes.shape[0]}")
        print(f"   - Visual prompt created: {visual is not None}")
        print(f"   - Boxes for geometric: {boxes_out.shape[0]}")
        return result
    
    model._get_visual_prompt = debug_visual
    
    out2 = model.add_prompt(
        inference_state,
        frame_idx=0,
        boxes_xywh=test_box,
        box_labels=np.array([1], dtype=np.int64),
        output_prob_thresh=0.5
    )
    
    print(f"\n   After box refinement: {len(out2['out_probs'])} detections")
    print(f"   Probabilities: {out2['out_probs']}")


def debug_combined_refactored():
    """Debug refactored model with combined prompts."""
    print("\n\n=== Refactored Model - Combined Prompts ===")
    
    model = build_sam3_model(
        bpe_path=BPE_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        device="cuda",
        eval_mode=True
    )
    
    image_pil = Image.open(TEST_IMAGE_PATH).convert("RGB")
    image_np = np.array(image_pil)
    h, w = image_np.shape[:2]
    
    session = Sam3Session(model)
    session.set_image(image_np)
    
    # First add text prompt
    print("\n1. Adding text prompt 'cars'")
    session.set_text_prompt("cars")
    
    # Debug predict
    original_predict = session.predict
    def debug_predict(*args, **kwargs):
        print(f"\n   Session.predict called:")
        print(f"   - has_predicted: {session.has_predicted}")
        print(f"   - has text: {session.text_features is not None}")
        print(f"   - num boxes: {session.prompts['boxes_cxcywh'].shape[0]}")
        
        # Check visual prompt logic
        is_visual = not session.has_predicted and session.prompts["boxes_cxcywh"].numel() > 0
        print(f"   - is_visual_prompt: {is_visual}")
        
        result = original_predict(*args, **kwargs)
        return result
    
    session.predict = debug_predict
    
    out1 = session.predict(output_prob_thresh=0.5)
    print(f"   After text prompt: {len(out1['out_probs'])} detections")
    print(f"   Probabilities: {out1['out_probs']}")
    
    # Add box refinement
    print("\n2. Adding box refinement")
    pixel_box = [0.5 * w, 0.1 * h, 0.65 * w, 0.2 * h]
    session.add_box_prompt([pixel_box])
    
    out2 = session.predict(output_prob_thresh=0.5)
    print(f"   After box refinement: {len(out2['out_probs'])} detections")
    print(f"   Probabilities: {out2['out_probs']}")
    
    # Check if we're treating the box as visual prompt incorrectly
    print(f"\n   Session state after refinement:")
    print(f"   - has_predicted: {session.has_predicted}")


def compare_thresholds():
    """Compare outputs with different thresholds."""
    print("\n\n=== Comparing with Different Thresholds ===")
    
    # Original
    original_model = build_sam3_original(
        bpe_path=BPE_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        device="cuda",
        eval_mode=True
    )
    
    # Refactored
    refactored_model = build_sam3_model(
        bpe_path=BPE_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        device="cuda",
        eval_mode=True
    )
    
    image_pil = Image.open(TEST_IMAGE_PATH).convert("RGB")
    image_np = np.array(image_pil)
    h, w = image_np.shape[:2]
    
    # Test with threshold 0.0 to see all outputs
    print("\nTesting with threshold=0.0")
    
    # Original
    inference_state = original_model.init_state(TEST_IMAGE_PATH)
    original_model.add_prompt(inference_state, frame_idx=0, text_str="cars", output_prob_thresh=0.0)
    test_box = np.array([[0.5, 0.1, 0.15, 0.1]], dtype=np.float32)
    out_orig = original_model.add_prompt(
        inference_state, frame_idx=0, boxes_xywh=test_box,
        box_labels=np.array([1], dtype=np.int64), output_prob_thresh=0.0
    )
    
    # Refactored
    session = Sam3Session(refactored_model)
    session.set_image(image_np)
    session.set_text_prompt("cars")
    session.predict(output_prob_thresh=0.0)
    pixel_box = [0.5 * w, 0.1 * h, 0.65 * w, 0.2 * h]
    session.add_box_prompt([pixel_box])
    out_ref = session.predict(output_prob_thresh=0.0)
    
    print(f"\nOriginal: {len(out_orig['out_probs'])} objects")
    print(f"Refactored: {len(out_ref['out_probs'])} objects")
    
    # Show probabilities sorted
    print(f"\nOriginal probs (sorted): {sorted(out_orig['out_probs'], reverse=True)[:10]}")
    print(f"Refactored probs (sorted): {sorted(out_ref['out_probs'], reverse=True)[:10]}")


if __name__ == "__main__":
    debug_combined_original()
    debug_combined_refactored()
    compare_thresholds()