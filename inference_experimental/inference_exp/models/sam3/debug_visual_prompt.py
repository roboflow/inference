"""Debug script to understand visual prompt handling differences."""
import torch
import numpy as np
from PIL import Image

from sam3.model_builder import build_sam3_image_model as build_sam3_original
from inference_exp.models.sam3.sam3_image_model import build_sam3_model
from inference_exp.models.sam3.sam3_session import Sam3Session

BPE_PATH = "/home/hansent/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
CHECKPOINT_PATH = "/home/hansent/weights-sam3/sam3_prod_v12_interactive_5box_image_only.pt"
TEST_IMAGE_PATH = "/home/hansent/images/traffic.jpg"


def debug_visual_prompt_original():
    """Debug how original handles visual prompts."""
    print("=== Original Model - Visual Prompt Handling ===")
    
    model = build_sam3_original(
        bpe_path=BPE_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        device="cuda",
        eval_mode=True
    )
    
    # Test box prompt (which becomes visual prompt on first use)
    inference_state = model.init_state(TEST_IMAGE_PATH)
    
    print("\nChecking initial state:")
    print(f"  per_frame_visual_prompt[0]: {inference_state['per_frame_visual_prompt'][0]}")
    print(f"  previous_stages_out[0]: {inference_state['previous_stages_out'][0]}")
    
    # Define test box
    test_box = np.array([[0.4, 0.3, 0.2, 0.3]], dtype=np.float32)  # [x, y, w, h]
    
    print(f"\nAdding box prompt: {test_box}")
    
    # Capture what happens in _get_visual_prompt
    original_get_visual_prompt = model._get_visual_prompt
    def debug_get_visual_prompt(inference_state, frame_idx, boxes_cxcywh, box_labels):
        print(f"\n_get_visual_prompt called:")
        print(f"  Number of input boxes: {boxes_cxcywh.shape[0]}")
        result = original_get_visual_prompt(inference_state, frame_idx, boxes_cxcywh, box_labels)
        boxes_out, labels_out, visual_prompt = result
        print(f"  Visual prompt created: {visual_prompt is not None}")
        if visual_prompt is not None:
            print(f"    Visual prompt box shape: {visual_prompt.box_embeddings.shape}")
        print(f"  Boxes for geometric prompt: {boxes_out.shape[0]}")
        return result
    
    model._get_visual_prompt = debug_get_visual_prompt
    
    # Add prompt
    out = model.add_prompt(
        inference_state,
        frame_idx=0,
        boxes_xywh=test_box,
        box_labels=np.array([1], dtype=np.int64),
        output_prob_thresh=0.5
    )
    
    print(f"\nFirst box prompt results:")
    print(f"  Number of detections: {len(out['out_probs'])}")
    print(f"  Probabilities: {out['out_probs']}")
    
    # Add another box to see refinement behavior
    print("\n\nAdding second box (should be refinement, not visual):")
    test_box2 = np.array([[0.5, 0.1, 0.15, 0.1]], dtype=np.float32)
    
    out2 = model.add_prompt(
        inference_state,
        frame_idx=0,
        boxes_xywh=test_box2,
        box_labels=np.array([1], dtype=np.int64),
        clear_old_boxes=False,  # Keep previous box
        output_prob_thresh=0.5
    )
    
    print(f"\nSecond box prompt results:")
    print(f"  Number of detections: {len(out2['out_probs'])}")


def debug_visual_prompt_refactored():
    """Debug how refactored handles visual prompts."""
    print("\n\n=== Refactored Model - Visual Prompt Handling ===")
    
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
    
    print("\nChecking initial state:")
    print(f"  has_predicted: {session.has_predicted}")
    
    # Convert box to pixel coordinates
    test_box_norm = [0.4, 0.3, 0.2, 0.3]  # [x, y, w, h] normalized
    pixel_box = [
        test_box_norm[0] * w,
        test_box_norm[1] * h,
        (test_box_norm[0] + test_box_norm[2]) * w,
        (test_box_norm[1] + test_box_norm[3]) * h
    ]
    
    print(f"\nAdding box prompt (pixel coords): {pixel_box}")
    
    # Debug the predict method
    original_predict = session.predict
    def debug_predict(*args, **kwargs):
        print(f"\nSession.predict called:")
        print(f"  has_predicted before: {session.has_predicted}")
        print(f"  Number of boxes: {session.prompts['boxes_cxcywh'].shape[0]}")
        
        # Check visual prompt logic
        is_visual_prompt = not session.has_predicted and session.prompts["boxes_cxcywh"].numel() > 0
        print(f"  Is visual prompt: {is_visual_prompt}")
        
        result = original_predict(*args, **kwargs)
        print(f"  has_predicted after: {session.has_predicted}")
        return result
    
    session.predict = debug_predict
    
    session.add_box_prompt([pixel_box])
    out = session.predict(output_prob_thresh=0.5)
    
    print(f"\nFirst box prompt results:")
    print(f"  Number of detections: {len(out['out_probs'])}")
    print(f"  Probabilities: {out['out_probs']}")
    
    # Add second box
    print("\n\nAdding second box:")
    pixel_box2 = [
        0.5 * w,
        0.1 * h,
        0.65 * w,
        0.2 * h
    ]
    session.add_box_prompt([pixel_box, pixel_box2])  # Note: this replaces all boxes
    out2 = session.predict(output_prob_thresh=0.5)
    
    print(f"\nSecond box prompt results:")
    print(f"  Number of detections: {len(out2['out_probs'])}")


def compare_prompt_encoding():
    """Compare how prompts are encoded in both implementations."""
    print("\n\n=== Comparing Prompt Encoding ===")
    
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
    
    # Test with original
    inference_state = original_model.init_state(TEST_IMAGE_PATH)
    
    # Hook to capture encoder inputs
    encoder_inputs = {}
    def capture_encoder_input(module, input):
        # Encoder inputs: src, src_key_padding_mask, src_pos, prompt, prompt_pos, prompt_key_padding_mask
        encoder_inputs['prompt_shape'] = input[3].shape if len(input) > 3 else None
        encoder_inputs['prompt_mask_shape'] = input[5].shape if len(input) > 5 else None
    
    hook = original_model.transformer.encoder.register_forward_pre_hook(capture_encoder_input)
    
    # Add box prompt
    test_box = np.array([[0.4, 0.3, 0.2, 0.3]], dtype=np.float32)
    original_model.add_prompt(
        inference_state,
        frame_idx=0,
        boxes_xywh=test_box,
        box_labels=np.array([1], dtype=np.int64),
        output_prob_thresh=0.5
    )
    
    hook.remove()
    
    print("\nOriginal model encoder inputs:")
    print(f"  Prompt shape: {encoder_inputs.get('prompt_shape')}")
    print(f"  Prompt mask shape: {encoder_inputs.get('prompt_mask_shape')}")
    
    # Test with refactored
    image_pil = Image.open(TEST_IMAGE_PATH).convert("RGB")
    image_np = np.array(image_pil)
    h, w = image_np.shape[:2]
    
    session = Sam3Session(refactored_model)
    session.set_image(image_np)
    
    # Hook refactored encoder
    encoder_inputs2 = {}
    hook2 = refactored_model.transformer.encoder.register_forward_pre_hook(
        lambda m, i: encoder_inputs2.update({
            'prompt_shape': i[3].shape if len(i) > 3 else None,
            'prompt_mask_shape': i[5].shape if len(i) > 5 else None
        })
    )
    
    pixel_box = [0.4 * w, 0.3 * h, 0.6 * w, 0.6 * h]
    session.add_box_prompt([pixel_box])
    session.predict(output_prob_thresh=0.5)
    
    hook2.remove()
    
    print("\nRefactored model encoder inputs:")
    print(f"  Prompt shape: {encoder_inputs2.get('prompt_shape')}")
    print(f"  Prompt mask shape: {encoder_inputs2.get('prompt_mask_shape')}")


if __name__ == "__main__":
    debug_visual_prompt_original()
    debug_visual_prompt_refactored()
    compare_prompt_encoding()