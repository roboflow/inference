"""Debug script to understand visual prompt encoding differences."""
import torch
import numpy as np
from PIL import Image

from sam3.model_builder import build_sam3_image_model as build_sam3_original
from inference_exp.models.sam3.sam3_image_model import build_sam3_model
from inference_exp.models.sam3.sam3_session import Sam3Session

BPE_PATH = "/home/hansent/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
CHECKPOINT_PATH = "/home/hansent/weights-sam3/sam3_prod_v12_interactive_5box_image_only.pt"
TEST_IMAGE_PATH = "/home/hansent/images/traffic.jpg"


def debug_original_visual_encoding():
    """Debug how original encodes visual prompts."""
    print("=== Original Model - Visual Prompt Encoding ===")
    
    model = build_sam3_original(
        bpe_path=BPE_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        device="cuda",
        eval_mode=True
    )
    
    inference_state = model.init_state(TEST_IMAGE_PATH)
    
    # Capture the visual prompt encoding
    captured_data = {}
    
    # Hook the _encode_prompt method
    original_encode_prompt = model._encode_prompt
    def debug_encode_prompt(backbone_out, find_input, geometric_prompt, 
                           visual_prompt_embed=None, visual_prompt_mask=None, 
                           encode_text=True, prev_mask_pred=None):
        result = original_encode_prompt(
            backbone_out, find_input, geometric_prompt,
            visual_prompt_embed, visual_prompt_mask, 
            encode_text, prev_mask_pred
        )
        prompt, prompt_mask, backbone_out_ret = result
        
        print(f"\n_encode_prompt called:")
        print(f"  encode_text: {encode_text}")
        print(f"  geometric_prompt boxes: {geometric_prompt.box_embeddings.shape if geometric_prompt.box_embeddings is not None else None}")
        print(f"  visual_prompt_embed shape: {visual_prompt_embed.shape if visual_prompt_embed is not None else None}")
        print(f"  Output prompt shape: {prompt.shape}")
        print(f"  Output prompt_mask shape: {prompt_mask.shape}")
        
        if not encode_text and visual_prompt_embed is None:
            # This is the visual prompt encoding call
            captured_data['visual_prompt_encoded'] = prompt
            captured_data['visual_prompt_mask'] = prompt_mask
            
        return result
    
    model._encode_prompt = debug_encode_prompt
    
    # Hook the geometry encoder
    original_geo_forward = model.geometry_encoder.forward
    def debug_geo_forward(geo_prompt, img_feats, img_sizes, img_pos_embeds):
        print(f"\nGeometry encoder called:")
        print(f"  Input boxes shape: {geo_prompt.box_embeddings.shape}")
        result = original_geo_forward(geo_prompt, img_feats, img_sizes, img_pos_embeds)
        print(f"  Output features shape: {result[0].shape}")
        print(f"  Output mask shape: {result[1].shape}")
        return result
    
    model.geometry_encoder.forward = debug_geo_forward
    
    # Add box prompt
    test_box = np.array([[0.4, 0.3, 0.2, 0.3]], dtype=np.float32)
    out = model.add_prompt(
        inference_state,
        frame_idx=0,
        boxes_xywh=test_box,
        box_labels=np.array([1], dtype=np.int64),
        output_prob_thresh=0.5
    )
    
    print(f"\nFinal results:")
    print(f"  Number of detections: {len(out['out_probs'])}")
    print(f"  Visual prompt was encoded: {'visual_prompt_encoded' in captured_data}")
    
    # Check what text ID is used
    print(f"\nText IDs used:")
    print(f"  TEXT_ID_FOR_VISUAL: {model.TEXT_ID_FOR_VISUAL}")
    print(f"  find_text_batch[1]: {inference_state['input_batch'].find_text_batch[1]}")


def debug_refactored_visual_encoding():
    """Debug how refactored encodes visual prompts."""
    print("\n\n=== Refactored Model - Visual Prompt Encoding ===")
    
    model = build_sam3_model(
        bpe_path=BPE_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        device="cuda",
        eval_mode=True
    )
    
    # Hook the geometry encoder
    original_geo_forward = model.geometry_encoder.forward
    def debug_geo_forward(geo_prompt, img_feats, img_sizes, img_pos_embeds):
        print(f"\nGeometry encoder called:")
        print(f"  Input boxes shape: {geo_prompt.box_embeddings.shape}")
        result = original_geo_forward(geo_prompt, img_feats, img_sizes, img_pos_embeds)
        print(f"  Output features shape: {result[0].shape}")
        print(f"  Output mask shape: {result[1].shape}")
        return result
    
    model.geometry_encoder.forward = debug_geo_forward
    
    # Hook the predict method to see prompt construction
    original_predict = model.predict
    def debug_predict(image_features, text_features, geometric_prompt, 
                     visual_prompt=None, multimask_output=False):
        print(f"\nModel.predict called:")
        print(f"  text_features: {text_features is not None}")
        if geometric_prompt.box_embeddings is not None:
            print(f"  geometric_prompt boxes: {geometric_prompt.box_embeddings.shape}")
        if visual_prompt is not None and visual_prompt.box_embeddings is not None:
            print(f"  visual_prompt boxes: {visual_prompt.box_embeddings.shape}")
        
        # Call original
        result = original_predict(image_features, text_features, geometric_prompt, 
                                visual_prompt, multimask_output)
        return result
    
    model.predict = debug_predict
    
    # Test
    image_pil = Image.open(TEST_IMAGE_PATH).convert("RGB")
    image_np = np.array(image_pil)
    h, w = image_np.shape[:2]
    
    session = Sam3Session(model)
    session.set_image(image_np)
    
    pixel_box = [0.4 * w, 0.3 * h, 0.6 * w, 0.6 * h]
    session.add_box_prompt([pixel_box])
    out = session.predict(output_prob_thresh=0.5)
    
    print(f"\nFinal results:")
    print(f"  Number of detections: {len(out['out_probs'])}")


def compare_text_encoding():
    """Compare how text is handled for visual prompts."""
    print("\n\n=== Comparing Text Handling for Visual Prompts ===")
    
    # Original model
    original_model = build_sam3_original(
        bpe_path=BPE_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        device="cuda",
        eval_mode=True
    )
    
    # Check the text encoder
    inference_state = original_model.init_state(TEST_IMAGE_PATH)
    
    print("\nOriginal model text setup:")
    print(f"  find_text_batch: {inference_state['input_batch'].find_text_batch}")
    
    # Hook backbone.forward_text
    text_outputs = {}
    original_forward_text = original_model.backbone.forward_text
    def capture_text_forward(captions, device):
        print(f"\nBackbone.forward_text called with: {captions}")
        result = original_forward_text(captions, device=device)
        text_outputs[str(captions)] = result
        print(f"  Output language_features shape: {result['language_features'].shape}")
        return result
    
    original_model.backbone.forward_text = capture_text_forward
    
    # Add box prompt to trigger visual prompt
    test_box = np.array([[0.4, 0.3, 0.2, 0.3]], dtype=np.float32)
    original_model.add_prompt(
        inference_state,
        frame_idx=0,
        boxes_xywh=test_box,
        box_labels=np.array([1], dtype=np.int64),
        output_prob_thresh=0.5
    )
    
    print(f"\nText outputs captured: {list(text_outputs.keys())}")


if __name__ == "__main__":
    debug_original_visual_encoding()
    debug_refactored_visual_encoding()
    compare_text_encoding()