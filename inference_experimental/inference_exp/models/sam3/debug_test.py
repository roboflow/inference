"""Debug test to understand the query and output shapes."""
import torch
import numpy as np
from PIL import Image

from sam3.model_builder import build_sam3_image_model as build_sam3_original
from inference_exp.models.sam3.sam3_image_model import build_sam3_model
from inference_exp.models.sam3.sam3_session import Sam3Session

BPE_PATH = "/home/hansent/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
CHECKPOINT_PATH = "/home/hansent/weights-sam3/sam3_prod_v12_interactive_5box_image_only.pt"
TEST_IMAGE_PATH = "/home/hansent/images/traffic.jpg"


def debug_original_model():
    """Debug the original model to understand its behavior."""
    print("=== Debugging Original Model ===")
    
    model = build_sam3_original(
        bpe_path=BPE_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        device="cuda",
        eval_mode=True
    )
    
    # Check decoder configuration
    print(f"\nDecoder configuration:")
    print(f"  num_queries: {model.transformer.decoder.num_queries}")
    print(f"  num_o2m_queries: {model.transformer.decoder.num_o2m_queries}")
    print(f"  use_instance_query: {model.use_instance_query}")
    print(f"  num_instances: {model.transformer.decoder.num_instances}")
    
    if hasattr(model.transformer.decoder, 'instance_query_embed'):
        print(f"  instance_query_embed shape: {model.transformer.decoder.instance_query_embed.weight.shape}")
    if hasattr(model.transformer.decoder, 'query_embed'):
        print(f"  query_embed shape: {model.transformer.decoder.query_embed.weight.shape if model.transformer.decoder.query_embed else 'None'}")
    
    # Run inference
    inference_state = model.init_state(TEST_IMAGE_PATH)
    
    # Add hooks to capture intermediate outputs
    outputs = {}
    def capture_decoder_output(module, input, output):
        outputs['decoder_hs'] = output[0]  # hs
        outputs['decoder_reference_boxes'] = output[1]  # reference_boxes
    
    hook = model.transformer.decoder.register_forward_hook(capture_decoder_output)
    
    out = model.add_prompt(
        inference_state,
        frame_idx=0,
        text_str="cars",
        output_prob_thresh=0.0  # Set to 0 to see all outputs
    )
    
    hook.remove()
    
    print(f"\nDecoder outputs:")
    if 'decoder_hs' in outputs:
        print(f"  hs shape: {outputs['decoder_hs'].shape}")
        print(f"  reference_boxes shape: {outputs['decoder_reference_boxes'].shape}")
    
    print(f"\nFinal outputs (thresh=0.0):")
    print(f"  Number of objects: {len(out['out_probs'])}")
    print(f"  Probabilities: {out['out_probs']}")


def debug_refactored_model():
    """Debug the refactored model."""
    print("\n=== Debugging Refactored Model ===")
    
    model = build_sam3_model(
        bpe_path=BPE_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        device="cuda",
        eval_mode=True
    )
    
    # Check decoder configuration
    print(f"\nDecoder configuration:")
    print(f"  num_queries: {model.transformer.decoder.num_queries}")
    print(f"  use_instance_query: {model.use_instance_query}")
    
    if hasattr(model.transformer.decoder, 'instance_query_embed'):
        print(f"  instance_query_embed shape: {model.transformer.decoder.instance_query_embed.weight.shape}")
    
    # Load image
    image_pil = Image.open(TEST_IMAGE_PATH).convert("RGB")
    image_np = np.array(image_pil)
    
    # Test query initialization
    print("\nTesting query initialization:")
    queries = model._init_instance_queries(B=1, multimask_output=False)
    print(f"  Query embed shape: {queries['embed'].shape}")
    print(f"  Reference boxes shape: {queries['reference_boxes'].shape}")
    
    # Run inference
    session = Sam3Session(model)
    session.set_image(image_np)
    session.set_text_prompt("cars")
    
    # Add hook to capture decoder output
    outputs = {}
    def capture_outputs(module, input, output):
        if isinstance(output, dict):
            for k, v in output.items():
                if isinstance(v, torch.Tensor):
                    outputs[f'model_{k}'] = v
        return output
    
    def capture_decoder_output(module, input, output):
        outputs['decoder_hs'] = output[0]
        outputs['decoder_reference_boxes'] = output[1]
        return output
    
    hook2 = model.transformer.decoder.register_forward_hook(capture_decoder_output)
    
    # Get raw model outputs before postprocessing
    model_outputs = model.predict(
        image_features=session.image_features,
        text_features=session.text_features,
        geometric_prompt=session.prompts,
        visual_prompt=None,
        multimask_output=False,
    )
    
    hook2.remove()
    
    print(f"\nDecoder outputs:")
    if 'decoder_hs' in outputs:
        print(f"  hs shape: {outputs['decoder_hs'].shape}")
    
    print(f"\nModel outputs before postprocessing:")
    print(f"  pred_logits shape: {model_outputs['pred_logits'].shape}")
    print(f"  pred_masks shape: {model_outputs['pred_masks'].shape}")
    
    # Test postprocessing with different thresholds
    for thresh in [0.0, 0.5]:
        post_out = model.postprocess_outputs(
            model_outputs,
            session.original_size,
            output_prob_thresh=thresh,
            multimask_output=False
        )
        print(f"\nPostprocessed outputs (thresh={thresh}):")
        print(f"  Number of objects: {len(post_out['out_probs'])}")
        if len(post_out['out_probs']) > 0:
            print(f"  Probabilities: {post_out['out_probs']}")


if __name__ == "__main__":
    debug_original_model()
    debug_refactored_model()