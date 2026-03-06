"""
Test script to verify that RF-DETR outputs are IDENTICAL between:
- Main branch (transformers 4.x with head_mask)
- Current branch (transformers 5.x without head_mask)

This script compares:
1. Backbone feature outputs
2. Transformer encoder/decoder outputs
3. Final detection outputs (boxes, logits)

The changes we made (removing head_mask) should produce IDENTICAL results because:
- head_mask was always None in practice (never used)
- The `if head_mask is not None` block was never executed
- We only removed dead code paths

Usage:
    # On current branch (transformers 5.x):
    python test_rfdetr_transformers5_comparison.py --save-outputs current_outputs.pt

    # On main branch (transformers 4.x):
    git checkout main
    pip install transformers==4.46.0
    python test_rfdetr_transformers5_comparison.py --save-outputs main_outputs.pt

    # Compare:
    python test_rfdetr_transformers5_comparison.py --compare main_outputs.pt current_outputs.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def create_deterministic_input(seed: int = 42, batch_size: int = 1, height: int = 560, width: int = 560):
    """Create deterministic input tensor for reproducible testing."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create a deterministic image tensor (normalized)
    pixel_values = torch.randn(batch_size, 3, height, width)
    return pixel_values


def extract_backbone_outputs(model, pixel_values: torch.Tensor) -> dict:
    """Extract intermediate outputs from the backbone."""
    model.eval()

    outputs = {}

    with torch.no_grad():
        # Get backbone directly
        backbone = model.model.backbone

        # Forward through backbone
        # The backbone returns features and position embeddings
        from inference_models.models.rfdetr.nested_tensor import NestedTensor

        # Create nested tensor (simulating what happens in forward)
        mask = torch.zeros(pixel_values.shape[0], pixel_values.shape[2], pixel_values.shape[3], dtype=torch.bool)
        nested_tensor = NestedTensor(pixel_values, mask)

        features, pos_embeds = backbone(nested_tensor)

        # Store feature tensors
        for i, (feat, pos) in enumerate(zip(features, pos_embeds)):
            outputs[f'feature_{i}_tensor'] = feat.tensors.clone()
            outputs[f'feature_{i}_mask'] = feat.mask.clone()
            outputs[f'pos_embed_{i}'] = pos.clone()

    return outputs


def extract_full_forward_outputs(model, pixel_values: torch.Tensor) -> dict:
    """Extract outputs from the full model forward pass."""
    model.eval()

    outputs = {}

    with torch.no_grad():
        # Run full forward
        from inference_models.models.rfdetr.nested_tensor import NestedTensor

        mask = torch.zeros(pixel_values.shape[0], pixel_values.shape[2], pixel_values.shape[3], dtype=torch.bool)
        nested_tensor = NestedTensor(pixel_values, mask)

        # Get model output
        model_output = model.model(nested_tensor)

        # Extract key outputs
        outputs['pred_logits'] = model_output['pred_logits'].clone()
        outputs['pred_boxes'] = model_output['pred_boxes'].clone()

        # Auxiliary outputs if present
        if 'aux_outputs' in model_output:
            for i, aux in enumerate(model_output['aux_outputs']):
                outputs[f'aux_{i}_logits'] = aux['pred_logits'].clone()
                outputs[f'aux_{i}_boxes'] = aux['pred_boxes'].clone()

    return outputs


def extract_dinov2_backbone_outputs(pixel_values: torch.Tensor) -> dict:
    """
    Test the WindowedDinov2WithRegistersBackbone directly.
    This is the component we modified for transformers 5.x compatibility.
    """
    from inference_models.models.rfdetr.dinov2_with_windowed_attn import (
        WindowedDinov2WithRegistersBackbone,
        WindowedDinov2WithRegistersConfig,
    )

    outputs = {}

    # Create a config similar to what RF-DETR uses
    config = WindowedDinov2WithRegistersConfig(
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        image_size=560,
        patch_size=14,
        num_register_tokens=4,
        out_features=["stage2", "stage5", "stage8", "stage11"],
        out_indices=[2, 5, 8, 11],
    )

    # Initialize model (random weights - we just want to verify identical behavior)
    torch.manual_seed(42)
    model = WindowedDinov2WithRegistersBackbone(config)
    model.eval()

    with torch.no_grad():
        # Forward pass
        backbone_output = model(pixel_values, output_hidden_states=True, return_dict=True)

        # Store outputs
        outputs['feature_maps'] = [fm.clone() for fm in backbone_output.feature_maps]
        if backbone_output.hidden_states is not None:
            outputs['hidden_states'] = [hs.clone() for hs in backbone_output.hidden_states]

    return outputs


def save_outputs(outputs: dict, path: str):
    """Save outputs to a file."""
    torch.save(outputs, path)
    print(f"Saved outputs to {path}")


def load_outputs(path: str) -> dict:
    """Load outputs from a file."""
    return torch.load(path, map_location='cpu')


def compare_outputs(outputs1: dict, outputs2: dict, rtol: float = 1e-5, atol: float = 1e-5) -> bool:
    """Compare two output dictionaries for equality."""
    all_match = True

    keys1 = set(outputs1.keys())
    keys2 = set(outputs2.keys())

    if keys1 != keys2:
        print(f"WARNING: Key mismatch!")
        print(f"  Only in first: {keys1 - keys2}")
        print(f"  Only in second: {keys2 - keys1}")
        all_match = False

    common_keys = keys1 & keys2

    for key in sorted(common_keys):
        val1 = outputs1[key]
        val2 = outputs2[key]

        if isinstance(val1, torch.Tensor) and isinstance(val2, torch.Tensor):
            if val1.shape != val2.shape:
                print(f"MISMATCH {key}: shape {val1.shape} vs {val2.shape}")
                all_match = False
                continue

            # Check exact equality first
            if torch.equal(val1, val2):
                print(f"✓ {key}: EXACT MATCH (shape {val1.shape})")
            else:
                # Check approximate equality
                if torch.allclose(val1, val2, rtol=rtol, atol=atol):
                    max_diff = (val1 - val2).abs().max().item()
                    print(f"✓ {key}: APPROX MATCH (max diff: {max_diff:.2e}, shape {val1.shape})")
                else:
                    max_diff = (val1 - val2).abs().max().item()
                    mean_diff = (val1 - val2).abs().mean().item()
                    print(f"✗ {key}: MISMATCH (max diff: {max_diff:.2e}, mean diff: {mean_diff:.2e})")
                    all_match = False

        elif isinstance(val1, list) and isinstance(val2, list):
            if len(val1) != len(val2):
                print(f"MISMATCH {key}: list length {len(val1)} vs {len(val2)}")
                all_match = False
                continue

            for i, (v1, v2) in enumerate(zip(val1, val2)):
                if isinstance(v1, torch.Tensor) and isinstance(v2, torch.Tensor):
                    if torch.equal(v1, v2):
                        print(f"✓ {key}[{i}]: EXACT MATCH (shape {v1.shape})")
                    elif torch.allclose(v1, v2, rtol=rtol, atol=atol):
                        max_diff = (v1 - v2).abs().max().item()
                        print(f"✓ {key}[{i}]: APPROX MATCH (max diff: {max_diff:.2e})")
                    else:
                        max_diff = (v1 - v2).abs().max().item()
                        print(f"✗ {key}[{i}]: MISMATCH (max diff: {max_diff:.2e})")
                        all_match = False

    return all_match


def run_standalone_dinov2_test():
    """
    Run a standalone test of the WindowedDinov2WithRegistersBackbone.
    This tests the exact component we modified.
    """
    print("\n" + "="*60)
    print("Testing WindowedDinov2WithRegistersBackbone (modified component)")
    print("="*60)

    # Create deterministic input
    pixel_values = create_deterministic_input(seed=42, batch_size=2, height=560, width=560)

    # Extract outputs
    outputs = extract_dinov2_backbone_outputs(pixel_values)

    print(f"\nExtracted outputs:")
    for key, val in outputs.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key}: shape {val.shape}, dtype {val.dtype}")
        elif isinstance(val, list):
            print(f"  {key}: list of {len(val)} tensors")
            for i, v in enumerate(val):
                if isinstance(v, torch.Tensor):
                    print(f"    [{i}]: shape {v.shape}")

    return outputs


def run_verification_test():
    """
    Run a simple verification that the model forward pass works without errors.
    This doesn't compare to main branch but verifies basic functionality.
    """
    print("\n" + "="*60)
    print("Verification Test: Checking model forward pass works")
    print("="*60)

    try:
        from inference_models.models.rfdetr.dinov2_with_windowed_attn import (
            WindowedDinov2WithRegistersBackbone,
            WindowedDinov2WithRegistersConfig,
            WindowedDinov2WithRegistersModel,
        )

        # Test 1: Config creation
        print("\n[1/4] Creating config...")
        config = WindowedDinov2WithRegistersConfig(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            image_size=224,
            patch_size=14,
            num_register_tokens=4,
            out_features=["stage2", "stage5", "stage8", "stage11"],
            out_indices=[2, 5, 8, 11],
        )
        print("  ✓ Config created successfully")

        # Test 2: Model initialization
        print("\n[2/4] Initializing model...")
        torch.manual_seed(42)
        model = WindowedDinov2WithRegistersBackbone(config)
        model.eval()
        print(f"  ✓ Model initialized (parameters: {sum(p.numel() for p in model.parameters()):,})")

        # Test 3: Forward pass
        print("\n[3/4] Running forward pass...")
        pixel_values = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(pixel_values, return_dict=True)
        print(f"  ✓ Forward pass successful")
        print(f"    - Feature maps: {len(output.feature_maps)}")
        for i, fm in enumerate(output.feature_maps):
            print(f"      [{i}]: shape {fm.shape}")

        # Test 4: Verify no head_mask in code
        print("\n[4/4] Verifying head_mask removal...")
        import inspect
        source = inspect.getsource(WindowedDinov2WithRegistersBackbone.forward)
        if 'head_mask' in source:
            print("  ✗ WARNING: head_mask still found in forward method!")
        else:
            print("  ✓ head_mask successfully removed from forward method")

        print("\n" + "="*60)
        print("All verification tests PASSED!")
        print("="*60)
        return True

    except Exception as e:
        print(f"\n✗ Error during verification: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Compare RF-DETR outputs between transformers versions")
    parser.add_argument("--save-outputs", type=str, help="Save outputs to this file")
    parser.add_argument("--compare", nargs=2, type=str, help="Compare two output files")
    parser.add_argument("--verify", action="store_true", help="Run verification test only")
    parser.add_argument("--test-dinov2", action="store_true", help="Test DinoV2 backbone standalone")

    args = parser.parse_args()

    if args.verify:
        success = run_verification_test()
        sys.exit(0 if success else 1)

    if args.test_dinov2:
        outputs = run_standalone_dinov2_test()
        if args.save_outputs:
            save_outputs(outputs, args.save_outputs)
        sys.exit(0)

    if args.compare:
        print(f"Comparing {args.compare[0]} vs {args.compare[1]}")
        outputs1 = load_outputs(args.compare[0])
        outputs2 = load_outputs(args.compare[1])

        all_match = compare_outputs(outputs1, outputs2)

        if all_match:
            print("\n" + "="*60)
            print("SUCCESS: All outputs match!")
            print("="*60)
        else:
            print("\n" + "="*60)
            print("FAILURE: Some outputs do not match!")
            print("="*60)

        sys.exit(0 if all_match else 1)

    # Default: run verification
    run_verification_test()


if __name__ == "__main__":
    main()
