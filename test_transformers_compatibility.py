#!/usr/bin/env python
"""
Test script to verify all transformer-based models can be imported
after transformers>=5.2.0 upgrade.

Each model is tested in a separate subprocess to avoid circular import issues.
"""

import subprocess
import sys
from typing import List, Tuple

# Models that use transformers and their import paths
TRANSFORMER_MODELS = [
    # (name, module_path, class_name, description)
    ("qwen3_5", "inference_models.models.qwen3_5.qwen3_5_hf", "Qwen35HF", "Qwen 3.5 VLM"),
    ("qwen3vl", "inference_models.models.qwen3vl.qwen3vl_hf", "Qwen3VLHF", "Qwen 3 VL"),
    ("qwen25vl", "inference_models.models.qwen25vl.qwen25vl_hf", "Qwen25VLHF", "Qwen 2.5 VL"),
    ("smolvlm", "inference_models.models.smolvlm.smolvlm_hf", "SmolVLMHF", "SmoL VLM"),
    ("paligemma", "inference_models.models.paligemma.paligemma_hf", "PaliGemmaHF", "PaliGemma"),
    ("florence2", "inference_models.models.florence2.florence2_hf", "Florence2HF", "Florence-2"),
    ("vit_cls", "inference_models.models.vit.vit_classification_huggingface", "VITForClassificationHF", "ViT Classification"),
    ("vit_multi", "inference_models.models.vit.vit_classification_huggingface", "VITForMultiLabelClassificationHF", "ViT Multi-Label"),
    ("trocr", "inference_models.models.trocr.trocr_hf", "TROcrHF", "TrOCR"),
    ("depth_v2", "inference_models.models.depth_anything_v2.depth_anything_v2_hf", "DepthAnythingV2HF", "Depth Anything V2"),
    ("depth_v3", "inference_models.models.depth_anything_v3.depth_anything_v3_torch", "DepthAnythingV3Torch", "Depth Anything V3"),
    ("owlv2", "inference_models.models.owlv2.owlv2_hf", "OWLv2HF", "OWLv2"),
    ("rfdetr_backbone", "inference_models.models.rfdetr.rfdetr_backbone_pytorch", "Backbone", "RF-DETR Backbone"),
    ("rfdetr_dinov2", "inference_models.models.rfdetr.dinov2_with_windowed_attn", "WindowedDinov2WithRegistersModel", "DINOv2 Windowed"),
]

TRANSFORMERS_MODEL_CLASSES = [
    "Qwen3_5ForConditionalGeneration",
    "Qwen3VLForConditionalGeneration",
    "Qwen2_5_VLForConditionalGeneration",
    "Qwen2_5_VLProcessor",
    "PaliGemmaForConditionalGeneration",
    "Florence2ForConditionalGeneration",
    "Florence2Processor",
    "ViTModel",
    "TrOCRProcessor",
    "VisionEncoderDecoderModel",
    "Owlv2ForObjectDetection",
    "Owlv2Processor",
    "AutoImageProcessor",
    "AutoModelForDepthEstimation",
    "AutoBackbone",
    "AutoModelForImageTextToText",
]


def test_import_in_subprocess(module_path: str, class_name: str) -> Tuple[bool, str]:
    """Test if a model class can be imported in a fresh subprocess."""
    code = f'''
import sys
try:
    module = __import__("{module_path}", fromlist=["{class_name}"])
    cls = getattr(module, "{class_name}")
    print("SUCCESS")
except Exception as e:
    print(f"FAIL: {{type(e).__name__}}: {{e}}")
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
    )
    output = result.stdout.strip()
    if output == "SUCCESS":
        return True, "OK"
    else:
        return False, output.replace("FAIL: ", "")


def test_transformers_class(class_name: str) -> Tuple[bool, str]:
    """Test importing a transformers class in subprocess."""
    code = f'''
try:
    from transformers import {class_name}
    print("SUCCESS")
except Exception as e:
    print(f"FAIL: {{type(e).__name__}}: {{e}}")
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
    )
    output = result.stdout.strip()
    if output == "SUCCESS":
        return True, "OK"
    else:
        return False, output.replace("FAIL: ", "")


def test_transformers_basic() -> Tuple[bool, str]:
    """Test basic transformers imports."""
    code = '''
try:
    import transformers
    version = transformers.__version__
    from transformers import AutoProcessor, AutoModel, BitsAndBytesConfig
    from transformers.utils import is_flash_attn_2_available
    print(f"SUCCESS: {version}")
except Exception as e:
    print(f"FAIL: {type(e).__name__}: {e}")
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
    )
    output = result.stdout.strip()
    if output.startswith("SUCCESS:"):
        version = output.split(": ")[1]
        return True, f"transformers {version} - basic imports OK"
    else:
        return False, output.replace("FAIL: ", "")


def main():
    print("=" * 70)
    print("Testing Transformer-based Models Compatibility")
    print("(Each test runs in a separate subprocess)")
    print("=" * 70)
    print()

    # Test basic transformers
    print("1. Testing transformers library...")
    success, msg = test_transformers_basic()
    status = "✓" if success else "✗"
    print(f"   [{status}] {msg}")
    print()

    # Test specific transformers model classes
    print("2. Testing transformers model classes...")
    transformers_passed = 0
    transformers_failed = 0
    for class_name in TRANSFORMERS_MODEL_CLASSES:
        success, msg = test_transformers_class(class_name)
        status = "✓" if success else "✗"
        print(f"   [{status}] {class_name}: {msg}")
        if success:
            transformers_passed += 1
        else:
            transformers_failed += 1
    print()

    # Test inference_models imports
    print("3. Testing inference_models model imports...")
    results: List[Tuple[str, str, bool, str]] = []

    for name, module_path, class_name, description in TRANSFORMER_MODELS:
        success, msg = test_import_in_subprocess(module_path, class_name)
        results.append((name, description, success, msg))
        status = "✓" if success else "✗"
        print(f"   [{status}] {description} ({class_name})")
        if not success:
            print(f"       Error: {msg}")

    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)

    passed = sum(1 for _, _, success, _ in results if success)
    failed = sum(1 for _, _, success, _ in results if not success)

    print(f"Transformers classes: {transformers_passed}/{transformers_passed + transformers_failed}")
    print(f"Inference models: {passed}/{len(results)}")

    if failed > 0:
        print("\nFailed models:")
        for name, description, success, msg in results:
            if not success:
                print(f"  - {description}: {msg}")
        return 1

    print("\nAll transformer-based models imported successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
