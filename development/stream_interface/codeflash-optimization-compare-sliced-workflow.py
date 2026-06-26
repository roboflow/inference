"""Benchmark RF-DETR depth-2 behavior on multiple slices per video frame.

This variant reuses ``codeflash-optimization-compare.py`` for all benchmark
plumbing, but replaces the single-block workflow with:

    image -> image_slicer@v2 -> RF-DETR instance segmentation

The purpose is to stress batch-order alignment when one input frame fans out
into multiple cropped images before RF-DETR runs. At depth=2, the delayed
RF-DETR response must remain paired with the correct source frame and slice
batch order, including the final flushed frame.
"""

import importlib.util
from pathlib import Path
import sys


_THIS_FILE = Path(__file__).resolve()
_BASE_SCRIPT = _THIS_FILE.with_name("codeflash-optimization-compare.py")


def _load_base_module():
    spec = importlib.util.spec_from_file_location(
        "codeflash_optimization_compare_base",
        _BASE_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load base benchmark script: {_BASE_SCRIPT}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_workflow(model_id: str, confidence: float) -> dict:
    """Build an image-slicer -> RF-DETR workflow for stream-pipeline testing."""
    return {
        "version": "1.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "steps": [
            {
                "type": "roboflow_core/image_slicer@v2",
                "name": "image_slicer",
                "image": "$inputs.image",
                "slice_width": 640,
                "slice_height": 640,
                "overlap_ratio_width": 0.0,
                "overlap_ratio_height": 0.0,
            },
            {
                "type": "roboflow_core/roboflow_instance_segmentation_model@v3",
                "name": "segmentation",
                "images": "$steps.image_slicer.slices",
                "model_id": model_id,
                "confidence_mode": "custom",
                "custom_confidence": confidence,
                "enforce_dense_masks_in_inference_models": False,
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "predictions",
                "selector": "$steps.segmentation.predictions",
            },
        ],
    }


def main() -> None:
    base_module = _load_base_module()
    base_module._SELF = _THIS_FILE
    base_module.build_workflow = build_workflow
    base_module.main()


if __name__ == "__main__":
    main()
