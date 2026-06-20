"""Benchmark RF-DETR depth-2 behavior inside a multi-step workflow.

This variant reuses ``codeflash-optimization-compare.py`` for all benchmark
plumbing, but replaces the single-block workflow with:

    image -> relative static crop -> RF-DETR instance segmentation -> confidence filter

The purpose is to stress the risky alignment path discussed during review:
RF-DETR may return Future-backed predictions at depth=2, and a downstream
workflow block must receive the resolved detections for the same cropped frame.
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
    """Build a crop -> RF-DETR -> filter workflow for stream-pipeline testing."""
    return {
        "version": "1.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "steps": [
            {
                "type": "roboflow_core/relative_statoic_crop@v1",
                "name": "center_crop",
                "images": "$inputs.image",
                "x_center": 0.5,
                "y_center": 0.5,
                "width": 0.8,
                "height": 0.8,
            },
            {
                "type": "roboflow_core/roboflow_instance_segmentation_model@v3",
                "name": "segmentation",
                "images": "$steps.center_crop.crops",
                "model_id": model_id,
                "confidence_mode": "custom",
                "custom_confidence": confidence,
                "enforce_dense_masks_in_inference_models": False,
            },
            {
                "type": "roboflow_core/per_class_confidence_filter@v1",
                "name": "confidence_filter",
                "predictions": "$steps.segmentation.predictions",
                "class_thresholds": {},
                "default_threshold": 0.0,
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "predictions",
                "selector": "$steps.confidence_filter.predictions",
            },
            {
                "type": "JsonField",
                "name": "raw_predictions",
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
