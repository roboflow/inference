"""Benchmark RF-DETR depth-2 behavior inside the automate-style workflow.

This variant reuses ``codeflash-optimization-compare.py`` for all benchmark
plumbing, but replaces the single-block workflow with the automate definition:

    image -> instance segmentation -> mask visualization -> label visualization
          -> (gated) vision events
          + detection count, CLIP embedding, and image-change detection

The purpose is to stress the risky alignment path discussed during review:
RF-DETR may return Future-backed predictions at depth=2, and downstream
workflow blocks must receive the resolved detections and masks for the same frame.

Vision events use the local event-store code path (prediction conversion, image
packaging, metadata assembly) with the final HTTP POST stubbed out so FPS is not
dominated by network I/O.
"""

import importlib.util
import os
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


def _patch_backend_for_multi_model_workflow(base_module) -> None:
    """Allow TRT for RF-DETR and ONNX for CLIP when ``--backend trt`` is set.

    The base benchmark pins a single backend globally via
    ``DISABLED_INFERENCE_MODELS_BACKENDS``. The automate workflow also runs CLIP,
    which only has onnx/torch packages, so TRT-only pinning breaks on the first
    frame. RF-DETR still uses the local TRT package; CLIP auto-negotiates ONNX.
    """
    original_configure_backend = base_module._configure_backend
    all_backends = base_module._ALL_BACKENDS

    def _configure_backend(backend: str) -> None:
        os.environ.setdefault(
            "ONNXRUNTIME_EXECUTION_PROVIDERS",
            "[TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider]",
        )
        if backend == "trt":
            disabled_backends = sorted(all_backends - {"trt", "onnx"})
            os.environ["DISABLED_INFERENCE_MODELS_BACKENDS"] = ",".join(
                disabled_backends
            )
            return

        original_configure_backend(backend=backend)

    base_module._configure_backend = _configure_backend


def _patch_vision_events_local_sink() -> None:
    """Stub the event-store HTTP POST while keeping payload assembly on the hot path."""
    from inference.core.workflows.core_steps.sinks.roboflow.vision_events import (
        v1 as vision_events_module,
    )

    def _noop_send_local_event(url: str, payload: dict) -> tuple[bool, str, str]:
        del url, payload
        return False, "Vision event sink mocked for benchmark", ""

    vision_events_module._send_local_event = _noop_send_local_event


def build_workflow(model_id: str, confidence: float) -> dict:
    """Build the automate-style segmentation and visualization workflow."""
    return {
        "version": "1.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "steps": [
            {
                "type": "roboflow_core/roboflow_instance_segmentation_model@v3",
                "name": "model",
                "images": "$inputs.image",
                "model_id": model_id,
                "confidence_mode": "custom",
                "iou_threshold": 0.4,
                "custom_confidence": confidence,
                "enforce_dense_masks_in_inference_models": False,
            },
            {
                "type": "roboflow_core/mask_visualization@v1",
                "name": "mask_visualization",
                "image": "$inputs.image",
                "predictions": "$steps.model.predictions",
            },
            {
                "type": "roboflow_core/label_visualization@v1",
                "name": "label_visualization",
                "image": "$steps.mask_visualization.image",
                "predictions": "$steps.model.predictions",
                "text": "Class",
            },
            {
                "type": "roboflow_core/roboflow_vision_events@v1",
                "name": "roboflow_vision_events",
                "event_type": "quality_check",
                "solution": "benchmark",
                "predictions": "$steps.model.predictions",
                "input_image": "$inputs.image",
                "output_image": "$steps.label_visualization.image",
                "custom_metadata": {
                    "source": "training-table",
                },
                "write_to_event_store": True,
                "fire_and_forget": True,
            },
            {
                "type": "roboflow_core/property_definition@v1",
                "name": "detection_count",
                "data": "$steps.model.predictions",
                "operations": [
                    {
                        "type": "SequenceLength",
                    }
                ],
            },
            {
                "type": "roboflow_core/clip@v1",
                "name": "clip_embedding",
                "data": "$inputs.image",
            },
            {
                "type": "roboflow_core/identify_changes@v1",
                "name": "image_change_detector",
                "strategy": "Sliding Window",
                "embedding": "$steps.clip_embedding.embedding",
                "threshold_percentile": 0.2,
                "warmup": 2,
                "window_size": 2,
            },
            {
                "type": "roboflow_core/continue_if@v1",
                "name": "gate_vision_events",
                "condition_statement": {
                    "type": "StatementGroup",
                    "operator": "and",
                    "statements": [
                        {
                            "type": "BinaryStatement",
                            "left_operand": {
                                "type": "DynamicOperand",
                                "operand_name": "detection_count",
                            },
                            "comparator": {
                                "type": "(Number) >",
                            },
                            "right_operand": {
                                "type": "StaticOperand",
                                "value": 0,
                            },
                        },
                        {
                            "type": "UnaryStatement",
                            "operand": {
                                "type": "DynamicOperand",
                                "operand_name": "image_changed",
                            },
                            "operator": {
                                "type": "(Boolean) is True",
                            },
                        },
                    ],
                },
                "evaluation_parameters": {
                    "detection_count": "$steps.detection_count.output",
                    "image_changed": "$steps.image_change_detector.is_outlier",
                },
                "next_steps": [
                    "$steps.roboflow_vision_events",
                ],
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "output_image",
                "coordinates_system": "own",
                "selector": "$steps.label_visualization.image",
            },
            {
                "type": "JsonField",
                "name": "predictions",
                "coordinates_system": "own",
                "selector": "$steps.model.predictions",
            },
            {
                "type": "JsonField",
                "name": "detection_count",
                "selector": "$steps.detection_count.output",
            },
            {
                "type": "JsonField",
                "name": "image_changed",
                "selector": "$steps.image_change_detector.is_outlier",
            },
        ],
    }


def main() -> None:
    _patch_vision_events_local_sink()
    base_module = _load_base_module()
    _patch_backend_for_multi_model_workflow(base_module)
    base_module._SELF = _THIS_FILE
    base_module.build_workflow = build_workflow
    base_module.main()


if __name__ == "__main__":
    main()
