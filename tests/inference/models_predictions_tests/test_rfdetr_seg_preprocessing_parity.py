"""End-to-end RF-DETR segmentation parity for Triton preprocessing.

This test uses a small set of images
already present in the repository and toggles the Triton path in-process.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
cv2 = pytest.importorskip("cv2")
pytest.importorskip("triton")
if not torch.cuda.is_available():  # pragma: no cover - host-dependent
    pytest.skip("CUDA not available", allow_module_level=True)

REPO_ROOT = Path(__file__).resolve().parents[3]
ASSETS_DIR = REPO_ROOT / "tests" / "inference" / "models_predictions_tests" / "assets"
DEFAULT_MODEL_ID = "rfdetr-seg-nano"
DEFAULT_CONFIDENCE = 0.4
BACKENDS = ("torch", "onnx", "trt")
IMAGE_PATHS = [
    ASSETS_DIR / "beer.jpg",
    ASSETS_DIR / "person_image.jpg",
    ASSETS_DIR / "truck.jpg",
    ASSETS_DIR / "melee.jpg",
]
MATCH_IOU_THRESHOLD = 0.5
MIN_BOX_IOU = 0.9
MIN_MEAN_BOX_IOU = 0.97
MAX_SCORE_DELTA = 0.02
MIN_MASK_IOU = 0.9
MIN_MEAN_MASK_IOU = 0.97


def ensure_backend_dependencies_available(backend: str) -> None:
    if backend == "onnx":
        onnxruntime = pytest.importorskip("onnxruntime")
        available_providers = set(onnxruntime.get_available_providers())
        if "CUDAExecutionProvider" not in available_providers:
            pytest.skip("onnxruntime CUDAExecutionProvider not available")
    elif backend == "trt":
        pytest.importorskip("tensorrt")
        pytest.importorskip("pycuda.driver")


def configure_backend_environment(
    backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    if backend == "onnx" and "ONNXRUNTIME_EXECUTION_PROVIDERS" not in os.environ:
        monkeypatch.setenv(
            "ONNXRUNTIME_EXECUTION_PROVIDERS",
            "[CUDAExecutionProvider,CPUExecutionProvider]",
        )
    elif backend == "trt" and "ONNXRUNTIME_EXECUTION_PROVIDERS" not in os.environ:
        monkeypatch.setenv(
            "ONNXRUNTIME_EXECUTION_PROVIDERS",
            "[TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider]",
        )


def load_parity_model(backend: str):
    from inference_models import AutoModel

    return AutoModel.from_pretrained(
        DEFAULT_MODEL_ID,
        backend=backend,
        device="cuda",
        trt_engine_host_code_allowed=True,
    )


def collect_parity_records(model, backend: str, use_triton: bool) -> dict:
    import inference_models.models.rfdetr.pre_processing as pre_processing

    original_triton_kernel = pre_processing.triton_preprocess_rfdetr_stretch
    if use_triton:
        assert (
            original_triton_kernel is not None
        ), "RF-DETR Triton preprocessing kernel is unavailable."

    triton_calls = {"count": 0}
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            pre_processing,
            "USE_TRITON_FOR_PREPROCESSING",
            use_triton,
        )
        if use_triton:

            def counting(*args, **kwargs):
                triton_calls["count"] += 1
                return original_triton_kernel(*args, **kwargs)

            monkeypatch.setattr(
                pre_processing,
                "triton_preprocess_rfdetr_stretch",
                counting,
            )

        records = []
        for image_path in IMAGE_PATHS:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Could not load test image: {image_path}")
            pre_processed, metadata = model.pre_process(image)
            raw_predictions = model.forward(pre_processed)
            detections = model.post_process(
                raw_predictions,
                metadata,
                confidence=DEFAULT_CONFIDENCE,
            )[0]
            detections_count = int(detections.class_id.numel())
            records.append(
                {
                    "path": str(image_path),
                    "xyxy": (
                        detections.xyxy.detach().cpu().numpy()
                        if detections_count
                        else np.zeros((0, 4), dtype=np.float32)
                    ),
                    "conf": (
                        detections.confidence.detach().cpu().numpy()
                        if detections_count
                        else np.zeros((0,), dtype=np.float32)
                    ),
                    "cls": (
                        detections.class_id.detach().cpu().numpy()
                        if detections_count
                        else np.zeros((0,), dtype=np.int32)
                    ),
                    "mask": (
                        detections.mask.detach().to(torch.bool).cpu().numpy()
                        if detections_count and detections.mask is not None
                        else None
                    ),
                }
            )

    return {
        "backend": backend,
        "records": records,
        "triton_calls": triton_calls["count"],
        "use_triton_for_preprocessing": use_triton,
    }


def iou_box(a: np.ndarray, b: np.ndarray) -> float:
    x0 = max(float(a[0]), float(b[0]))
    y0 = max(float(a[1]), float(b[1]))
    x1 = min(float(a[2]), float(b[2]))
    y1 = min(float(a[3]), float(b[3]))
    inter_w = max(0.0, x1 - x0)
    inter_h = max(0.0, y1 - y0)
    inter = inter_w * inter_h
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(
        0.0, float(a[3]) - float(a[1])
    )
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(
        0.0, float(b[3]) - float(b[1])
    )
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def summarize_parity(triton_run: dict, reference_run: dict) -> dict:
    triton_records = triton_run["records"]
    reference_records = reference_run["records"]
    if len(triton_records) != len(reference_records):
        raise AssertionError(
            "Collected different numbers of images between parity runs: "
            f"{len(triton_records)} != {len(reference_records)}"
        )

    total_triton_detections = 0
    total_reference_detections = 0
    matched_detections = 0
    count_mismatch_images = 0
    class_disagreements = 0
    mask_presence_mismatches = 0
    box_ious = []
    score_deltas = []
    mask_ious = []

    for triton_record, reference_record in zip(triton_records, reference_records):
        if triton_record["path"] != reference_record["path"]:
            raise AssertionError(
                "Image order mismatch between parity runs: "
                f"{triton_record['path']} != {reference_record['path']}"
            )

        triton_boxes = triton_record["xyxy"]
        reference_boxes = reference_record["xyxy"]
        triton_scores = triton_record["conf"]
        reference_scores = reference_record["conf"]
        triton_classes = triton_record["cls"]
        reference_classes = reference_record["cls"]
        triton_masks = triton_record["mask"]
        reference_masks = reference_record["mask"]

        triton_count = len(triton_boxes)
        reference_count = len(reference_boxes)
        total_triton_detections += triton_count
        total_reference_detections += reference_count
        if triton_count != reference_count:
            count_mismatch_images += 1
        if triton_count == 0 and reference_count == 0:
            continue

        used_triton_indices = set()
        for reference_index in range(reference_count):
            best_triton_index = -1
            best_iou = MATCH_IOU_THRESHOLD
            for triton_index in range(triton_count):
                if triton_index in used_triton_indices:
                    continue
                iou = iou_box(
                    triton_boxes[triton_index],
                    reference_boxes[reference_index],
                )
                if iou > best_iou:
                    best_iou = iou
                    best_triton_index = triton_index
            if best_triton_index < 0:
                continue

            used_triton_indices.add(best_triton_index)
            matched_detections += 1
            box_ious.append(best_iou)
            score_deltas.append(
                abs(
                    float(triton_scores[best_triton_index])
                    - float(reference_scores[reference_index])
                )
            )
            if int(triton_classes[best_triton_index]) != int(
                reference_classes[reference_index]
            ):
                class_disagreements += 1

            if (triton_masks is None) != (reference_masks is None):
                mask_presence_mismatches += 1
            elif triton_masks is not None and reference_masks is not None:
                triton_mask = triton_masks[best_triton_index]
                reference_mask = reference_masks[reference_index]
                intersection = np.logical_and(triton_mask, reference_mask).sum()
                union = np.logical_or(triton_mask, reference_mask).sum()
                mask_ious.append(float(intersection) / float(union) if union else 0.0)

    unmatched_reference_detections = total_reference_detections - matched_detections
    return {
        "backend": reference_run["backend"],
        "images": len(reference_records),
        "triton_calls_enabled": int(triton_run["triton_calls"]),
        "triton_calls_disabled": int(reference_run["triton_calls"]),
        "total_triton_detections": int(total_triton_detections),
        "total_reference_detections": int(total_reference_detections),
        "matched_detections": int(matched_detections),
        "unmatched_reference_detections": int(unmatched_reference_detections),
        "count_mismatch_images": int(count_mismatch_images),
        "class_disagreements": int(class_disagreements),
        "mask_presence_mismatches": int(mask_presence_mismatches),
        "mean_box_iou": float(np.mean(box_ious)) if box_ious else None,
        "min_box_iou": float(np.min(box_ious)) if box_ious else None,
        "mean_abs_score_delta": (
            float(np.mean(score_deltas)) if score_deltas else None
        ),
        "max_abs_score_delta": float(np.max(score_deltas)) if score_deltas else None,
        "mean_mask_iou": float(np.mean(mask_ious)) if mask_ious else None,
        "min_mask_iou": float(np.min(mask_ious)) if mask_ious else None,
    }


def format_summary(summary: dict) -> str:
    return json.dumps(summary, indent=2, sort_keys=True)


@pytest.mark.timeout(1200)
@pytest.mark.slow
@pytest.mark.parametrize("backend", BACKENDS, ids=BACKENDS)
def test_rfdetr_seg_nano_triton_preprocessing_matches_reference_path(
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_images = [
        str(image_path) for image_path in IMAGE_PATHS if not image_path.is_file()
    ]
    assert not missing_images, f"Missing parity images: {missing_images}"

    ensure_backend_dependencies_available(backend=backend)
    configure_backend_environment(backend=backend, monkeypatch=monkeypatch)
    model = load_parity_model(backend=backend)

    try:
        enabled_run = collect_parity_records(
            model=model,
            backend=backend,
            use_triton=True,
        )
        disabled_run = collect_parity_records(
            model=model,
            backend=backend,
            use_triton=False,
        )
    finally:
        del model
        torch.cuda.empty_cache()

    summary = summarize_parity(
        triton_run=enabled_run,
        reference_run=disabled_run,
    )
    summary_text = format_summary(summary)

    assert enabled_run["backend"] == disabled_run["backend"] == backend
    assert enabled_run["triton_calls"] == len(IMAGE_PATHS), summary_text
    assert disabled_run["triton_calls"] == 0, summary_text
    assert summary["total_reference_detections"] > 0, summary_text
    assert summary["count_mismatch_images"] == 0, summary_text
    assert summary["class_disagreements"] == 0, summary_text
    assert summary["mask_presence_mismatches"] == 0, summary_text
    assert summary["unmatched_reference_detections"] == 0, summary_text
    assert summary["mean_box_iou"] is not None, summary_text
    assert summary["min_box_iou"] is not None, summary_text
    assert summary["mean_abs_score_delta"] is not None, summary_text
    assert summary["max_abs_score_delta"] is not None, summary_text
    assert summary["mean_mask_iou"] is not None, summary_text
    assert summary["min_mask_iou"] is not None, summary_text
    assert summary["mean_box_iou"] >= MIN_MEAN_BOX_IOU, summary_text
    assert summary["min_box_iou"] >= MIN_BOX_IOU, summary_text
    assert summary["max_abs_score_delta"] <= MAX_SCORE_DELTA, summary_text
    assert summary["mean_mask_iou"] >= MIN_MEAN_MASK_IOU, summary_text
    assert summary["min_mask_iou"] >= MIN_MASK_IOU, summary_text
