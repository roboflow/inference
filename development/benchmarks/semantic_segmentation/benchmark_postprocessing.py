"""Benchmark the semantic segmentation post-processing path.

Measures, per frame, the two stages this PR stack optimizes:

* ``model_side`` - what the model does to hand masks to a consumer:
  PNG-encode + base64 both full-resolution masks (``base64_png`` transport),
  or nothing but a ``present_class_ids`` bincount (``numpy`` transport).
* ``convert`` - the workflow block's ``_convert_to_sv_detections``:
  decode (if base64), class scan, per-class bbox/confidence reductions and
  RLE encoding.

Modes (selected automatically based on what the checked-out tree supports):

* ``baseline``   - the pre-optimization algorithm, inlined below verbatim
                   (PNG/base64 transport, ``np.unique`` scan, per-class
                   ``asfortranarray`` copies). This is what inference<=1.3.7
                   executes.
* ``pr-base64``  - the production code with PNG/base64 transport plus the
                   ``present_class_ids`` hint (PR "class-ids + F-order-once").
* ``pr-numpy``   - the production code with in-process numpy masks plus the
                   hint (PR "response_mask_format=numpy" stacked on the
                   former). Skipped automatically when the checked-out tree
                   does not support ndarray masks.

Inputs are synthetic by default (blobby label map + smooth confidence field,
sized to the 16.1 MP camera that motivated the work) and fully described in
the report; supply ``--label-map`` / ``--confidence-map`` (grayscale PNG or
``.npy``) to measure real data.

Latency run (defaults: 3 warmup + 30 measured iterations):

    PYTHONPATH=. python development/benchmarks/semantic_segmentation/benchmark_postprocessing.py \
        --iterations 100 --json results.json

Memory run (per-mode subprocesses so RSS high-water marks do not bleed
between modes; reports tracemalloc peak + RSS):

    PYTHONPATH=. python development/benchmarks/semantic_segmentation/benchmark_postprocessing.py --memory

The on-device numbers in the PR descriptions came from the same code paths
measured end-to-end on a Jetson AGX Orin (SEMSEG_TIMING stage profiler in
roboflow-edge); this script is the portable reproduction of the CPU
post-processing cost those tables attribute.
"""

import argparse
import base64
import io
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pycocotools.mask as mask_utils
from PIL import Image

from inference.core.workflows.core_steps.models.roboflow.semantic_segmentation.v2 import (
    RoboflowSemanticSegmentationModelBlockV2,
)

try:  # torch is required by inference-models; tolerate its absence anyway
    import torch

    from inference.core.models.semantic_segmentation_utils import (
        present_class_ids_from_label_map,
    )
except ImportError:  # pragma: no cover - environment without torch
    torch = None
    present_class_ids_from_label_map = None

CONVERT = RoboflowSemanticSegmentationModelBlockV2._convert_to_sv_detections
DEFAULT_HEIGHT = 3032
DEFAULT_WIDTH = 5320
PERCENTILES = (50, 95, 99)


# --------------------------------------------------------------------------
# Baseline reference: the pre-optimization `_convert_to_sv_detections` body
# (inference<=1.3.7), inlined verbatim so the benchmark can compare against
# it after the optimized code replaced it in the tree. Kept in sync with the
# parity tests in tests/workflows/unit_tests/.../semantic_segmentation/.
# --------------------------------------------------------------------------


def baseline_convert_to_sv_detections(predictions_dict: Dict) -> Any:
    from uuid import uuid4

    import supervision as sv

    from inference.core.workflows.execution_engine.constants import (
        DETECTION_ID_KEY,
        RLE_MASK_KEY_IN_SV_DETECTIONS,
    )

    seg_mask = predictions_dict.get("segmentation_mask", "")
    conf_mask = predictions_dict.get("confidence_mask", "")
    class_map: Dict[str, str] = predictions_dict.get("class_map", {})

    mask_bytes = base64.b64decode(seg_mask)
    nparr = np.frombuffer(mask_bytes, np.uint8)
    mask_array = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    if mask_array is None:
        return sv.Detections.empty()

    unique_class_ids = [cid for cid in np.unique(mask_array).tolist() if cid != 0]
    if not unique_class_ids:
        return sv.Detections.empty()

    conf_array = None
    if conf_mask:
        conf_bytes = base64.b64decode(conf_mask)
        conf_nparr = np.frombuffer(conf_bytes, np.uint8)
        conf_array = cv2.imdecode(conf_nparr, cv2.IMREAD_GRAYSCALE)

    xyxy_list, masks_list, class_id_list, class_name_list, confidence_list = (
        [],
        [],
        [],
        [],
        [],
    )
    for class_id in unique_class_ids:
        binary_mask = mask_array == class_id
        rows = np.where(np.any(binary_mask, axis=1))[0]
        cols = np.where(np.any(binary_mask, axis=0))[0]
        if rows.size == 0:
            continue
        xyxy_list.append([cols[0], rows[0], cols[-1], rows[-1]])
        masks_list.append(binary_mask)
        class_id_list.append(class_id)
        class_name_list.append(class_map.get(str(class_id), str(class_id)))
        if conf_array is not None:
            confidence_list.append(float(conf_array[binary_mask].mean()) / 255.0)
        else:
            confidence_list.append(1.0)

    if not class_id_list:
        return sv.Detections.empty()

    rle_list = []
    for mask in masks_list:
        rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
        rle["counts"] = rle["counts"].decode("utf-8")
        rle_list.append(rle)

    detection_ids = np.array([str(uuid4()) for _ in class_id_list])
    result = sv.Detections(
        xyxy=np.array(xyxy_list, dtype=np.float64),
        mask=None,
        class_id=np.array(class_id_list),
        confidence=np.array(confidence_list, dtype=np.float32),
        data={
            "class_name": np.array(class_name_list),
            DETECTION_ID_KEY: detection_ids,
            RLE_MASK_KEY_IN_SV_DETECTIONS: np.array(rle_list, dtype=object),
        },
    )
    if conf_array is not None:
        result["confidence_mask"] = conf_array
    return result


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


def synthetic_label_map(
    height: int, width: int, foreground_classes: int, seed: int
) -> np.ndarray:
    """Blobby multi-class label map: a few large regions per class plus light
    speckle, so PNG payloads land near what real segmentation masks produce
    (large uniform runs, some boundary entropy)."""
    rng = np.random.default_rng(seed)
    label_map = np.zeros((height, width), dtype=np.uint8)
    class_ids = list(range(1, foreground_classes + 1))
    for class_id in class_ids:
        for _ in range(rng.integers(2, 5)):
            center = (int(rng.integers(0, width)), int(rng.integers(0, height)))
            axes = (
                int(rng.integers(width // 20, width // 5)),
                int(rng.integers(height // 20, height // 5)),
            )
            angle = float(rng.uniform(0, 180))
            cv2.ellipse(label_map, center, axes, angle, 0, 360, int(class_id), -1)
    speckle = rng.random((height, width)) < 0.002
    label_map[speckle] = rng.choice(class_ids, size=int(speckle.sum()))
    return label_map


def synthetic_confidence_map(label_map: np.ndarray, seed: int) -> np.ndarray:
    """Confidence field shaped like real uint8 softmax surfaces: per-class
    saturated plateaus, smoothed dips at class boundaries, light
    quantization. A continuous random gradient would make every pixel differ
    from its neighbours and blow the PNG payload an order of magnitude past
    what real confidence masks produce (~250 KiB at 16 MP on the reference
    device), unfairly penalising the base64 modes; this lands in the same
    ballpark (exact payload sizes are recorded in the report metadata)."""
    rng = np.random.default_rng(seed)
    per_class_confidence = rng.uniform(190, 255, size=256).astype(np.uint8)
    plateaus = per_class_confidence[label_map]
    smoothed = cv2.GaussianBlur(plateaus, (0, 0), 5)
    quantized = np.round(smoothed.astype(np.float32) / 4.0) * 4.0
    return np.clip(quantized, 0, 255).astype(np.uint8)


def load_grayscale(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        array = np.load(path)
    else:
        array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if array is None:
            raise ValueError(f"could not read grayscale image from {path}")
    if array.dtype != np.uint8:
        raise ValueError(f"{path}: expected uint8 data, got {array.dtype}")
    return array


def encode_mask_png_b64(mask: np.ndarray) -> str:
    """Mirror of the model-side img_to_b64_str: PIL PNG encode + base64."""
    buffer = io.BytesIO()
    Image.fromarray(mask).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def detect_numpy_mask_support() -> bool:
    """True when the checked-out `_convert_to_sv_detections` accepts ndarray
    masks (the response_mask_format=numpy fast path).

    Behavioral probe: trees without the fast path hand the ndarray's raw
    buffer to base64.b64decode, which silently strips the non-alphabet bytes
    (yielding empty detections) or trips on the array elsewhere - so support
    is judged by the produced detection, not by exception type."""
    tiny = np.zeros((4, 4), dtype=np.uint8)
    tiny[1:3, 1:3] = 1
    try:
        result = CONVERT(
            {
                "segmentation_mask": tiny,
                "confidence_mask": np.full((4, 4), 200, dtype=np.uint8),
                "class_map": {"1": "a"},
            }
        )
    except Exception:
        return False
    return len(result) == 1 and result.class_id.tolist() == [1]


def compute_hint(label_map: np.ndarray, fallback_hint: List[int]) -> List[int]:
    """present_class_ids via the production torch helper when available (so
    its per-frame cost lands in the timed model_side stage, as on a real
    server); otherwise return the precomputed hint at zero cost rather than
    charging a non-production np.unique substitute to the PR modes."""
    if present_class_ids_from_label_map is not None:
        return present_class_ids_from_label_map(torch.from_numpy(label_map))
    return fallback_hint


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------


def run_mode(
    mode: str,
    label_map: np.ndarray,
    confidence_map: np.ndarray,
    class_map: Dict[str, str],
    warmup: int,
    iterations: int,
) -> Dict[str, List[float]]:
    samples: Dict[str, List[float]] = {
        "model_side_ms": [],
        "convert_ms": [],
        "total_ms": [],
    }
    fallback_hint = [int(v) for v in np.unique(label_map)]
    for iteration in range(warmup + iterations):
        start = time.perf_counter()
        if mode == "baseline":
            payload = {
                "segmentation_mask": encode_mask_png_b64(label_map),
                "confidence_mask": encode_mask_png_b64(confidence_map),
                "class_map": class_map,
            }
            model_side_end = time.perf_counter()
            result = baseline_convert_to_sv_detections(payload)
        elif mode == "pr-base64":
            hint = compute_hint(label_map, fallback_hint)
            payload = {
                "segmentation_mask": encode_mask_png_b64(label_map),
                "confidence_mask": encode_mask_png_b64(confidence_map),
                "class_map": class_map,
                "present_class_ids": hint,
            }
            model_side_end = time.perf_counter()
            result = CONVERT(payload)
        elif mode == "pr-numpy":
            hint = compute_hint(label_map, fallback_hint)
            payload = {
                "segmentation_mask": label_map,
                "confidence_mask": confidence_map,
                "class_map": class_map,
                "present_class_ids": hint,
            }
            model_side_end = time.perf_counter()
            result = CONVERT(payload)
        else:
            raise ValueError(f"unknown mode {mode}")
        end = time.perf_counter()
        if len(result) == 0:
            raise RuntimeError(f"mode {mode} produced no detections - bad input?")
        if iteration >= warmup:
            samples["model_side_ms"].append((model_side_end - start) * 1000.0)
            samples["convert_ms"].append((end - model_side_end) * 1000.0)
            samples["total_ms"].append((end - start) * 1000.0)
    return samples


def summarize(samples: List[float]) -> Dict[str, float]:
    array = np.asarray(samples, dtype=np.float64)
    p50, p95, p99 = np.percentile(array, PERCENTILES)
    return {
        "mean": float(array.mean()),
        "p50": float(p50),
        "p95": float(p95),
        "p99": float(p99),
        "min": float(array.min()),
        "max": float(array.max()),
        "stdev": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "n": int(array.size),
    }


def measure_memory(mode: str, args: argparse.Namespace) -> Dict[str, Any]:
    """Run one mode in a fresh subprocess so each gets its own RSS high-water
    mark, tracemalloc peak, and allocator state."""
    command = [
        sys.executable,
        os.path.abspath(__file__),
        "--memory-worker",
        mode,
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--foreground-classes",
        str(args.foreground_classes),
        "--seed",
        str(args.seed),
        "--memory-iterations",
        str(args.memory_iterations),
    ]
    if args.label_map:
        command += ["--label-map", args.label_map]
    if args.confidence_map:
        command += ["--confidence-map", args.confidence_map]
    environment = dict(os.environ)
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
    )
    environment["PYTHONPATH"] = os.pathsep.join(
        p for p in [repo_root, environment.get("PYTHONPATH")] if p
    )
    completed = subprocess.run(
        command, capture_output=True, text=True, env=environment, check=False
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"memory worker for {mode} failed:\n{completed.stderr[-2000:]}"
        )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def memory_worker(mode: str, args: argparse.Namespace) -> Dict[str, Any]:
    import tracemalloc

    try:
        import psutil

        process = psutil.Process()
    except ImportError:  # pragma: no cover
        process = None

    label_map, confidence_map, class_map, _ = build_inputs(args)
    if mode == "pr-numpy" and not detect_numpy_mask_support():
        return {"mode": mode, "skipped": "numpy masks unsupported by this tree"}

    # one untraced warmup so imports/caches do not count against the mode
    run_mode(mode, label_map, confidence_map, class_map, warmup=0, iterations=1)

    rss_before = process.memory_info().rss if process else None
    rss_peak = rss_before or 0
    tracemalloc.start()
    for _ in range(args.memory_iterations):
        run_mode(mode, label_map, confidence_map, class_map, warmup=0, iterations=1)
        if process:
            rss_peak = max(rss_peak, process.memory_info().rss)
    _, traced_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    result = {
        "mode": mode,
        "iterations": args.memory_iterations,
        "tracemalloc_peak_mb": traced_peak / 1e6,
        "rss_before_mb": rss_before / 1e6 if rss_before is not None else None,
        "rss_peak_mb": rss_peak / 1e6 if process else None,
        "rss_growth_mb": (rss_peak - rss_before) / 1e6 if process else None,
    }
    return result


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def build_inputs(
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, str], Dict[str, Any]]:
    if args.label_map:
        label_map = load_grayscale(args.label_map)
        label_source = args.label_map
    else:
        label_map = synthetic_label_map(
            args.height, args.width, args.foreground_classes, args.seed
        )
        label_source = "synthetic"
    if args.confidence_map:
        confidence_map = load_grayscale(args.confidence_map)
        confidence_source = args.confidence_map
    else:
        confidence_map = synthetic_confidence_map(label_map, args.seed + 1)
        confidence_source = "synthetic"
    if confidence_map.shape != label_map.shape:
        raise ValueError(
            f"label map {label_map.shape} and confidence map "
            f"{confidence_map.shape} shapes differ"
        )
    present = [int(v) for v in np.unique(label_map)]
    class_map = {str(cid): f"class_{cid}" for cid in present if cid != 0}
    height, width = label_map.shape
    seg_payload = encode_mask_png_b64(label_map)
    conf_payload = encode_mask_png_b64(confidence_map)
    metadata = {
        "label_map_source": label_source,
        "confidence_map_source": confidence_source,
        "height": height,
        "width": width,
        "megapixels": round(height * width / 1e6, 2),
        "dtype": str(label_map.dtype),
        "present_class_ids": present,
        "foreground_classes": len([c for c in present if c != 0]),
        "foreground_pixel_share": round(float((label_map != 0).mean()), 4),
        "seg_mask_png_b64_bytes": len(seg_payload),
        "conf_mask_png_b64_bytes": len(conf_payload),
        "seed": args.seed,
    }
    return label_map, confidence_map, class_map, metadata


def environment_metadata() -> Dict[str, Any]:
    import pycocotools
    import supervision

    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "pillow": Image.__version__ if hasattr(Image, "__version__") else "unknown",
        "supervision": supervision.__version__,
        "pycocotools": getattr(pycocotools, "__version__", "unknown"),
        "torch": torch.__version__ if torch is not None else None,
    }
    try:
        import inference.core.version

        versions["inference"] = inference.core.version.__version__
    except Exception:
        versions["inference"] = "unknown"
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "versions": versions,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def format_markdown(report: Dict[str, Any]) -> str:
    lines = []
    metadata = report["input"]
    lines.append(
        f"Input: {metadata['width']}x{metadata['height']} "
        f"({metadata['megapixels']} MP) uint8 label map, "
        f"{metadata['foreground_classes']} foreground classes "
        f"({metadata['foreground_pixel_share'] * 100:.1f}% fg pixels), "
        f"seg/conf PNG+b64 payloads "
        f"{metadata['seg_mask_png_b64_bytes'] / 1024:.0f} / "
        f"{metadata['conf_mask_png_b64_bytes'] / 1024:.0f} KiB "
        f"(source: {metadata['label_map_source']})"
    )
    lines.append("")
    lines.append(
        "| mode | stage | mean ms | p50 ms | p95 ms | p99 ms | min | max | n |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for mode, stages in report["latency"].items():
        if "skipped" in stages:
            lines.append(f"| {mode} | - | skipped: {stages['skipped']} | | | | | | |")
            continue
        for stage in ("model_side_ms", "convert_ms", "total_ms"):
            s = stages[stage]
            label = stage.replace("_ms", "")
            lines.append(
                f"| {mode} | {label} | {s['mean']:.1f} | {s['p50']:.1f} | "
                f"{s['p95']:.1f} | {s['p99']:.1f} | {s['min']:.1f} | "
                f"{s['max']:.1f} | {s['n']} |"
            )
    if report.get("memory"):
        lines.append("")
        lines.append(
            "| mode | tracemalloc peak MB | RSS before MB | RSS peak MB | RSS growth MB |"
        )
        lines.append("|---|---|---|---|---|")
        for entry in report["memory"]:
            if "skipped" in entry:
                lines.append(f"| {entry['mode']} | skipped: {entry['skipped']} | | | |")
                continue
            lines.append(
                f"| {entry['mode']} | {entry['tracemalloc_peak_mb']:.1f} | "
                f"{entry['rss_before_mb']:.1f} | {entry['rss_peak_mb']:.1f} | "
                f"{entry['rss_growth_mb']:.1f} |"
            )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument(
        "--foreground-classes",
        type=int,
        default=4,
        help="synthetic foreground class count (device workload averaged ~3.8)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument(
        "--label-map", help="grayscale PNG or .npy uint8 label map to use instead"
    )
    parser.add_argument(
        "--confidence-map", help="grayscale PNG or .npy uint8 confidence map"
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["baseline", "pr-base64", "pr-numpy"],
        help="restrict to specific modes (default: all supported)",
    )
    parser.add_argument("--json", help="write the full report (raw samples) here")
    parser.add_argument(
        "--memory",
        action="store_true",
        help="also measure per-mode memory in fresh subprocesses",
    )
    parser.add_argument("--memory-iterations", type=int, default=5)
    parser.add_argument("--memory-worker", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.memory_worker:
        print(json.dumps(memory_worker(args.memory_worker, args)))
        return 0

    label_map, confidence_map, class_map, input_metadata = build_inputs(args)
    numpy_supported = detect_numpy_mask_support()
    modes = args.modes or ["baseline", "pr-base64", "pr-numpy"]

    latency: Dict[str, Any] = {}
    raw: Dict[str, Any] = {}
    for mode in modes:
        if mode == "pr-numpy" and not numpy_supported:
            latency[mode] = {"skipped": "numpy masks unsupported by this tree"}
            print(f"[{mode}] skipped: tree lacks response_mask_format=numpy support")
            continue
        print(
            f"[{mode}] running {args.warmup} warmup + {args.iterations} "
            "measured iterations..."
        )
        samples = run_mode(
            mode, label_map, confidence_map, class_map, args.warmup, args.iterations
        )
        latency[mode] = {stage: summarize(values) for stage, values in samples.items()}
        raw[mode] = samples

    memory: List[Dict[str, Any]] = []
    if args.memory:
        for mode in modes:
            if mode == "pr-numpy" and not numpy_supported:
                memory.append(
                    {"mode": mode, "skipped": "numpy masks unsupported by this tree"}
                )
                continue
            print(f"[{mode}] measuring memory in subprocess...")
            memory.append(measure_memory(mode, args))

    report = {
        "input": input_metadata,
        "environment": environment_metadata(),
        "hint_helper": (
            "torch present_class_ids_from_label_map"
            if present_class_ids_from_label_map is not None
            else "np.unique substitute (torch unavailable)"
        ),
        "latency": latency,
        "memory": memory,
        "raw_samples": raw,
    }
    markdown = format_markdown(report)
    print()
    print(markdown)
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(report, handle, indent=2)
        print(f"\nfull report written to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
