"""Isolated RF-DETR segmentation post_process() microbenchmark.

Benchmarks only model.post_process() on frozen TensorRT outputs for a few
representative original-image sizes. Run once with
RFDETR_TRITON_POSTPROC=true and once with RFDETR_TRITON_POSTPROC=false.
"""

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch

os.environ.setdefault(
    "DISABLED_INFERENCE_MODELS_BACKENDS",
    "torch,torch-script,onnx,hugging-face,ultralytics,mediapipe,custom",
)

DEFAULT_MODEL_ID = "rfdetr-seg-nano"
PREFERRED_LOCAL_TRT_PACKAGE = "rfdetr-seg-nano-orin-trt-package"


def _is_local_trt_package(path: Path) -> bool:
    if not path.is_dir():
        return False
    required_files = ("engine.plan", "model_config.json", "inference_config.json")
    if not all((path / file_name).is_file() for file_name in required_files):
        return False
    try:
        model_config = json.loads((path / "model_config.json").read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return model_config.get("backend_type") == "trt"


def _find_local_trt_package() -> str | None:
    preferred = Path.cwd() / PREFERRED_LOCAL_TRT_PACKAGE
    if _is_local_trt_package(preferred):
        return str(preferred.resolve())

    candidates = sorted(
        path.resolve() for path in Path.cwd().iterdir() if _is_local_trt_package(path)
    )
    if len(candidates) == 1:
        return str(candidates[0])
    return None


LOCAL_TRT_PACKAGE = _find_local_trt_package()
if LOCAL_TRT_PACKAGE is not None:
    os.environ.setdefault("ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES", "True")

from inference_models import AutoModel


DEFAULT_SIZES = ("176x312", "720x1280", "1080x1920")
DEFAULT_VIDEO = Path("vehicles_312px.mp4")


def _parse_hw(spec: str) -> tuple[int, int]:
    try:
        h, w = spec.lower().split("x", 1)
        return int(h), int(w)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid size '{spec}', expected HxW") from exc


def _read_seed_frame(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"failed to read a frame from {video_path}")
    return frame


def _sync_detection(det) -> None:
    done_event = getattr(det, "_postproc_done_event", None)
    if done_event is not None:
        done_event.synchronize()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _prepare_case(model, frame, height: int, width: int):
    resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
    with torch.inference_mode():
        preprocessed, metadata = model.pre_process(resized)
        outputs = model.forward(preprocessed)
    return outputs, metadata


def _benchmark_case(
    model,
    outputs,
    metadata,
    confidence: float,
    warmup: int,
    iterations: int,
):
    latencies_ms = []
    with torch.inference_mode():
        for _ in range(warmup):
            det = model.post_process(outputs, metadata, confidence=confidence)[0]
            _sync_detection(det)

        det_count = 0
        for _ in range(iterations):
            start = time.perf_counter()
            det = model.post_process(outputs, metadata, confidence=confidence)[0]
            _sync_detection(det)
            latencies_ms.append((time.perf_counter() - start) * 1000.0)
            det_count += int(det.class_id.numel())
    samples = np.array(latencies_ms, dtype=np.float64)
    return {
        "mean_ms": float(samples.mean()),
        "p50_ms": float(np.percentile(samples, 50)),
        "p90_ms": float(np.percentile(samples, 90)),
        "p95_ms": float(np.percentile(samples, 95)),
        "detections": det_count // max(1, iterations),
    }


def _resolve_model_id(model_id: str) -> str:
    if model_id == DEFAULT_MODEL_ID and LOCAL_TRT_PACKAGE is not None:
        return LOCAL_TRT_PACKAGE
    return model_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_reference", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--confidence", type=float, default=0.4)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--size", dest="sizes", action="append")
    args = parser.parse_args()

    specs = args.sizes if args.sizes else list(DEFAULT_SIZES)
    sizes = [_parse_hw(spec) for spec in specs]
    flag = os.environ.get("RFDETR_TRITON_POSTPROC", "<unset>")

    print(
        f"[setup] RFDETR_TRITON_POSTPROC={flag} "
        f"video={args.video_reference} warmup={args.warmup} iterations={args.iterations}",
        flush=True,
    )

    frame = _read_seed_frame(args.video_reference)
    model_id = _resolve_model_id(args.model_id)
    if model_id != args.model_id:
        print(f"[model] using local TRT package: {model_id}", flush=True)
    model = AutoModel.from_pretrained(model_id)
    cases = []
    for height, width in sizes:
        outputs, metadata = _prepare_case(model, frame, height, width)
        cases.append((height, width, outputs, metadata))

    print("size,detections,mean_ms,p50_ms,p90_ms,p95_ms", flush=True)
    for height, width, outputs, metadata in cases:
        stats = _benchmark_case(
            model=model,
            outputs=outputs,
            metadata=metadata,
            confidence=args.confidence,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        print(
            f"{height}x{width},{stats['detections']},{stats['mean_ms']:.4f},"
            f"{stats['p50_ms']:.4f},{stats['p90_ms']:.4f},{stats['p95_ms']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
