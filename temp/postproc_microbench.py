"""Isolated RF-DETR segmentation post_process() microbenchmark.

Benchmarks only model.post_process() on frozen TensorRT outputs for a few
representative original-image sizes. Run once with
RFDETR_TRITON_POSTPROC=true and once with RFDETR_TRITON_POSTPROC=false.
"""

import argparse
import os
import time
from pathlib import Path

import cv2
import torch

os.environ.setdefault(
    "DISABLED_INFERENCE_MODELS_BACKENDS",
    "torch,torch-script,onnx,hugging-face,ultralytics,mediapipe,custom",
)

from inference_models import AutoModel


DEFAULT_SIZES = ("176x312", "720x1280", "1080x1920")
DEFAULT_VIDEO = Path("/home/ubuntu/inference/vehicles_312px.mp4")


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
    with torch.inference_mode():
        for _ in range(warmup):
            det = model.post_process(outputs, metadata, confidence=confidence)[0]
            _sync_detection(det)

        start = time.perf_counter()
        det_count = 0
        for _ in range(iterations):
            det = model.post_process(outputs, metadata, confidence=confidence)[0]
            _sync_detection(det)
            det_count += int(det.class_id.numel())
        elapsed = time.perf_counter() - start
    return (elapsed * 1000.0) / iterations, det_count // max(1, iterations)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_reference", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--model_id", default="rfdetr-seg-nano")
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
    model = AutoModel.from_pretrained(args.model_id)
    cases = []
    for height, width in sizes:
        outputs, metadata = _prepare_case(model, frame, height, width)
        cases.append((height, width, outputs, metadata))

    print("size,detections,mean_ms", flush=True)
    for height, width, outputs, metadata in cases:
        mean_ms, detections = _benchmark_case(
            model=model,
            outputs=outputs,
            metadata=metadata,
            confidence=args.confidence,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        print(f"{height}x{width},{detections},{mean_ms:.4f}", flush=True)


if __name__ == "__main__":
    main()
