"""Isolate pre_process() timing. Path selected by
USE_TRITON_FOR_PREPROCESSING. Reports the Triton
kernel call count so the reviewer can see which path ran.

Usage (run twice and compare):
  USE_TRITON_FOR_PREPROCESSING=true  python temp/preproc_microbench.py
  USE_TRITON_FOR_PREPROCESSING=false python temp/preproc_microbench.py
"""
import os
import time

os.environ.setdefault(
    "DISABLED_INFERENCE_MODELS_BACKENDS",
    "torch,torch-script,onnx,hugging-face,ultralytics,mediapipe,custom",
)

import numpy as np
import torch
from inference_models import AutoModel
import inference_models.models.rfdetr.pre_processing as trt_mod


RANDOM_SEED = 42


def bench(label, fn, n=200):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()

    samples_ms = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples_ms.append((time.perf_counter() - t0) * 1000)

    samples_fps = [1000 / sample_ms for sample_ms in samples_ms]
    mean_ms = float(np.mean(samples_ms))
    p50_ms, p95_ms, p99_ms = np.percentile(samples_ms, [50, 95, 99])
    mean_fps = float(np.mean(samples_fps))
    p50_fps, p95_fps, p99_fps = np.percentile(samples_fps, [50, 95, 99])
    print(
        f"{label:>18s} ms/frame: mean {mean_ms:7.3f}  "
        f"p50 {p50_ms:7.3f}  p95 {p95_ms:7.3f}  p99 {p99_ms:7.3f}"
    )
    print(
        f"{label:>18s} fps:      mean {mean_fps:7.1f}  "
        f"p50 {p50_fps:7.1f}  p95 {p95_fps:7.1f}  p99 {p99_fps:7.1f}"
    )


def main():
    rng = np.random.default_rng(RANDOM_SEED)
    env_flag = os.environ.get("USE_TRITON_FOR_PREPROCESSING", "<unset>")

    triton_calls = {"count": 0}
    original = trt_mod.triton_preprocess_rfdetr_stretch
    if original is not None:
        def counting(*a, **kw):
            triton_calls["count"] += 1
            return original(*a, **kw)
        trt_mod.triton_preprocess_rfdetr_stretch = counting

    print(f"USE_TRITON_FOR_PREPROCESSING={env_flag}")
    print(f"USE_TRITON_FOR_PREPROCESSING={trt_mod.USE_TRITON_FOR_PREPROCESSING}")

    m = AutoModel.from_pretrained("rfdetr-seg-nano")
    for src_h, src_w in [(312, 312), (720, 1280), (1080, 1920)]:
        img = rng.integers(0, 255, (src_h, src_w, 3), dtype=np.uint8)
        print(f"\nsrc {src_h}x{src_w} -> 312x312:")
        bench("pre_process", lambda img=img: m.pre_process(img))

    print(f"\n  Triton kernel calls: {triton_calls['count']} "
          f"(expected {'>0' if env_flag.lower() in ('true','1') else '0'})")


if __name__ == "__main__":
    main()
