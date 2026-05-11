"""Isolate pre_process() timing. Path selected by
INFERENCE_MODELS_RFDETR_TRITON_PREPROC_ENABLED. Reports the Triton
kernel call count so the reviewer can see which path ran.

Usage (run twice and compare):
  INFERENCE_MODELS_RFDETR_TRITON_PREPROC_ENABLED=true  python temp/preproc_microbench.py
  INFERENCE_MODELS_RFDETR_TRITON_PREPROC_ENABLED=false python temp/preproc_microbench.py
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


def bench(label, fn, n=200):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / n * 1000
    print(f"{label:>18s}: {dt:7.3f} ms/frame  ({1000/dt:6.1f} fps)")


def main():
    env_flag = os.environ.get("INFERENCE_MODELS_RFDETR_TRITON_PREPROC_ENABLED", "<unset>")

    triton_calls = {"count": 0}
    original = trt_mod.triton_preprocess_rfdetr_stretch
    if original is not None:
        def counting(*a, **kw):
            triton_calls["count"] += 1
            return original(*a, **kw)
        trt_mod.triton_preprocess_rfdetr_stretch = counting

    print(f"INFERENCE_MODELS_RFDETR_TRITON_PREPROC_ENABLED={env_flag}")
    print(f"_FAST_PATH_ENABLED={trt_mod._FAST_PATH_ENABLED}")

    m = AutoModel.from_pretrained("rfdetr-seg-nano")
    for src_h, src_w in [(312, 312), (720, 1280), (1080, 1920)]:
        img = np.random.randint(0, 255, (src_h, src_w, 3), dtype=np.uint8)
        print(f"\nsrc {src_h}x{src_w} -> 312x312:")
        bench("pre_process", lambda img=img: m.pre_process(img))

    print(f"\n  Triton kernel calls: {triton_calls['count']} "
          f"(expected {'>0' if env_flag.lower() in ('true','1') else '0'})")


if __name__ == "__main__":
    main()
