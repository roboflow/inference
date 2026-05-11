"""Compare Triton PIL-antialias-bilinear kernel against the production PIL path
on a handful of COCO val2017 images at 3 target sizes.

Prod path reference: inference_models/models/rfdetr/pre_processing.py:_pre_process_numpy
(BGR uint8 -> RGB -> PIL.Image.fromarray -> TF.resize antialias -> TF.to_tensor -> TF.normalize).

Triton path: build_resample_tables -> triton_preprocess_rfdetr_stretch with swap_rb=True.
"""
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from inference_models.models.rfdetr.triton_preprocess import (
    build_resample_tables,
    triton_preprocess_rfdetr_stretch,
)

COCO_DIR = Path("/home/ubuntu/inference/coco/val2017")
MEANS = (0.485, 0.456, 0.406)
STDS = (0.229, 0.224, 0.225)
TARGETS = [(312, 312), (640, 640), (512, 384)]
N_IMAGES = 20
DEVICE = torch.device("cuda:0")


def pil_reference(bgr_uint8: np.ndarray, th: int, tw: int) -> torch.Tensor:
    rgb = bgr_uint8[:, :, ::-1]
    pil = Image.fromarray(np.ascontiguousarray(rgb))
    resized = TF.resize(pil, (th, tw), antialias=True)
    tensor = TF.to_tensor(resized)
    tensor = TF.normalize(tensor, mean=list(MEANS), std=list(STDS))
    return tensor


def main() -> None:
    paths = sorted(COCO_DIR.glob("*.jpg"))[:N_IMAGES]
    assert paths, f"no images in {COCO_DIR}"

    print(f"{'target':>11}  {'n':>3}  {'max|Δ|':>10}  {'mean|Δ|':>10}  {'≈|Δ|<1e-3':>11}")
    for th, tw in TARGETS:
        max_abs = 0.0
        sum_abs = 0.0
        count = 0
        near = 0
        tables = None
        last_src_hw = None
        for p in paths:
            bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            # Reference
            ref = pil_reference(bgr, th, tw).unsqueeze(0).to(DEVICE)
            # Triton
            src_gpu = torch.from_numpy(bgr).to(DEVICE)
            if tables is None or last_src_hw != (bgr.shape[0], bgr.shape[1]):
                tables = build_resample_tables(
                    src_h=bgr.shape[0],
                    src_w=bgr.shape[1],
                    target_h=th,
                    target_w=tw,
                    device=DEVICE,
                )
                last_src_hw = (bgr.shape[0], bgr.shape[1])
            out = triton_preprocess_rfdetr_stretch(
                src=src_gpu,
                tables=tables,
                target_h=th,
                target_w=tw,
                means=MEANS,
                stds=STDS,
                swap_rb=True,
            )
            torch.cuda.synchronize()
            diff = (out - ref).abs()
            cur_max = float(diff.max().item())
            max_abs = max(max_abs, cur_max)
            sum_abs += float(diff.sum().item())
            count += diff.numel()
            if cur_max < 1e-3:
                near += 1
        mean_abs = sum_abs / count
        print(
            f"  {th:>4}x{tw:<4}  {len(paths):>3}  {max_abs:>10.3e}  "
            f"{mean_abs:>10.3e}  {near:>5}/{len(paths):<4}"
        )


if __name__ == "__main__":
    main()
