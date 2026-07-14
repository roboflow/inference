"""Stage breakdown of the vectorized overlap_analysis bbox path, n100 x n100."""

import statistics
import time
from uuid import uuid4

import numpy as np
import supervision as sv
import torch

from inference.core.workflows.core_steps.fusion.overlap_analysis import v1_tensor as m

W, H = 1280, 720
N = 100
device = "cuda" if torch.cuda.is_available() else "cpu"


def make(seed):
    rng = np.random.default_rng(seed)
    centers = rng.uniform([100, 100], [W - 100, H - 100], size=(N, 2))
    sizes = rng.uniform(40, 160, size=(N, 2))
    xyxy = np.concatenate([centers - sizes / 2, centers + sizes / 2], axis=1).astype(
        np.float32
    )
    from inference_models.models.base.object_detection import Detections

    return Detections(
        xyxy=torch.as_tensor(xyxy, device=device),
        class_id=torch.zeros(N, dtype=torch.int64, device=device),
        confidence=torch.full((N,), 0.9, device=device),
        image_metadata={"class_names": {0: "object"}},
        bboxes_metadata=[{"detection_id": str(uuid4())} for _ in range(N)],
    )


ref, cand = make(1), make(2)


def bench(name, fn, iters=300):
    for _ in range(30):
        fn()
    torch.cuda.synchronize()
    s = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        s.append((time.perf_counter() - t0) * 1000)
    print(f"{name:28s} {statistics.median(s):8.4f} ms")


bench("xyxy_d2h_x2", lambda: (ref.xyxy.detach().to("cpu").numpy(), cand.xyxy.detach().to("cpu").numpy()))
ref_np = ref.xyxy.detach().to("cpu").numpy()
cand_np = cand.xyxy.detach().to("cpu").numpy()
bench("box_iou_batch", lambda: sv.box_iou_batch(ref_np, cand_np))
iou = sv.box_iou_batch(ref_np, cand_np)
bench("detection_ids_x2", lambda: (m._detection_ids(ref), m._detection_ids(cand)))
bench("class_names_x2", lambda: (m._class_names(ref), m._class_names(cand)))
bench("confidence_d2h_x2", lambda: (m._confidence_values(ref), m._confidence_values(cand)))

ref_ids, cand_ids = m._detection_ids(ref), m._detection_ids(cand)
ref_cls, cand_cls = m._class_names(ref), m._class_names(cand)
ref_conf, cand_conf = m._confidence_values(ref), m._confidence_values(cand)


def clip_math():
    rb = ref_np.astype(np.float64, copy=False)
    cb = cand_np.astype(np.float64, copy=False)
    rx1 = np.minimum(rb[:, 0], rb[:, 2]); ry1 = np.minimum(rb[:, 1], rb[:, 3])
    rx2 = np.maximum(rb[:, 0], rb[:, 2]); ry2 = np.maximum(rb[:, 1], rb[:, 3])
    cx1 = np.minimum(cb[:, 0], cb[:, 2]); cy1 = np.minimum(cb[:, 1], cb[:, 3])
    cx2 = np.maximum(cb[:, 0], cb[:, 2]); cy2 = np.maximum(cb[:, 1], cb[:, 3])
    iw = np.clip(np.minimum(rx2[:, None], cx2[None, :]) - np.maximum(rx1[:, None], cx1[None, :]), 0.0, None)
    ih = np.clip(np.minimum(ry2[:, None], cy2[None, :]) - np.maximum(ry1[:, None], cy1[None, :]), 0.0, None)
    inter = iw * ih
    ra = (rx2 - rx1) * (ry2 - ry1)
    ratios = np.zeros_like(inter)
    np.divide(inter, ra[:, None], out=ratios, where=ra[:, None] > 0.0)
    pairs = np.argwhere((iou > 0.0) & (ra[:, None] > 0.0) & (ratios >= 0.1))
    return ratios, pairs


bench("clip_math_argwhere", clip_math)
ratios, pairs = clip_math()
print(f"pairs: {len(pairs)}")


def build_records():
    out = []
    for i, j in pairs:
        out.append(
            m._build_record(
                i=int(i), j=int(j), overlap_ratio=ratios[i, j],
                ref_class_names=ref_cls, cand_class_names=cand_cls,
                ref_confidences=ref_conf, cand_confidences=cand_conf,
                ref_ids=ref_ids, cand_ids=cand_ids,
            )
        )
    return out


bench("build_records_376", build_records)
bench("full_block_run", lambda: m.OverlapAnalysisBlockV1().run(ref, cand, 0.1))
