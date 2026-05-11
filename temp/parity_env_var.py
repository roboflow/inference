"""Parity sanity + kill-switch verification on perf/rfdetr-seg-triton-preproc.

Two modes:
  --mode run   : run N_IMAGES through the model in *this* process (env var
                 controls fast path); save predictions + Triton kernel call
                 count to --out.
  --mode compare : load two prediction files and report per-detection parity.

Driver: first invokes mode=run twice (env=true, env=false) via subprocess so
each import sees a fresh env-var read, then invokes mode=compare.
"""
import argparse
import os
import pickle
import random
import subprocess
import sys
from pathlib import Path


COCO = Path("/home/ubuntu/inference/coco/val2017")
N_IMAGES = 100
CONFIDENCE = 0.4
SEED = 0
PY = "/home/ubuntu/inference/.venv/bin/python"
SELF = Path(__file__).resolve()
OUT_ON = "/tmp/parity_env_on.pkl"
OUT_OFF = "/tmp/parity_env_off.pkl"


def do_run(out_path: str):
    os.environ.setdefault(
        "DISABLED_INFERENCE_MODELS_BACKENDS",
        "torch,torch-script,onnx,hugging-face,ultralytics,mediapipe,custom",
    )
    import cv2
    import torch
    from inference_models import AutoModel
    import inference_models.models.rfdetr.pre_processing as trt_mod

    triton_calls = {"count": 0}
    original = trt_mod.triton_preprocess_rfdetr_stretch
    if original is not None:
        def counting(*args, **kwargs):
            triton_calls["count"] += 1
            return original(*args, **kwargs)
        trt_mod.triton_preprocess_rfdetr_stretch = counting

    random.seed(SEED)
    all_paths = sorted(COCO.glob("*.jpg"))
    paths = random.sample(all_paths, N_IMAGES)

    model = AutoModel.from_pretrained("rfdetr-seg-nano")
    print(f"[child] env={os.environ.get('INFERENCE_MODELS_RFDETR_TRITON_PREPROC_ENABLED','<unset>')}  "
          f"_FAST_PATH_ENABLED={trt_mod._FAST_PATH_ENABLED}  _TRITON_AVAILABLE={trt_mod._TRITON_AVAILABLE}",
          flush=True)

    records = []
    for p in paths:
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im is None:
            continue
        pre, meta = model.pre_process(im)
        out = model.forward(pre)
        det = model.post_process(out, meta, confidence=CONFIDENCE)[0]
        n = int(det.class_id.numel())
        records.append({
            "path": str(p),
            "xyxy": det.xyxy.cpu().numpy() if n else None,
            "conf": det.confidence.cpu().numpy() if n else None,
            "cls":  det.class_id.cpu().numpy() if n else None,
            "mask": det.mask.cpu().to(torch.bool).numpy() if (n and det.mask is not None) else None,
        })

    with open(out_path, "wb") as f:
        pickle.dump({"records": records, "triton_calls": triton_calls["count"]}, f)
    print(f"[child] triton_kernel_calls = {triton_calls['count']}  saved -> {out_path}", flush=True)


def iou_box(a, b):
    import numpy as np
    x0 = max(a[0], b[0]); y0 = max(a[1], b[1])
    x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
    iw = max(0, x1 - x0); ih = max(0, y1 - y0)
    inter = iw * ih
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    u = area_a + area_b - inter
    return inter / u if u > 0 else 0.0


def do_compare(on_path: str, off_path: str):
    import numpy as np
    with open(on_path, "rb") as f:
        on = pickle.load(f)
    with open(off_path, "rb") as f:
        off = pickle.load(f)

    assert len(on["records"]) == len(off["records"])
    n_imgs = len(on["records"])

    tot_on = tot_off = matched = 0
    ious = []
    dscores = []
    mask_iou = []
    pixel_identical = 0
    count_mm_images = 0
    class_disagree = 0

    for r_on, r_off in zip(on["records"], off["records"]):
        assert r_on["path"] == r_off["path"]
        nf = 0 if r_on["xyxy"] is None else len(r_on["xyxy"])
        nr = 0 if r_off["xyxy"] is None else len(r_off["xyxy"])
        tot_on += nf; tot_off += nr
        if nf != nr:
            count_mm_images += 1
        if nf == 0 and nr == 0:
            continue
        bf = r_on["xyxy"] if nf else np.zeros((0, 4))
        br = r_off["xyxy"] if nr else np.zeros((0, 4))
        sf = r_on["conf"] if nf else np.zeros(0)
        sr = r_off["conf"] if nr else np.zeros(0)
        cf = r_on["cls"] if nf else np.zeros(0, dtype=int)
        cr = r_off["cls"] if nr else np.zeros(0, dtype=int)
        mf = r_on["mask"]; mr_m = r_off["mask"]

        used = set()
        for j in range(nr):
            best_i, best_iou = -1, 0.5
            for i in range(nf):
                if i in used:
                    continue
                iou = iou_box(bf[i], br[j])
                if iou > best_iou:
                    best_iou, best_i = iou, i
            if best_i >= 0:
                used.add(best_i)
                matched += 1
                ious.append(best_iou)
                dscores.append(abs(float(sf[best_i]) - float(sr[j])))
                if int(cf[best_i]) != int(cr[j]):
                    class_disagree += 1
                if mf is not None and mr_m is not None:
                    a = mf[best_i]; b = mr_m[j]
                    inter = np.logical_and(a, b).sum()
                    u = np.logical_or(a, b).sum()
                    mask_iou.append(float(inter) / float(u) if u else 0.0)
                    if np.array_equal(a, b):
                        pixel_identical += 1

    print()
    print(f"==== parity: env=true vs env=false  ({n_imgs} images) ====")
    print(f"  triton calls (env=true)       : {on['triton_calls']}")
    print(f"  triton calls (env=false)      : {off['triton_calls']}")
    print(f"  dets env=true                 : {tot_on}")
    print(f"  dets env=false                : {tot_off}")
    print(f"  matched (IoU>0.5)             : {matched} "
          f"({100*matched/max(1,tot_off):.2f}% of env=false)")
    print(f"  count-mismatch images         : {count_mm_images}")
    print(f"  class-id disagreements        : {class_disagree}")
    if ious:
        print(f"  mean box IoU                  : {np.mean(ious):.6f}")
    if dscores:
        print(f"  mean / max |Δscore|           : {np.mean(dscores):.3e} / {np.max(dscores):.3e}")
    if mask_iou:
        a = np.array(mask_iou)
        print(f"  mean / min mask IoU           : {a.mean():.6f} / {a.min():.6f}")
        print(f"  pixel-identical masks         : {pixel_identical}/{len(mask_iou)}")

    print()
    expected_calls_on = n_imgs  # 1 kernel launch per image (single-item batch)
    if on["triton_calls"] == expected_calls_on:
        print(f"  [PASS] env=true  -> Triton kernel fired {on['triton_calls']}/{expected_calls_on} times")
    else:
        print(f"  [FAIL] env=true  -> Triton kernel fired {on['triton_calls']} times "
              f"(expected {expected_calls_on})")
    if off["triton_calls"] == 0:
        print(f"  [PASS] env=false -> Triton kernel never fired")
    else:
        print(f"  [FAIL] env=false -> Triton kernel fired {off['triton_calls']} times (expected 0)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("driver", "run", "compare"), default="driver")
    ap.add_argument("--out")
    args = ap.parse_args()

    if args.mode == "run":
        do_run(args.out)
        return
    if args.mode == "compare":
        do_compare(OUT_ON, OUT_OFF)
        return

    for env_value, out in (("true", OUT_ON), ("false", OUT_OFF)):
        env = os.environ.copy()
        env["INFERENCE_MODELS_RFDETR_TRITON_PREPROC_ENABLED"] = env_value
        print(f"\n---- running child: env={env_value} out={out} ----", flush=True)
        subprocess.run(
            [PY, str(SELF), "--mode", "run", "--out", out],
            check=True,
            cwd="/tmp",
            env=env,
        )
    print("\n---- comparing ----", flush=True)
    do_compare(OUT_ON, OUT_OFF)


if __name__ == "__main__":
    main()
