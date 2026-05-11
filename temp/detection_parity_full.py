"""Full coco/val2017 detection + mask parity: Triton fast path vs PIL.

Driven by INFERENCE_MODELS_RFDETR_TRITON_PREPROC_ENABLED. Because the env
var is read once at import time, we run two subprocesses (true, false),
dump per-image detections to pickle, then compare.

Usage:
  python temp/detection_parity_full.py           # driver: runs both passes
  python temp/detection_parity_full.py --mode run --env true  --out /tmp/full_on.pkl
  python temp/detection_parity_full.py --mode compare --on /tmp/full_on.pkl --off /tmp/full_off.pkl
"""
import argparse
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

COCO = Path("/home/ubuntu/inference/coco/val2017")
CONFIDENCE = 0.4
PY = sys.executable
SELF = Path(__file__).resolve()
OUT_ON = "/tmp/det_parity_full_on.pkl"
OUT_OFF = "/tmp/det_parity_full_off.pkl"


def do_run(out_path):
    os.environ.setdefault(
        "DISABLED_INFERENCE_MODELS_BACKENDS",
        "torch,torch-script,onnx,hugging-face,ultralytics,mediapipe,custom",
    )
    import cv2
    import numpy as np
    import torch

    from inference_models import AutoModel
    import inference_models.models.rfdetr.pre_processing as trt_mod

    triton_calls = {"count": 0}
    original = trt_mod.triton_preprocess_rfdetr_stretch
    if original is not None:
        def counting(*a, **kw):
            triton_calls["count"] += 1
            return original(*a, **kw)
        trt_mod.triton_preprocess_rfdetr_stretch = counting

    env_flag = os.environ.get("INFERENCE_MODELS_RFDETR_TRITON_PREPROC_ENABLED", "<unset>")
    print(f"[run] env={env_flag}  _FAST_PATH_ENABLED={trt_mod._FAST_PATH_ENABLED}")

    paths = sorted(COCO.glob("*.jpg"))
    model = AutoModel.from_pretrained("rfdetr-seg-nano")

    records = []
    t0 = time.perf_counter()
    for idx, p in enumerate(paths):
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
        if (idx + 1) % 500 == 0:
            print(f"  [{env_flag}] {idx+1}/{len(paths)}  ({time.perf_counter()-t0:.0f}s)", flush=True)

    with open(out_path, "wb") as f:
        pickle.dump({"records": records, "triton_calls": triton_calls["count"]}, f)
    print(f"[run] triton_kernel_calls={triton_calls['count']}  saved -> {out_path}")


def iou_box(a, b):
    x0 = max(a[0], b[0]); y0 = max(a[1], b[1])
    x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
    iw = max(0, x1 - x0); ih = max(0, y1 - y0)
    inter = iw * ih
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    u = area_a + area_b - inter
    return inter / u if u > 0 else 0.0


def do_compare(on_path, off_path):
    import numpy as np
    with open(on_path, "rb") as f:
        on = pickle.load(f)
    with open(off_path, "rb") as f:
        off = pickle.load(f)

    assert len(on["records"]) == len(off["records"])
    n_imgs = len(on["records"])

    tot_on = tot_off = matched = class_disagree = count_mm = pixel_identical = 0
    ious, dscores, mask_iou = [], [], []

    for r_on, r_off in zip(on["records"], off["records"]):
        assert r_on["path"] == r_off["path"]
        nf = 0 if r_on["xyxy"] is None else len(r_on["xyxy"])
        nr = 0 if r_off["xyxy"] is None else len(r_off["xyxy"])
        tot_on += nf; tot_off += nr
        if nf != nr:
            count_mm += 1
        if nf == 0 and nr == 0:
            continue
        bf = r_on["xyxy"] if nf else np.zeros((0, 4))
        br = r_off["xyxy"] if nr else np.zeros((0, 4))
        sf = r_on["conf"] if nf else np.zeros(0)
        sr = r_off["conf"] if nr else np.zeros(0)
        cf = r_on["cls"] if nf else np.zeros(0, dtype=int)
        cr = r_off["cls"] if nr else np.zeros(0, dtype=int)
        mf, mr_m = r_on["mask"], r_off["mask"]

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
    print(f"==== full coco/val2017 parity: env=true vs env=false ({n_imgs} images) ====")
    print(f"  triton calls (env=true)      : {on['triton_calls']}")
    print(f"  triton calls (env=false)     : {off['triton_calls']}")
    print(f"  dets env=true / env=false    : {tot_on} / {tot_off}")
    print(f"  matched (IoU>0.5)            : {matched} ({100*matched/max(1,tot_off):.2f}% of env=false)")
    print(f"  count-mismatch images        : {count_mm}")
    print(f"  class-id disagreements       : {class_disagree}")
    if ious:
        print(f"  mean box IoU                 : {np.mean(ious):.6f}")
    if dscores:
        print(f"  mean / max |Δscore|          : {np.mean(dscores):.3e} / {np.max(dscores):.3e}")
    if mask_iou:
        a = np.array(mask_iou)
        print(f"  mean / min mask IoU          : {a.mean():.6f} / {a.min():.6f}")
        print(f"  pixel-identical masks        : {pixel_identical}/{len(mask_iou)}")
    print()
    expected = n_imgs
    ok_on  = "[PASS]" if on["triton_calls"]  == expected else "[FAIL]"
    ok_off = "[PASS]" if off["triton_calls"] == 0 else "[FAIL]"
    print(f"  {ok_on} env=true  -> Triton fired {on['triton_calls']}/{expected}")
    print(f"  {ok_off} env=false -> Triton fired {off['triton_calls']} (expected 0)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("driver", "run", "compare"), default="driver")
    ap.add_argument("--out")
    ap.add_argument("--on")
    ap.add_argument("--off")
    args = ap.parse_args()

    if args.mode == "run":
        do_run(args.out)
        return
    if args.mode == "compare":
        do_compare(args.on, args.off)
        return

    for env_value, out in (("true", OUT_ON), ("false", OUT_OFF)):
        env = os.environ.copy()
        env["INFERENCE_MODELS_RFDETR_TRITON_PREPROC_ENABLED"] = env_value
        print(f"\n---- child: env={env_value} out={out} ----", flush=True)
        subprocess.run([PY, str(SELF), "--mode", "run", "--out", out], check=True, env=env)

    do_compare(OUT_ON, OUT_OFF)


if __name__ == "__main__":
    main()
