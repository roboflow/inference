"""Full coco/val2017 detection + mask parity: fused-postproc Triton path vs reference.

Driven by RFDETR_TRITON_POSTPROC (true/false). The env var is read once at
import time, so we run two subprocesses (postproc on, postproc off), stream
per-image detections to a pickle file (one pickle.dump per record), then
compare in lockstep.

Usage:
  python temp/detection_parity_full.py           # driver: runs both passes
  python temp/detection_parity_full.py --mode run --out /tmp/full_on.pkl
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
# Cap images per pass: postproc caches a per-(orig_h, orig_w) GPU buffer in
# _get_mask_bin_buffer, so the working set grows with distinct image sizes and
# OOMs near ~5k images on a 14 GiB card. 1500 covers a wide variety of shapes
# while staying well under that ceiling.
MAX_IMAGES = int(os.environ.get("PARITY_MAX_IMAGES", "1500"))


def _iter_records(path):
    with open(path, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                return


def do_run(out_path):
    os.environ.setdefault(
        "DISABLED_INFERENCE_MODELS_BACKENDS",
        "torch,torch-script,onnx,hugging-face,ultralytics,mediapipe,custom",
    )
    import cv2
    import numpy as np
    import torch

    from inference_models import AutoModel
    import inference_models.models.rfdetr.common as common_mod

    postproc_calls = {"count": 0}
    original_fp = getattr(common_mod, "rfdetr_triton_postproc", None)
    if original_fp is not None:
        def counting_fp(*a, **kw):
            postproc_calls["count"] += 1
            return original_fp(*a, **kw)
        common_mod.rfdetr_triton_postproc = counting_fp

    fp_flag = os.environ.get("RFDETR_TRITON_POSTPROC", "<unset>")
    print(
        f"[run] RFDETR_TRITON_POSTPROC={fp_flag} "
        f"(module ready={getattr(common_mod, '_TRITON_POSTPROC_READY', False)})"
    )

    paths = sorted(COCO.glob("*.jpg"))[:MAX_IMAGES]
    model = AutoModel.from_pretrained("rfdetr-seg-nano")

    n_records = 0
    t0 = time.perf_counter()
    with open(out_path, "wb") as f:
        # placeholder header — rewritten at end
        pickle.dump({"_kind": "header", "postproc_calls": -1, "n_records": -1}, f)
        for idx, p in enumerate(paths):
            im = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if im is None:
                continue
            pre, meta = model.pre_process(im)
            out = model.forward(pre)
            det = model.post_process(out, meta, confidence=CONFIDENCE)[0]
            n = int(det.class_id.numel())
            if n and det.mask is not None:
                m_np = det.mask.cpu().to(torch.bool).numpy()
                packed = np.packbits(m_np.reshape(n, -1), axis=1)
                mask_shape = m_np.shape[1:]
            else:
                packed = None
                mask_shape = None
            rec = {
                "_kind": "rec",
                "path": str(p),
                "xyxy": det.xyxy.cpu().numpy() if n else None,
                "conf": det.confidence.cpu().numpy() if n else None,
                "cls":  det.class_id.cpu().numpy() if n else None,
                "mask_packed": packed,
                "mask_shape": mask_shape,
            }
            pickle.dump(rec, f)
            n_records += 1
            if (idx + 1) % 500 == 0:
                print(f"  [fp={fp_flag}] {idx+1}/{len(paths)}  ({time.perf_counter()-t0:.0f}s)", flush=True)

    # append a footer with the totals (header is ignored on read; iterator just walks records + footer)
    with open(out_path, "ab") as f:
        pickle.dump({"_kind": "footer", "postproc_calls": postproc_calls["count"], "n_records": n_records}, f)
    print(f"[run] postproc_kernel_calls={postproc_calls['count']}  records={n_records}  saved -> {out_path}")


def iou_box(a, b):
    x0 = max(a[0], b[0]); y0 = max(a[1], b[1])
    x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
    iw = max(0, x1 - x0); ih = max(0, y1 - y0)
    inter = iw * ih
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    u = area_a + area_b - inter
    return inter / u if u > 0 else 0.0


def _unpack_masks(rec):
    import numpy as np
    if rec["mask_packed"] is None:
        return None
    n = len(rec["mask_packed"])
    h, w = rec["mask_shape"]
    flat = np.unpackbits(rec["mask_packed"], axis=1, count=h * w)
    return flat.reshape(n, h, w).astype(bool)


def do_compare(on_path, off_path):
    import numpy as np

    tot_on = tot_off = matched = class_disagree = count_mm = pixel_identical = 0
    ious, dscores, mask_iou = [], [], []
    on_fp_calls = off_fp_calls = -1
    n_imgs = 0

    on_iter = _iter_records(on_path)
    off_iter = _iter_records(off_path)

    for r_on, r_off in zip(on_iter, off_iter):
        if r_on.get("_kind") == "header":
            r_on = next(on_iter)
        if r_off.get("_kind") == "header":
            r_off = next(off_iter)
        if r_on.get("_kind") == "footer" or r_off.get("_kind") == "footer":
            on_fp_calls = r_on.get("postproc_calls", on_fp_calls)
            off_fp_calls = r_off.get("postproc_calls", off_fp_calls)
            break

        assert r_on["path"] == r_off["path"], (r_on["path"], r_off["path"])
        n_imgs += 1
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
        mf = _unpack_masks(r_on) if nf else None
        mr_m = _unpack_masks(r_off) if nr else None

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

    # drain footers if not already pulled
    for it, current_calls_attr in ((on_iter, "on_fp_calls"), (off_iter, "off_fp_calls")):
        for r in it:
            if r.get("_kind") == "footer":
                if current_calls_attr == "on_fp_calls":
                    on_fp_calls = r["postproc_calls"]
                else:
                    off_fp_calls = r["postproc_calls"]

    print()
    print(f"==== full coco/val2017 parity: postproc=true vs postproc=false ({n_imgs} images) ====")
    print(f"  postproc calls (fp=true)     : {on_fp_calls}")
    print(f"  postproc calls (fp=false)    : {off_fp_calls}")
    print(f"  dets fp=true / fp=false      : {tot_on} / {tot_off}")
    print(f"  matched (IoU>0.5)            : {matched} ({100*matched/max(1,tot_off):.2f}% of fp=false)")
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
    ok_on  = "[PASS]" if on_fp_calls  == expected else "[FAIL]"
    ok_off = "[PASS]" if off_fp_calls == 0 else "[FAIL]"
    print(f"  {ok_on} fp=true  -> postproc fired {on_fp_calls}/{expected}")
    print(f"  {ok_off} fp=false -> postproc fired {off_fp_calls} (expected 0)")


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

    for fp_value, out in (("true", OUT_ON), ("false", OUT_OFF)):
        env = os.environ.copy()
        env["RFDETR_TRITON_POSTPROC"] = fp_value
        print(f"\n---- child: postproc={fp_value} out={out} ----", flush=True)
        subprocess.run([PY, str(SELF), "--mode", "run", "--out", out], check=True, env=env)

    do_compare(OUT_ON, OUT_OFF)


if __name__ == "__main__":
    main()
