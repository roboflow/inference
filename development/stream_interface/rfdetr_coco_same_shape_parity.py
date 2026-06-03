"""Compare RF-DETR instance-segmentation outputs on same-shape COCO images.

This harness is used to reproduce the correctness table in the RF-DETR Triton
postprocess PR. It runs a baseline git ref with all RF-DETR fast paths disabled
and a candidate ref with only Triton RLE postprocess enabled, then compares
detection counts, classes, boxes, scores, and RLE masks.

Example:

    env PARITY_MODEL_PATH=/path/to/rfdetr-seg-nano-orin-trt-package \
      python development/stream_interface/rfdetr_coco_same_shape_parity.py \
        --base-ref main \
        --candidate-ref opt-python-postproc \
        --height 480 \
        --width 640 \
        --image-count 1000
"""

import argparse
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
from collections import deque
from pathlib import Path
from typing import Deque, Optional

import cv2
import numpy as np

SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[2]
SELF = Path(__file__).resolve()
PY = sys.executable
MODEL_ID = "rfdetr-seg-nano"
CONFIDENCE = 0.4
DEFAULT_BASE_OUT = "/tmp/rfdetr_coco_same_shape_base.pkl"
DEFAULT_CANDIDATE_OUT = "/tmp/rfdetr_coco_same_shape_candidate.pkl"

BASE_FLAGS_OFF = {
    "INFERENCE_MODELS_RFDETR_TRITON_POSTPROC_ENABLED": "false",
    "INFERENCE_MODELS_RFDETR_TRITON_PREPROC_ENABLED": "false",
    "RFDETR_PIPELINE_DEPTH": "1",
    "ENABLE_AUTO_CUDA_GRAPHS_FOR_TRT_BACKEND": "false",
    "RFDETR_NSIGHT_MARKERS": "false",
}
CANDIDATE_FLAGS_ON = {
    "INFERENCE_MODELS_RFDETR_TRITON_POSTPROC_ENABLED": "true",
    "INFERENCE_MODELS_RFDETR_TRITON_PREPROC_ENABLED": "false",
    "RFDETR_PIPELINE_DEPTH": "1",
    "ENABLE_AUTO_CUDA_GRAPHS_FOR_TRT_BACKEND": "false",
    "RFDETR_NSIGHT_MARKERS": "false",
}
TRT_PACKAGE_REQUIRED_FILES = (
    "model_config.json",
    "class_names.txt",
    "inference_config.json",
    "engine.plan",
)


def _repo_import_roots(repo_root: Path) -> list[Path]:
    return [repo_root, repo_root / "inference_models"]


def _child_pythonpath(repo_root: Path, existing_pythonpath: Optional[str]) -> str:
    entries = [str(path) for path in _repo_import_roots(repo_root) if path.exists()]
    if existing_pythonpath:
        entries.append(existing_pythonpath)
    return os.pathsep.join(entries)


def _prioritize_local_packages(repo_root: Path) -> None:
    for search_root in reversed(_repo_import_roots(repo_root)):
        search_root_str = str(search_root)
        if search_root_str in sys.path:
            sys.path.remove(search_root_str)
        if search_root.exists():
            sys.path.insert(0, search_root_str)
    for module_name in list(sys.modules):
        if module_name == "inference" or module_name.startswith("inference."):
            sys.modules.pop(module_name, None)
        if module_name == "inference_models" or module_name.startswith(
            "inference_models."
        ):
            sys.modules.pop(module_name, None)


def _bootstrap_repo_root(repo_root: str) -> Path:
    repo_path = Path(repo_root).resolve()
    os.chdir(repo_path)
    _prioritize_local_packages(repo_path)
    return repo_path


def _git_output(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=str(repo_root),
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


def _safe_git_output(repo_root: Path, *args: str, default: str = "<unknown>") -> str:
    try:
        return _git_output(repo_root, *args)
    except subprocess.CalledProcessError:
        return default


def _remove_worktree(worktree_root: Path) -> None:
    subprocess.run(
        ["git", "worktree", "remove", "--force", str(worktree_root)],
        cwd=str(SCRIPT_REPO_ROOT),
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    shutil.rmtree(worktree_root, ignore_errors=True)


def _materialize_target(ref: str) -> dict:
    if ref.lower() in {"working-tree", "worktree", "current"}:
        return {
            "label": (
                f"{_safe_git_output(SCRIPT_REPO_ROOT, 'rev-parse', '--abbrev-ref', 'HEAD')} "
                "(working-tree)"
            ),
            "repo_root": SCRIPT_REPO_ROOT,
            "cleanup": None,
        }
    worktree_root = Path(tempfile.mkdtemp(prefix="rfdetr-coco-parity-"))
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(worktree_root), ref],
        cwd=str(SCRIPT_REPO_ROOT),
        check=True,
    )
    return {
        "label": ref,
        "repo_root": worktree_root,
        "cleanup": lambda: _remove_worktree(worktree_root),
    }


def _is_trt_package(package_dir: Path) -> bool:
    return package_dir.is_dir() and all(
        (package_dir / filename).exists() for filename in TRT_PACKAGE_REQUIRED_FILES
    )


def _resolve_model_reference() -> str:
    explicit_model_path = os.environ.get("PARITY_MODEL_PATH")
    if explicit_model_path and _is_trt_package(Path(explicit_model_path)):
        return str(Path(explicit_model_path).resolve())
    for root in (SCRIPT_REPO_ROOT, Path.cwd(), Path(tempfile.gettempdir())):
        for name in (
            "rfdetr-seg-nano-orin-trt-package",
            "rfdetr-seg-nano-trt-package",
        ):
            package = root / name
            if _is_trt_package(package):
                return str(package.resolve())
    return MODEL_ID


def _select_same_shape_images(
    coco_dir: Path, shape: tuple[int, int], limit: int
) -> list[str]:
    target_h, target_w = shape
    selected = []
    for path in sorted(coco_dir.glob("*.jpg")):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        h, w = image.shape[:2]
        if (h, w) == (target_h, target_w):
            selected.append(str(path.resolve()))
            if len(selected) >= limit:
                break
    if len(selected) < limit:
        raise RuntimeError(
            f"Found only {len(selected)} images with shape {(target_h, target_w)}"
        )
    return selected


def _normalized_rle(rle: dict) -> dict:
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return {"size": list(rle["size"]), "counts": counts}


def _rle_for_coco_iou(rle: dict) -> dict:
    counts = rle["counts"]
    if isinstance(counts, str):
        counts = counts.encode("ascii")
    return {"size": list(rle["size"]), "counts": counts}


def _rles_equal(left: dict, right: dict) -> bool:
    left_norm = _rle_for_coco_iou(left)
    right_norm = _rle_for_coco_iou(right)
    return (
        left_norm["size"] == right_norm["size"]
        and left_norm["counts"] == right_norm["counts"]
    )


def _rle_iou(left: dict, right: dict) -> float:
    from pycocotools import mask as mask_utils

    return float(
        mask_utils.iou([_rle_for_coco_iou(left)], [_rle_for_coco_iou(right)], [False])[
            0, 0
        ]
    )


def _record_from_response(image_path: str, response) -> dict:
    predictions = response.predictions
    if not predictions:
        return {
            "_kind": "rec",
            "image_path": image_path,
            "xyxy": None,
            "conf": None,
            "cls": None,
            "rles": None,
        }

    xyxy = np.empty((len(predictions), 4), dtype=np.float32)
    conf = np.empty((len(predictions),), dtype=np.float32)
    cls = np.empty((len(predictions),), dtype=np.int32)
    rles = []
    for idx, pred in enumerate(predictions):
        x1 = float(pred.x) - float(pred.width) / 2.0
        y1 = float(pred.y) - float(pred.height) / 2.0
        x2 = float(pred.x) + float(pred.width) / 2.0
        y2 = float(pred.y) + float(pred.height) / 2.0
        xyxy[idx] = (x1, y1, x2, y2)
        conf[idx] = float(pred.confidence)
        cls[idx] = int(pred.class_id)
        rle = getattr(pred, "rle", None)
        if rle is None:
            raise ValueError("Expected RLE predictions; got polygon response.")
        rles.append(_normalized_rle(rle))
    return {
        "_kind": "rec",
        "image_path": image_path,
        "xyxy": xyxy,
        "conf": conf,
        "cls": cls,
        "rles": rles,
    }


def _run_warmup(model, frame, warmup_frames: int) -> None:
    for _ in range(warmup_frames):
        preprocessed, metadata = model.preprocess(
            frame,
            confidence=CONFIDENCE,
            response_mask_format="rle",
        )
        prediction_handle = model.predict(
            preprocessed,
            confidence=CONFIDENCE,
            response_mask_format="rle",
        )
        model.postprocess(
            prediction_handle,
            metadata,
            confidence=CONFIDENCE,
            response_mask_format="rle",
        )
    if hasattr(model, "flush"):
        model.flush()


def do_run(
    out_path: str,
    repo_root: str,
    label: str,
    image_list_path: str,
    warmup_frames: int,
) -> None:
    repo_path = _bootstrap_repo_root(repo_root)
    os.environ.setdefault(
        "DISABLED_INFERENCE_MODELS_BACKENDS",
        "torch,torch-script,onnx,hugging-face,ultralytics,custom",
    )
    os.environ.setdefault(
        "ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES", "true"
    )

    import torch
    from inference.core.models.inference_models_adapters import (
        InferenceModelsInstanceSegmentationAdapter,
    )

    with open(image_list_path, "rb") as f:
        image_paths = pickle.load(f)

    model_reference = _resolve_model_reference()
    model = InferenceModelsInstanceSegmentationAdapter(model_reference)
    pipeline_depth = int(getattr(model, "_pipeline_depth", 1))
    response_delay = max(0, int(getattr(model, "_response_delay", 0)))
    signature = {
        "git_head": _safe_git_output(repo_path, "rev-parse", "--short", "HEAD"),
        "git_describe": _safe_git_output(
            repo_path, "describe", "--always", "--dirty", "--broken"
        ),
    }

    first_frame = cv2.imread(image_paths[0], cv2.IMREAD_COLOR)
    if first_frame is None:
        raise RuntimeError(f"Could not read image: {image_paths[0]}")

    print(
        "[run] "
        f"label={label} repo_root={repo_path} head={signature['git_head']} "
        f"model_reference={model_reference} pipeline_depth={pipeline_depth}",
        flush=True,
    )
    _run_warmup(model=model, frame=first_frame, warmup_frames=warmup_frames)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    header = {
        "_kind": "header",
        "label": label,
        "repo_root": str(repo_path),
        "model_reference": model_reference,
        "confidence": CONFIDENCE,
        "git_head": signature["git_head"],
        "git_describe": signature["git_describe"],
        "pipeline_depth": pipeline_depth,
        "response_delay": response_delay,
        "image_count": len(image_paths),
        "flags": {
            key: os.environ.get(key)
            for key in sorted({*BASE_FLAGS_OFF.keys(), *CANDIDATE_FLAGS_ON.keys()})
        },
    }

    pending: Deque[str] = deque()
    n_records = 0
    with open(out_path, "wb") as f:
        pickle.dump(header, f)
        for index, image_path in enumerate(image_paths):
            frame = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"Could not read image: {image_path}")
            preprocessed, metadata = model.preprocess(
                frame,
                confidence=CONFIDENCE,
                response_mask_format="rle",
            )
            prediction_handle = model.predict(
                preprocessed,
                confidence=CONFIDENCE,
                response_mask_format="rle",
            )
            responses = model.postprocess(
                prediction_handle,
                metadata,
                confidence=CONFIDENCE,
                response_mask_format="rle",
            )

            if pipeline_depth <= 1:
                if len(responses) != 1:
                    raise ValueError(f"image {index}: expected one response")
                pickle.dump(_record_from_response(image_path, responses[0]), f)
                n_records += 1
            else:
                pending.append(image_path)
                if len(pending) > response_delay:
                    if len(responses) != 1:
                        raise ValueError(f"image {index}: expected one response")
                    response_image_path = pending.popleft()
                    pickle.dump(
                        _record_from_response(response_image_path, responses[0]), f
                    )
                    n_records += 1
            if (index + 1) % 25 == 0:
                print(f"  [{label}] images={index + 1} records={n_records}", flush=True)

        flush_responses = model.flush() if hasattr(model, "flush") else []
        for response in flush_responses:
            if not pending:
                raise ValueError("flush returned response with no pending image")
            response_image_path = pending.popleft()
            pickle.dump(_record_from_response(response_image_path, response), f)
            n_records += 1
        if pending:
            raise ValueError(f"pending images left after flush: {len(pending)}")
        pickle.dump(
            {
                "_kind": "footer",
                "label": label,
                "n_records": n_records,
            },
            f,
        )
    print(f"[run] label={label} records={n_records} saved={out_path}", flush=True)


def _iter_pickles(path: str):
    with open(path, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                return


def _box_iou(left, right) -> float:
    x0 = max(left[0], right[0])
    y0 = max(left[1], right[1])
    x1 = min(left[2], right[2])
    y1 = min(left[3], right[3])
    iw = max(0.0, float(x1 - x0))
    ih = max(0.0, float(y1 - y0))
    inter = iw * ih
    left_area = max(0.0, float(left[2] - left[0])) * max(0.0, float(left[3] - left[1]))
    right_area = max(0.0, float(right[2] - right[0])) * max(
        0.0, float(right[3] - right[1])
    )
    union = left_area + right_area - inter
    return inter / union if union > 0 else 0.0


def do_compare(base_path: str, candidate_path: str) -> None:
    base_iter = _iter_pickles(base_path)
    candidate_iter = _iter_pickles(candidate_path)
    base_header = next(base_iter)
    candidate_header = next(candidate_iter)

    n_images = 0
    tot_base = 0
    tot_candidate = 0
    matched = 0
    count_mismatch = 0
    class_disagree = 0
    pixel_identical = 0
    box_ious = []
    score_deltas = []
    mask_ious = []
    first_mismatches = []
    base_footer = None
    candidate_footer = None

    for base_record, candidate_record in zip(base_iter, candidate_iter):
        if (
            base_record.get("_kind") == "footer"
            or candidate_record.get("_kind") == "footer"
        ):
            base_footer = base_record
            candidate_footer = candidate_record
            break
        if base_record["image_path"] != candidate_record["image_path"]:
            raise AssertionError(
                (base_record["image_path"], candidate_record["image_path"])
            )
        n_images += 1
        n_base = 0 if base_record["xyxy"] is None else len(base_record["xyxy"])
        n_candidate = (
            0 if candidate_record["xyxy"] is None else len(candidate_record["xyxy"])
        )
        tot_base += n_base
        tot_candidate += n_candidate
        if n_base != n_candidate:
            count_mismatch += 1
            if len(first_mismatches) < 10:
                first_mismatches.append(
                    (Path(base_record["image_path"]).name, n_base, n_candidate)
                )
        if n_base == 0 and n_candidate == 0:
            continue

        base_boxes = base_record["xyxy"] if n_base else np.zeros((0, 4), dtype=float)
        candidate_boxes = (
            candidate_record["xyxy"] if n_candidate else np.zeros((0, 4), dtype=float)
        )
        base_scores = base_record["conf"] if n_base else np.zeros(0, dtype=float)
        candidate_scores = (
            candidate_record["conf"] if n_candidate else np.zeros(0, dtype=float)
        )
        base_classes = base_record["cls"] if n_base else np.zeros(0, dtype=np.int32)
        candidate_classes = (
            candidate_record["cls"] if n_candidate else np.zeros(0, dtype=np.int32)
        )
        base_rles = base_record["rles"] or []
        candidate_rles = candidate_record["rles"] or []

        used = set()
        for candidate_idx in range(n_candidate):
            best_base_idx = -1
            best_iou = 0.5
            for base_idx in range(n_base):
                if base_idx in used:
                    continue
                if int(base_classes[base_idx]) != int(candidate_classes[candidate_idx]):
                    continue
                box_iou = _box_iou(base_boxes[base_idx], candidate_boxes[candidate_idx])
                if box_iou > best_iou:
                    best_iou = box_iou
                    best_base_idx = base_idx
            if best_base_idx < 0:
                continue

            used.add(best_base_idx)
            matched += 1
            box_ious.append(best_iou)
            score_deltas.append(
                abs(
                    float(base_scores[best_base_idx])
                    - float(candidate_scores[candidate_idx])
                )
            )
            if int(base_classes[best_base_idx]) != int(
                candidate_classes[candidate_idx]
            ):
                class_disagree += 1
            if base_rles and candidate_rles:
                base_rle = base_rles[best_base_idx]
                candidate_rle = candidate_rles[candidate_idx]
                mask_ious.append(_rle_iou(base_rle, candidate_rle))
                if _rles_equal(base_rle, candidate_rle):
                    pixel_identical += 1

    if base_footer is None:
        for obj in base_iter:
            if obj.get("_kind") == "footer":
                base_footer = obj
                break
    if candidate_footer is None:
        for obj in candidate_iter:
            if obj.get("_kind") == "footer":
                candidate_footer = obj
                break

    print()
    print(
        f"==== COCO same-shape parity: {base_header['label']} vs "
        f"{candidate_header['label']} ===="
    )
    print(f"  base repo                    : {base_header['git_describe']}")
    print(f"  candidate repo               : {candidate_header['git_describe']}")
    print(
        f"  pipeline depth (base/cand)   : "
        f"{base_header['pipeline_depth']} / {candidate_header['pipeline_depth']}"
    )
    print(
        f"  images base / candidate      : "
        f"{base_footer['n_records']} / {candidate_footer['n_records']}"
    )
    print(f"  detections base / candidate  : {tot_base} / {tot_candidate}")
    print(
        f"  matched same-class IoU>0.5   : {matched} "
        f"({100 * matched / max(1, tot_base):.2f}% of base)"
    )
    print(f"  count-mismatch images        : {count_mismatch}")
    if first_mismatches:
        print(f"  first count mismatches       : {first_mismatches}")
    print(f"  class-id disagreements       : {class_disagree}")
    if box_ious:
        print(
            f"  mean / min box IoU           : {np.mean(box_ious):.6f} / {np.min(box_ious):.6f}"
        )
    if score_deltas:
        print(
            f"  mean / max |delta score|     : "
            f"{np.mean(score_deltas):.3e} / {np.max(score_deltas):.3e}"
        )
    if mask_ious:
        mask_iou_array = np.array(mask_ious)
        print(
            f"  mean / min mask IoU          : "
            f"{mask_iou_array.mean():.6f} / {mask_iou_array.min():.6f}"
        )
        print(f"  pixel-identical masks        : {pixel_identical}/{len(mask_ious)}")


def _run_child(
    repo_root: Path,
    label: str,
    out_path: str,
    image_list_path: str,
    flags: dict,
    warmup_frames: int,
) -> None:
    env = os.environ.copy()
    env.update(flags)
    env["MPLCONFIGDIR"] = "/tmp/mpl"
    env["PARITY_MODEL_PATH"] = str(
        (SCRIPT_REPO_ROOT / "rfdetr-seg-nano-orin-trt-package").resolve()
    )
    env["PYTHONPATH"] = _child_pythonpath(repo_root, env.get("PYTHONPATH"))
    args = [
        PY,
        str(SELF),
        "--mode",
        "run",
        "--repo-root",
        str(repo_root),
        "--label",
        label,
        "--out",
        out_path,
        "--image-list",
        image_list_path,
        "--warmup-frames",
        str(warmup_frames),
    ]
    print(
        "\n---- child ----\n"
        f"  label={label}\n"
        f"  repo_root={repo_root}\n"
        f"  out={out_path}\n"
        f"  flags={flags}",
        flush=True,
    )
    subprocess.run(args, cwd=str(SCRIPT_REPO_ROOT), env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("driver", "run", "compare"), default="driver"
    )
    parser.add_argument("--repo-root")
    parser.add_argument("--label")
    parser.add_argument("--out")
    parser.add_argument("--image-list")
    parser.add_argument("--base", default=DEFAULT_BASE_OUT)
    parser.add_argument("--candidate", default=DEFAULT_CANDIDATE_OUT)
    parser.add_argument("--base-ref", default="main")
    parser.add_argument("--candidate-ref", default="working-tree")
    parser.add_argument("--coco-dir", default="coco/val2017")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--image-count", type=int, default=100)
    parser.add_argument("--warmup-frames", type=int, default=10)
    parser.add_argument("--keep-worktrees", action="store_true")
    args = parser.parse_args()

    if args.mode == "run":
        if not args.out or not args.image_list:
            raise ValueError("--out and --image-list are required in run mode")
        do_run(
            out_path=args.out,
            repo_root=args.repo_root or str(SCRIPT_REPO_ROOT),
            label=args.label or "run",
            image_list_path=args.image_list,
            warmup_frames=args.warmup_frames,
        )
        return
    if args.mode == "compare":
        do_compare(args.base, args.candidate)
        return

    coco_dir = (SCRIPT_REPO_ROOT / args.coco_dir).resolve()
    image_paths = _select_same_shape_images(
        coco_dir=coco_dir,
        shape=(args.height, args.width),
        limit=args.image_count,
    )
    image_list_path = Path(
        tempfile.mkstemp(prefix="rfdetr-coco-images-", suffix=".pkl")[1]
    )
    with open(image_list_path, "wb") as f:
        pickle.dump(image_paths, f)
    print(
        f"[driver] selected {len(image_paths)} images with shape "
        f"{(args.height, args.width)} from {coco_dir}",
        flush=True,
    )

    base_target = _materialize_target(args.base_ref)
    candidate_target = _materialize_target(args.candidate_ref)
    cleanup_callbacks = [
        target["cleanup"]
        for target in (base_target, candidate_target)
        if callable(target["cleanup"])
    ]
    try:
        _run_child(
            repo_root=Path(base_target["repo_root"]),
            label=f"{base_target['label']} flags-off",
            out_path=args.base,
            image_list_path=str(image_list_path),
            flags=BASE_FLAGS_OFF,
            warmup_frames=args.warmup_frames,
        )
        _run_child(
            repo_root=Path(candidate_target["repo_root"]),
            label=f"{candidate_target['label']} flags-on",
            out_path=args.candidate,
            image_list_path=str(image_list_path),
            flags=CANDIDATE_FLAGS_ON,
            warmup_frames=args.warmup_frames,
        )
    finally:
        image_list_path.unlink(missing_ok=True)
        if not args.keep_worktrees:
            for cleanup in reversed(cleanup_callbacks):
                cleanup()
    do_compare(args.base, args.candidate)


if __name__ == "__main__":
    main()
