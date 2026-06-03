"""Compare RF-DETR workflow outputs on a video across two git refs.

The driver mode runs the same one-block RF-DETR instance-segmentation workflow
twice: a baseline ref with RF-DETR fast paths disabled, and a candidate ref with
the full stack enabled. Each child run writes sink outputs keyed by video frame;
the final compare checks that both runs emitted the same frame ids and
semantically equivalent serialized workflow predictions.

Example:

    env PARITY_MODEL_PATH=/app/helloworld/inference/rfdetr-seg-nano-orin-trt-package \
      python development/stream_interface/rfdetr_workflow_video_parity.py \
        --video_reference vehicles_1080p.mp4 \
        --base-ref main \
        --candidate-ref opt-pipeline-integration
"""

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Optional

import numpy as np

SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[2]
SELF = Path(__file__).resolve()
PY = sys.executable
MODEL_ID = "rfdetr-seg-nano"
LOCAL_WORKFLOW_MODEL_ID = f"{MODEL_ID}/1"
CONFIDENCE = 0.4
DEFAULT_BASE_OUT = "/tmp/rfdetr_workflow_video_base.pkl"
DEFAULT_CANDIDATE_OUT = "/tmp/rfdetr_workflow_video_candidate.pkl"
TRT_PACKAGE_REQUIRED_FILES = (
    "model_config.json",
    "class_names.txt",
    "inference_config.json",
    "engine.plan",
)

BASE_FLAGS_OFF = {
    "INFERENCE_MODELS_RFDETR_TRITON_POSTPROC_ENABLED": "false",
    "INFERENCE_MODELS_RFDETR_TRITON_PREPROC_ENABLED": "false",
    "RFDETR_PIPELINE_DEPTH": "1",
    "ENABLE_AUTO_CUDA_GRAPHS_FOR_TRT_BACKEND": "false",
}
CANDIDATE_FLAGS_ON = {
    "INFERENCE_MODELS_RFDETR_TRITON_POSTPROC_ENABLED": "true",
    "INFERENCE_MODELS_RFDETR_TRITON_PREPROC_ENABLED": "true",
    "RFDETR_PIPELINE_DEPTH": "2",
    "ENABLE_AUTO_CUDA_GRAPHS_FOR_TRT_BACKEND": "true",
}
ALL_BACKENDS = {
    "torch",
    "torch-script",
    "onnx",
    "trt",
    "hugging-face",
    "ultralytics",
    "custom",
}
_SV_DETECTIONS_SERIALIZER = None
_VOLATILE_OUTPUT_KEYS = {"detection_id"}


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
    worktree_root = Path(tempfile.mkdtemp(prefix="rfdetr-video-parity-"))
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


def _resolve_model_package() -> Optional[Path]:
    explicit_model_path = os.environ.get("PARITY_MODEL_PATH")
    if explicit_model_path and _is_trt_package(Path(explicit_model_path)):
        return Path(explicit_model_path).resolve()
    for root in (SCRIPT_REPO_ROOT, Path.cwd(), Path(tempfile.gettempdir())):
        for name in (
            "rfdetr-seg-nano-orin-trt-package",
            "rfdetr-seg-nano-trt-package",
        ):
            package = root / name
            if _is_trt_package(package):
                return package.resolve()
    return None


def _prepare_local_workflow_model_bundle(repo_root: Path) -> str:
    package = _resolve_model_package()
    if package is None:
        return MODEL_ID
    os.environ.setdefault(
        "ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES", "true"
    )
    model_dir = repo_root / LOCAL_WORKFLOW_MODEL_ID
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    if not model_dir.exists():
        model_dir.symlink_to(package, target_is_directory=True)

    model_cache_dir = (
        Path(os.environ.get("MODEL_CACHE_DIR", "/tmp/cache")) / MODEL_ID / "1"
    )
    model_cache_dir.mkdir(parents=True, exist_ok=True)
    model_type_path = model_cache_dir / "model_type.json"
    model_metadata = {
        "project_task_type": "instance-segmentation",
        "model_type": "rfdetr-seg-nano",
    }
    model_type_path.write_text(json.dumps(model_metadata, indent=4))
    return LOCAL_WORKFLOW_MODEL_ID


def _load_local_inference(repo_root: Path):
    spec = importlib.util.spec_from_file_location(
        "inference",
        repo_root / "inference" / "__init__.py",
        submodule_search_locations=[str(repo_root / "inference")],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load local inference package from {repo_root}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["inference"] = module
    spec.loader.exec_module(module)
    return module


def _build_workflow(model_id: str, confidence: float) -> dict:
    return {
        "version": "1.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "steps": [
            {
                "type": "roboflow_core/roboflow_instance_segmentation_model@v3",
                "name": "segmentation",
                "images": "$inputs.image",
                "model_id": model_id,
                "confidence_mode": "custom",
                "custom_confidence": confidence,
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "predictions",
                "selector": "$steps.segmentation.predictions",
            },
        ],
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _jsonable(val)
            for key, val in sorted(value.items())
            if str(key) not in _VOLATILE_OUTPUT_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(element) for element in value]
    if all(hasattr(value, attr) for attr in ("xyxy", "confidence", "class_id")):
        if _SV_DETECTIONS_SERIALIZER is None:
            raise RuntimeError("sv.Detections serializer was not initialized")
        return _jsonable(_SV_DETECTIONS_SERIALIZER(value))
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("ascii")
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(by_alias=True, exclude_none=True))
    if hasattr(value, "dict"):
        return _jsonable(value.dict())
    return value


def _as_list(value: Any) -> list:
    return value if isinstance(value, list) else [value]


def do_run(
    out_path: str,
    repo_root: str,
    label: str,
    video_reference: str,
    backend: str,
    confidence: float,
) -> None:
    global _SV_DETECTIONS_SERIALIZER
    repo_path = _bootstrap_repo_root(repo_root)
    os.environ.setdefault(
        "ONNXRUNTIME_EXECUTION_PROVIDERS",
        "[TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider]",
    )
    os.environ["DISABLED_INFERENCE_MODELS_BACKENDS"] = ",".join(
        sorted(ALL_BACKENDS - {backend})
    )
    model_id = _prepare_local_workflow_model_bundle(repo_root=repo_path)
    inference_module = _load_local_inference(repo_root=repo_path)
    from inference.core.workflows.core_steps.common.serializers import (
        serialise_sv_detections,
    )

    _SV_DETECTIONS_SERIALIZER = serialise_sv_detections
    inference_pipeline = inference_module.InferencePipeline
    signature = {
        "git_head": _safe_git_output(repo_path, "rev-parse", "--short", "HEAD"),
        "git_describe": _safe_git_output(
            repo_path, "describe", "--always", "--dirty", "--broken"
        ),
    }
    header = {
        "_kind": "header",
        "label": label,
        "repo_root": str(repo_path),
        "model_id": model_id,
        "video_reference": video_reference,
        "confidence": confidence,
        "git_head": signature["git_head"],
        "git_describe": signature["git_describe"],
        "flags": {
            key: os.environ.get(key)
            for key in sorted({*BASE_FLAGS_OFF.keys(), *CANDIDATE_FLAGS_ON.keys()})
        },
    }
    errors = []
    records = 0
    with open(out_path, "wb") as f:
        pickle.dump(header, f)

        def sink(predictions, video_frames) -> None:
            nonlocal records
            prediction_list = _as_list(predictions)
            frame_list = _as_list(video_frames)
            if len(prediction_list) != len(frame_list):
                errors.append(
                    f"sink length mismatch: {len(prediction_list)} predictions "
                    f"for {len(frame_list)} frames"
                )
                return
            for prediction, video_frame in zip(prediction_list, frame_list):
                pickle.dump(
                    {
                        "_kind": "record",
                        "frame_id": int(video_frame.frame_id),
                        "source_id": video_frame.source_id,
                        "prediction": _jsonable(prediction),
                    },
                    f,
                )
                records += 1

        print(
            "[run] "
            f"label={label} repo_root={repo_path} head={signature['git_head']} "
            f"model_id={model_id} video={video_reference}",
            flush=True,
        )
        pipeline = inference_pipeline.init_with_workflow(
            video_reference=video_reference,
            workflow_specification=_build_workflow(model_id, confidence),
            on_prediction=sink,
            serialize_results=False,
        )
        pipeline.start()
        pipeline.join()
        if errors:
            raise RuntimeError("; ".join(errors))
        if records == 0:
            raise RuntimeError("video workflow emitted no prediction records")
        pickle.dump(
            {
                "_kind": "footer",
                "label": label,
                "n_records": records,
            },
            f,
        )
    print(f"[run] label={label} records={records} saved={out_path}", flush=True)


def _iter_pickles(path: str):
    with open(path, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                return


def _compare_values(
    base: Any, candidate: Any, path: str, atol: float, errors: list
) -> None:
    if isinstance(base, dict) and isinstance(candidate, dict):
        if set(base) != set(candidate):
            errors.append(f"{path}: key mismatch {sorted(base)} != {sorted(candidate)}")
            return
        for key in sorted(base):
            _compare_values(base[key], candidate[key], f"{path}.{key}", atol, errors)
        return
    if isinstance(base, list) and isinstance(candidate, list):
        if len(base) != len(candidate):
            errors.append(f"{path}: length mismatch {len(base)} != {len(candidate)}")
            return
        for index, (base_item, candidate_item) in enumerate(zip(base, candidate)):
            _compare_values(base_item, candidate_item, f"{path}[{index}]", atol, errors)
        return
    if isinstance(base, (int, float)) and isinstance(candidate, (int, float)):
        if not math.isclose(float(base), float(candidate), abs_tol=atol, rel_tol=0.0):
            errors.append(f"{path}: numeric mismatch {base} != {candidate}")
        return
    if base != candidate:
        errors.append(f"{path}: value mismatch {base!r} != {candidate!r}")


def _extract_detection_list(prediction: Any) -> Optional[list]:
    if not isinstance(prediction, dict) or "predictions" not in prediction:
        return None
    value = prediction["predictions"]
    if isinstance(value, dict) and isinstance(value.get("predictions"), list):
        return value["predictions"]
    if isinstance(value, list):
        return value
    return None


def _xyxy_from_detection(
    detection: dict,
) -> Optional[tuple[float, float, float, float]]:
    try:
        width = float(detection["width"])
        height = float(detection["height"])
        center_x = float(detection["x"])
        center_y = float(detection["y"])
    except (KeyError, TypeError, ValueError):
        return None
    half_width = width / 2.0
    half_height = height / 2.0
    return (
        center_x - half_width,
        center_y - half_height,
        center_x + half_width,
        center_y + half_height,
    )


def _box_iou(
    left: tuple[float, float, float, float], right: tuple[float, float, float, float]
) -> float:
    x0 = max(left[0], right[0])
    y0 = max(left[1], right[1])
    x1 = min(left[2], right[2])
    y1 = min(left[3], right[3])
    iw = max(0.0, x1 - x0)
    ih = max(0.0, y1 - y0)
    inter = iw * ih
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - inter
    return inter / union if union > 0 else 0.0


def _compare_detection_lists(
    base_detections: list,
    candidate_detections: list,
    frame_id: int,
    min_box_iou: float,
    max_score_delta: float,
    stats: dict,
    errors: list,
) -> None:
    stats["base_detections"] += len(base_detections)
    stats["candidate_detections"] += len(candidate_detections)
    if len(base_detections) != len(candidate_detections):
        stats["count_mismatch_frames"] += 1
        errors.append(
            f"frame {frame_id}: detection count mismatch "
            f"{len(base_detections)} != {len(candidate_detections)}"
        )
        return

    used_base_indices = set()
    for candidate_index, candidate_detection in enumerate(candidate_detections):
        candidate_class = candidate_detection.get("class_id")
        candidate_box = _xyxy_from_detection(candidate_detection)
        if candidate_box is None:
            errors.append(f"frame {frame_id}: candidate detection has no xyxy box")
            continue
        best_base_index = -1
        best_iou = min_box_iou
        for base_index, base_detection in enumerate(base_detections):
            if base_index in used_base_indices:
                continue
            if base_detection.get("class_id") != candidate_class:
                continue
            base_box = _xyxy_from_detection(base_detection)
            if base_box is None:
                continue
            box_iou = _box_iou(base_box, candidate_box)
            if box_iou > best_iou:
                best_iou = box_iou
                best_base_index = base_index
        if best_base_index < 0:
            stats["unmatched_candidate_detections"] += 1
            errors.append(
                f"frame {frame_id}: candidate detection {candidate_index} "
                f"class_id={candidate_class!r} has no matching base detection"
            )
            continue

        used_base_indices.add(best_base_index)
        stats["matched_detections"] += 1
        stats["box_ious"].append(best_iou)
        base_detection = base_detections[best_base_index]
        base_score = float(base_detection.get("confidence", 0.0))
        candidate_score = float(candidate_detection.get("confidence", 0.0))
        score_delta = abs(base_score - candidate_score)
        stats["score_deltas"].append(score_delta)
        if score_delta > max_score_delta:
            errors.append(
                f"frame {frame_id}: score delta {score_delta:.6f} exceeds "
                f"{max_score_delta:.6f}"
            )
        base_points = base_detection.get("points") or []
        candidate_points = candidate_detection.get("points") or []
        if len(base_points) != len(candidate_points):
            stats["polygon_point_count_mismatches"] += 1

    unmatched_base = len(base_detections) - len(used_base_indices)
    stats["unmatched_base_detections"] += unmatched_base
    if unmatched_base:
        errors.append(f"frame {frame_id}: {unmatched_base} base detections unmatched")


def do_compare(
    base_path: str,
    candidate_path: str,
    atol: float,
    min_box_iou: float,
    max_score_delta: float,
) -> None:
    base_iter = _iter_pickles(base_path)
    candidate_iter = _iter_pickles(candidate_path)
    base_header = next(base_iter)
    candidate_header = next(candidate_iter)
    compared = 0
    errors = []
    stats = {
        "base_detections": 0,
        "candidate_detections": 0,
        "matched_detections": 0,
        "count_mismatch_frames": 0,
        "unmatched_candidate_detections": 0,
        "unmatched_base_detections": 0,
        "polygon_point_count_mismatches": 0,
        "box_ious": [],
        "score_deltas": [],
    }
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
        if base_record["frame_id"] != candidate_record["frame_id"]:
            errors.append(
                f"frame id mismatch {base_record['frame_id']} != "
                f"{candidate_record['frame_id']}"
            )
            break
        compared += 1
        base_detections = _extract_detection_list(base_record["prediction"])
        candidate_detections = _extract_detection_list(candidate_record["prediction"])
        if base_detections is not None and candidate_detections is not None:
            _compare_detection_lists(
                base_detections=base_detections,
                candidate_detections=candidate_detections,
                frame_id=base_record["frame_id"],
                min_box_iou=min_box_iou,
                max_score_delta=max_score_delta,
                stats=stats,
                errors=errors,
            )
        else:
            _compare_values(
                base_record["prediction"],
                candidate_record["prediction"],
                f"frame[{base_record['frame_id']}]",
                atol,
                errors,
            )
        if len(errors) >= 20:
            break

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
        f"==== RF-DETR workflow video parity: {base_header['label']} vs "
        f"{candidate_header['label']} ===="
    )
    print(f"  base repo                  : {base_header['git_describe']}")
    print(f"  candidate repo             : {candidate_header['git_describe']}")
    print(
        f"  frames base / candidate    : "
        f"{base_footer['n_records']} / {candidate_footer['n_records']}"
    )
    print(f"  compared frames            : {compared}")
    print(
        f"  detections base / candidate: "
        f"{stats['base_detections']} / {stats['candidate_detections']}"
    )
    print(f"  matched detections         : {stats['matched_detections']}")
    print(f"  count-mismatch frames      : {stats['count_mismatch_frames']}")
    print(
        f"  unmatched base / candidate : "
        f"{stats['unmatched_base_detections']} / "
        f"{stats['unmatched_candidate_detections']}"
    )
    if stats["box_ious"]:
        print(
            f"  mean / min box IoU         : "
            f"{np.mean(stats['box_ious']):.6f} / {np.min(stats['box_ious']):.6f}"
        )
    if stats["score_deltas"]:
        print(
            f"  mean / max |delta score|   : "
            f"{np.mean(stats['score_deltas']):.3e} / "
            f"{np.max(stats['score_deltas']):.3e}"
        )
    print(
        f"  polygon point-count diffs  : " f"{stats['polygon_point_count_mismatches']}"
    )
    if errors:
        print("  first mismatches:")
        for error in errors[:20]:
            print(f"    - {error}")
        raise AssertionError(f"{len(errors)} parity mismatches found")
    print("  result                     : all detections matched semantic thresholds")


def _run_child(
    repo_root: Path,
    label: str,
    out_path: str,
    video_reference: str,
    backend: str,
    confidence: float,
    flags: dict,
) -> None:
    env = os.environ.copy()
    env.update(flags)
    env["MPLCONFIGDIR"] = "/tmp/mpl"
    model_package = _resolve_model_package()
    if model_package is not None:
        env["PARITY_MODEL_PATH"] = str(model_package)
    env["PYTHONPATH"] = _child_pythonpath(repo_root, env.get("PYTHONPATH"))
    video_path = Path(video_reference)
    if not video_path.is_absolute():
        candidate = SCRIPT_REPO_ROOT / video_path
        if candidate.exists():
            video_reference = str(candidate.resolve())
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
        "--video_reference",
        video_reference,
        "--backend",
        backend,
        "--confidence",
        str(confidence),
    ]
    print(
        "\n---- child ----\n"
        f"  label={label}\n"
        f"  repo_root={repo_root}\n"
        f"  out={out_path}\n"
        f"  flags={flags}",
        flush=True,
    )
    subprocess.run(args, cwd=str(repo_root), env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("driver", "run", "compare"), default="driver"
    )
    parser.add_argument("--repo-root")
    parser.add_argument("--label")
    parser.add_argument("--out")
    parser.add_argument("--base", default=DEFAULT_BASE_OUT)
    parser.add_argument("--candidate", default=DEFAULT_CANDIDATE_OUT)
    parser.add_argument("--base-ref", default="main")
    parser.add_argument("--candidate-ref", default="working-tree")
    parser.add_argument("--video_reference", default="vehicles_1080p.mp4")
    parser.add_argument("--confidence", type=float, default=CONFIDENCE)
    parser.add_argument("--backend", choices=("trt", "onnx", "torch"), default="trt")
    parser.add_argument("--float-atol", type=float, default=1e-4)
    parser.add_argument("--min-box-iou", type=float, default=0.5)
    parser.add_argument("--max-score-delta", type=float, default=0.25)
    parser.add_argument("--keep-worktrees", action="store_true")
    args = parser.parse_args()

    if args.mode == "run":
        if not args.out:
            raise ValueError("--out is required in run mode")
        do_run(
            out_path=args.out,
            repo_root=args.repo_root or str(SCRIPT_REPO_ROOT),
            label=args.label or "run",
            video_reference=args.video_reference,
            backend=args.backend,
            confidence=args.confidence,
        )
        return
    if args.mode == "compare":
        do_compare(
            args.base,
            args.candidate,
            args.float_atol,
            args.min_box_iou,
            args.max_score_delta,
        )
        return

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
            video_reference=args.video_reference,
            backend=args.backend,
            confidence=args.confidence,
            flags=BASE_FLAGS_OFF,
        )
        _run_child(
            repo_root=Path(candidate_target["repo_root"]),
            label=f"{candidate_target['label']} flags-on",
            out_path=args.candidate,
            video_reference=args.video_reference,
            backend=args.backend,
            confidence=args.confidence,
            flags=CANDIDATE_FLAGS_ON,
        )
    finally:
        if not args.keep_worktrees:
            for cleanup in reversed(cleanup_callbacks):
                cleanup()
    do_compare(
        args.base,
        args.candidate,
        args.float_atol,
        args.min_box_iou,
        args.max_score_delta,
    )


if __name__ == "__main__":
    main()
