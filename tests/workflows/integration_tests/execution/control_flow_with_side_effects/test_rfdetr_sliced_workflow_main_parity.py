import argparse
import copy
import json
import math
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import pytest

SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[5]
SELF = Path(__file__).resolve()
PY = sys.executable

MODEL_TYPE = "rfdetr-seg-nano"
MODEL_ALIAS = "__workflow_parity_rfdetr_seg_nano__"
LOCAL_WORKFLOW_MODEL_ID = f"{MODEL_ALIAS}/1"
CONFIDENCE = 0.4
IMAGE_COUNT = 4
IMAGE_SIZE = 2560
VIDEO_REFERENCE = "vehicles_1080p.mp4"
DEFAULT_BASE_OUT = "/tmp/rfdetr_sliced_workflow_main_base.pkl"
DEFAULT_CANDIDATE_OUT = "/tmp/rfdetr_sliced_workflow_main_candidate.pkl"
TRT_PACKAGE_REQUIRED_FILES = (
    "model_config.json",
    "class_names.txt",
    "inference_config.json",
    "engine.plan",
)
ALL_BACKENDS = {
    "torch",
    "torch-script",
    "onnx",
    "trt",
    "hugging-face",
    "ultralytics",
    "custom",
}
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
_VOLATILE_OUTPUT_KEYS = {"detection_id", "inference_id"}
BASE_WORKFLOW_DEFINITION = {
    "version": "1.0",
    "inputs": [{"type": "WorkflowImage", "name": "image"}],
    "steps": [
        {
            "type": "roboflow_core/image_slicer@v2",
            "name": "image_slicer",
            "image": "$inputs.image",
            "slice_width": 640,
            "slice_height": 640,
            "overlap_ratio_width": 0.0,
            "overlap_ratio_height": 0.0,
        },
        {
            "type": "roboflow_core/roboflow_instance_segmentation_model@v3",
            "name": "segmentation",
            "images": "$steps.image_slicer.slices",
            "model_id": LOCAL_WORKFLOW_MODEL_ID,
            "confidence_mode": "custom",
            "custom_confidence": CONFIDENCE,
            "iou_threshold": 0.3,
            "max_detections": 300,
            "max_candidates": 3000,
            "class_agnostic_nms": False,
            "mask_decode_mode": "accurate",
            "tradeoff_factor": 0.0,
            "disable_active_learning": True,
            "enforce_dense_masks_in_inference_models": False,
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "predictions",
            "selector": "$steps.segmentation.predictions",
        }
    ],
}


def bool_env(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    return str(val).lower() in {"true", "1", "t", "y", "yes"}


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
    worktree_root = Path(tempfile.mkdtemp(prefix="rfdetr-sliced-workflow-parity-"))
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
    for root in (
        SCRIPT_REPO_ROOT,
        SCRIPT_REPO_ROOT / "temp",
        Path.cwd(),
        Path(tempfile.gettempdir()),
    ):
        for name in (
            "rfdetr-seg-nano-t4-trt-package",
            "rfdetr-seg-nano-orin-trt-package",
            "rfdetr-seg-nano-trt-package",
        ):
            package = root / name
            if _is_trt_package(package):
                return package.resolve()
    return None


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _prepare_local_workflow_model_bundle(repo_root: Path) -> tuple[str, callable]:
    package = _resolve_model_package()
    if package is None:
        return MODEL_TYPE, lambda: None
    os.environ.setdefault(
        "ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES", "true"
    )
    model_root = repo_root / MODEL_ALIAS
    model_dir = model_root / "1"
    if model_dir.exists() or model_dir.is_symlink():
        raise RuntimeError(
            f"Refusing to reuse existing parity model alias path: {model_dir}"
        )
    model_root.mkdir(parents=True, exist_ok=True)
    model_dir.symlink_to(package, target_is_directory=True)

    model_cache_root = (
        Path(os.environ.get("MODEL_CACHE_DIR", "/tmp/cache")) / MODEL_ALIAS
    )
    model_cache_dir = model_cache_root / "1"
    model_cache_dir.mkdir(parents=True, exist_ok=True)
    model_type_path = model_cache_dir / "model_type.json"
    model_metadata = {
        "project_task_type": "instance-segmentation",
        "model_type": MODEL_TYPE,
    }
    model_type_path.write_text(json.dumps(model_metadata, indent=4))

    def cleanup() -> None:
        if model_dir.is_symlink():
            _safe_unlink(model_dir)
        elif model_dir.exists():
            shutil.rmtree(model_dir, ignore_errors=True)
        try:
            model_root.rmdir()
        except OSError:
            pass
        _safe_unlink(model_type_path)
        try:
            model_cache_dir.rmdir()
        except OSError:
            pass
        try:
            model_cache_root.rmdir()
        except OSError:
            pass

    return LOCAL_WORKFLOW_MODEL_ID, cleanup


def _letterbox_to_square(image: np.ndarray, size: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = size / max(height, width)
    target_width = max(1, int(round(width * scale)))
    target_height = max(1, int(round(height * scale)))
    resized = cv2.resize(
        image,
        (target_width, target_height),
        interpolation=cv2.INTER_LINEAR,
    )
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    offset_x = (size - target_width) // 2
    offset_y = (size - target_height) // 2
    canvas[offset_y : offset_y + target_height, offset_x : offset_x + target_width] = (
        resized
    )
    return canvas


def _load_video_frames_as_square_images(
    video_reference: str,
    image_count: int,
    image_size: int,
) -> list[np.ndarray]:
    video_path = Path(video_reference)
    if not video_path.is_absolute():
        candidate = SCRIPT_REPO_ROOT / video_path
        if candidate.exists():
            video_path = candidate.resolve()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video reference: {video_path}")
    frame_count = max(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
    indices = np.linspace(0, frame_count - 1, num=image_count, dtype=int).tolist()
    images = []
    try:
        for frame_index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            success, frame = capture.read()
            if not success or frame is None:
                raise RuntimeError(
                    f"Could not read frame {frame_index} from {video_path}"
                )
            images.append(_letterbox_to_square(frame, size=image_size))
    finally:
        capture.release()
    return images


def _build_workflow(model_id: str) -> dict:
    workflow_definition = copy.deepcopy(BASE_WORKFLOW_DEFINITION)
    workflow_definition["steps"][1]["model_id"] = model_id
    return workflow_definition


def _normalize_output(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _normalize_output(val)
            for key, val in sorted(value.items())
            if str(key) not in _VOLATILE_OUTPUT_KEYS
        }
    if isinstance(value, list):
        return [_normalize_output(element) for element in value]
    if isinstance(value, tuple):
        return [_normalize_output(element) for element in value]
    if isinstance(value, np.ndarray):
        return _normalize_output(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


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


def _rle_iou(left: dict, right: dict) -> float:
    from pycocotools import mask as mask_utils

    return float(
        mask_utils.iou([_rle_for_coco_iou(left)], [_rle_for_coco_iou(right)], [False])[
            0, 0
        ]
    )


def _extract_detection_list(value: Any) -> Optional[list]:
    if not isinstance(value, dict):
        return None
    predictions = value.get("predictions")
    if isinstance(predictions, list):
        return predictions
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
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
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
    path: str,
    min_box_iou: float,
    max_score_delta: float,
    min_mask_iou: float,
    stats: dict,
    errors: list[str],
) -> None:
    stats["base_detections"] += len(base_detections)
    stats["candidate_detections"] += len(candidate_detections)
    if len(base_detections) != len(candidate_detections):
        stats["count_mismatch_containers"] += 1
        errors.append(
            f"{path}: detection count mismatch "
            f"{len(base_detections)} != {len(candidate_detections)}"
        )
        return

    used_base_indices = set()
    for candidate_index, candidate_detection in enumerate(candidate_detections):
        candidate_class = candidate_detection.get("class_id")
        candidate_box = _xyxy_from_detection(candidate_detection)
        if candidate_box is None:
            errors.append(f"{path}[{candidate_index}]: candidate detection has no box")
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
                f"{path}[{candidate_index}]: no base match for class_id={candidate_class!r}"
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
                f"{path}[{candidate_index}]: score delta {score_delta:.6f} exceeds "
                f"{max_score_delta:.6f}"
            )

        base_rle = base_detection.get("rle")
        candidate_rle = candidate_detection.get("rle")
        if base_rle is None and candidate_rle is None:
            continue
        if base_rle is None or candidate_rle is None:
            errors.append(
                f"{path}[{candidate_index}]: RLE presence mismatch "
                f"{base_rle is not None} != {candidate_rle is not None}"
            )
            continue
        base_rle = _normalized_rle(base_rle)
        candidate_rle = _normalized_rle(candidate_rle)
        mask_iou = _rle_iou(base_rle, candidate_rle)
        stats["mask_ious"].append(mask_iou)
        if base_rle == candidate_rle:
            stats["pixel_identical_masks"] += 1
        if mask_iou < min_mask_iou:
            errors.append(
                f"{path}[{candidate_index}]: mask IoU {mask_iou:.6f} is below "
                f"{min_mask_iou:.6f}"
            )

    unmatched_base = len(base_detections) - len(used_base_indices)
    stats["unmatched_base_detections"] += unmatched_base
    if unmatched_base:
        errors.append(f"{path}: {unmatched_base} base detections left unmatched")


def _compare_values(
    base: Any,
    candidate: Any,
    path: str,
    atol: float,
    min_box_iou: float,
    max_score_delta: float,
    min_mask_iou: float,
    stats: dict,
    errors: list[str],
) -> None:
    base_detections = _extract_detection_list(base)
    candidate_detections = _extract_detection_list(candidate)
    if (
        base_detections is not None
        and candidate_detections is not None
        and isinstance(base, dict)
        and isinstance(candidate, dict)
        and isinstance(base.get("image"), dict)
        and isinstance(candidate.get("image"), dict)
    ):
        stats["prediction_containers"] += 1
        _compare_values(
            base["image"],
            candidate["image"],
            f"{path}.image",
            atol,
            min_box_iou,
            max_score_delta,
            min_mask_iou,
            stats,
            errors,
        )
        _compare_detection_lists(
            base_detections=base_detections,
            candidate_detections=candidate_detections,
            path=f"{path}.predictions",
            min_box_iou=min_box_iou,
            max_score_delta=max_score_delta,
            min_mask_iou=min_mask_iou,
            stats=stats,
            errors=errors,
        )
        return
    if isinstance(base, dict) and isinstance(candidate, dict):
        if set(base) != set(candidate):
            errors.append(f"{path}: key mismatch {sorted(base)} != {sorted(candidate)}")
            return
        for key in sorted(base):
            _compare_values(
                base[key],
                candidate[key],
                f"{path}.{key}",
                atol,
                min_box_iou,
                max_score_delta,
                min_mask_iou,
                stats,
                errors,
            )
        return
    if isinstance(base, list) and isinstance(candidate, list):
        if len(base) != len(candidate):
            errors.append(f"{path}: length mismatch {len(base)} != {len(candidate)}")
            return
        for index, (base_item, candidate_item) in enumerate(zip(base, candidate)):
            _compare_values(
                base_item,
                candidate_item,
                f"{path}[{index}]",
                atol,
                min_box_iou,
                max_score_delta,
                min_mask_iou,
                stats,
                errors,
            )
        return
    if isinstance(base, (int, float)) and isinstance(candidate, (int, float)):
        if not math.isclose(float(base), float(candidate), abs_tol=atol, rel_tol=0.0):
            errors.append(f"{path}: numeric mismatch {base} != {candidate}")
        return
    if base != candidate:
        errors.append(f"{path}: value mismatch {base!r} != {candidate!r}")


def do_run(
    out_path: str,
    repo_root: str,
    label: str,
    video_reference: str,
    image_count: int,
    image_size: int,
) -> None:
    repo_path = _bootstrap_repo_root(repo_root)
    os.environ.setdefault(
        "ONNXRUNTIME_EXECUTION_PROVIDERS",
        "[TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider]",
    )
    os.environ["DISABLED_INFERENCE_MODELS_BACKENDS"] = ",".join(
        sorted(ALL_BACKENDS - {"trt"})
    )
    model_id, cleanup_model_bundle = _prepare_local_workflow_model_bundle(repo_path)
    try:
        from inference.core.managers.base import ModelManager
        from inference.core.registries.roboflow import RoboflowModelRegistry
        from inference.core.workflows.core_steps.common.entities import (
            StepExecutionMode,
        )
        from inference.core.workflows.execution_engine.core import ExecutionEngine
        from inference.models.utils import ROBOFLOW_MODEL_TYPES

        workflow_definition = _build_workflow(model_id=model_id)
        model_manager = ModelManager(
            model_registry=RoboflowModelRegistry(ROBOFLOW_MODEL_TYPES)
        )
        execution_engine = ExecutionEngine.init(
            workflow_definition=workflow_definition,
            init_parameters={
                "workflows_core.model_manager": model_manager,
                "workflows_core.api_key": None,
                "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
            },
            max_concurrent_steps=4,
        )
        images = _load_video_frames_as_square_images(
            video_reference=video_reference,
            image_count=image_count,
            image_size=image_size,
        )
        result = execution_engine.run(
            runtime_parameters={"image": images},
            serialize_results=True,
        )
        payload = {
            "label": label,
            "repo_root": str(repo_path),
            "git_head": _safe_git_output(repo_path, "rev-parse", "--short", "HEAD"),
            "git_describe": _safe_git_output(
                repo_path,
                "describe",
                "--always",
                "--dirty",
                "--broken",
            ),
            "video_reference": video_reference,
            "image_count": image_count,
            "image_size": image_size,
            "flags": {
                key: os.environ.get(key)
                for key in sorted({*BASE_FLAGS_OFF.keys(), *CANDIDATE_FLAGS_ON.keys()})
            },
            "result": _normalize_output(result),
        }
        with open(out_path, "wb") as f:
            pickle.dump(payload, f)
        print(
            "[run] "
            f"label={label} repo_root={repo_path} head={payload['git_head']} "
            f"images={image_count} image_size={image_size} out={out_path}",
            flush=True,
        )
    finally:
        cleanup_model_bundle()


def _load_payload(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def do_compare(
    base_path: str,
    candidate_path: str,
    atol: float,
    min_box_iou: float,
    max_score_delta: float,
    min_mask_iou: float,
) -> None:
    base_payload = _load_payload(base_path)
    candidate_payload = _load_payload(candidate_path)
    errors: list[str] = []
    stats = {
        "prediction_containers": 0,
        "base_detections": 0,
        "candidate_detections": 0,
        "matched_detections": 0,
        "count_mismatch_containers": 0,
        "unmatched_base_detections": 0,
        "unmatched_candidate_detections": 0,
        "box_ious": [],
        "score_deltas": [],
        "mask_ious": [],
        "pixel_identical_masks": 0,
    }
    _compare_values(
        base=base_payload["result"],
        candidate=candidate_payload["result"],
        path="result",
        atol=atol,
        min_box_iou=min_box_iou,
        max_score_delta=max_score_delta,
        min_mask_iou=min_mask_iou,
        stats=stats,
        errors=errors,
    )

    print()
    print(
        f"==== RF-DETR sliced workflow parity: {base_payload['label']} vs "
        f"{candidate_payload['label']} ===="
    )
    print(f"  base repo                    : {base_payload['git_describe']}")
    print(f"  candidate repo               : {candidate_payload['git_describe']}")
    print(f"  prediction containers        : {stats['prediction_containers']}")
    print(
        f"  detections base / candidate  : "
        f"{stats['base_detections']} / {stats['candidate_detections']}"
    )
    print(f"  matched detections           : {stats['matched_detections']}")
    print(f"  count-mismatch containers    : " f"{stats['count_mismatch_containers']}")
    print(
        f"  unmatched base / candidate   : "
        f"{stats['unmatched_base_detections']} / "
        f"{stats['unmatched_candidate_detections']}"
    )
    if stats["box_ious"]:
        print(
            f"  mean / min box IoU           : "
            f"{np.mean(stats['box_ious']):.6f} / {np.min(stats['box_ious']):.6f}"
        )
    if stats["score_deltas"]:
        print(
            f"  mean / max |delta score|     : "
            f"{np.mean(stats['score_deltas']):.3e} / "
            f"{np.max(stats['score_deltas']):.3e}"
        )
    if stats["mask_ious"]:
        print(
            f"  mean / min mask IoU          : "
            f"{np.mean(stats['mask_ious']):.6f} / {np.min(stats['mask_ious']):.6f}"
        )
        print(
            f"  pixel-identical masks        : "
            f"{stats['pixel_identical_masks']}/{len(stats['mask_ious'])}"
        )
    if errors:
        print("  first mismatches:")
        for error in errors[:20]:
            print(f"    - {error}")
        raise AssertionError(f"{len(errors)} parity mismatches found")
    print("  result                       : outputs matched parity thresholds")


def _run_child(
    repo_root: Path,
    label: str,
    out_path: str,
    video_reference: str,
    image_count: int,
    image_size: int,
    flags: dict[str, str],
) -> None:
    env = os.environ.copy()
    env.update(flags)
    model_package = _resolve_model_package()
    if model_package is not None:
        env["PARITY_MODEL_PATH"] = str(model_package)
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
        "--video-reference",
        video_reference,
        "--image-count",
        str(image_count),
        "--image-size",
        str(image_size),
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


def _run_driver(
    base_ref: str,
    candidate_ref: str,
    base_out: str,
    candidate_out: str,
    video_reference: str,
    image_count: int,
    image_size: int,
    float_atol: float,
    min_box_iou: float,
    max_score_delta: float,
    min_mask_iou: float,
    keep_worktrees: bool,
) -> None:
    base_target = _materialize_target(base_ref)
    candidate_target = _materialize_target(candidate_ref)
    cleanup_callbacks = [
        target["cleanup"]
        for target in (base_target, candidate_target)
        if callable(target["cleanup"])
    ]
    try:
        _run_child(
            repo_root=Path(base_target["repo_root"]),
            label=f"{base_target['label']} flags-off",
            out_path=base_out,
            video_reference=video_reference,
            image_count=image_count,
            image_size=image_size,
            flags=BASE_FLAGS_OFF,
        )
        _run_child(
            repo_root=Path(candidate_target["repo_root"]),
            label=f"{candidate_target['label']} flags-on",
            out_path=candidate_out,
            video_reference=video_reference,
            image_count=image_count,
            image_size=image_size,
            flags=CANDIDATE_FLAGS_ON,
        )
    finally:
        if not keep_worktrees:
            for cleanup in reversed(cleanup_callbacks):
                cleanup()
    do_compare(
        base_path=base_out,
        candidate_path=candidate_out,
        atol=float_atol,
        min_box_iou=min_box_iou,
        max_score_delta=max_score_delta,
        min_mask_iou=min_mask_iou,
    )


@pytest.mark.slow
@pytest.mark.workflows
@pytest.mark.skipif(
    bool_env(os.getenv("SKIP_RFDETR_SLICED_WORKFLOW_MAIN_PARITY_TEST", True)),
    reason="Skipping RF-DETR sliced workflow parity test",
)
def test_rfdetr_sliced_workflow_optimized_flow_matches_main_branch() -> None:
    base_out = Path(
        tempfile.mkstemp(prefix="rfdetr-sliced-workflow-base-", suffix=".pkl")[1]
    )
    candidate_out = Path(
        tempfile.mkstemp(prefix="rfdetr-sliced-workflow-candidate-", suffix=".pkl")[1]
    )
    try:
        _run_driver(
            base_ref=os.getenv("RFDETR_WORKFLOW_PARITY_BASE_REF", "main"),
            candidate_ref=os.getenv(
                "RFDETR_WORKFLOW_PARITY_CANDIDATE_REF", "working-tree"
            ),
            base_out=str(base_out),
            candidate_out=str(candidate_out),
            video_reference=os.getenv(
                "RFDETR_WORKFLOW_PARITY_VIDEO_REFERENCE",
                VIDEO_REFERENCE,
            ),
            image_count=int(
                os.getenv("RFDETR_WORKFLOW_PARITY_IMAGE_COUNT", str(IMAGE_COUNT))
            ),
            image_size=int(
                os.getenv("RFDETR_WORKFLOW_PARITY_IMAGE_SIZE", str(IMAGE_SIZE))
            ),
            float_atol=float(os.getenv("RFDETR_WORKFLOW_PARITY_FLOAT_ATOL", "1e-4")),
            min_box_iou=float(os.getenv("RFDETR_WORKFLOW_PARITY_MIN_BOX_IOU", "0.5")),
            max_score_delta=float(
                os.getenv("RFDETR_WORKFLOW_PARITY_MAX_SCORE_DELTA", "0.25")
            ),
            min_mask_iou=float(
                os.getenv("RFDETR_WORKFLOW_PARITY_MIN_MASK_IOU", "0.95")
            ),
            keep_worktrees=bool_env(
                os.getenv("RFDETR_WORKFLOW_PARITY_KEEP_WORKTREES", False)
            ),
        )
    finally:
        base_out.unlink(missing_ok=True)
        candidate_out.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("driver", "run", "compare"),
        default="driver",
    )
    parser.add_argument("--repo-root")
    parser.add_argument("--label")
    parser.add_argument("--out")
    parser.add_argument("--base", default=DEFAULT_BASE_OUT)
    parser.add_argument("--candidate", default=DEFAULT_CANDIDATE_OUT)
    parser.add_argument("--base-ref", default="main")
    parser.add_argument("--candidate-ref", default="working-tree")
    parser.add_argument("--video-reference", default=VIDEO_REFERENCE)
    parser.add_argument("--image-count", type=int, default=IMAGE_COUNT)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--float-atol", type=float, default=1e-4)
    parser.add_argument("--min-box-iou", type=float, default=0.5)
    parser.add_argument("--max-score-delta", type=float, default=0.25)
    parser.add_argument("--min-mask-iou", type=float, default=0.95)
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
            image_count=args.image_count,
            image_size=args.image_size,
        )
        return
    if args.mode == "compare":
        do_compare(
            base_path=args.base,
            candidate_path=args.candidate,
            atol=args.float_atol,
            min_box_iou=args.min_box_iou,
            max_score_delta=args.max_score_delta,
            min_mask_iou=args.min_mask_iou,
        )
        return

    _run_driver(
        base_ref=args.base_ref,
        candidate_ref=args.candidate_ref,
        base_out=args.base,
        candidate_out=args.candidate,
        video_reference=args.video_reference,
        image_count=args.image_count,
        image_size=args.image_size,
        float_atol=args.float_atol,
        min_box_iou=args.min_box_iou,
        max_score_delta=args.max_score_delta,
        min_mask_iou=args.min_mask_iou,
        keep_worktrees=args.keep_worktrees,
    )


if __name__ == "__main__":
    main()
