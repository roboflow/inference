"""Compare RF-DETR depth-2 predictions through a visualization workflow.

This script reuses the package-loading and benchmark setup helpers from
``codeflash-optimization-compare.py``, but runs a workflow shaped as:

    image -> relative static crop -> RF-DETR instance segmentation
          -> polygon visualization

It stops after a small number of dispatched frame results, saves visualization
artifacts for manual inspection, and compares baseline vs optimized prediction
values with an absolute tolerance.
"""

import argparse
import base64
import dataclasses
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from time import perf_counter
from typing import Any

import numpy as np


_THIS_FILE = Path(__file__).resolve()
_BASE_SCRIPT = _THIS_FILE.with_name("codeflash-optimization-compare.py")
_DEFAULT_ARTIFACTS_DIR = (
    _THIS_FILE.parent / "codeflash-visualization-parity-artifacts"
)
_IGNORE_COMPARISON_KEYS = {
    "detection_id",
    "inference_id",
    "parent_id",
    "time",
    "visualization",
}


def _load_base_module():
    spec = importlib.util.spec_from_file_location(
        "codeflash_optimization_compare_base",
        _BASE_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load base benchmark script: {_BASE_SCRIPT}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_workflow(model_id: str, confidence: float) -> dict:
    """Build a crop -> RF-DETR -> polygon visualization workflow."""
    return {
        "version": "1.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "steps": [
            {
                "type": "roboflow_core/relative_statoic_crop@v1",
                "name": "center_crop",
                "images": "$inputs.image",
                "x_center": 0.5,
                "y_center": 0.5,
                "width": 0.8,
                "height": 0.8,
            },
            {
                "type": "roboflow_core/roboflow_instance_segmentation_model@v3",
                "name": "segmentation",
                "images": "$steps.center_crop.crops",
                "model_id": model_id,
                "confidence_mode": "custom",
                "custom_confidence": confidence,
                "enforce_dense_masks_in_inference_models": False,
            },
            {
                "type": "roboflow_core/polygon_visualization@v1",
                "name": "polygon_visualization",
                "image": "$steps.center_crop.crops",
                "predictions": "$steps.segmentation.predictions",
                "copy_image": True,
                "thickness": 2,
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "predictions",
                "selector": "$steps.segmentation.predictions",
            },
            {
                "type": "JsonField",
                "name": "visualization",
                "selector": "$steps.polygon_visualization.image",
            },
        ],
    }


def _normalise_prediction_value(value: Any) -> Any:
    if _is_supervision_detections(value):
        return _normalise_supervision_detections(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _normalise_prediction_value(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {
            str(key): _normalise_prediction_value(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
            if key not in _IGNORE_COMPARISON_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_normalise_prediction_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "detach") and callable(value.detach):
        return _normalise_prediction_value(value.detach().cpu().numpy())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return _normalise_prediction_value(
            value.model_dump(by_alias=True, exclude_none=True)
        )
    return repr(value)


def _is_supervision_detections(value: Any) -> bool:
    return (
        hasattr(value, "xyxy")
        and hasattr(value, "confidence")
        and hasattr(value, "class_id")
        and hasattr(value, "data")
    )


def _normalise_supervision_detections(detections: Any) -> list[dict]:
    xyxy = np.asarray(detections.xyxy).tolist()
    confidence = (
        np.asarray(detections.confidence).tolist()
        if detections.confidence is not None
        else [None] * len(detections)
    )
    class_id = (
        np.asarray(detections.class_id).tolist()
        if detections.class_id is not None
        else [None] * len(detections)
    )
    class_names = detections.data.get("class_name", [])
    class_names = np.asarray(class_names).tolist() if len(class_names) else []

    rows = []
    for index, box in enumerate(xyxy):
        row = {
            "xyxy": box,
            "confidence": confidence[index] if index < len(confidence) else None,
            "class_id": class_id[index] if index < len(class_id) else None,
        }
        if index < len(class_names):
            row["class_name"] = class_names[index]
        rows.append(_normalise_prediction_value(row))
    return rows


def _extract_visualization_image(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (list, tuple)):
        for element in value:
            image = _extract_visualization_image(element)
            if image is not None:
                return image
        return None
    if hasattr(value, "numpy_image"):
        return _extract_visualization_image(value.numpy_image)
    if isinstance(value, dict):
        if "numpy_image" in value:
            return _extract_visualization_image(value["numpy_image"])
        if value.get("type") == "numpy_object" and "value" in value:
            return _extract_visualization_image(value["value"])
        for key in ("image", "value", "data"):
            if key in value:
                image = _extract_visualization_image(value[key])
                if image is not None:
                    return image
        return None
    if isinstance(value, str):
        try:
            from PIL import Image
            from io import BytesIO

            decoded = base64.b64decode(value, validate=True)
            image = Image.open(BytesIO(decoded))
            return np.asarray(image)
        except Exception:
            return None
    return None


def _save_visualization(
    *,
    value: Any,
    artifacts_dir: Path,
    frame_id: int,
    profile: str,
) -> tuple[str | None, bool]:
    image = _extract_visualization_image(value)
    if image is None:
        return None, True

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    output_path = artifacts_dir / f"{profile}_frame_{frame_id:06d}.png"
    try:
        import cv2

        cv2.imwrite(str(output_path), image)
    except Exception:
        from PIL import Image

        Image.fromarray(image).save(output_path)
    return str(output_path), False


def _make_sink(
    *,
    profile: str,
    max_frames: int,
    artifacts_dir: Path,
    records: list[dict],
    pipeline_holder: dict[str, Any],
):
    def sink(prediction, video_frame) -> None:
        frame_id = getattr(video_frame, "frame_id", len(records) + 1)
        prediction_payload = prediction if isinstance(prediction, dict) else {}
        visual_value = prediction_payload.get("visualization")
        visualization_path, missing_visualization = _save_visualization(
            value=visual_value,
            artifacts_dir=artifacts_dir,
            frame_id=int(frame_id),
            profile=profile,
        )
        records.append(
            {
                "frame_id": int(frame_id),
                "predictions": _normalise_prediction_value(
                    prediction_payload.get("predictions", prediction)
                ),
                "visualization_path": visualization_path,
                "missing_visualization": missing_visualization,
                "output_keys": (
                    sorted(prediction_payload.keys())
                    if isinstance(prediction_payload, dict)
                    else []
                ),
            }
        )
        if len(records) >= max_frames:
            pipeline = pipeline_holder.get("pipeline")
            if pipeline is not None:
                pipeline.terminate()

    return sink


def _write_result(
    *,
    result_out: str | None,
    result: dict,
) -> None:
    if result_out is None:
        return
    Path(result_out).write_text(json.dumps(result, indent=2))


def do_run(
    *,
    video_reference: str,
    model_id: str,
    confidence: float,
    backend: str,
    model_package_id: str | None,
    local_package: str | None,
    benchmark_profile: str,
    result_out: str | None,
    max_frames: int,
    artifacts_dir: Path,
) -> dict:
    base = _load_base_module()
    records: list[dict] = []
    pipeline_holder: dict[str, Any] = {}
    start_time = None

    print(f"[benchmark] profile={benchmark_profile}", flush=True)
    print(f"[benchmark] flags: {base._format_optimization_flags()}", flush=True)
    base._log_compute_environment()

    resolved_local_package = base._resolve_local_package(
        backend=backend,
        model_id=model_id,
        model_package_id=model_package_id,
        local_package=local_package,
    )
    if resolved_local_package is not None:
        os.environ.setdefault(
            "ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES",
            "True",
        )

    workflow_model_id = base._resolve_model_id(
        model_id=model_id,
        local_package=resolved_local_package,
    )
    if resolved_local_package is not None:
        base._prepare_local_workflow_model_bundle(
            workflow_model_id=workflow_model_id,
            local_package=resolved_local_package,
        )
        print(
            f"[model] using package via workflow model id: {workflow_model_id}",
            flush=True,
        )

    inference_pipeline = base._load_inference_pipeline(backend=backend)
    pipeline = inference_pipeline.init_with_workflow(
        video_reference=base._resolve_video_reference(video_reference),
        workflow_specification=build_workflow(workflow_model_id, confidence),
        on_prediction=_make_sink(
            profile=benchmark_profile,
            max_frames=max_frames,
            artifacts_dir=artifacts_dir,
            records=records,
            pipeline_holder=pipeline_holder,
        ),
    )
    pipeline_holder["pipeline"] = pipeline
    start_time = perf_counter()
    pipeline.start()
    pipeline.join()

    elapsed = perf_counter() - start_time if start_time is not None else 0.0
    fps = len(records) / elapsed if elapsed > 0 else 0.0
    result = {
        "profile": benchmark_profile,
        "frames": len(records),
        "elapsed": elapsed,
        "fps": fps,
        "flags": {key: os.environ.get(key) for key in base._OPTIMIZATION_FLAG_KEYS},
        "records": records,
    }
    print(
        f"[benchmark] profile={benchmark_profile} frames={len(records)} "
        f"elapsed={elapsed:.2f}s fps={fps:.2f}",
        flush=True,
    )
    _write_result(result_out=result_out, result=result)
    return result


def _compare_values(
    *,
    baseline: Any,
    optimized: Any,
    path: str,
    abs_tol: float,
    errors: list[str],
) -> None:
    if isinstance(baseline, (int, float)) and isinstance(optimized, (int, float)):
        if abs(float(baseline) - float(optimized)) > abs_tol:
            errors.append(f"{path}: {baseline!r} != {optimized!r}")
        return
    if isinstance(baseline, list) and isinstance(optimized, list):
        if len(baseline) != len(optimized):
            errors.append(f"{path}: list length {len(baseline)} != {len(optimized)}")
            return
        for index, (left, right) in enumerate(zip(baseline, optimized)):
            _compare_values(
                baseline=left,
                optimized=right,
                path=f"{path}[{index}]",
                abs_tol=abs_tol,
                errors=errors,
            )
        return
    if isinstance(baseline, dict) and isinstance(optimized, dict):
        if set(baseline) != set(optimized):
            errors.append(
                f"{path}: dict keys {sorted(baseline)} != {sorted(optimized)}"
            )
            return
        for key in sorted(baseline):
            _compare_values(
                baseline=baseline[key],
                optimized=optimized[key],
                path=f"{path}.{key}",
                abs_tol=abs_tol,
                errors=errors,
            )
        return
    if baseline != optimized:
        errors.append(f"{path}: {baseline!r} != {optimized!r}")


def _compare_results(
    *,
    baseline: dict,
    optimized: dict,
    abs_tol: float,
) -> list[str]:
    errors: list[str] = []
    baseline_records = baseline.get("records", [])
    optimized_records = optimized.get("records", [])
    if len(baseline_records) != len(optimized_records):
        errors.append(
            f"record count {len(baseline_records)} != {len(optimized_records)}"
        )
        return errors
    for index, (baseline_record, optimized_record) in enumerate(
        zip(baseline_records, optimized_records)
    ):
        record_path = f"records[{index}]"
        if baseline_record.get("frame_id") != optimized_record.get("frame_id"):
            errors.append(
                f"{record_path}.frame_id: {baseline_record.get('frame_id')} != "
                f"{optimized_record.get('frame_id')}"
            )
        if optimized_record.get("missing_visualization"):
            errors.append(
                f"{record_path}: optimized frame is missing visualization output"
            )
        if baseline_record.get("missing_visualization"):
            errors.append(
                f"{record_path}: baseline frame is missing visualization output"
            )
        _compare_values(
            baseline=baseline_record.get("predictions"),
            optimized=optimized_record.get("predictions"),
            path=f"{record_path}.predictions",
            abs_tol=abs_tol,
            errors=errors,
        )
    return errors


def _build_child_command(
    *,
    video_reference: str,
    model_id: str,
    confidence: float,
    backend: str,
    model_package_id: str | None,
    local_package: str | None,
    benchmark_profile: str,
    result_out: str,
    max_frames: int,
    artifacts_dir: Path,
    abs_tol: float,
) -> list[str]:
    command = [
        sys.executable,
        str(_THIS_FILE),
        "--mode",
        "run",
        "--video_reference",
        video_reference,
        "--model_id",
        model_id,
        "--confidence",
        str(confidence),
        "--backend",
        backend,
        "--benchmark-profile",
        benchmark_profile,
        "--result-out",
        result_out,
        "--max-frames",
        str(max_frames),
        "--artifacts-dir",
        str(artifacts_dir),
        "--abs-tol",
        str(abs_tol),
    ]
    if model_package_id is not None:
        command.extend(["--model_package_id", model_package_id])
    if local_package is not None:
        command.extend(["--local_package", local_package])
    return command


def _run_child_benchmark(
    *,
    benchmark_profile: str,
    flags: dict[str, str],
    video_reference: str,
    model_id: str,
    confidence: float,
    backend: str,
    model_package_id: str | None,
    local_package: str | None,
    result_out: str,
    max_frames: int,
    artifacts_dir: Path,
    abs_tol: float,
) -> dict:
    base = _load_base_module()
    env = os.environ.copy()
    env.update(flags)
    env["PYTHONPATH"] = base._child_pythonpath(env.get("PYTHONPATH"))
    command = _build_child_command(
        video_reference=video_reference,
        model_id=model_id,
        confidence=confidence,
        backend=backend,
        model_package_id=model_package_id,
        local_package=local_package,
        benchmark_profile=benchmark_profile,
        result_out=result_out,
        max_frames=max_frames,
        artifacts_dir=artifacts_dir,
        abs_tol=abs_tol,
    )
    print(
        "\n---- child ----\n"
        f"  profile={benchmark_profile}\n"
        f"  flags={flags}\n"
        f"  result_out={result_out}\n"
        f"  artifacts_dir={artifacts_dir}",
        flush=True,
    )
    subprocess.run(
        command,
        cwd=str(base._REPO_ROOT),
        env=env,
        check=True,
    )
    return json.loads(Path(result_out).read_text())


def do_compare(
    *,
    video_reference: str,
    model_id: str,
    confidence: float,
    backend: str,
    model_package_id: str | None,
    local_package: str | None,
    max_frames: int,
    artifacts_dir: Path,
    abs_tol: float,
) -> None:
    base = _load_base_module()
    resolved_video_reference = base._resolve_video_reference(video_reference)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rfdetr-visualization-parity-") as tmp_dir:
        baseline_result_path = str(Path(tmp_dir) / "baseline.json")
        optimized_result_path = str(Path(tmp_dir) / "optimized.json")
        baseline = _run_child_benchmark(
            benchmark_profile="baseline",
            flags=base._BASELINE_FLAGS,
            video_reference=resolved_video_reference,
            model_id=model_id,
            confidence=confidence,
            backend=backend,
            model_package_id=model_package_id,
            local_package=local_package,
            result_out=baseline_result_path,
            max_frames=max_frames,
            artifacts_dir=artifacts_dir / "baseline",
            abs_tol=abs_tol,
        )
        optimized = _run_child_benchmark(
            benchmark_profile="optimized",
            flags=base._OPTIMIZED_FLAGS,
            video_reference=resolved_video_reference,
            model_id=model_id,
            confidence=confidence,
            backend=backend,
            model_package_id=model_package_id,
            local_package=local_package,
            result_out=optimized_result_path,
            max_frames=max_frames,
            artifacts_dir=artifacts_dir / "optimized",
            abs_tol=abs_tol,
        )

    errors = _compare_results(
        baseline=baseline,
        optimized=optimized,
        abs_tol=abs_tol,
    )
    baseline_fps = baseline["fps"]
    optimized_fps = optimized["fps"]
    speedup = optimized_fps / baseline_fps if baseline_fps > 0 else 0.0
    print("\n---- compare ----", flush=True)
    print(
        f"  baseline   frames={baseline['frames']} "
        f"elapsed={baseline['elapsed']:.2f}s fps={baseline_fps:.2f}",
        flush=True,
    )
    print(
        f"  optimized  frames={optimized['frames']} "
        f"elapsed={optimized['elapsed']:.2f}s fps={optimized_fps:.2f}",
        flush=True,
    )
    print(f"  speedup    {speedup:.2f}x", flush=True)
    print(f"  artifacts  {artifacts_dir}", flush=True)
    if errors:
        print("\n---- parity failures ----", flush=True)
        for error in errors[:50]:
            print(f"  - {error}", flush=True)
        if len(errors) > 50:
            print(f"  ... {len(errors) - 50} more", flush=True)
        raise SystemExit(1)
    print("  parity     ok", flush=True)


def main() -> None:
    base = _load_base_module()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("run", "compare"),
        default="run",
    )
    parser.add_argument("--video_reference", required=True)
    parser.add_argument("--model_id", default=base._DEFAULT_MODEL_ID)
    parser.add_argument("--confidence", type=float, default=0.4)
    parser.add_argument(
        "--backend",
        choices=("trt", "onnx", "torch"),
        default="trt",
    )
    parser.add_argument("--local_package", default=None)
    parser.add_argument("--model_package_id", default=None)
    parser.add_argument("--benchmark-profile", default="run")
    parser.add_argument("--result-out", default=None)
    parser.add_argument("--max-frames", type=int, default=4)
    parser.add_argument("--abs-tol", type=float, default=1e-5)
    parser.add_argument(
        "--artifacts-dir",
        default=str(_DEFAULT_ARTIFACTS_DIR),
    )
    args = parser.parse_args()

    if args.local_package is not None and args.model_package_id is not None:
        parser.error("--local_package and --model_package_id are mutually exclusive")
    if args.max_frames < 1:
        parser.error("--max-frames must be >= 1")

    artifacts_dir = Path(args.artifacts_dir)
    if not artifacts_dir.is_absolute():
        artifacts_dir = base._REPO_ROOT / artifacts_dir

    if args.mode == "compare":
        do_compare(
            video_reference=args.video_reference,
            model_id=args.model_id,
            confidence=args.confidence,
            backend=args.backend,
            model_package_id=args.model_package_id,
            local_package=args.local_package,
            max_frames=args.max_frames,
            artifacts_dir=artifacts_dir,
            abs_tol=args.abs_tol,
        )
        return

    do_run(
        video_reference=args.video_reference,
        model_id=args.model_id,
        confidence=args.confidence,
        backend=args.backend,
        model_package_id=args.model_package_id,
        local_package=args.local_package,
        benchmark_profile=args.benchmark_profile,
        result_out=args.result_out,
        max_frames=args.max_frames,
        artifacts_dir=artifacts_dir,
    )


if __name__ == "__main__":
    main()
