"""Benchmark remote workflow stream pipelining against sequential execution.

Runs the same video workflow (remote object-detection model -> ByteTrack ->
bounding-box visualization) through `InferencePipeline.init_with_workflow`
once per requested pipeline depth. Depth 1 is the sequential baseline; depths
above 1 enable `WORKFLOWS_REMOTE_EXECUTION_PIPELINE_DEPTH`, which keeps that
many remote model requests in flight while ByteTrack consumes results in
strict frame order.

Each scenario runs in a subprocess because the pipeline depth env variable is
read at import time. The parent aggregates per-scenario results, verifies that
every pipelined run emits frames in order with tracker ids identical to the
sequential baseline, and writes a combined summary.

Example:
    ROBOFLOW_API_KEY=... python development/benchmark_scripts/benchmark_remote_stream_pipeline.py \
        --api-url https://serverless.roboflow.com \
        --model-id rfdetr-nano \
        --frames 300 --warmup 10 --resize-width 640 \
        --depths 1,8,16,32 \
        --output-dir /tmp/remote_stream_pipeline_benchmark
"""

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_API_URL = "https://serverless.roboflow.com"
DEFAULT_MODEL_ID = "rfdetr-nano"
DEFAULT_VIDEO_URL = (
    "https://media.roboflow.com/supervision/video-examples/vehicles-2.mp4"
)
SCENARIO_RESULT_MARKER = "SCENARIO_RESULT_JSON "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare sequential remote workflow execution against remote "
            "stream pipelining across pipeline depths."
        )
    )
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument(
        "--core-model-url",
        default="https://infer.roboflow.com",
        help="Hosted core-model API used by the sam3 workflow.",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--video-url", default=DEFAULT_VIDEO_URL)
    parser.add_argument(
        "--frames", type=int, default=300, help="Measured frames per scenario."
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Frames run before measurement to warm the model and sessions.",
    )
    parser.add_argument(
        "--resize-width",
        type=int,
        default=640,
        help="Width the benchmark clip is scaled to; 0 keeps the original.",
    )
    parser.add_argument(
        "--depths",
        default="1,8,16,32",
        help="Comma-separated pipeline depths; depth 1 is the baseline.",
    )
    parser.add_argument(
        "--workflow",
        choices=["tracking", "sam3", "two-models", "preprocessed-tracking"],
        default="tracking",
        help=(
            "Workflow shape: tracking (model -> ByteTrack -> viz), sam3 "
            "(SAM3 text-prompt segmentation -> ByteTrack -> viz), two-models "
            "(two detection models -> consensus -> ByteTrack -> viz), "
            "preprocessed-tracking (static crop -> model -> ByteTrack -> viz "
            "plus a stateless blur side branch)."
        ),
    )
    parser.add_argument(
        "--second-model-id",
        default="yolov8n-640",
        help="Second model for the two-models workflow.",
    )
    parser.add_argument(
        "--sam3-class-names",
        default="car,truck",
        help="Comma-separated text prompts for the sam3 workflow.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--run-single-depth",
        type=int,
        default=None,
        help="Internal: run one scenario in this process and print its result.",
    )
    return parser.parse_args()


def prepare_clip(
    video_url: str,
    frames: int,
    resize_width: int,
    output_dir: Path,
) -> Dict[str, Any]:
    import cv2

    source_path = output_dir / "source-video.mp4"
    if not source_path.exists():
        urllib.request.urlretrieve(video_url, source_path)
    clip_path = output_dir / f"clip-{frames}f-w{resize_width}.mp4"
    capture = cv2.VideoCapture(str(source_path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if resize_width and resize_width < source_width:
        clip_width = resize_width
        clip_height = round(source_height * resize_width / source_width)
    else:
        clip_width, clip_height = source_width, source_height
    if not clip_path.exists():
        writer = cv2.VideoWriter(
            str(clip_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (clip_width, clip_height),
        )
        written = 0
        while written < frames:
            success, frame = capture.read()
            if not success:
                break
            if (clip_width, clip_height) != (source_width, source_height):
                frame = cv2.resize(
                    frame, (clip_width, clip_height), interpolation=cv2.INTER_AREA
                )
            writer.write(frame)
            written += 1
        writer.release()
    capture.release()
    return {
        "clip_path": str(clip_path),
        "fps": fps,
        "width": clip_width,
        "height": clip_height,
    }


def build_workflow_specification(args: argparse.Namespace) -> Dict[str, Any]:
    if args.workflow == "tracking":
        detection_steps = [
            {
                "type": "roboflow_core/roboflow_object_detection_model@v3",
                "name": "model",
                "images": "$inputs.image",
                "model_id": args.model_id,
                "confidence_mode": "custom",
                "custom_confidence": 0.35,
            },
        ]
        tracker_detections_selector = "$steps.model.predictions"
    elif args.workflow == "sam3":
        detection_steps = [
            {
                "type": "roboflow_core/sam3@v3",
                "name": "sam3",
                "images": "$inputs.image",
                "class_names": args.sam3_class_names.split(","),
                "output_format": "rle",
            },
        ]
        tracker_detections_selector = "$steps.sam3.predictions"
    elif args.workflow == "preprocessed-tracking":
        # Stateless preprocessing UPSTREAM of the model plus a stateless side
        # branch — shapes only executable ahead of stream order with the
        # frontier scheduler.
        detection_steps = [
            {
                "type": "roboflow_core/absolute_static_crop@v1",
                "name": "crop",
                "images": "$inputs.image",
                "x_center": 320,
                "y_center": 180,
                "width": 600,
                "height": 340,
            },
            {
                "type": "roboflow_core/image_blur@v1",
                "name": "blur",
                "image": "$inputs.image",
            },
            {
                "type": "roboflow_core/roboflow_object_detection_model@v3",
                "name": "model",
                "images": "$steps.crop.crops",
                "model_id": args.model_id,
                "confidence_mode": "custom",
                "custom_confidence": 0.35,
            },
        ]
        tracker_detections_selector = "$steps.model.predictions"
    else:
        detection_steps = [
            {
                "type": "roboflow_core/roboflow_object_detection_model@v3",
                "name": "model_a",
                "images": "$inputs.image",
                "model_id": args.model_id,
                "confidence_mode": "custom",
                "custom_confidence": 0.35,
            },
            {
                "type": "roboflow_core/roboflow_object_detection_model@v3",
                "name": "model_b",
                "images": "$inputs.image",
                "model_id": args.second_model_id,
                "confidence_mode": "custom",
                "custom_confidence": 0.35,
            },
            {
                "type": "roboflow_core/detections_consensus@v1",
                "name": "consensus",
                "predictions_batches": [
                    "$steps.model_a.predictions",
                    "$steps.model_b.predictions",
                ],
                "required_votes": 1,
            },
        ]
        tracker_detections_selector = "$steps.consensus.predictions"
    return {
        "version": "1.0.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "steps": detection_steps
        + [
            {
                "type": "roboflow_core/byte_tracker@v3",
                "name": "byte_tracker",
                "image": "$inputs.image",
                "detections": tracker_detections_selector,
            },
            {
                "type": "roboflow_core/bounding_box_visualization@v1",
                "name": "bounding_box_visualization",
                "image": "$inputs.image",
                "predictions": "$steps.byte_tracker.tracked_detections",
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "tracked_detections",
                "coordinates_system": "own",
                "selector": "$steps.byte_tracker.tracked_detections",
            },
            {
                "type": "JsonField",
                "name": "visualization",
                "coordinates_system": "own",
                "selector": "$steps.bounding_box_visualization.image",
            },
        ]
        + (
            [
                {
                    "type": "JsonField",
                    "name": "blurred",
                    "coordinates_system": "own",
                    "selector": "$steps.blur.image",
                }
            ]
            if args.workflow == "preprocessed-tracking"
            else []
        ),
    }


def run_scenario(args: argparse.Namespace, depth: int) -> Dict[str, Any]:
    from inference.core.interfaces.stream.inference_pipeline import InferencePipeline

    workflow_specification = build_workflow_specification(args=args)
    clip = json.loads((args.output_dir / "clip-metadata.json").read_text())

    def execute_pipeline(video_path: str) -> Dict[str, Any]:
        emissions: List[Dict[str, Any]] = []

        def on_prediction(predictions, video_frame) -> None:
            tracked = predictions.get("tracked_detections") if predictions else None
            tracker_ids = (
                [int(tracker_id) for tracker_id in tracked.tracker_id]
                if tracked is not None and tracked.tracker_id is not None
                else []
            )
            emissions.append(
                {
                    "frame_id": video_frame.frame_id,
                    "emitted_at": time.perf_counter(),
                    "tracker_ids": tracker_ids,
                    "has_visualization": (
                        predictions.get("visualization") is not None
                        if predictions
                        else False
                    ),
                }
            )

        pipeline = InferencePipeline.init_with_workflow(
            video_reference=video_path,
            workflow_specification=workflow_specification,
            api_key=os.environ["ROBOFLOW_API_KEY"],
            on_prediction=on_prediction,
        )
        runner_type = type(pipeline._on_video_frame).__name__
        started_at = time.perf_counter()
        pipeline.start()
        pipeline.join()
        finished_at = time.perf_counter()
        return {
            "runner_type": runner_type,
            "started_at": started_at,
            "finished_at": finished_at,
            "emissions": emissions,
        }

    warmup_clip = args.output_dir / f"warmup-{args.warmup}f.mp4"
    if args.warmup > 0 and warmup_clip.exists():
        execute_pipeline(video_path=str(warmup_clip))
    measured = execute_pipeline(video_path=clip["clip_path"])

    emissions = measured["emissions"]
    if not emissions:
        raise RuntimeError(
            "Scenario emitted no frames — the pipeline likely hit an error on "
            "the inference thread (check credentials/URLs and rerun with logs)."
        )
    frame_ids = [emission["frame_id"] for emission in emissions]
    emission_times = [emission["emitted_at"] for emission in emissions]
    wall_seconds = measured["finished_at"] - measured["started_at"]
    steady_seconds = (
        emission_times[-1] - emission_times[0] if len(emission_times) > 1 else None
    )
    inter_emission_ms = [
        (later - earlier) * 1000
        for earlier, later in zip(emission_times, emission_times[1:])
    ]
    return {
        "depth": depth,
        "runner_type": measured["runner_type"],
        "frames_emitted": len(emissions),
        "wall_seconds": round(wall_seconds, 3),
        "wall_fps": round(len(emissions) / wall_seconds, 2) if wall_seconds else None,
        "steady_fps": (
            round((len(emissions) - 1) / steady_seconds, 2) if steady_seconds else None
        ),
        "first_emission_latency_seconds": (
            round(emission_times[0] - measured["started_at"], 3)
            if emission_times
            else None
        ),
        "inter_emission_ms": summarize_values(values=inter_emission_ms),
        "emitted_in_order": frame_ids == sorted(frame_ids),
        "unique_frames": len(set(frame_ids)) == len(frame_ids),
        "all_have_visualization": all(
            emission["has_visualization"] for emission in emissions
        ),
        "total_tracked_detections": sum(
            len(emission["tracker_ids"]) for emission in emissions
        ),
        "tracker_ids_per_frame": {
            str(emission["frame_id"]): emission["tracker_ids"] for emission in emissions
        },
    }


def summarize_values(values: List[float]) -> Optional[Dict[str, float]]:
    if not values:
        return None
    ordered = sorted(values)
    return {
        "count": len(values),
        "min": round(ordered[0], 2),
        "max": round(ordered[-1], 2),
        "avg": round(statistics.fmean(values), 2),
        "median": round(statistics.median(values), 2),
        "p95": round(percentile(ordered, 0.95), 2),
    }


def percentile(ordered_values: List[float], fraction: float) -> float:
    if len(ordered_values) == 1:
        return ordered_values[0]
    position = fraction * (len(ordered_values) - 1)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered_values) - 1)
    weight = position - lower_index
    return (
        ordered_values[lower_index] * (1 - weight)
        + ordered_values[upper_index] * weight
    )


def check_tracker_parity(
    baseline: Dict[str, Any], candidate: Dict[str, Any]
) -> Dict[str, Any]:
    baseline_ids = baseline["tracker_ids_per_frame"]
    candidate_ids = candidate["tracker_ids_per_frame"]
    mismatched_frames = [
        frame_id
        for frame_id, tracker_ids in baseline_ids.items()
        if candidate_ids.get(frame_id) != tracker_ids
    ]
    return {
        "frames_compared": len(baseline_ids),
        "mismatched_frames": len(mismatched_frames),
        "matches_baseline": not mismatched_frames
        and len(candidate_ids) == len(baseline_ids),
        "first_mismatches": mismatched_frames[:5],
    }


def run_child_scenarios(args: argparse.Namespace, depths: List[int]) -> List[dict]:
    scenario_results = []
    for depth in depths:
        child_env = {
            **os.environ,
            "WORKFLOWS_STEP_EXECUTION_MODE": "remote",
            "WORKFLOWS_REMOTE_API_TARGET": "hosted",
            "HOSTED_DETECT_URL": args.api_url,
            "HOSTED_CORE_MODEL_URL": args.core_model_url,
            "WORKFLOWS_REMOTE_EXECUTION_PIPELINE_DEPTH": str(depth),
        }
        command = [
            sys.executable,
            os.path.abspath(__file__),
            "--api-url",
            args.api_url,
            "--model-id",
            args.model_id,
            "--video-url",
            args.video_url,
            "--frames",
            str(args.frames),
            "--warmup",
            str(args.warmup),
            "--resize-width",
            str(args.resize_width),
            "--workflow",
            args.workflow,
            "--second-model-id",
            args.second_model_id,
            "--sam3-class-names",
            args.sam3_class_names,
            "--output-dir",
            str(args.output_dir),
            "--run-single-depth",
            str(depth),
        ]
        print(f"Running scenario depth={depth}...", flush=True)
        completed = subprocess.run(
            command, env=child_env, capture_output=True, text=True
        )
        result_lines = [
            line
            for line in completed.stdout.splitlines()
            if line.startswith(SCENARIO_RESULT_MARKER)
        ]
        if completed.returncode != 0 or not result_lines:
            raise RuntimeError(
                f"Scenario depth={depth} failed (exit {completed.returncode}).\n"
                f"stdout tail: {completed.stdout[-2000:]}\n"
                f"stderr tail: {completed.stderr[-2000:]}"
            )
        scenario_results.append(
            json.loads(result_lines[-1][len(SCENARIO_RESULT_MARKER) :])
        )
    return scenario_results


def main() -> None:
    args = parse_args()
    if "ROBOFLOW_API_KEY" not in os.environ:
        raise RuntimeError("ROBOFLOW_API_KEY must be set")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.run_single_depth is not None:
        result = run_scenario(args=args, depth=args.run_single_depth)
        print(SCENARIO_RESULT_MARKER + json.dumps(result), flush=True)
        return

    clip = prepare_clip(
        video_url=args.video_url,
        frames=args.frames,
        resize_width=args.resize_width,
        output_dir=args.output_dir,
    )
    (args.output_dir / "clip-metadata.json").write_text(json.dumps(clip))
    if args.warmup > 0:
        warmup_clip = prepare_clip(
            video_url=args.video_url,
            frames=args.warmup,
            resize_width=args.resize_width,
            output_dir=args.output_dir,
        )
        os.replace(
            warmup_clip["clip_path"], args.output_dir / f"warmup-{args.warmup}f.mp4"
        )

    depths = [int(depth.strip()) for depth in args.depths.split(",")]
    scenario_results = run_child_scenarios(args=args, depths=depths)

    baseline = next(
        (result for result in scenario_results if result["depth"] == 1), None
    )
    for result in scenario_results:
        if baseline is not None and result["depth"] != 1:
            result["tracker_parity_vs_sequential"] = check_tracker_parity(
                baseline=baseline, candidate=result
            )
            if baseline["wall_fps"]:
                result["speedup_vs_sequential"] = round(
                    result["wall_fps"] / baseline["wall_fps"], 2
                )

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "api_url": args.api_url,
        "workflow": args.workflow,
        "model_id": args.model_id,
        "video_url": args.video_url,
        "frames": args.frames,
        "warmup": args.warmup,
        "clip": clip,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "scenarios": [
            {
                key: value
                for key, value in result.items()
                if key != "tracker_ids_per_frame"
            }
            for result in scenario_results
        ],
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    for result in scenario_results:
        scenario_path = args.output_dir / f"scenario-depth-{result['depth']}.json"
        scenario_path.write_text(json.dumps(result, indent=2))

    print("SUMMARY_JSON_START")
    print(json.dumps(summary, indent=2))
    print("SUMMARY_JSON_END")
    print(f"Results written to {summary_path}")


if __name__ == "__main__":
    main()
