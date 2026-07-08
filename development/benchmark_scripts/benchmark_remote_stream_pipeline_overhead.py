"""Measure the scheduler-side overhead of stream-lookahead execution.

All remote calls are mocked to return instantly, so the numbers isolate the
execution engine's own costs — workflow compilation, frontier computation,
and the per-frame difference between the classic single-pass run and the
lookahead two-pass (deferred run + resume) — with zero network time.

Example:
    python development/benchmark_scripts/benchmark_lookahead_overhead.py \
        --frames 200 --compiles 30 --output /tmp/lookahead_overhead.json
"""

import argparse
import json
import statistics
import time
from datetime import datetime
from typing import Any, Callable, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np

WORKFLOW_SPECIFICATION = {
    "version": "1.0.0",
    "inputs": [{"type": "WorkflowImage", "name": "image"}],
    "steps": [
        {
            "type": "roboflow_core/roboflow_object_detection_model@v3",
            "name": "model",
            "images": "$inputs.image",
            "model_id": "some-project/1",
            "confidence_mode": "custom",
            "custom_confidence": 0.35,
        },
        {
            "type": "roboflow_core/byte_tracker@v3",
            "name": "byte_tracker",
            "image": "$inputs.image",
            "detections": "$steps.model.predictions",
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
    ],
}

CANNED_PREDICTION = {
    "predictions": [
        {
            "x": 100.0,
            "y": 100.0,
            "width": 50.0,
            "height": 50.0,
            "confidence": 0.9,
            "class": "car",
            "class_id": 0,
            "detection_id": "00000000-0000-0000-0000-000000000000",
        }
    ],
    "image": {"width": 640, "height": 360},
    "time": 0.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure compilation, frontier and per-frame scheduler overhead."
    )
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--compiles", type=int, default=30)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def time_repeated(task: Callable[[], Any], repeats: int) -> Dict[str, float]:
    durations = []
    for _ in range(repeats):
        started_at = time.perf_counter()
        task()
        durations.append((time.perf_counter() - started_at) * 1000)
    return {
        "repeats": repeats,
        "avg_ms": round(statistics.fmean(durations), 4),
        "median_ms": round(statistics.median(durations), 4),
        "min_ms": round(min(durations), 4),
        "max_ms": round(max(durations), 4),
    }


def build_engine():
    from inference.core.workflows.execution_engine.core import ExecutionEngine

    return ExecutionEngine.init(
        workflow_definition=WORKFLOW_SPECIFICATION,
        init_parameters={
            "workflows_core.api_key": "overhead-benchmark",
            "workflows_core.model_manager": MagicMock(),
            "workflows_core.step_execution_mode": _step_execution_mode(),
        },
    )


def _step_execution_mode():
    from inference.core.workflows.core_steps.common.entities import StepExecutionMode

    return StepExecutionMode.REMOTE


def make_video_frame(frame_id: int, image: np.ndarray):
    from inference.core.interfaces.camera.entities import VideoFrame

    return VideoFrame(
        image=image,
        frame_id=frame_id,
        frame_timestamp=datetime.fromtimestamp(1_700_000_000 + frame_id / 30),
        fps=30,
        measured_fps=30,
        source_id=0,
        comes_from_video_file=True,
    )


def run_per_frame_measurement(
    runner: Callable[[List[Any]], Any],
    image: np.ndarray,
    frames: int,
    warmup: int,
    drain: Callable[[], Any],
) -> Dict[str, float]:
    for frame_id in range(warmup):
        runner([make_video_frame(frame_id, image)])
    started_at = time.perf_counter()
    for frame_id in range(warmup, warmup + frames):
        runner([make_video_frame(frame_id, image)])
    drain()
    total_ms = (time.perf_counter() - started_at) * 1000
    return {"frames": frames, "avg_ms_per_frame": round(total_ms / frames, 4)}


def main() -> None:
    args = parse_args()
    from inference.core.interfaces.stream.model_handlers.workflows import (
        LookaheadPipelinedWorkflowRunner,
        WorkflowRunner,
        wrap_workflow_runner_for_stream_pipeline,
    )
    from inference.core.workflows.core_steps.models.roboflow.object_detection import (
        v3 as object_detection_v3,
    )
    from inference.core.workflows.execution_engine.v1.executor.core import (
        compute_stream_lookahead_frontier,
    )

    mock_client = MagicMock()
    mock_client.infer.return_value = CANNED_PREDICTION
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    results: Dict[str, Any] = {}

    with patch.object(
        object_detection_v3, "InferenceHTTPClient", MagicMock(return_value=mock_client)
    ), patch.object(object_detection_v3, "InferenceConfiguration", MagicMock()):
        results["compilation"] = time_repeated(build_engine, repeats=args.compiles)

        engine = build_engine()
        compiled_workflow = engine._engine._compiled_workflow
        results["frontier_computation"] = time_repeated(
            lambda: compute_stream_lookahead_frontier(workflow=compiled_workflow),
            repeats=1000,
        )

        def build_workflow_runner():
            return WorkflowRunner(
                workflows_parameters=None,
                execution_engine=engine,
                image_input_name="image",
                video_metadata_input_name="video_metadata",
            )

        object_detection_v3.WORKFLOWS_REMOTE_EXECUTION_PIPELINE_DEPTH = 1
        sequential_runner = build_workflow_runner()
        results["sequential_per_frame"] = run_per_frame_measurement(
            runner=sequential_runner,
            image=image,
            frames=args.frames,
            warmup=args.warmup,
            drain=lambda: None,
        )

        object_detection_v3.WORKFLOWS_REMOTE_EXECUTION_PIPELINE_DEPTH = args.depth
        lookahead_engine = build_engine()
        lookahead_runner = wrap_workflow_runner_for_stream_pipeline(
            workflow_runner=WorkflowRunner(
                workflows_parameters=None,
                execution_engine=lookahead_engine,
                image_input_name="image",
                video_metadata_input_name="video_metadata",
            ),
            execution_engine=lookahead_engine,
        )
        if not isinstance(lookahead_runner, LookaheadPipelinedWorkflowRunner):
            raise RuntimeError(
                f"Expected the lookahead runner to activate, got "
                f"{type(lookahead_runner).__name__} — results would be invalid."
            )
        try:
            results["lookahead_per_frame"] = run_per_frame_measurement(
                runner=lookahead_runner,
                image=image,
                frames=args.frames,
                warmup=args.warmup,
                drain=lookahead_runner.flush,
            )
        finally:
            lookahead_runner.close()

    results["scheduler_overhead_ms_per_frame"] = round(
        results["lookahead_per_frame"]["avg_ms_per_frame"]
        - results["sequential_per_frame"]["avg_ms_per_frame"],
        4,
    )
    print("OVERHEAD_RESULT " + json.dumps(results, indent=2))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
