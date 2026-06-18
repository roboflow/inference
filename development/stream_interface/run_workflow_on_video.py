#!/usr/bin/env python3
"""Run an InferencePipeline against a registered Roboflow workflow over a local video
file and report the processing FPS (throughput).

Every frame of the video is streamed through the workflow as fast as the workflow
allows (no max-fps cap by default), and FPS is counted synchronously in the prediction
sink — both a rolling FPS (supervision's FPSMonitor) and a cumulative average.

The workflow is fetched from the Roboflow platform by workspace + id, so an API key is
required (via --api-key or the ROBOFLOW_API_KEY / API_KEY environment variable).

Examples:
    # minimal
    python development/stream_interface/run_workflow_on_video.py \\
        --workspace my-workspace --workflow-id my-workflow --video /path/clip.mp4

    # steer an optional `model_id` workflow parameter from the CLI
    python development/stream_interface/run_workflow_on_video.py \\
        --workspace my-workspace --workflow-id my-workflow --video /path/clip.mp4 \\
        --model-id yolov8n-640
"""
import argparse
import os
import sys
import time
from typing import Any, Optional

import supervision as sv

from inference import InferencePipeline
from inference.core.interfaces.stream.watchdog import BasePipelineWatchDog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a registered Roboflow workflow over a local video file and count FPS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--workspace", required=True, help="Roboflow workspace name (the URL slug)."
    )
    parser.add_argument(
        "--workflow-id", required=True, help="Registered workflow id."
    )
    parser.add_argument(
        "--video", required=True, help="Path to a local video file."
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Optional value for the workflow's `model_id` parameter "
        "(only sent when provided; otherwise the workflow's own default is used).",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("ROBOFLOW_API_KEY") or os.environ.get("API_KEY"),
        help="Roboflow API key (defaults to the ROBOFLOW_API_KEY / API_KEY env var).",
    )
    parser.add_argument(
        "--image-input-name",
        default="image",
        help="Name of the workflow's image input that frames are injected into.",
    )
    parser.add_argument(
        "--max-fps",
        type=float,
        default=None,
        help="Cap processing FPS. Leave unset to measure full throughput.",
    )
    parser.add_argument(
        "--report-every",
        type=int,
        default=30,
        help="Print a rolling FPS line every N processed frames (0 to disable).",
    )
    return parser.parse_args()


class FPSCounter:
    """Counts processed frames and reports a rolling FPS (supervision's FPSMonitor)
    plus a cumulative average over the whole run."""

    def __init__(self) -> None:
        self._monitor = sv.FPSMonitor()
        self.frames = 0
        self._start: Optional[float] = None

    def tick(self) -> None:
        if self._start is None:
            self._start = time.monotonic()
        self._monitor.tick()
        self.frames += 1

    @property
    def rolling_fps(self) -> float:
        # supervision >= 0.18 exposes `.fps`; older versions are callable.
        return float(self._monitor.fps if hasattr(self._monitor, "fps") else self._monitor())

    @property
    def overall_fps(self) -> float:
        if self._start is None:
            return 0.0
        elapsed = time.monotonic() - self._start
        return self.frames / elapsed if elapsed > 0 else 0.0


def build_sink(counter: FPSCounter, report_every: int):
    def sink(predictions: Any, video_frames: Any) -> None:
        # A single video source delivers one (prediction, frame) per call; the list
        # form is used for multi-source pipelines. Normalise to a list either way.
        if not isinstance(predictions, list):
            predictions, video_frames = [predictions], [video_frames]
        for _prediction, frame in zip(predictions, video_frames):
            if frame is None:
                continue
            counter.tick()
            if report_every and counter.frames % report_every == 0:
                print(
                    f"[{counter.frames:>7} frames] rolling FPS: {counter.rolling_fps:6.1f} "
                    f"| avg FPS: {counter.overall_fps:6.1f}",
                    flush=True,
                )

    return sink


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise SystemExit(
            "A Roboflow API key is required (registered workflows are fetched from the "
            "platform). Pass --api-key or set ROBOFLOW_API_KEY."
        )
    if not os.path.isfile(args.video):
        raise SystemExit(f"Video file not found: {args.video}")

    # Only steer `model_id` when explicitly provided, so the workflow keeps its own
    # default otherwise.
    workflows_parameters = (
        {"model_id": args.model_id} if args.model_id is not None else None
    )

    counter = FPSCounter()
    watchdog = BasePipelineWatchDog()
    pipeline = InferencePipeline.init_with_workflow(
        video_reference=args.video,
        workspace_name=args.workspace,
        workflow_id=args.workflow_id,
        api_key=args.api_key,
        image_input_name=args.image_input_name,
        workflows_parameters=workflows_parameters,
        on_prediction=build_sink(counter, args.report_every),
        max_fps=args.max_fps,  # None => run as fast as the workflow allows
        watchdog=watchdog,
    )

    cap = f"capped at {args.max_fps} FPS" if args.max_fps else "uncapped"
    model = f" model_id='{args.model_id}'" if args.model_id else ""
    print(
        f"Running '{args.workspace}/{args.workflow_id}' over '{args.video}' "
        f"({cap}){model}\n"
    )
    wall_start = time.monotonic()
    try:
        pipeline.start()  # blocks (use_main_thread) until the video file is exhausted
        pipeline.join()
    except KeyboardInterrupt:
        print("\nInterrupted — terminating pipeline...")
        pipeline.terminate()
        pipeline.join()
    elapsed = time.monotonic() - wall_start

    print("\n=== Summary ===")
    print(f"frames processed     : {counter.frames}")
    print(f"total wall time      : {elapsed:.2f} s (includes model load / warm-up)")
    print(f"average processing FPS: {counter.overall_fps:.2f}")


if __name__ == "__main__":
    main()
