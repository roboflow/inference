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
import json
import os
import sys
import time
from collections import deque
from typing import Any, List, Optional

import supervision as sv

from inference import InferencePipeline
from inference.core.interfaces.stream.watchdog import BasePipelineWatchDog

# Modules whose module-level globals we patch to wire in the workflow profiler.
# Both functions read these names as module globals at call time, so patching the
# attribute after import is enough to steer the behaviour.
import inference.core.interfaces.stream.inference_pipeline as _ip_module
import inference.core.interfaces.stream.utils as _ip_utils
from inference.core.workflows.execution_engine.profiling.core import (
    BaseWorkflowsProfiler,
)


def _write_trace(path: str, trace: List[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(trace, f)


class _FirstNRunsProfiler(BaseWorkflowsProfiler):
    """Captures only the FIRST `max_runs_in_buffer` workflow runs (~ first N frames),
    then freezes: every later event is dropped, so the trace stays small and stable
    no matter how long the pipeline keeps running.

    Unlike the stock profiler (a rolling `deque` that keeps the *last* N runs), this
    keeps the *first* N. It is flushed to disk the moment it freezes — so a later hard
    kill still keeps the trace — and again on pipeline end via `on_pipeline_end`.
    """

    # Set by `_enable_workflow_profiler` before the pipeline starts.
    _trace_path: Optional[str] = None

    @classmethod
    def init(cls, max_runs_in_buffer: int = 50, **kwargs) -> "_FirstNRunsProfiler":
        inst = cls(runs_buffer=deque())  # unbounded; we cap ourselves by freezing
        inst._max_runs = max_runs_in_buffer
        inst._frozen = False
        return inst

    def start_workflow_run(self) -> None:
        if self._frozen:
            return
        super().start_workflow_run()

    def end_workflow_run(self) -> None:
        if self._frozen:
            return
        super().end_workflow_run()
        if len(self._runs_buffer) >= self._max_runs:
            self._frozen = True
            self._current_run_events = []
            if self._trace_path is not None:
                _write_trace(self._trace_path, self.export_trace())
                print(
                    f"[profiler] captured first {self._max_runs} frame(s) -> "
                    f"{self._trace_path}",
                    flush=True,
                )

    def _add_event(self, *args, **kwargs) -> None:
        if self._frozen:
            return
        super()._add_event(*args, **kwargs)


def _enable_workflow_profiler(trace_path: str, max_frames: int) -> None:
    """Turn on the workflow profiler so `init_with_workflow` builds our first-N-runs
    profiler and dumps the Chrome trace to the exact `trace_path` we ask for."""
    _FirstNRunsProfiler._trace_path = trace_path
    # `init_with_workflow` reads these two as module globals at call time.
    _ip_module.ENABLE_WORKFLOWS_PROFILING = True
    _ip_module.WORKFLOWS_PROFILER_BUFFER_SIZE = max_frames
    _ip_module.BaseWorkflowsProfiler = _FirstNRunsProfiler
    # `on_pipeline_end` (in utils) gates on this flag and calls this save function on
    # shutdown — point it at our exact path instead of the timestamped default name.
    _ip_utils.ENABLE_WORKFLOWS_PROFILING = True
    _ip_utils.save_workflows_profiler_trace = (
        lambda directory, profiler_trace: _write_trace(trace_path, profiler_trace)
    )


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
        "-n",
        "--n",
        type=int,
        default=1,
        dest="n",
        help="Number of concurrent streams to run, all decoding the SAME --video "
        "(multi-source / multi-camera throughput test). The video is passed N times "
        "as the pipeline's video_reference; aggregate (and per-stream) FPS is reported.",
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
    parser.add_argument(
        "--profile-trace",
        default=None,
        metavar="PATH",
        help="Enable the workflow profiler and dump a Chrome trace (JSON) to this exact "
        "file path (e.g. ./traces/my_run.json). View it in chrome://tracing or "
        "https://ui.perfetto.dev. Leave unset to disable profiling.",
    )
    parser.add_argument(
        "--profile-frames",
        type=int,
        default=50,
        help="Number of leading frames (workflow runs) to capture before the profiler "
        "freezes. Keeps the trace small; later frames are not recorded. Only used when "
        "--profile-trace is given.",
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
    if args.n < 1:
        raise SystemExit("--n must be >= 1")
    if args.profile_trace:
        if args.profile_frames < 1:
            raise SystemExit("--profile-frames must be >= 1")
        _enable_workflow_profiler(
            trace_path=args.profile_trace, max_frames=args.profile_frames
        )
        print(
            f"Profiler enabled: capturing first {args.profile_frames} frame(s) -> "
            f"'{args.profile_trace}'"
        )

    # Multiply the same video into N concurrent sources. A single reference keeps the
    # exact legacy behaviour; a list makes InferencePipeline run one VideoSource per copy.
    video_reference = args.video if args.n == 1 else [args.video] * args.n

    # Only steer `model_id` when explicitly provided, so the workflow keeps its own
    # default otherwise.
    workflows_parameters = (
        {"model_id": args.model_id} if args.model_id is not None else None
    )

    counter = FPSCounter()
    watchdog = BasePipelineWatchDog()
    pipeline = InferencePipeline.init_with_workflow(
        video_reference=video_reference,
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
        f"x{args.n} stream(s) ({cap}){model}\n"
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

    per_stream = f"  ({counter.overall_fps / args.n:.2f}/stream)" if args.n > 1 else ""
    print("\n=== Summary ===")
    print(f"streams (same video) : {args.n}")
    print(f"frames processed     : {counter.frames}  (aggregate across streams)")
    print(f"total wall time      : {elapsed:.2f} s (includes model load / warm-up)")
    print(f"average processing FPS: {counter.overall_fps:.2f} aggregate{per_stream}")
    if args.profile_trace:
        print(
            f"profiler trace        : {os.path.abspath(args.profile_trace)} "
            f"(first {args.profile_frames} frame(s))"
        )


if __name__ == "__main__":
    main()
