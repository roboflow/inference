#!/usr/bin/env python3
"""Run an InferencePipeline against a registered Roboflow workflow over one or more
video sources (local files or live streams) and report the processing FPS (throughput).

Every frame of every source is streamed through the workflow as fast as the workflow
allows (no max-fps cap by default), and FPS is counted synchronously in the prediction
sink — both a rolling FPS (supervision's FPSMonitor) and a cumulative average.

The workflow is fetched from the Roboflow platform by workspace + id, so an API key is
required (via --api-key or the ROBOFLOW_API_KEY / API_KEY environment variable).

Sources: `--video` accepts a local file path or a stream reference (rtsp://...,
/dev/video0, ...) and may be repeated for multiple concurrent sources. `--n` instead
replicates a SINGLE --video into N identical streams — the two forms are mutually
exclusive. Live streams run until Ctrl+C.

Examples:
    # minimal
    python development/stream_interface/run_workflow_on_video.py \\
        --workspace my-workspace --workflow-id my-workflow --video /path/clip.mp4

    # the same file decoded by 4 concurrent streams (multi-camera throughput test)
    python development/stream_interface/run_workflow_on_video.py \\
        --workspace my-workspace --workflow-id my-workflow --video /path/clip.mp4 --n 4

    # two different live sources
    python development/stream_interface/run_workflow_on_video.py \\
        --workspace my-workspace --workflow-id my-workflow \\
        --video rtsp://camera-1/stream --video rtsp://camera-2/stream

    # steer an optional `model_id` workflow parameter from the CLI
    python development/stream_interface/run_workflow_on_video.py \\
        --workspace my-workspace --workflow-id my-workflow --video /path/clip.mp4 \\
        --model-id yolov8n-640
"""

import argparse
import json
import os
import time
from collections import deque
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import supervision as sv

# Modules whose module-level globals we patch to wire in the workflow profiler.
# Both functions read these names as module globals at call time, so patching the
# attribute after import is enough to steer the behaviour.
import inference.core.interfaces.stream.inference_pipeline as _ip_module
import inference.core.interfaces.stream.utils as _ip_utils
from inference import InferencePipeline
from inference.core.interfaces.stream.watchdog import BasePipelineWatchDog
from inference.core.utils.drawing import create_tiles
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
        description="Run a registered Roboflow workflow over video files / streams and count FPS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--workspace", required=True, help="Roboflow workspace name (the URL slug)."
    )
    parser.add_argument("--workflow-id", required=True, help="Registered workflow id.")
    parser.add_argument(
        "--video",
        required=True,
        action="append",
        help="Video source: a local file path or a stream reference (rtsp://..., "
        "/dev/video0, ...). Repeat the flag for multiple concurrent sources "
        "(mutually exclusive with --n).",
    )
    parser.add_argument(
        "-n",
        "--n",
        type=int,
        default=None,
        dest="n",
        help="Number of concurrent streams to run, all decoding the SAME single "
        "--video (multi-source / multi-camera throughput test). The source is passed "
        "N times as the pipeline's video_reference; aggregate (and per-stream) FPS is "
        "reported. Mutually exclusive with passing multiple --video arguments.",
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
    parser.add_argument(
        "--output-video",
        default=None,
        metavar="PATH",
        help="Optionally write the visualization video to this exact path (e.g. "
        "./out.mp4). Requires --output-key. With a single stream, source [0] is "
        "written verbatim; with multiple streams, every source's output is tiled "
        "into one grid frame (inference's create_tiles) per processing round.",
    )
    parser.add_argument(
        "--output-key",
        default=None,
        help="Name of the workflow output field that holds the visualization image "
        "(e.g. 'label_visualization_output'). Required when --output-video is given.",
    )
    parser.add_argument(
        "--output-fps",
        type=float,
        default=None,
        help="FPS for the written output video. Defaults to the source video's FPS "
        "(falling back to 30 if it cannot be read). Only used with --output-video.",
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
        return float(
            self._monitor.fps if hasattr(self._monitor, "fps") else self._monitor()
        )

    @property
    def overall_fps(self) -> float:
        if self._start is None:
            return 0.0
        elapsed = time.monotonic() - self._start
        return self.frames / elapsed if elapsed > 0 else 0.0


class VisualizationVideoWriter:
    """Lazily opens a ``cv2.VideoWriter`` on the first frame (so it can size itself to
    the visualization output) and writes subsequent BGR frames to a video file.

    Frame size is fixed at the first frame; any later frame of a different size is
    resized to match (cv2 requires a constant frame size).
    """

    def __init__(self, path: str, fps: float) -> None:
        self._path = path
        self._fps = fps if fps and fps > 0 else 30.0
        self._writer: Optional[cv2.VideoWriter] = None
        self._size: Optional[tuple] = None
        self.frames_written = 0

    def write(self, image_bgr: np.ndarray) -> None:
        height, width = image_bgr.shape[:2]
        if self._writer is None:
            os.makedirs(
                os.path.dirname(os.path.abspath(self._path)) or ".", exist_ok=True
            )
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(
                self._path, fourcc, self._fps, (width, height)
            )
            if not self._writer.isOpened():
                raise SystemExit(
                    f"Could not open a VideoWriter for '{self._path}' "
                    f"({width}x{height} @ {self._fps:.2f} fps). Check the path/extension "
                    "(try .mp4) and codec availability."
                )
            self._size = (width, height)
        elif (width, height) != self._size:
            image_bgr = cv2.resize(image_bgr, self._size)
        self._writer.write(np.ascontiguousarray(image_bgr, dtype=np.uint8))
        self.frames_written += 1

    def release(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None


def _to_bgr_image(value: Any) -> Optional[np.ndarray]:
    """Extract an HWC BGR uint8 image from a workflow output value.

    Visualization outputs arrive (unserialized) as ``WorkflowImageData`` whose
    ``.numpy_image`` is HWC BGR uint8 (cv2-native, exactly what VideoWriter wants);
    a bare ndarray is taken as-is. Anything else returns ``None``.
    """
    numpy_image = getattr(value, "numpy_image", None)
    if numpy_image is not None:
        return numpy_image
    if isinstance(value, np.ndarray):
        return value
    return None


class OutputCapture:
    """Pulls the named visualization output out of workflow results and feeds it to
    the video writer.

    With a single stream, source [0]'s frames are written directly (legacy
    behaviour). With multiple streams, the latest image of EVERY source is kept and
    one `create_tiles` mosaic is written per sink round (see `end_round`) — the
    first frame waits until every source has delivered once so the grid never
    changes size mid-video, and a source that ends early keeps showing its last
    frame. Warns once (then stays quiet) on a missing key or a non-image value so a
    misconfigured run is obvious without spamming per frame."""

    def __init__(
        self,
        writer: VisualizationVideoWriter,
        output_key: str,
        num_streams: int = 1,
    ) -> None:
        self.writer = writer
        self.output_key = output_key
        self._num_streams = num_streams
        self._latest: Dict[int, np.ndarray] = {}
        self._round_dirty = False
        self._warned_missing = False
        self._warned_type = False

    def handle(self, prediction: Any, source_id: int) -> None:
        if not isinstance(prediction, dict):
            return
        if self.output_key not in prediction:
            if not self._warned_missing:
                self._warned_missing = True
                print(
                    f"[output-capture] WARNING: output '{self.output_key}' not found in "
                    f"the workflow result. Available outputs: {sorted(prediction.keys())}. "
                    "No video will be written.",
                    flush=True,
                )
            return
        image = _to_bgr_image(prediction[self.output_key])
        if image is None:
            if not self._warned_type:
                self._warned_type = True
                print(
                    f"[output-capture] WARNING: output '{self.output_key}' is "
                    f"{type(prediction[self.output_key]).__name__}, not an image; "
                    "cannot write video.",
                    flush=True,
                )
            return
        if self._num_streams == 1:
            self.writer.write(image)
            return
        self._latest[source_id] = image
        self._round_dirty = True

    def end_round(self) -> None:
        """Multi-stream only: write ONE tiled frame per sink round in which any
        source updated, once all sources have been seen (stable tile grid)."""
        if self._num_streams == 1 or not self._round_dirty:
            return
        self._round_dirty = False
        if len(self._latest) < self._num_streams:
            return
        tile = create_tiles(
            images=[self._latest[source_id] for source_id in sorted(self._latest)]
        )
        self.writer.write(tile)


def build_sink(
    counter: FPSCounter,
    report_every: int,
    capture: Optional[OutputCapture] = None,
):
    def sink(predictions: Any, video_frames: Any) -> None:
        # A single video source delivers one (prediction, frame) per call; the list
        # form is used for multi-source pipelines. Normalise to a list either way.
        if not isinstance(predictions, list):
            predictions, video_frames = [predictions], [video_frames]
        for prediction, frame in zip(predictions, video_frames):
            if frame is None:
                continue
            counter.tick()
            if capture is not None:
                # source_id is an int for multi-source pipelines and None for a
                # single source - normalise so the capture can key its tiles.
                source_id = getattr(frame, "source_id", None)
                capture.handle(prediction, source_id if source_id is not None else 0)
            if report_every and counter.frames % report_every == 0:
                print(
                    f"[{counter.frames:>7} frames] rolling FPS: {counter.rolling_fps:6.1f} "
                    f"| avg FPS: {counter.overall_fps:6.1f}",
                    flush=True,
                )
        if capture is not None:
            capture.end_round()

    return sink


def _looks_like_stream(reference: str) -> bool:
    """Anything with a URI scheme (rtsp://, http://, ...) or a V4L2 device path is a
    live stream/camera; only bare paths are validated (and FPS-probed) as local
    files."""
    return "://" in reference or reference.startswith("/dev/video")


def _source_video_fps(video_path: str) -> Optional[float]:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps and fps > 0 else None


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise SystemExit(
            "A Roboflow API key is required (registered workflows are fetched from the "
            "platform). Pass --api-key or set ROBOFLOW_API_KEY."
        )
    references: List[str] = args.video
    if args.n is not None:
        if len(references) > 1:
            raise SystemExit(
                "--n and multiple --video arguments are mutually exclusive: --n "
                "replicates a SINGLE --video into N identical streams, while "
                "repeating --video enumerates distinct sources explicitly."
            )
        if args.n < 1:
            raise SystemExit("--n must be >= 1")
    for reference in references:
        if not _looks_like_stream(reference) and not os.path.isfile(reference):
            raise SystemExit(f"Video file not found: {reference}")
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

    # --n multiplies the single source into N concurrent copies; repeated --video
    # enumerates distinct sources. A single reference keeps the exact legacy
    # behaviour; a list makes InferencePipeline run one VideoSource per entry.
    video_references = (
        references * args.n if args.n is not None and args.n > 1 else list(references)
    )
    num_streams = len(video_references)
    video_reference = video_references[0] if num_streams == 1 else video_references

    # Optional capture of the visualization output: source [0] verbatim for a single
    # stream, a create_tiles mosaic of every source for multi-stream runs.
    capture: Optional[OutputCapture] = None
    if args.output_video:
        if not args.output_key:
            raise SystemExit(
                "--output-key (the workflow output field holding the visualization "
                "image) is required when --output-video is given."
            )
        # cv2.VideoCapture can block on live sources, so only local files are
        # FPS-probed for the writer default; streams fall back to --output-fps / 30.
        probed_fps = (
            _source_video_fps(references[0])
            if not _looks_like_stream(references[0])
            else None
        )
        out_fps = args.output_fps or probed_fps or 30.0
        capture = OutputCapture(
            writer=VisualizationVideoWriter(args.output_video, out_fps),
            output_key=args.output_key,
            num_streams=num_streams,
        )
        target_desc = (
            "source [0]"
            if num_streams == 1
            else f"all {num_streams} sources (create_tiles mosaic)"
        )
        print(
            f"Capturing workflow output '{args.output_key}' from {target_desc} -> "
            f"'{args.output_video}' @ {out_fps:.2f} fps"
        )
    elif args.output_key:
        raise SystemExit("--output-key requires --output-video to also be given.")

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
        on_prediction=build_sink(counter, args.report_every, capture),
        max_fps=args.max_fps,  # None => run as fast as the workflow allows
        watchdog=watchdog,
    )

    cap = f"capped at {args.max_fps} FPS" if args.max_fps else "uncapped"
    model = f" model_id='{args.model_id}'" if args.model_id else ""
    sources_desc = (
        f"'{video_references[0]}' x{num_streams} stream(s)"
        if len(set(video_references)) == 1
        else f"{num_streams} sources: " + ", ".join(f"'{r}'" for r in video_references)
    )
    print(
        f"Running '{args.workspace}/{args.workflow_id}' over {sources_desc} "
        f"({cap}){model}\n"
    )
    wall_start = time.monotonic()
    try:
        try:
            # Blocks (use_main_thread) until file sources are exhausted; live
            # streams run until Ctrl+C.
            pipeline.start()
            pipeline.join()
        except KeyboardInterrupt:
            print("\nInterrupted — terminating pipeline...")
            pipeline.terminate()
            pipeline.join()
    finally:
        # Always finalise the output video — even on interrupt — so it stays playable.
        if capture is not None:
            capture.writer.release()
    elapsed = time.monotonic() - wall_start

    per_stream = (
        f"  ({counter.overall_fps / num_streams:.2f}/stream)" if num_streams > 1 else ""
    )
    same_video = (
        "  (same source)" if num_streams > 1 and len(set(video_references)) == 1 else ""
    )
    print("\n=== Summary ===")
    print(f"streams              : {num_streams}{same_video}")
    print(f"frames processed     : {counter.frames}  (aggregate across streams)")
    print(f"total wall time      : {elapsed:.2f} s (includes model load / warm-up)")
    print(f"average processing FPS: {counter.overall_fps:.2f} aggregate{per_stream}")
    if args.profile_trace:
        print(
            f"profiler trace        : {os.path.abspath(args.profile_trace)} "
            f"(first {args.profile_frames} frame(s))"
        )
    if capture is not None:
        captured_desc = (
            "from source [0]"
            if num_streams == 1
            else f"tiled across {num_streams} sources"
        )
        print(
            f"output video          : {os.path.abspath(args.output_video)} "
            f"({capture.writer.frames_written} frame(s) {captured_desc})"
        )


if __name__ == "__main__":
    main()
