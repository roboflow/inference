"""Tests of the live runner: slots, statistics, lifecycle and the real graph.

Fake captures, fake sessions and a fake window replace the camera, the model
and the GUI; the last tests run the compiled workflow with a fake native
model and the real painters. Run from the repository root::

    PYTHONPATH=.:workflows:inference_models:stream_vision python -m pytest \
        development/workflows-2.0/08-live-detection/tests
"""

import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Union

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.append(str(EXAMPLE_DIR))

import cv2  # noqa: E402
import live  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from live import (  # noqa: E402
    Frame,
    LatestSlot,
    StillImage,
    frame_to_image,
    image_to_bgr,
    run_live,
)
from roboflow_workflows.execution_engine.v2.pipelining import (  # noqa: E402
    PipelineFullError,
)
from stats import (  # noqa: E402
    STAGES,
    LiveStats,
    RollingRate,
    RollingSamples,
    overlay_lines,
)

from inference_models.models.base.object_detection import Detections  # noqa: E402

WAIT_SECONDS = 10


def _frame_pixels(index: int, *, size_hw=(6, 8)) -> np.ndarray:
    pixels = np.zeros((*size_hw, 3), dtype=np.uint8)
    pixels[0, 0] = (index % 256, 0, 0)

    return pixels


class FakeCapture:
    """VideoCapture-like: ``count`` frames, optionally paced, then end."""

    def __init__(self, count: int, *, interval: float = 0.0):
        self.count = count
        self.interval = interval
        self.reads = 0
        self.released = threading.Event()

    def read(self):
        if self.interval:
            time.sleep(self.interval)
        if self.reads >= self.count:
            return False, None

        pixels = _frame_pixels(self.reads)
        self.reads += 1

        return True, pixels

    def release(self) -> None:
        self.released.set()


def _fake_result(inputs: Dict[str, Any]) -> Any:
    values = {
        "annotated": inputs["image"],
        "predictions": [],
        **{stage: 1.0 for stage in STAGES},
    }
    result = SimpleNamespace(
        selections={name: {name: name} for name in values},
        outputs=SimpleNamespace(data=values),
    )

    return result


class FakeSession:
    """``run`` and ``pipeline`` with optional delay, failure and a hold gate.

    ``delay`` is seconds, or seconds by call number. ``stalled_workers`` pool
    workers of the pipeline never become idle.
    """

    def __init__(
        self,
        *,
        delay: Union[float, Callable[[int], float]] = 0.0,
        fail_at: Optional[int] = None,
        hold: Optional[threading.Event] = None,
        stalled_workers: int = 0,
    ):
        self.delay = delay if callable(delay) else lambda _: delay
        self.fail_at = fail_at
        self.hold = hold
        self.stalled_workers = stalled_workers
        self.started = threading.Event()
        self.finished: List[str] = []
        self.peak_outstanding = 0
        self._calls = 0
        self._lock = threading.Lock()

    def run(self, inputs: Dict[str, Any]) -> Any:
        with self._lock:
            call = self._calls
            self._calls += 1
        self.started.set()
        if self.hold is not None:
            self.hold.wait(WAIT_SECONDS)
        time.sleep(self.delay(call))
        if call == self.fail_at:
            raise RuntimeError(f"step failed on call {call}")

        self.finished.append(inputs["image"].image_id)
        result = _fake_result(inputs)

        return result

    def pipeline(self, *, options: Any) -> "FakePipeline":
        pipeline = FakePipeline(
            self,
            max_in_flight=options.max_in_flight,
            stalled_workers=self.stalled_workers,
        )

        return pipeline


class FakePipeline:
    """Like ``PassivePipeline.submit``: wait up to ``timeout`` for an idle worker."""

    def __init__(
        self, session: FakeSession, *, max_in_flight: int, stalled_workers: int
    ):
        self.session = session
        self.outstanding = 0
        self._idle = threading.Semaphore(max_in_flight - stalled_workers)
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_in_flight)

    def __enter__(self) -> "FakePipeline":
        return self

    def __exit__(self, *_: Any) -> None:
        self._executor.shutdown(wait=True)

    def submit(self, inputs: Dict[str, Any], *, timeout: float) -> Future:
        if not self._idle.acquire(timeout=timeout):
            raise PipelineFullError("All pipeline workers are busy")

        with self._lock:
            self.outstanding += 1
            self.session.peak_outstanding = max(
                self.session.peak_outstanding, self.outstanding
            )

        future = self._executor.submit(self.session.run, inputs)
        future.add_done_callback(self._release)

        return future

    def _release(self, _: Future) -> None:
        with self._lock:
            self.outstanding -= 1
        self._idle.release()


class FakeWindow:
    """Replaces the cv2 GUI calls; keys and window state are scripted.

    ``shown`` holds every imshow, the startup placeholder first. Keys and the
    close (after ``closes_after`` imshow calls) wait for ``ready`` when given;
    ``on_close`` runs when the window first reports closed.
    """

    def __init__(
        self,
        *,
        keys: Optional[List[int]] = None,
        closes_after: int = 0,
        ready: Optional[threading.Event] = None,
        on_close: Callable[[], None] = lambda: None,
        destroy_error: Optional[Exception] = None,
    ):
        self.keys = list(keys or [])
        self.closes_after = closes_after
        self.ready = ready
        self.on_close = on_close
        self.destroy_error = destroy_error
        self.shown: List[np.ndarray] = []
        self.destroyed = False
        self._closed = False

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(cv2, "namedWindow", lambda *_: None)
        monkeypatch.setattr(cv2, "imshow", lambda _, pixels: self.shown.append(pixels))
        monkeypatch.setattr(cv2, "waitKey", self._wait_key)
        monkeypatch.setattr(cv2, "getWindowProperty", self._visible)
        monkeypatch.setattr(cv2, "destroyAllWindows", self._destroy)

    def _is_ready(self) -> bool:
        ready = self.ready is None or self.ready.is_set()

        return ready

    def _wait_key(self, _: int) -> int:
        time.sleep(0.001)
        key = self.keys.pop(0) if self.keys and self._is_ready() else -1

        return key

    def _visible(self, *_: Any) -> float:
        if self.closes_after and len(self.shown) >= self.closes_after:
            if self._is_ready() and not self._closed:
                self._closed = True
                self.on_close()
        visible = 0.0 if self._closed else 1.0

        return visible

    def _destroy(self) -> None:
        self.destroyed = True
        if self.destroy_error is not None:
            raise self.destroy_error


def _run(session: Any, capture: Any, **options: Any) -> live.LiveSummary:
    settings = {
        "lossless": True,
        "confidence": 0.5,
        "max_in_flight": None,
        "header": "test",
        "display": False,
    }
    settings.update(options)
    summary = run_live(session, capture=capture, **settings)

    return summary


def test_latest_slot_keeps_only_the_newest_item() -> None:
    slot: LatestSlot[int] = LatestSlot()

    replaced = [slot.put(value) for value in range(3)]

    assert replaced == [False, True, True]
    assert slot.take(timeout=0) == 2
    assert slot.take(timeout=0) is None


def test_waiting_put_resumes_after_take_and_after_close() -> None:
    slot: LatestSlot[int] = LatestSlot()
    slot.put(1)
    second = threading.Thread(
        target=slot.put, args=(2,), kwargs={"wait_until_free": True}
    )
    second.start()

    time.sleep(0.05)
    assert second.is_alive()
    assert slot.take(timeout=0) == 1
    second.join(WAIT_SECONDS)
    assert slot.take(timeout=0) == 2

    slot.put(3)
    blocked = threading.Thread(
        target=slot.put, args=(4,), kwargs={"wait_until_free": True}
    )
    blocked.start()
    slot.close()
    blocked.join(WAIT_SECONDS)
    assert not blocked.is_alive()
    assert slot.take(timeout=0) == 3
    assert slot.exhausted


def test_wake_returns_take_without_an_item() -> None:
    slot: LatestSlot[int] = LatestSlot()
    threading.Timer(0.05, slot.wake).start()

    started = time.perf_counter()
    item = slot.take(timeout=WAIT_SECONDS)

    assert item is None
    assert time.perf_counter() - started < WAIT_SECONDS / 2


def test_rate_counts_only_the_window_and_memory_is_bounded() -> None:
    rate = RollingRate(window_seconds=1.0, max_events=10)
    for second in range(100):
        rate.add(second * 0.1)

    assert rate.total == 100
    assert len(rate._times) == 10
    assert rate.per_second(now=9.9) == pytest.approx(10.0)
    assert rate.per_second(now=100.0) is None


def test_overlay_before_any_result_shows_dashes() -> None:
    lines = overlay_lines(LiveStats().snapshot(time.perf_counter()), header="h")

    assert lines[0] == "h"
    assert "output - FPS" in lines[1]
    assert "0x0" in lines[4]


def test_frame_boundaries_swap_bgr_and_rgb_once_each() -> None:
    pixels_bgr = np.zeros((2, 3, 3), dtype=np.uint8)
    pixels_bgr[0, 1] = (255, 0, 0)  # blue in BGR
    pixels_bgr[1, 2] = (0, 0, 255)  # red in BGR

    image = frame_to_image(Frame(index=7, pixels_bgr=pixels_bgr, captured_at=0.0))

    assert image.image_id == "frame-7"
    assert image.tensor_image.is_contiguous()
    assert image.tensor_image[:, 0, 1].tolist() == [0, 0, 255]
    assert image.tensor_image[:, 1, 2].tolist() == [255, 0, 0]
    assert np.array_equal(image_to_bgr(image.tensor_image), pixels_bgr)


@pytest.mark.parametrize("max_in_flight", [None, 3])
def test_file_frames_are_all_delivered_in_order(
    max_in_flight: Optional[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    delivered: List[int] = []
    deliver = live.GraphWorker._deliver

    def record(worker: Any, frame: Frame, *, result: Any) -> None:
        delivered.append(frame.index)
        deliver(worker, frame, result=result)

    monkeypatch.setattr(live.GraphWorker, "_deliver", record)
    # Every third run is slow, so the fake pipeline completes out of order.
    session = FakeSession(delay=lambda call: 0.02 if call % 3 == 0 else 0.001)
    capture = FakeCapture(25)

    summary = _run(session, capture, max_in_flight=max_in_flight)

    assert summary.reason == "end of input"
    assert delivered == list(range(25))
    completed = [int(name[len("frame-") :]) for name in session.finished]
    assert sorted(completed) == delivered
    if max_in_flight is not None:
        assert completed != delivered, "completions were out of order"
    assert summary.snapshot["results_completed"] == 25
    assert summary.snapshot["drops"]["capture_overwritten"] == 0
    assert summary.last_result.frame_index == 24
    assert capture.released.is_set()


def test_pipeline_holds_at_most_max_in_flight_runs() -> None:
    session = FakeSession(delay=0.01)

    summary = _run(session, FakeCapture(30), max_in_flight=3)

    assert summary.snapshot["results_completed"] == 30
    assert 1 < session.peak_outstanding <= 3


def test_slow_graph_drops_old_camera_frames_and_counts_them() -> None:
    session = FakeSession(delay=0.03)
    capture = FakeCapture(60, interval=0.002)

    summary = _run(session, capture, lossless=False, max_in_flight=2)

    snapshot = summary.snapshot
    overwritten = snapshot["drops"]["capture_overwritten"]
    assert overwritten > 0
    assert snapshot["frames_read"] == 60
    assert snapshot["results_completed"] + overwritten <= 60
    assert snapshot["results_completed"] == len(session.finished)


def test_failure_ends_the_run_and_releases_the_capture() -> None:
    capture = FakeCapture(100)

    summary = _run(FakeSession(fail_at=2), capture, max_in_flight=2)

    assert summary.reason == "error"
    assert "call 2" in str(summary.error)
    assert capture.released.is_set()


def test_stop_waits_for_the_running_inference() -> None:
    # The first inference is held well past the 0.05 s duration.
    hold = threading.Event()
    session = FakeSession(hold=hold)
    capture = FakeCapture(1000, interval=0.001)
    threading.Thread(
        target=lambda: (
            session.started.wait(WAIT_SECONDS),
            time.sleep(0.3),
            hold.set(),
        ),
        daemon=True,
    ).start()

    started = time.perf_counter()
    summary = _run(session, capture, lossless=False, duration_seconds=0.05)

    assert summary.reason == "duration"
    assert time.perf_counter() - started >= 0.3
    assert session.finished == ["frame-0"], "the held inference finished, not killed"
    assert summary.snapshot["results_completed"] == 1
    assert capture.released.is_set()


def test_q_key_closes_the_window(monkeypatch: pytest.MonkeyPatch) -> None:
    window = FakeWindow(keys=[-1, ord("q")])
    window.install(monkeypatch)
    capture = FakeCapture(10_000, interval=0.001)

    summary = _run(FakeSession(), capture, lossless=False, display=True)

    assert summary.reason == "q/Esc"
    assert window.destroyed
    assert capture.released.is_set()


def test_closed_window_ends_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    window = FakeWindow(closes_after=3)
    window.install(monkeypatch)

    summary = _run(FakeSession(), FakeCapture(10_000), display=True)

    assert summary.reason == "window closed"
    assert len(window.shown) == 3


def test_output_fps_counts_each_result_once(monkeypatch: pytest.MonkeyPatch) -> None:
    window = FakeWindow()
    window.install(monkeypatch)

    # A slow graph leaves many display iterations without a new result.
    summary = _run(FakeSession(delay=0.02), FakeCapture(5), display=True)

    assert summary.reason == "end of input"
    assert summary.snapshot["results_presented"] == len(window.shown[1:]) == 5
    assert summary.snapshot["capture_to_present_ms"]["median"] > 0


def test_display_shows_bgr_pixels_with_overlay(monkeypatch: pytest.MonkeyPatch) -> None:
    window = FakeWindow()
    window.install(monkeypatch)
    pixels_bgr = np.zeros((400, 600, 3), dtype=np.uint8)
    pixels_bgr[-1, -1] = (255, 0, 0)

    _run(FakeSession(), StillImage(pixels_bgr), display=True)

    _placeholder, shown = window.shown
    assert shown[-1, -1].tolist() == [255, 0, 0]
    assert shown[:20, :200].any(), "overlay text drawn in the top-left corner"


class FakeNativeModel:
    """``inference_models`` API with one fixed box for the real graph."""

    class_names = ["person", "car"]

    def pre_process(self, images: torch.Tensor, input_color_format: str):
        assert input_color_format == "rgb"
        batch = images.unsqueeze(0).float()

        return batch, [None]

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return batch

    def post_process(self, raw: torch.Tensor, metadata: Any, confidence: float):
        detections = Detections(
            xyxy=torch.tensor([[10, 20, 60, 70]], dtype=torch.int32),
            class_id=torch.tensor([1], dtype=torch.int32),
            confidence=torch.tensor([0.9]),
        )

        return [detections]


@pytest.mark.parametrize("mode", ["serial", "pipeline"])
def test_real_graph_annotates_every_frame(mode: str) -> None:
    from run_demo import compile_live_workflow, warm_up

    session = compile_live_workflow(mode=mode).create_session(
        {"detection_model": FakeNativeModel()}
    )
    warm_up(session, confidence=0.5)

    summary = _run(
        session,
        FakeCapture(4),
        max_in_flight=3 if mode == "pipeline" else None,
    )

    assert summary.reason == "end of input", summary.error
    assert summary.snapshot["results_completed"] == 4
    assert all(
        value is not None and value >= 0
        for value in summary.snapshot["stage_median_ms"].values()
    )
    last = summary.last_result
    assert last.predictions.image_metadata["parent_id"] == "frame-3"
    assert last.predictions.image_metadata["class_names"][1] == "car"
    assert last.annotated.shape == (3, 6, 8)


def test_real_graph_pixels_equal_in_both_modes() -> None:
    from run_demo import compile_live_workflow

    pixels_bgr = np.full((120, 160, 3), 40, dtype=np.uint8)
    annotated = {}
    for mode, max_in_flight in (("serial", None), ("pipeline", 2)):
        session = compile_live_workflow(mode=mode).create_session(
            {"detection_model": FakeNativeModel()}
        )
        summary = _run(session, StillImage(pixels_bgr), max_in_flight=max_in_flight)
        annotated[mode] = summary.last_result.annotated

    assert torch.equal(annotated["serial"], annotated["pipeline"])
    assert (annotated["serial"] != 40).any(), "boxes and labels were drawn"
    assert (pixels_bgr == 40).all(), "the source frame is unchanged"


def test_backend_is_read_from_the_loaded_model() -> None:
    from run_demo import describe_backend

    options = SimpleNamespace(intra_op_num_threads=0, inter_op_num_threads=2)
    onnx_like = SimpleNamespace(
        _session=SimpleNamespace(
            get_providers=lambda: ["CPUExecutionProvider"],
            get_session_options=lambda: options,
        ),
        _device=torch.device("cpu"),
    )

    described = describe_backend(onnx_like)
    assert described["providers"] == "CPUExecutionProvider"
    assert described["device"] == "cpu"
    assert described["ort_intra_op_num_threads"] == 0
    assert described["ort_inter_op_num_threads"] == 2
    other = describe_backend(FakeNativeModel())
    assert other["providers"] == "not ONNX Runtime"
    assert other["ort_intra_op_num_threads"] is None


def test_image_command_writes_annotation_and_stats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    import run_demo
    from click.testing import CliRunner

    image_path = tmp_path / "input.png"
    cv2.imwrite(str(image_path), np.full((90, 120, 3), 40, dtype=np.uint8))
    monkeypatch.setattr(run_demo, "load_model", lambda *_, **__: FakeNativeModel())

    outcome = CliRunner().invoke(
        run_demo.main,
        ["--image", str(image_path), "--output-dir", str(tmp_path / "out")],
    )

    assert outcome.exit_code == 0, outcome.output
    assert "car" in outcome.output
    annotated = cv2.imread(str(tmp_path / "out" / "annotated.png"))
    assert annotated.shape == (90, 120, 3)
    assert (annotated != 40).any()
    stats = json.loads((tmp_path / "out" / "stats.json").read_text())
    assert stats["stopped"] == "end of input"
    assert stats["results_completed"] == 1
    assert stats["results_presented"] is None
    assert stats["ort_intra_op_num_threads"] is None
    assert stats["warmup_blank_runs_excluded"] == 3


@pytest.mark.parametrize(
    ("size", "expected"), [(1, 1), (20, 19), (100, 95), (120, 114)]
)
def test_p95_is_the_nearest_rank(size: int, expected: int) -> None:
    samples = RollingSamples(size=size)
    for value in reversed(range(1, size + 1)):
        samples.add(value)

    assert samples.p95() == expected


def _live_threads() -> List[str]:
    names = [
        thread.name
        for thread in threading.enumerate()
        if thread.name in ("live-capture", "live-graph")
    ]

    return names


@pytest.mark.parametrize("removed_window", [False, True])
def test_window_closed_before_the_first_result(
    removed_window: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The window closes while the first inference is still held; releasing
    # it 0.1 s later lets shutdown wait for it.
    hold = threading.Event()
    session = FakeSession(hold=hold)
    window = FakeWindow(
        closes_after=1,
        ready=session.started,
        on_close=lambda: threading.Timer(0.1, hold.set).start(),
    )
    window.install(monkeypatch)
    if removed_window:

        def visible(*args: Any) -> float:
            value = window._visible(*args)
            if value == 0:
                error = cv2.error("NULL window")
                error.code = cv2.Error.StsNullPtr
                raise error
            return value

        monkeypatch.setattr(cv2, "getWindowProperty", visible)
    capture = FakeCapture(10_000, interval=0.001)

    summary = _run(session, capture, lossless=False, display=True)

    assert summary.reason == "window closed"
    assert len(window.shown) == 1, "no imshow after the close"
    assert session.finished == ["frame-0"]
    assert summary.snapshot["results_presented"] == 0
    assert capture.released.is_set()
    assert _live_threads() == []


@pytest.mark.parametrize("max_in_flight", [None, 2])
def test_failing_window_close_still_joins_the_worker(
    max_in_flight: Optional[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    hold = threading.Event()
    session = FakeSession(hold=hold)
    window = FakeWindow(
        keys=[ord("q")],
        ready=session.started,
        destroy_error=RuntimeError("destroy failed"),
    )
    window.install(monkeypatch)
    threading.Timer(0.2, hold.set).start()
    capture = FakeCapture(10_000, interval=0.001)

    with pytest.raises(RuntimeError, match="destroy failed"):
        _run(
            session,
            capture,
            lossless=False,
            display=True,
            max_in_flight=max_in_flight,
        )

    assert session.finished, "the inference in flight finished before returning"
    assert capture.released.is_set()
    assert _live_threads() == []


def test_worker_start_failure_stops_the_started_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(_: Any) -> None:
        raise RuntimeError("no graph thread")

    monkeypatch.setattr(live.GraphWorker, "start", fail)
    capture = FakeCapture(10_000, interval=0.001)

    with pytest.raises(RuntimeError, match="no graph thread"):
        _run(FakeSession(), capture, lossless=False)

    assert capture.released.is_set()
    assert _live_threads() == []


def test_reader_start_failure_releases_the_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(_: Any) -> None:
        raise RuntimeError("no capture thread")

    monkeypatch.setattr(live.FrameReader, "start", fail)
    capture = FakeCapture(10)

    # The original error surfaces, not a join of the never-started worker.
    with pytest.raises(RuntimeError, match="no capture thread"):
        _run(FakeSession(), capture)

    assert capture.released.is_set()
    assert _live_threads() == []


class ReleaseFails(FakeCapture):
    def release(self) -> None:
        super().release()
        raise OSError("release failed")


def test_failing_release_still_ends_the_input() -> None:
    started = time.perf_counter()

    summary = _run(FakeSession(), ReleaseFails(3), duration_seconds=WAIT_SECONDS)

    assert time.perf_counter() - started < WAIT_SECONDS / 2
    assert summary.snapshot["results_completed"] == 3
    assert summary.reason == "error"
    assert isinstance(summary.error, OSError)


def test_stalled_pipeline_fails_instead_of_dropping_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(live, "STALLED_SUBMIT_SECONDS", 0.05)
    capture = FakeCapture(5)

    summary = _run(FakeSession(stalled_workers=1), capture, max_in_flight=1)

    assert summary.reason == "error"
    assert "Pipeline stalled: no worker accepted frame 0" in str(summary.error)
    assert summary.snapshot["results_completed"] == 0
    assert capture.released.is_set()


def test_other_window_errors_surface_after_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = FakeWindow()
    window.install(monkeypatch)
    error = cv2.error("unexpected GUI failure")
    error.code = cv2.Error.StsError

    def fail(*_: Any) -> float:
        raise error

    monkeypatch.setattr(cv2, "getWindowProperty", fail)
    capture = FakeCapture(10_000, interval=0.001)
    with pytest.raises(cv2.error, match="unexpected GUI failure"):
        _run(FakeSession(), capture, lossless=False, display=True)

    assert capture.released.is_set()
    assert _live_threads() == []
