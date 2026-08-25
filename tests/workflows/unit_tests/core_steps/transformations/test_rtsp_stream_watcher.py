import threading
import time
from unittest.mock import MagicMock

import numpy as np
import pytest
import supervision as sv

from inference.core.workflows.core_steps.transformations.rtsp_stream_watcher import (
    v1 as watcher_module,
)
from inference.core.workflows.core_steps.transformations.rtsp_stream_watcher.v1 import (
    BlockManifest,
    RTSPStreamWatcherBlockV1,
    _redact,
    _StreamReader,
)


class FakeCapture:
    """Stands in for cv2.VideoCapture: serves numbered frames, can be told to fail."""

    def __init__(self, opened=True, fail_after=None, shape=(360, 640, 3)):
        self._opened = opened
        self._n = 0
        self._fail_after = fail_after
        self._shape = shape
        self.released = False

    def isOpened(self):
        return self._opened

    def grab(self):
        if self._fail_after is not None and self._n >= self._fail_after:
            return False
        self._n += 1
        time.sleep(0.005)
        return True

    def retrieve(self):
        frame = np.full(self._shape, self._n % 255, dtype=np.uint8)
        return True, frame

    def release(self):
        self.released = True


class CountingDetector:
    """Stands in for the background model: counts calls, returns one 'person'."""

    def __init__(self, fail=False):
        self.calls = 0
        self.fail = fail
        self._lock = threading.Lock()

    def __call__(self, frame):
        with self._lock:
            self.calls += 1
        if self.fail:
            raise RuntimeError("model exploded")
        return sv.Detections(
            xyxy=np.array([[10.0, 10.0, 50.0, 90.0]]),
            confidence=np.array([0.9]),
            class_id=np.array([0]),
            data={"class_name": np.array(["person"])},
        )


@pytest.fixture(autouse=True)
def clean_readers():
    watcher_module.stop_all_readers()
    yield
    watcher_module.stop_all_readers()


def _block():
    return RTSPStreamWatcherBlockV1(model_manager=MagicMock(), api_key="key")


def _run(url="rtsp://cam/live", block=None, **overrides):
    kwargs = dict(
        stream_url=url,
        check_interval_seconds=0.02,
        model_id=None,
        confidence=0.4,
        class_filter=None,
        max_frame_age_seconds=5.0,
        reconnect_delay_seconds=0.05,
        idle_timeout_seconds=60.0,
    )
    kwargs.update(overrides)
    return (block or _block()).run(**kwargs)


def _wait_for(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_first_run_returns_placeholder_while_connecting(monkeypatch):
    monkeypatch.setattr(_StreamReader, "_open", lambda self: FakeCapture(opened=False))

    result = _run()

    assert result["frame_available"] is False
    assert result["frame"].numpy_image.shape == watcher_module.PLACEHOLDER_SHAPE
    assert result["frame_age_seconds"] == -1.0
    assert len(result["predictions"]) == 0


def test_frames_flow_once_stream_is_open(monkeypatch):
    monkeypatch.setattr(_StreamReader, "_open", lambda self: FakeCapture())
    _run()  # starts the reader

    assert _wait_for(lambda: _run()["frame_available"])
    result = _run()
    assert result["status"] == "streaming"
    assert result["frame"].numpy_image.shape == (360, 640, 3)
    assert 0 <= result["frame_age_seconds"] < 5
    assert len(result["predictions"]) == 0  # no model configured


def test_readers_are_shared_per_url_and_model(monkeypatch):
    monkeypatch.setattr(_StreamReader, "_open", lambda self: FakeCapture())
    monkeypatch.setattr(
        RTSPStreamWatcherBlockV1,
        "_build_detector",
        lambda self, **kw: CountingDetector(),
    )

    _run("rtsp://a")
    _run("rtsp://a")
    _run("rtsp://a", model_id="m/1")
    _run("rtsp://b")

    assert set(watcher_module._READERS) == {"rtsp://a#", "rtsp://a#m/1", "rtsp://b#"}


def test_model_runs_in_background_at_check_interval(monkeypatch):
    monkeypatch.setattr(_StreamReader, "_open", lambda self: FakeCapture())
    detector = CountingDetector()
    monkeypatch.setattr(
        RTSPStreamWatcherBlockV1, "_build_detector", lambda self, **kw: detector
    )
    block = _block()
    _run(block=block, model_id="m/1", check_interval_seconds=0.1)

    assert _wait_for(lambda: detector.calls >= 1)
    time.sleep(0.55)
    # ~5 checks in 0.55s at 0.1s; nothing here ran the model on the request path
    assert 3 <= detector.calls <= 8

    result = _run(block=block, model_id="m/1", check_interval_seconds=0.1)
    assert result["frame_available"] is True
    assert list(result["predictions"].data["class_name"]) == ["person"]


def test_predictions_persist_between_checks(monkeypatch):
    monkeypatch.setattr(_StreamReader, "_open", lambda self: FakeCapture())
    detector = CountingDetector()
    monkeypatch.setattr(
        RTSPStreamWatcherBlockV1, "_build_detector", lambda self, **kw: detector
    )
    block = _block()
    _run(block=block, model_id="m/1", check_interval_seconds=5.0)
    assert _wait_for(lambda: detector.calls == 1)

    results = [
        _run(block=block, model_id="m/1", check_interval_seconds=5.0) for _ in range(5)
    ]

    assert detector.calls == 1  # five workflow runs, still one model call
    assert all(len(r["predictions"]) == 1 for r in results)


def test_model_fault_yields_empty_predictions_and_keeps_streaming(monkeypatch):
    monkeypatch.setattr(_StreamReader, "_open", lambda self: FakeCapture())
    detector = CountingDetector(fail=True)
    monkeypatch.setattr(
        RTSPStreamWatcherBlockV1, "_build_detector", lambda self, **kw: detector
    )
    block = _block()
    _run(block=block, model_id="m/1", check_interval_seconds=0.05)
    assert _wait_for(lambda: detector.calls >= 2)

    result = _run(block=block, model_id="m/1", check_interval_seconds=0.05)

    assert result["status"] == "streaming"
    assert result["frame_available"] is True
    assert len(result["predictions"]) == 0


def test_stale_frame_is_reported_unavailable_with_empty_predictions(monkeypatch):
    monkeypatch.setattr(_StreamReader, "_open", lambda self: FakeCapture())
    detector = CountingDetector()
    monkeypatch.setattr(
        RTSPStreamWatcherBlockV1, "_build_detector", lambda self, **kw: detector
    )
    block = _block()
    _run(block=block, model_id="m/1")
    assert _wait_for(lambda: _run(block=block, model_id="m/1")["frame_available"])
    reader = watcher_module._READERS["rtsp://cam/live#m/1"]
    with reader._lock:
        reader._frame_time -= 100  # pretend the newest sample is very old

    result = _run(block=block, model_id="m/1", max_frame_age_seconds=5.0)

    assert result["frame_available"] is False
    assert result["status"] == "stale"
    assert not result["frame"].numpy_image.any()
    assert len(result["predictions"]) == 0


def test_reader_reconnects_after_stream_failure(monkeypatch):
    captures = []

    def make(self):
        captures.append(FakeCapture(fail_after=3))
        return captures[-1]

    monkeypatch.setattr(_StreamReader, "_open", make)
    _run(reconnect_delay_seconds=0.02)

    assert _wait_for(lambda: len(captures) >= 2, timeout=3.0)
    assert captures[0].released is True


def test_reader_stops_when_idle_and_restarts_on_next_request(monkeypatch):
    monkeypatch.setattr(_StreamReader, "_open", lambda self: FakeCapture())
    _run(idle_timeout_seconds=1.0)
    reader = watcher_module._READERS["rtsp://cam/live#"]
    assert _wait_for(lambda: reader.latest()[2] == "streaming")

    reader._last_request -= 10  # nobody has asked for a frame in a while
    assert _wait_for(lambda: reader.latest()[2] == "idle", timeout=3.0)
    assert not reader._thread.is_alive()

    _run(idle_timeout_seconds=1.0)
    assert reader._thread.is_alive()


def test_urls_with_credentials_are_redacted_in_logs():
    assert (
        _redact("rtsp://admin:hunter2@192.168.1.17:554/live")
        == "rtsp://***@192.168.1.17:554/live"
    )
    assert _redact("rtsp://192.168.1.17:554/live") == "rtsp://192.168.1.17:554/live"


def test_manifest_defaults_are_demo_friendly():
    m = BlockManifest.model_validate(
        {
            "type": "roboflow_core/rtsp_stream_watcher@v1",
            "name": "cam",
            "stream_url": "rtsp://cam/live",
        }
    )
    assert m.check_interval_seconds == 3.0
    assert m.model_id is None
    assert m.max_frame_age_seconds == 15.0
    assert m.idle_timeout_seconds == 60.0
    assert m.discover_dependent_resources() is None


def test_manifest_declares_model_dependency_when_set():
    m = BlockManifest.model_validate(
        {
            "type": "roboflow_core/rtsp_stream_watcher@v1",
            "name": "cam",
            "stream_url": "rtsp://cam/live",
            "model_id": "rfdetr-nano",
        }
    )
    resources = m.discover_dependent_resources()
    assert len(resources) == 1 and resources[0].metadata.model_id == "rfdetr-nano"
