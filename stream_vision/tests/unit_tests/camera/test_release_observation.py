"""Cleanup failures remain observable without changing legacy camera events."""

from threading import Event

import pytest
from streamvision.camera import video_source
from streamvision.camera.entities import SourceProperties
from streamvision.camera.exceptions import SourceConnectionError

WAIT = 5.0
REFERENCE = "rtsp://alice:hidden-password@camera.example/live"


class EmptyProducer:
    def __init__(self, *, release_failure=None, startup_failure=None):
        self.release_failure = release_failure
        self.startup_failure = startup_failure
        self.release_calls = 0

    def isOpened(self):
        return self.startup_failure != "open"

    def connection_error_message(self):
        return "connection refused"

    def initialize_source_properties(self, properties):
        pass

    def discover_source_properties(self):
        if self.startup_failure == "properties":
            raise ValueError("properties unavailable")

        return SourceProperties(width=2, height=2, fps=30, total_frames=0, is_file=True)

    def grab(self):
        return False

    def release(self):
        self.release_calls += 1
        if self.release_failure is not None:
            raise self.release_failure


def _source(*, reference=REFERENCE):
    events = []
    finished = Event()

    def record(update):
        events.append((update.event_type, update.payload))
        if update.event_type == video_source.VIDEO_CONSUMPTION_FINISHED_EVENT:
            finished.set()

    source = video_source.VideoSource.init(
        video_reference=reference, buffer_size=2, status_update_handlers=[record]
    )
    return source, events, finished


def _finish(source, finished):
    assert finished.wait(WAIT), "capture thread did not finish"
    source.terminate(wait_on_frames_consumption=False, purge_frames_buffer=True)


def test_release_error_is_initially_none_read_only_and_does_not_wait_for_state_lock():
    source, _, _ = _source()
    with source._state_change_lock:
        assert source.release_error is None
    with pytest.raises(AttributeError):
        source.release_error = "replacement"


def test_release_failure_keeps_legacy_events_and_exception_behavior(monkeypatch):
    event_runs = []
    for failure in (None, RuntimeError(f"cannot release {REFERENCE}")):
        producer = EmptyProducer(release_failure=failure)
        monkeypatch.setattr(
            video_source, "_build_default_producer", lambda *args, **kwargs: producer
        )
        source, events, finished = _source()
        source.start()
        _finish(source, finished)
        event_runs.append(events)
        assert producer.release_calls == 1
        assert source.get_state() is video_source.StreamState.ENDED
        if failure is None:
            assert source.release_error is None
        else:
            assert source.release_error.startswith("RuntimeError: cannot release")
            assert "camera.example" in source.release_error
            assert "hidden-password" not in source.release_error
            assert "alice" not in source.release_error

    assert event_runs[0] == event_runs[1]


@pytest.mark.parametrize("startup_failure", ["open", "properties"])
def test_partial_start_failure_keeps_original_exception_and_events(
    monkeypatch, startup_failure
):
    outcomes = []
    # An exact CV2 type does not take the hardware fallback; exercise that same
    # startup route with and without a second failure while releasing.
    monkeypatch.setattr(video_source, "CV2VideoFrameProducer", EmptyProducer)
    for release_failure in (None, RuntimeError("release failed")):
        producer = EmptyProducer(
            release_failure=release_failure, startup_failure=startup_failure
        )
        monkeypatch.setattr(
            video_source, "_build_default_producer", lambda *args, **kwargs: producer
        )
        source, events, _ = _source()
        with pytest.raises((ValueError, SourceConnectionError)) as caught:
            source.start()
        assert source.get_state() is video_source.StreamState.ERROR
        assert producer.release_calls == 1
        outcomes.append((type(caught.value), str(caught.value), events))
        assert source.release_error == (
            None if release_failure is None else "RuntimeError: release failed"
        )

    assert outcomes[0] == outcomes[1]


def test_auto_fallback_still_runs_and_keeps_failed_hardware_release(monkeypatch):
    hardware = EmptyProducer(
        startup_failure="properties", release_failure=RuntimeError("hardware cleanup")
    )
    fallback = EmptyProducer()
    selected = []

    def build_fallback(reference):
        selected.append(reference)
        return fallback

    monkeypatch.setattr(
        video_source, "_build_default_producer", lambda *args, **kwargs: hardware
    )
    monkeypatch.setattr(video_source, "CV2VideoFrameProducer", build_fallback)
    source, _, finished = _source()
    source.start()
    _finish(source, finished)

    assert selected == [REFERENCE]
    assert hardware.release_calls == fallback.release_calls == 1
    assert source.release_error == "RuntimeError: hardware cleanup"


def test_restart_retains_failure_and_a_later_failure_replaces_it():
    producers = iter(
        [
            EmptyProducer(release_failure=RuntimeError("first cleanup")),
            EmptyProducer(),
            EmptyProducer(release_failure=ValueError("latest cleanup")),
        ]
    )
    source, _, finished = _source(reference=lambda: next(producers))
    source.start()
    _finish(source, finished)
    assert source.release_error == "RuntimeError: first cleanup"

    for expected in ("RuntimeError: first cleanup", "ValueError: latest cleanup"):
        finished.clear()
        source.restart(wait_on_frames_consumption=False)
        _finish(source, finished)
        assert source.release_error == expected
