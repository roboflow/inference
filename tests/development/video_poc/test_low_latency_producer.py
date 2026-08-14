import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace


PROCESSOR_DIR = (
    Path(__file__).resolve().parents[3] / "development" / "video_poc" / "processor"
)
sys.path.insert(0, str(PROCESSOR_DIR))

import low_latency_producer  # noqa: E402


class InvalidDataError(Exception):
    pass


class FakeCodecContext:
    width = 3840
    height = 2160
    name = "h264"
    flags = 0
    thread_count = 0


class FakeContainer:
    def __init__(self):
        self.stream = SimpleNamespace(
            codec_context=FakeCodecContext(),
            average_rate=Fraction(60000, 1001),
            guessed_rate=None,
        )
        self.streams = SimpleNamespace(video=[self.stream])

    def demux(self, stream):
        assert stream is self.stream
        return iter(())

    def close(self):
        pass


def test_fresh_rtsp_publisher_is_retried_and_metadata_is_exact(monkeypatch):
    attempts = []
    sleeps = []

    def fake_open(url, options, timeout):
        attempts.append((url, options, timeout))
        if len(attempts) < 3:
            raise InvalidDataError("publisher is not ready")
        return FakeContainer()

    fake_av = SimpleNamespace(
        open=fake_open,
        error=SimpleNamespace(InvalidDataError=InvalidDataError),
    )
    monkeypatch.setitem(sys.modules, "av", fake_av)
    monkeypatch.setattr(low_latency_producer.time, "sleep", sleeps.append)

    producer = low_latency_producer.LowLatencyRtspProducer(
        "rtsp://credentials@relay/source",
        open_attempts=3,
        open_retry_delay_seconds=0.25,
    )

    assert len(attempts) == 3
    assert sleeps == [0.25, 0.25]
    assert producer.source_stream_metadata == {
        "width": 3840,
        "height": 2160,
        "codec": "h264",
        "fps": float(Fraction(60000, 1001)),
        "fpsNumerator": 60000,
        "fpsDenominator": 1001,
    }


def test_non_transient_open_error_fails_without_retry(monkeypatch):
    attempts = []

    def fake_open(url, options, timeout):
        attempts.append(url)
        raise ValueError("bad options")

    fake_av = SimpleNamespace(
        open=fake_open,
        error=SimpleNamespace(InvalidDataError=InvalidDataError),
    )
    monkeypatch.setitem(sys.modules, "av", fake_av)

    try:
        low_latency_producer.LowLatencyRtspProducer("rtsp://relay/source")
    except ValueError as error:
        assert str(error) == "bad options"
    else:
        raise AssertionError("expected ValueError")

    assert attempts == ["rtsp://relay/source"]
