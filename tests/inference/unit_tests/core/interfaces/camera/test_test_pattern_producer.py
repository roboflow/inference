import time

import numpy as np
import pytest

from inference.core.interfaces.camera.entities import SourceProperties
from inference.core.interfaces.camera.test_pattern_producer import (
    HEIGHT,
    WIDTH,
    FPS,
    TestPatternStreamProducer,
    generate_frame,
)
from inference.core.interfaces.camera.video_source import (
    BufferConsumptionStrategy,
    BufferFillingStrategy,
    StreamState,
    VideoSource,
    _is_test_pattern_reference,
)


def test_is_test_pattern_reference_accepts_valid_and_rejects_invalid():
    assert _is_test_pattern_reference("TestPatternStreamProducer") is True
    assert _is_test_pattern_reference("  TestPatternStreamProducer") is True
    assert _is_test_pattern_reference("TestPatternStreamProducer(foo=bar)") is True
    assert _is_test_pattern_reference(0) is False
    assert _is_test_pattern_reference("rtsp://camera.local/stream") is False
    assert _is_test_pattern_reference("TestPattern") is False
    assert _is_test_pattern_reference("") is False


def test_generate_frame_produces_valid_image():
    frame = generate_frame()
    assert frame.shape == (HEIGHT, WIDTH, 3)
    assert frame.dtype == np.uint8
    assert frame.sum() > 0


def test_discover_source_properties_for_test_pattern():
    producer = TestPatternStreamProducer()
    try:
        producer.isOpened()
        result = producer.discover_source_properties()
        assert result == SourceProperties(
            width=WIDTH,
            height=HEIGHT,
            total_frames=-1,
            is_file=False,
            fps=float(FPS),
            is_reconnectable=True,
        )
    finally:
        producer.release()


@pytest.mark.timeout(30)
def test_video_source_starts_and_reads_frames_from_test_pattern():
    source = VideoSource.init(
        video_reference="TestPatternStreamProducer",
        source_id=99,
    )
    try:
        source.start()
        frame = source.read_frame()
        assert frame is not None
        assert frame.image.shape == (HEIGHT, WIDTH, 3)

        metadata = source.describe_source()
        assert metadata.state is StreamState.RUNNING
        assert metadata.source_properties.is_file is False
        assert metadata.source_properties.width == WIDTH
        assert metadata.source_properties.height == HEIGHT
        assert metadata.source_properties.fps == float(FPS)
        assert metadata.source_id == 99
        assert metadata.buffer_filling_strategy is BufferFillingStrategy.ADAPTIVE_DROP_OLDEST
        assert metadata.buffer_consumption_strategy is BufferConsumptionStrategy.EAGER
    finally:
        source.terminate(wait_on_frames_consumption=False, purge_frames_buffer=True)
