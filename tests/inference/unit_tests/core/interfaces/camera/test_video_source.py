import time
from datetime import datetime
from queue import Queue
from threading import Thread
from unittest.mock import MagicMock

import cv2
import numpy as np
import supervision as sv
import pytest

from inference.core.interfaces.camera.entities import StatusUpdate, UpdateSeverity
from inference.core.interfaces.camera.exceptions import (
    SourceConnectionError,
    StreamOperationNotAllowedError,
)
from inference.core.interfaces.camera.video_source import (
    BufferConsumptionStrategy,
    BufferFillingStrategy,
    SourceMetadata,
    StreamState,
    VideoSource,
    discover_source_properties,
    purge_queue, drop_single_frame_from_buffer, decode_video_frame_to_buffer, VideoConsumer, SourceProperties,
)


def test_purge_queue_when_empty_queue_given_and_await_not_desired() -> None:
    # given
    queue = Queue()

    # when
    result = purge_queue(queue=queue, wait_on_empty=False)

    # then
    assert result is None


def test_purge_queue_when_non_empty_queue_given() -> None:
    # given
    queue = Queue()
    queue.put(1)
    queue.put(2)
    queue.put(3)

    # when
    result = purge_queue(queue=queue)

    # then
    assert result is 3
    assert queue.empty() is True


def test_purge_queue_when_non_empty_queue_given_with_fps_monitor() -> None:
    # given
    successful_reads = []

    def on_successful_read() -> None:
        successful_reads.append(1)

    queue = Queue()
    queue.put(1)
    queue.put(2)
    queue.put(3)

    # when
    result = purge_queue(queue=queue, on_successful_read=on_successful_read)

    # then
    assert result is 3
    assert queue.empty() is True
    assert len(successful_reads) == 3


def test_discover_source_properties_when_local_file_given(
    local_video_path: str,
) -> None:
    # given
    video = cv2.VideoCapture(local_video_path)

    # when
    result = discover_source_properties(stream=video)

    # then
    assert result.is_file is True
    assert result.total_frames == 431
    assert result.height == 240
    assert result.width == 426
    assert abs(result.fps - 30.0) < 1e-5


def test_video_source_throwing_error_when_invalid_video_reference_given() -> None:
    # given
    video_source = VideoSource.init(video_reference="invalid")

    # when
    with pytest.raises(SourceConnectionError):
        video_source.start()


def test_video_source_describe_source_when_stream_consumption_not_yet_started() -> None:
    # given
    video_source = VideoSource.init(video_reference="invalid")

    # when
    result = video_source.describe_source()

    # then
    assert result == SourceMetadata(
        source_properties=None,
        source_reference="invalid",
        buffer_size=64,
        state=StreamState.NOT_STARTED,
        buffer_filling_strategy=None,
        buffer_consumption_strategy=None,
    )


def test_video_source_describe_source_when_invalid_video_reference_consumption_started() -> (
    None
):
    # given
    video_source = VideoSource.init(video_reference="invalid")

    # when
    with pytest.raises(SourceConnectionError):
        video_source.start()
    result = video_source.describe_source()

    # then
    assert result == SourceMetadata(
        source_properties=None,
        source_reference="invalid",
        buffer_size=64,
        state=StreamState.ERROR,
        buffer_filling_strategy=None,
        buffer_consumption_strategy=None,
    )


@pytest.mark.timeout(90)
@pytest.mark.slow
def test_video_source_describe_source_when_valid_video_reference_consumption_started(
    local_video_path: str,
) -> None:
    # given
    video_source = VideoSource.init(video_reference=local_video_path)

    try:
        # when
        video_source.start()
        result = video_source.describe_source()
        _ = video_source.read_frame()

        # then
        assert result.source_properties.is_file is True
        assert result.source_reference == local_video_path
        assert result.state is StreamState.RUNNING
        assert result.buffer_filling_strategy is BufferFillingStrategy.WAIT
        assert result.buffer_consumption_strategy is BufferConsumptionStrategy.LAZY

    finally:
        tear_down_source(video_source=video_source)


@pytest.mark.timeout(90)
@pytest.mark.slow
def test_pausing_video_stream(local_video_path: str) -> None:
    # given
    video_source = VideoSource.init(video_reference=local_video_path)
    pause_resume_delay = 0.2

    try:
        # when
        video_source.start()
        _ = video_source.read_frame()
        video_source.pause()
        time.sleep(pause_resume_delay)
        timestamp_of_resume = datetime.now()
        video_source.resume()
        last_frame_before_resume, first_frame_after_resume = None, None
        while True:
            frame = video_source.read_frame()
            if (frame[0] - timestamp_of_resume).total_seconds() >= 0:
                first_frame_after_resume = frame
                break
            last_frame_before_resume = frame

        # then
        assert (
            first_frame_after_resume[0] - last_frame_before_resume[0]
        ).total_seconds() >= pause_resume_delay
    finally:
        tear_down_source(video_source=video_source)


@pytest.mark.timeout(90)
@pytest.mark.slow
def test_muting_video_stream(local_video_path: str) -> None:
    # given
    video_source = VideoSource.init(video_reference=local_video_path)

    try:
        # when
        video_source.start()
        _ = video_source.read_frame()
        video_source.mute()
        try:
            video_source.resume()
        except StreamOperationNotAllowedError:
            pass

    finally:
        tear_down_source(video_source=video_source)


@pytest.mark.timeout(90)
@pytest.mark.slow
def test_restart_paused_stream(local_video_path: str) -> None:
    # given
    video_source = VideoSource.init(video_reference=local_video_path)
    frames_before_restart = []

    def capture_frames() -> None:
        for f in video_source:
            frames_before_restart.append(f)

    capture_thread = Thread(target=capture_frames)

    try:
        # when
        video_source.start()
        frame = video_source.read_frame()
        last_id_before_restart = frame[1]
        capture_thread.start()
        video_source.pause()
        restart_timestamp = datetime.now()
        video_source.restart()
        capture_thread.join()
        frame_after_restart = video_source.read_frame()

        # then
        for frame_before_restart in frames_before_restart:
            last_id_before_restart = frame_before_restart[1]
        assert frame_after_restart[1] == last_id_before_restart + 1
        assert (frame_after_restart[0] - restart_timestamp).total_seconds() > 0
    finally:
        tear_down_source(video_source=video_source)


@pytest.mark.timeout(90)
@pytest.mark.slow
def test_restart_muted_stream(local_video_path: str) -> None:
    # given
    video_source = VideoSource.init(video_reference=local_video_path)
    frames_before_restart = []

    def capture_frames() -> None:
        for f in video_source:
            frames_before_restart.append(f)

    capture_thread = Thread(target=capture_frames)

    try:
        # when
        video_source.start()
        frame = video_source.read_frame()
        last_id_before_restart = frame[1]
        capture_thread.start()
        video_source.mute()
        restart_timestamp = datetime.now()
        video_source.restart()
        capture_thread.join()
        frame_after_restart = video_source.read_frame()

        # then
        for frame_before_restart in frames_before_restart:
            last_id_before_restart = frame_before_restart[1]
        assert frame_after_restart[1] == last_id_before_restart + 1
        assert (frame_after_restart[0] - restart_timestamp).total_seconds() > 0
    finally:
        tear_down_source(video_source=video_source)


@pytest.mark.timeout(90)
@pytest.mark.slow
def test_restart_running_stream(local_video_path: str) -> None:
    # given
    video_source = VideoSource.init(video_reference=local_video_path)
    frames_before_restart = []

    def capture_frames() -> None:
        for f in video_source:
            frames_before_restart.append(f)

    capture_thread = Thread(target=capture_frames)

    try:
        # when
        video_source.start()
        frame = video_source.read_frame()
        last_id_before_restart = frame[1]
        capture_thread.start()
        restart_timestamp = datetime.now()
        video_source.restart()
        capture_thread.join()
        frame_after_restart = video_source.read_frame()

        # then
        for frame_before_restart in frames_before_restart:
            last_id_before_restart = frame_before_restart[1]
        assert frame_after_restart[1] == last_id_before_restart + 1
        assert (frame_after_restart[0] - restart_timestamp).total_seconds() > 0
    finally:
        tear_down_source(video_source=video_source)


@pytest.mark.timeout(90)
@pytest.mark.slow
def test_terminate_running_stream(local_video_path: str) -> None:
    # given
    video_source = VideoSource.init(video_reference=local_video_path)
    frames_captured = []

    def capture_frames() -> None:
        for f in video_source:
            frames_captured.append(f)

    capture_thread = Thread(target=capture_frames)

    # when
    video_source.start()
    _ = video_source.read_frame()
    capture_thread.start()
    video_source.terminate()
    capture_thread.join()

    # then - nothing hangs


@pytest.mark.timeout(90)
@pytest.mark.slow
def test_terminate_paused_stream(local_video_path: str) -> None:
    # given
    video_source = VideoSource.init(video_reference=local_video_path)
    frames_captured = []

    def capture_frames() -> None:
        for f in video_source:
            frames_captured.append(f)

    capture_thread = Thread(target=capture_frames)

    # when
    video_source.start()
    _ = video_source.read_frame()
    video_source.pause()
    capture_thread.start()
    video_source.terminate()
    capture_thread.join()

    # then - nothing hangs


@pytest.mark.timeout(90)
@pytest.mark.slow
def test_terminate_muted_stream(local_video_path: str) -> None:
    # given
    video_source = VideoSource.init(video_reference=local_video_path)
    frames_captured = []

    def capture_frames() -> None:
        for f in video_source:
            frames_captured.append(f)

    capture_thread = Thread(target=capture_frames)

    # when
    video_source.start()
    _ = video_source.read_frame()
    video_source.mute()
    capture_thread.start()
    video_source.terminate()
    capture_thread.join()

    # then - nothing hangs


def test_pause_not_started_stream(local_video_path: str) -> None:
    # given
    video_source = VideoSource.init(video_reference=local_video_path)

    # when
    with pytest.raises(StreamOperationNotAllowedError):
        video_source.pause()


def test_mute_not_started_stream(local_video_path: str) -> None:
    # given
    video_source = VideoSource.init(video_reference=local_video_path)

    # when
    with pytest.raises(StreamOperationNotAllowedError):
        video_source.mute()


def test_restart_not_started_stream(local_video_path: str) -> None:
    # given
    video_source = VideoSource.init(video_reference=local_video_path)

    # when
    with pytest.raises(StreamOperationNotAllowedError):
        video_source.restart()


def test_terminate_not_started_stream(local_video_path: str) -> None:
    # given
    video_source = VideoSource.init(video_reference=local_video_path)

    # when
    with pytest.raises(StreamOperationNotAllowedError):
        video_source.terminate()


@pytest.mark.timeout(90)
@pytest.mark.slow
def test_pause_muted_stream(local_video_path: str) -> None:
    # given
    video_source = VideoSource.init(video_reference=local_video_path)

    try:
        # when
        video_source.start()
        video_source.mute()

        # then
        with pytest.raises(StreamOperationNotAllowedError):
            video_source.pause()
    finally:
        tear_down_source(video_source=video_source)


@pytest.mark.timeout(90)
@pytest.mark.slow
def test_mute_paused_stream(local_video_path: str) -> None:
    # given
    video_source = VideoSource.init(video_reference=local_video_path)

    try:
        # when
        video_source.start()
        video_source.pause()

        # then
        with pytest.raises(StreamOperationNotAllowedError):
            video_source.mute()
    finally:
        tear_down_source(video_source=video_source)


@pytest.mark.timeout(90)
@pytest.mark.slow
def test_pause_paused_stream(local_video_path: str) -> None:
    # given
    video_source = VideoSource.init(video_reference=local_video_path)

    try:
        # when
        video_source.start()
        video_source.pause()

        # then
        with pytest.raises(StreamOperationNotAllowedError):
            video_source.pause()
    finally:
        tear_down_source(video_source=video_source)


@pytest.mark.timeout(90)
@pytest.mark.slow
def test_mute_muted_stream(local_video_path: str) -> None:
    # given
    video_source = VideoSource.init(video_reference=local_video_path)

    try:
        # when
        video_source.start()
        video_source.mute()

        # then
        with pytest.raises(StreamOperationNotAllowedError):
            video_source.mute()
    finally:
        tear_down_source(video_source=video_source)


@pytest.mark.timeout(90)
@pytest.mark.slow
def test_consumption_of_video_file_in_eager_mode(local_video_path: str) -> None:
    # given
    video_source = VideoSource.init(
        video_reference=local_video_path,
        buffer_filling_strategy=BufferFillingStrategy.DROP_OLDEST,
        buffer_consumption_strategy=BufferConsumptionStrategy.EAGER,
    )

    try:
        # when
        frames_consumed = 0
        video_source.start()
        for _ in video_source:
            frames_consumed += 1

        # then
        assert 0 <= frames_consumed <= 431
    finally:
        tear_down_source(video_source=video_source)


def tear_down_source(video_source: VideoSource) -> None:
    video_source.terminate(wait_on_frames_consumption=False)


def test_drop_single_frame_from_buffer_when_buffer_is_empty() -> None:
    # given
    buffer = Queue()
    updates = []

    def handle_status_updates(status_update: StatusUpdate) -> None:
        updates.append(status_update)

    # when
    drop_single_frame_from_buffer(
        buffer=buffer,
        cause="some",
        status_update_handlers=[handle_status_updates]
    )

    # then
    assert len(updates) == 0


def test_drop_single_frame_from_buffer_when_buffer_has_video_frame() -> None:
    # given
    buffer = Queue()
    updates = []
    frame_timestamp = datetime.now()
    buffer.put((frame_timestamp, 37, np.zeros((128, 128, 3), dtype=np.uint8)))

    def handle_status_updates(status_update: StatusUpdate) -> None:
        updates.append(status_update)

    # when
    drop_single_frame_from_buffer(
        buffer=buffer,
        cause="some",
        status_update_handlers=[handle_status_updates]
    )

    # then
    assert len(updates) == 1
    assert updates[0].payload == {
        "frame_timestamp": frame_timestamp,
        "frame_id": 37,
        "cause": "some",
    }
    assert updates[0].severity is UpdateSeverity.DEBUG
    assert updates[0].event_type == "FRAME_DROPPED"


def test_decode_video_frame_to_buffer_when_frame_could_not_be_retrieved() -> None:
    # given
    video = MagicMock()
    video.retrieve.return_value = (False, None)
    fps_monitor = sv.FPSMonitor()
    fps_monitor.tick()
    buffer = Queue()

    # when
    result = decode_video_frame_to_buffer(
        frame_timestamp=datetime.now(),
        frame_id=1,
        video=video,
        buffer=buffer,
        decoding_pace_monitor=fps_monitor,
    )

    # then
    assert result is False
    assert abs(fps_monitor()) < 1e-5
    assert buffer.empty() is True


def test_decode_video_frame_to_buffer_when_frame_could_be_retrieved() -> None:
    # given
    video = MagicMock()
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    video.retrieve.return_value = (True, image)
    fps_monitor = sv.FPSMonitor()
    fps_monitor.tick()
    frame_timestamp = datetime.now()
    buffer = Queue()

    # when
    result = decode_video_frame_to_buffer(
        frame_timestamp=frame_timestamp,
        frame_id=1,
        video=video,
        buffer=buffer,
        decoding_pace_monitor=fps_monitor,
    )

    # then
    assert result is True
    assert fps_monitor() > 0
    assert buffer.get_nowait() == (frame_timestamp, 1, image)


def test_stream_consumption_when_frame_cannot_be_grabbed() -> None:
    # given
    consumer = VideoConsumer.init(
        buffer_filling_strategy=None,
        adaptive_mode_stream_pace_tolerance=0.1,
        adaptive_mode_reader_pace_tolerance=5.0,
        minimum_adaptive_mode_samples=10,
        maximum_adaptive_frames_dropped_in_row=16,
        status_update_handlers=[],
    )
    video = MagicMock()
    video.grab.return_value = False
    source_properties = assembly_dummy_source_properties(is_file=True, fps=16.0)
    buffer = Queue()

    # when
    consumer.reset(source_properties=source_properties)
    result = consumer.consume_frame(
        video=video,
        declared_source_fps=source_properties.fps,
        buffer=buffer,
        frames_buffering_allowed=True
    )

    # then
    assert result is False
    assert buffer.empty() is True


def test_stream_consumption_when_buffering_not_allowed() -> None:
    # given
    consumer = VideoConsumer.init(
        buffer_filling_strategy=None,
        adaptive_mode_stream_pace_tolerance=0.1,
        adaptive_mode_reader_pace_tolerance=5.0,
        minimum_adaptive_mode_samples=10,
        maximum_adaptive_frames_dropped_in_row=16,
        status_update_handlers=[],
    )
    video = MagicMock()
    video.grab.return_value = True
    source_properties = assembly_dummy_source_properties(is_file=True, fps=16.0)
    buffer = Queue()

    # when
    consumer.reset(source_properties=source_properties)
    result = consumer.consume_frame(
        video=video,
        declared_source_fps=source_properties.fps,
        buffer=buffer,
        frames_buffering_allowed=False
    )

    # then
    assert result is True
    assert buffer.empty() is True


def test_stream_consumption_when_buffer_is_ready_to_accept_frame_but_decoding_failed() -> None:
    # given
    consumer = VideoConsumer.init(
        buffer_filling_strategy=None,
        adaptive_mode_stream_pace_tolerance=0.1,
        adaptive_mode_reader_pace_tolerance=5.0,
        minimum_adaptive_mode_samples=10,
        maximum_adaptive_frames_dropped_in_row=16,
        status_update_handlers=[],
    )
    video = MagicMock()
    video.grab.return_value = True
    video.retrieve.return_value = (False, None)
    source_properties = assembly_dummy_source_properties(is_file=True, fps=16.0)
    buffer = Queue()

    # when
    consumer.reset(source_properties=source_properties)
    result = consumer.consume_frame(
        video=video,
        declared_source_fps=source_properties.fps,
        buffer=buffer,
        frames_buffering_allowed=True
    )

    # then
    assert result is False
    assert buffer.empty() is True


def test_stream_consumption_when_buffer_is_ready_to_accept_frame_and_decoding_succeed() -> None:
    # given
    consumer = VideoConsumer.init(
        buffer_filling_strategy=None,
        adaptive_mode_stream_pace_tolerance=0.1,
        adaptive_mode_reader_pace_tolerance=5.0,
        minimum_adaptive_mode_samples=10,
        maximum_adaptive_frames_dropped_in_row=16,
        status_update_handlers=[],
    )
    video = MagicMock()
    video.grab.return_value = True
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    video.retrieve.return_value = (True, image)
    source_properties = assembly_dummy_source_properties(is_file=True, fps=16.0)
    buffer = Queue()
    buffer.put((datetime.now(), -2, image))

    # when
    consumer.reset(source_properties=source_properties)
    result = consumer.consume_frame(
        video=video,
        declared_source_fps=source_properties.fps,
        buffer=buffer,
        frames_buffering_allowed=True
    )

    # then
    assert result is True
    assert buffer.get_nowait()[1] == -2
    buffered_result = buffer.get_nowait()
    assert issubclass(type(buffered_result[0]), datetime)
    assert buffered_result[1] == 1
    assert buffered_result[2] is image


def test_stream_consumption_when_buffer_full_and_latest_frames_to_be_dropped() -> None:
    # given
    consumer = VideoConsumer.init(
        buffer_filling_strategy=BufferFillingStrategy.DROP_LATEST,
        adaptive_mode_stream_pace_tolerance=0.1,
        adaptive_mode_reader_pace_tolerance=5.0,
        minimum_adaptive_mode_samples=10,
        maximum_adaptive_frames_dropped_in_row=16,
        status_update_handlers=[],
    )
    video = MagicMock()
    video.grab.return_value = True
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    video.retrieve.return_value = (True, image)
    source_properties = assembly_dummy_source_properties(is_file=True, fps=16.0)
    buffer = Queue(maxsize=1)
    buffer.put((datetime.now(), -2, image))

    # when
    consumer.reset(source_properties=source_properties)
    result = consumer.consume_frame(
        video=video,
        declared_source_fps=source_properties.fps,
        buffer=buffer,
        frames_buffering_allowed=True
    )

    # then
    assert result is True
    assert buffer.get_nowait()[1] == -2


def test_stream_consumption_when_buffer_full_and_oldest_frames_to_be_dropped() -> None:
    # given
    consumer = VideoConsumer.init(
        buffer_filling_strategy=BufferFillingStrategy.DROP_OLDEST,
        adaptive_mode_stream_pace_tolerance=0.1,
        adaptive_mode_reader_pace_tolerance=5.0,
        minimum_adaptive_mode_samples=10,
        maximum_adaptive_frames_dropped_in_row=16,
        status_update_handlers=[],
    )
    video = MagicMock()
    video.grab.return_value = True
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    video.retrieve.return_value = (True, image)
    source_properties = assembly_dummy_source_properties(is_file=True, fps=16.0)
    buffer = Queue(maxsize=2)
    buffer.put((datetime.now(), -2, image))
    buffer.put((datetime.now(), -1, image))

    # when
    consumer.reset(source_properties=source_properties)
    result = consumer.consume_frame(
        video=video,
        declared_source_fps=source_properties.fps,
        buffer=buffer,
        frames_buffering_allowed=True
    )

    # then
    assert result is True
    assert buffer.get_nowait()[1] == -1
    buffered_result = buffer.get_nowait()
    assert issubclass(type(buffered_result[0]), datetime)
    assert buffered_result[1] == 1
    assert buffered_result[2] is image
    assert buffer.empty() is True


def test_stream_consumption_when_adaptive_strategy_does_not_prevent_decoding_due_to_not_enough_observations() -> None:
    # given
    consumer = VideoConsumer.init(
        buffer_filling_strategy=BufferFillingStrategy.ADAPTIVE_DROP_OLDEST,
        adaptive_mode_stream_pace_tolerance=0.1,
        adaptive_mode_reader_pace_tolerance=5.0,
        minimum_adaptive_mode_samples=10,
        maximum_adaptive_frames_dropped_in_row=16,
        status_update_handlers=[],
    )
    video = MagicMock()
    video.grab.return_value = True
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    video.retrieve.return_value = (True, image)
    source_properties = assembly_dummy_source_properties(is_file=True, fps=16.0)
    buffer = Queue(maxsize=2)
    buffer.put((datetime.now(), -2, image))
    buffer.put((datetime.now(), -1, image))

    # when
    consumer.reset(source_properties=source_properties)
    result = consumer.consume_frame(
        video=video,
        declared_source_fps=source_properties.fps,
        buffer=buffer,
        frames_buffering_allowed=True
    )

    # then
    assert result is True
    assert buffer.get_nowait()[1] == -1
    buffered_result = buffer.get_nowait()
    assert issubclass(type(buffered_result[0]), datetime)
    assert buffered_result[1] == 1
    assert buffered_result[2] is image
    assert buffer.empty() is True


@pytest.mark.slow
def test_stream_consumption_when_adaptive_strategy_eventually_stops_preventing_decoding_after_series_of_preventions() -> None:
    # given
    consumer = VideoConsumer.init(
        buffer_filling_strategy=BufferFillingStrategy.ADAPTIVE_DROP_OLDEST,
        adaptive_mode_stream_pace_tolerance=0.1,
        adaptive_mode_reader_pace_tolerance=200.0,
        minimum_adaptive_mode_samples=2,
        maximum_adaptive_frames_dropped_in_row=4,
        status_update_handlers=[],
    )
    video = MagicMock()
    video.grab.return_value = True
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    video.retrieve.return_value = (True, image)
    source_properties = assembly_dummy_source_properties(is_file=True, fps=200)
    buffer = Queue(maxsize=2)
    buffer.put((datetime.now(), -2, image))
    buffer.put((datetime.now(), -1, image))

    # when
    results = []
    buffer_content = []
    consumer.reset(source_properties=source_properties)
    for _ in range(2):
        result = consumer.consume_frame(
            video=video,
            declared_source_fps=source_properties.fps,
            buffer=buffer,
            frames_buffering_allowed=True
        )
        results.append(result)
        time.sleep(0.01)
    buffer_content.append(buffer.get_nowait())
    consumer.notify_frame_consumed()
    buffer_content.append(buffer.get_nowait())
    consumer.notify_frame_consumed()
    for _ in range(6):
        result = consumer.consume_frame(
            video=video,
            declared_source_fps=source_properties.fps,
            buffer=buffer,
            frames_buffering_allowed=True
        )
        results.append(result)
        time.sleep(0.01)
    buffer_content.append(buffer.get_nowait())
    consumer.notify_frame_consumed()

    # then
    # First two frames will cause decoding, then 4 will be dropped due to not
    # keeping up to 200fps stream (we emit at 100fps), next will be submitted to buffer,
    # but since we are still lagging, the last frame will be rejected
    assert results == [True] * 8
    assert len(buffer_content) == 3
    assert buffer_content[0][1] == 1
    assert buffer_content[1][1] == 2
    assert buffer_content[2][1] == 7
    assert buffer.empty() is True


@pytest.mark.slow
def test_stream_consumption_when_adaptive_strategy_is_disabled_when_announced_fps_is_not_given_and_consumer_actively_read() -> None:
    # given
    consumer = VideoConsumer.init(
        buffer_filling_strategy=BufferFillingStrategy.ADAPTIVE_DROP_OLDEST,
        adaptive_mode_stream_pace_tolerance=5.0,
        adaptive_mode_reader_pace_tolerance=200.0,
        minimum_adaptive_mode_samples=2,
        maximum_adaptive_frames_dropped_in_row=4,
        status_update_handlers=[],
    )
    video = MagicMock()
    video.grab.return_value = True
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    video.retrieve.return_value = (True, image)
    source_properties = assembly_dummy_source_properties(is_file=True, fps=200)
    buffer = Queue(maxsize=2)
    buffer.put((datetime.now(), -2, image))
    buffer.put((datetime.now(), -1, image))

    # when
    results = []
    buffer_content = []
    consumer.reset(source_properties=source_properties)
    for _ in range(2):
        result = consumer.consume_frame(
            video=video,
            declared_source_fps=None,
            buffer=buffer,
            frames_buffering_allowed=True
        )
        results.append(result)
        time.sleep(0.01)
    buffer_content.append(buffer.get_nowait())
    consumer.notify_frame_consumed()
    buffer_content.append(buffer.get_nowait())
    consumer.notify_frame_consumed()
    for _ in range(2):
        result = consumer.consume_frame(
            video=video,
            declared_source_fps=None,
            buffer=buffer,
            frames_buffering_allowed=True
        )
        results.append(result)
        buffer_content.append(buffer.get_nowait())
        consumer.notify_frame_consumed()
        time.sleep(0.01)

    # then
    # As stream FPS is not announced - we cannot reject any frame
    assert results == [True] * 4
    assert len(buffer_content) == 4
    assert buffer_content[0][1] == 1
    assert buffer_content[1][1] == 2
    assert buffer_content[2][1] == 3
    assert buffer_content[3][1] == 4
    assert buffer.empty() is True


def assembly_dummy_source_properties(is_file: bool, fps: float) -> SourceProperties:
    return SourceProperties(
        width=128,
        height=128,
        total_frames=-1 if is_file else 10,
        is_file=is_file,
        fps=fps,
    )
