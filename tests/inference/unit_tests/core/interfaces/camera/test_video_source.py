import time
from datetime import datetime
from queue import Queue
from threading import Thread
from unittest import mock
from unittest.mock import MagicMock

import cv2
import numpy as np
import supervision as sv
import pytest

from inference.core.interfaces.camera.entities import (
    StatusUpdate,
    UpdateSeverity,
    VideoFrame,
)
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
    purge_queue,
    drop_single_frame_from_buffer,
    decode_video_frame_to_buffer,
    VideoConsumer,
    SourceProperties,
    get_fps_if_tick_happens_now,
)
from inference.core.interfaces.camera import video_source


def tear_down_source(source: VideoSource) -> None:
    source.terminate(wait_on_frames_consumption=False)


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
    source = VideoSource.init(video_reference="invalid")

    # when
    with pytest.raises(SourceConnectionError):
        source.start()


def test_video_source_describe_source_when_stream_consumption_not_yet_started() -> None:
    # given
    source = VideoSource.init(video_reference="invalid")

    # when
    result = source.describe_source()

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
    source = VideoSource.init(video_reference="invalid")

    # when
    with pytest.raises(SourceConnectionError):
        source.start()
    result = source.describe_source()

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
    source = VideoSource.init(video_reference=local_video_path)

    try:
        # when
        source.start()
        _ = source.read_frame()
        result = source.describe_source()

        # then
        assert result.source_properties.is_file is True
        assert result.source_reference == local_video_path
        assert result.state is StreamState.RUNNING
        assert result.buffer_filling_strategy is BufferFillingStrategy.WAIT
        assert result.buffer_consumption_strategy is BufferConsumptionStrategy.LAZY

    finally:
        tear_down_source(source=source)


@pytest.mark.timeout(90)
@pytest.mark.slow
def test_pausing_video_stream(local_video_path: str) -> None:
    # given
    source = VideoSource.init(video_reference=local_video_path)
    pause_resume_delay = 0.2

    try:
        # when
        source.start()
        _ = source.read_frame()
        source.pause()
        time.sleep(pause_resume_delay)
        timestamp_of_resume = datetime.now()
        source.resume()
        last_frame_before_resume, first_frame_after_resume = None, None
        while True:
            frame = source.read_frame()
            if (frame.frame_timestamp - timestamp_of_resume).total_seconds() >= 0:
                first_frame_after_resume = frame
                break
            last_frame_before_resume = frame

        # then
        assert (
            first_frame_after_resume.frame_timestamp
            - last_frame_before_resume.frame_timestamp
        ).total_seconds() >= pause_resume_delay
    finally:
        tear_down_source(source=source)


@pytest.mark.timeout(90)
@pytest.mark.slow
def test_muting_video_stream_succeeds(local_video_path: str) -> None:
    # given
    source = VideoSource.init(video_reference=local_video_path)

    try:
        # when
        source.start()
        _ = source.read_frame()
        source.mute()
        try:
            source.resume()
        except StreamOperationNotAllowedError:
            pass

    finally:
        tear_down_source(source=source)


@pytest.mark.timeout(90)
@pytest.mark.slow
def test_restart_paused_stream_for_video_preserves_frames_continuity(
    local_video_path: str,
) -> None:
    # given
    source = VideoSource.init(video_reference=local_video_path)
    frames_before_restart = []

    def capture_frames() -> None:
        for f in source:
            frames_before_restart.append(f)

    capture_thread = Thread(target=capture_frames)

    try:
        # when
        source.start()
        frame = source.read_frame()
        last_id_before_restart = frame.frame_id
        capture_thread.start()
        source.pause()
        restart_timestamp = datetime.now()
        source.restart()
        capture_thread.join()
        frame_after_restart = source.read_frame()

        # then
        for frame_before_restart in frames_before_restart:
            last_id_before_restart = frame_before_restart.frame_id
        assert frame_after_restart.frame_id == last_id_before_restart + 1
        assert (
            frame_after_restart.frame_timestamp - restart_timestamp
        ).total_seconds() > 0
    finally:
        tear_down_source(source=source)


@pytest.mark.timeout(90)
@pytest.mark.slow
def test_restart_muted_stream_completes_successfully(local_video_path: str) -> None:
    # given
    source = VideoSource.init(video_reference=local_video_path)
    frames_before_restart = []

    def capture_frames() -> None:
        for f in source:
            frames_before_restart.append(f)

    capture_thread = Thread(target=capture_frames)

    try:
        # when
        source.start()
        _ = source.read_frame()
        capture_thread.start()
        source.mute()
        restart_timestamp = datetime.now()
        source.restart()
        capture_thread.join()
        frame_after_restart = source.read_frame()

        # then
        assert (
            frame_after_restart.frame_timestamp - restart_timestamp
        ).total_seconds() > 0
    finally:
        tear_down_source(source=source)


@pytest.mark.timeout(90)
@pytest.mark.slow
def test_restart_running_stream_preserves_frame_id_continuity(
    local_video_path: str,
) -> None:
    # given
    source = VideoSource.init(video_reference=local_video_path)
    frames_before_restart = []

    def capture_frames() -> None:
        for f in source:
            frames_before_restart.append(f)

    capture_thread = Thread(target=capture_frames)

    try:
        # when
        source.start()
        frame = source.read_frame()
        last_id_before_restart = frame.frame_id
        capture_thread.start()
        restart_timestamp = datetime.now()
        source.restart()
        capture_thread.join()
        frame_after_restart = source.read_frame()

        # then
        for frame_before_restart in frames_before_restart:
            last_id_before_restart = frame_before_restart.frame_id
        assert frame_after_restart.frame_id == last_id_before_restart + 1
        assert (
            frame_after_restart.frame_timestamp - restart_timestamp
        ).total_seconds() > 0
    finally:
        tear_down_source(source=source)


@pytest.mark.timeout(90)
@pytest.mark.slow
def test_terminate_running_stream_succeeds(local_video_path: str) -> None:
    # given
    source = VideoSource.init(video_reference=local_video_path)
    frames_captured = []

    def capture_frames() -> None:
        for f in source:
            frames_captured.append(f)

    capture_thread = Thread(target=capture_frames)

    # when
    source.start()
    _ = source.read_frame()
    capture_thread.start()
    source.terminate()
    capture_thread.join()

    # then - nothing hangs


@pytest.mark.timeout(90)
@pytest.mark.slow
def test_terminate_paused_stream_succeeds(local_video_path: str) -> None:
    # given
    source = VideoSource.init(video_reference=local_video_path)
    frames_captured = []

    def capture_frames() -> None:
        for f in source:
            frames_captured.append(f)

    capture_thread = Thread(target=capture_frames)

    # when
    source.start()
    _ = source.read_frame()
    source.pause()
    capture_thread.start()
    source.terminate()
    capture_thread.join()

    # then - nothing hangs


@pytest.mark.timeout(90)
@pytest.mark.slow
def test_terminate_muted_stream_succeeds(local_video_path: str) -> None:
    # given
    source = VideoSource.init(video_reference=local_video_path)
    frames_captured = []

    def capture_frames() -> None:
        for f in source:
            frames_captured.append(f)

    capture_thread = Thread(target=capture_frames)

    # when
    source.start()
    _ = source.read_frame()
    source.mute()
    capture_thread.start()
    source.terminate()
    capture_thread.join()

    # then - nothing hangs


def test_pause_not_started_stream_fails(local_video_path: str) -> None:
    # given
    source = VideoSource.init(video_reference=local_video_path)

    # when
    with pytest.raises(StreamOperationNotAllowedError):
        source.pause()


def test_mute_not_started_stream_fails(local_video_path: str) -> None:
    # given
    source = VideoSource.init(video_reference=local_video_path)

    # when
    with pytest.raises(StreamOperationNotAllowedError):
        source.mute()


def test_restart_not_started_stream_fails(local_video_path: str) -> None:
    # given
    source = VideoSource.init(video_reference=local_video_path)

    # when
    with pytest.raises(StreamOperationNotAllowedError):
        source.restart()


def test_terminate_not_started_stream_fails(local_video_path: str) -> None:
    # given
    source = VideoSource.init(video_reference=local_video_path)

    # when
    with pytest.raises(StreamOperationNotAllowedError):
        source.terminate()


@pytest.mark.timeout(90)
@pytest.mark.slow
def test_pause_muted_stream_fails(local_video_path: str) -> None:
    # given
    source = VideoSource.init(video_reference=local_video_path)

    try:
        # when
        source.start()
        _ = source.read_frame()
        source.mute()

        # then
        with pytest.raises(StreamOperationNotAllowedError):
            source.pause()
    finally:
        tear_down_source(source=source)


@pytest.mark.timeout(90)
@pytest.mark.slow
def test_mute_paused_stream_fails(local_video_path: str) -> None:
    # given
    source = VideoSource.init(video_reference=local_video_path)

    try:
        # when
        source.start()
        _ = source.read_frame()
        source.pause()

        # then
        with pytest.raises(StreamOperationNotAllowedError):
            source.mute()
    finally:
        tear_down_source(source=source)


@pytest.mark.timeout(90)
@pytest.mark.slow
def test_pause_paused_stream_fails(local_video_path: str) -> None:
    # given
    source = VideoSource.init(video_reference=local_video_path)

    try:
        # when
        source.start()
        _ = source.read_frame()
        source.pause()

        # then
        with pytest.raises(StreamOperationNotAllowedError):
            source.pause()
    finally:
        tear_down_source(source=source)


@pytest.mark.timeout(90)
@pytest.mark.slow
def test_mute_muted_stream_fails(local_video_path: str) -> None:
    # given
    source = VideoSource.init(video_reference=local_video_path)

    try:
        # when
        source.start()
        _ = source.read_frame()
        source.mute()

        # then
        with pytest.raises(StreamOperationNotAllowedError):
            source.mute()
    finally:
        tear_down_source(source=source)


@pytest.mark.timeout(90)
@pytest.mark.slow
def test_consumption_of_video_file_in_eager_mode_ends_successfully(
    local_video_path: str,
) -> None:
    # given
    source = VideoSource.init(
        video_reference=local_video_path,
        buffer_filling_strategy=BufferFillingStrategy.DROP_OLDEST,
        buffer_consumption_strategy=BufferConsumptionStrategy.EAGER,
    )

    try:
        # when
        frames_consumed = 0
        source.start()
        for _ in source:
            frames_consumed += 1

        # then
        assert 0 <= frames_consumed <= 431
    finally:
        tear_down_source(source=source)


def test_drop_single_frame_from_buffer_when_buffer_is_empty() -> None:
    # given
    buffer = Queue()
    updates = []

    def handle_status_updates(status_update: StatusUpdate) -> None:
        updates.append(status_update)

    # when
    drop_single_frame_from_buffer(
        buffer=buffer, cause="some", status_update_handlers=[handle_status_updates]
    )

    # then
    assert len(updates) == 0


def test_drop_single_frame_from_buffer_when_buffer_has_video_frame() -> None:
    # given
    buffer = Queue()
    updates = []
    frame_timestamp = datetime.now()
    video_frame = VideoFrame(
        image=np.zeros((128, 128, 3), dtype=np.uint8),
        frame_timestamp=frame_timestamp,
        frame_id=37,
    )
    buffer.put(video_frame)

    def handle_status_updates(status_update: StatusUpdate) -> None:
        updates.append(status_update)

    # when
    drop_single_frame_from_buffer(
        buffer=buffer, cause="some", status_update_handlers=[handle_status_updates]
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
    assert buffer.get_nowait() == VideoFrame(
        image=image, frame_id=1, frame_timestamp=frame_timestamp
    )


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
        frames_buffering_allowed=True,
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
        frames_buffering_allowed=False,
    )

    # then
    assert result is True
    assert buffer.empty() is True


def test_stream_consumption_when_buffer_is_ready_to_accept_frame_but_decoding_failed() -> (
    None
):
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
        frames_buffering_allowed=True,
    )

    # then
    assert result is False
    assert buffer.empty() is True


def test_stream_consumption_when_buffer_is_ready_to_accept_frame_and_decoding_succeed() -> (
    None
):
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
    buffer.put(VideoFrame(image=image, frame_timestamp=datetime.now(), frame_id=-2))

    # when
    consumer.reset(source_properties=source_properties)
    result = consumer.consume_frame(
        video=video,
        declared_source_fps=source_properties.fps,
        buffer=buffer,
        frames_buffering_allowed=True,
    )

    # then
    assert result is True
    assert buffer.get_nowait().frame_id == -2
    buffered_result = buffer.get_nowait()
    assert buffered_result.frame_id == 1
    assert buffered_result.image is image


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
    buffer.put(VideoFrame(image=image, frame_timestamp=datetime.now(), frame_id=-2))

    # when
    consumer.reset(source_properties=source_properties)
    result = consumer.consume_frame(
        video=video,
        declared_source_fps=source_properties.fps,
        buffer=buffer,
        frames_buffering_allowed=True,
    )

    # then
    assert result is True
    assert buffer.get_nowait().frame_id == -2


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
    buffer.put(VideoFrame(image=image, frame_timestamp=datetime.now(), frame_id=-2))
    buffer.put(VideoFrame(image=image, frame_timestamp=datetime.now(), frame_id=-1))

    # when
    consumer.reset(source_properties=source_properties)
    result = consumer.consume_frame(
        video=video,
        declared_source_fps=source_properties.fps,
        buffer=buffer,
        frames_buffering_allowed=True,
    )

    # then
    assert result is True
    assert buffer.get_nowait().frame_id == -1
    buffered_result = buffer.get_nowait()
    assert buffered_result.frame_id == 1
    assert buffered_result.image is image
    assert buffer.empty() is True


def test_stream_consumption_when_adaptive_strategy_does_not_prevent_decoding_due_to_not_enough_observations() -> (
    None
):
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
    buffer.put(VideoFrame(image=image, frame_timestamp=datetime.now(), frame_id=-2))
    buffer.put(VideoFrame(image=image, frame_timestamp=datetime.now(), frame_id=-1))

    # when
    consumer.reset(source_properties=source_properties)
    result = consumer.consume_frame(
        video=video,
        declared_source_fps=source_properties.fps,
        buffer=buffer,
        frames_buffering_allowed=True,
    )

    # then
    assert result is True
    assert buffer.get_nowait().frame_id == -1
    buffered_result = buffer.get_nowait()
    assert buffered_result.frame_id == 1
    assert buffered_result.image is image
    assert buffer.empty() is True


@pytest.mark.slow
def test_stream_consumption_when_adaptive_strategy_eventually_stops_preventing_decoding_after_series_of_preventions() -> (
    None
):
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
    buffer.put(VideoFrame(image=image, frame_timestamp=datetime.now(), frame_id=-2))
    buffer.put(VideoFrame(image=image, frame_timestamp=datetime.now(), frame_id=-1))

    # when
    results = []
    buffer_content = []
    consumer.reset(source_properties=source_properties)
    for _ in range(2):
        result = consumer.consume_frame(
            video=video,
            declared_source_fps=source_properties.fps,
            buffer=buffer,
            frames_buffering_allowed=True,
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
            frames_buffering_allowed=True,
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
    assert buffer_content[0].frame_id == 1
    assert buffer_content[1].frame_id == 2
    assert buffer_content[2].frame_id == 7
    assert buffer.empty() is True


@pytest.mark.slow
def test_stream_consumption_when_adaptive_strategy_is_disabled_as_announced_fps_is_not_given_and_consumer_actively_read() -> (
    None
):
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
    buffer.put(VideoFrame(image=image, frame_timestamp=datetime.now(), frame_id=-2))
    buffer.put(VideoFrame(image=image, frame_timestamp=datetime.now(), frame_id=-1))

    # when
    results = []
    buffer_content = []
    consumer.reset(source_properties=source_properties)
    for _ in range(2):
        result = consumer.consume_frame(
            video=video,
            declared_source_fps=None,
            buffer=buffer,
            frames_buffering_allowed=True,
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
            frames_buffering_allowed=True,
        )
        results.append(result)
        buffer_content.append(buffer.get_nowait())
        consumer.notify_frame_consumed()
        time.sleep(0.01)

    # then
    # As stream FPS is not announced - we cannot reject any frame
    assert results == [True] * 4
    assert len(buffer_content) == 4
    assert buffer_content[0].frame_id == 1
    assert buffer_content[1].frame_id == 2
    assert buffer_content[2].frame_id == 3
    assert buffer_content[3].frame_id == 4
    assert buffer.empty() is True


@pytest.mark.slow
def test_stream_consumption_when_adaptive_strategy_drops_frames_due_to_reader_lag() -> (
    None
):
    # given
    consumer = VideoConsumer.init(
        buffer_filling_strategy=BufferFillingStrategy.ADAPTIVE_DROP_OLDEST,
        adaptive_mode_stream_pace_tolerance=100.0,
        adaptive_mode_reader_pace_tolerance=0.1,
        minimum_adaptive_mode_samples=2,
        maximum_adaptive_frames_dropped_in_row=10,
        status_update_handlers=[],
    )
    video = MagicMock()
    video.grab.return_value = True
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    video.retrieve.return_value = (True, image)
    source_properties = assembly_dummy_source_properties(is_file=True, fps=100)
    buffer = Queue()

    # when
    results = []
    buffer_content = []
    consumer.reset(source_properties=source_properties)
    for _ in range(3):
        result = consumer.consume_frame(
            video=video,
            declared_source_fps=None,
            buffer=buffer,
            frames_buffering_allowed=True,
        )
        results.append(result)
        time.sleep(0.01)
        buffer_content.append(buffer.get_nowait())
        consumer.notify_frame_consumed()
    for _ in range(100):
        result = consumer.consume_frame(
            video=video,
            declared_source_fps=None,
            buffer=buffer,
            frames_buffering_allowed=True,
        )
        results.append(result)
    while not buffer.empty():
        buffer_content.append(buffer.get_nowait())
        consumer.notify_frame_consumed()

    # then
    # Reader acked only minimal number of frames - over time we expect decoding pace to vanish, and
    # we will start dropping frames
    assert results == [True] * 103
    assert len(buffer_content) < 103
    assert buffer.empty() is True


def test_get_fps_if_tick_happens_now_when_monitor_has_no_ticks_registered() -> None:
    # given
    monitor = sv.FPSMonitor()

    # when
    result = get_fps_if_tick_happens_now(fps_monitor=monitor)

    # then
    assert abs(result) < 1e-5


@mock.patch.object(video_source.time, "monotonic")
def test_get_fps_if_tick_happens_now_when_monitor_has_tick_registered(
    monotonic_mock: MagicMock,
) -> None:
    # given
    monitor = sv.FPSMonitor()
    monitor.all_timestamps.append(0.1)
    monotonic_mock.return_value = 0.2

    # when
    result = get_fps_if_tick_happens_now(fps_monitor=monitor)

    # then
    # 10ms per two tics, so 20fps
    assert abs(result - 20) < 1e-5


def assembly_dummy_source_properties(is_file: bool, fps: float) -> SourceProperties:
    return SourceProperties(
        width=128,
        height=128,
        total_frames=-1 if is_file else 10,
        is_file=is_file,
        fps=fps,
    )
