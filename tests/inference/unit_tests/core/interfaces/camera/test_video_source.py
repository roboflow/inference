import time
from datetime import datetime
from queue import Queue
from threading import Thread
from unittest import mock
from unittest.mock import MagicMock, call, patch

import cv2
import numpy as np
import pytest
import supervision as sv

from inference.core.interfaces.camera import video_source
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
    CV2VideoFrameProducer,
    SourceMetadata,
    SourceProperties,
    StreamState,
    VideoConsumer,
    VideoSource,
    decode_video_frame_to_buffer,
    drop_single_frame_from_buffer,
    get_fps_if_tick_happens_now,
    get_from_queue,
)


def tear_down_source(source: VideoSource) -> None:
    source.terminate(wait_on_frames_consumption=False)


def test_get_from_queue_when_empty_queue_given_and_await_not_desired() -> None:
    # given
    queue = Queue()

    # when
    result = get_from_queue(queue=queue, timeout=0.0)

    # then
    assert (
        result is None
    ), "Purging empty queue should yield empty result when waiting is not desired"


def test_get_from_queue_when_non_empty_queue_given_and_purge_disabled() -> None:
    # given
    queue = Queue()
    queue.put(1)
    queue.put(2)
    queue.put(3)

    # when
    result = get_from_queue(queue=queue, purge=False)

    # then
    assert (
        result == 1
    ), "As a result of non-empty queue purge - last inserted value should be returned"
    assert (
        queue.empty() is False
    ), "After purge - queue must be empty if there is no external producer"


def test_get_from_queue_when_non_empty_queue_given_and_purge_enabled() -> None:
    # given
    queue = Queue()
    queue.put(1)
    queue.put(2)
    queue.put(3)

    # when
    result = get_from_queue(queue=queue, purge=True)

    # then
    assert (
        result == 3
    ), "As a result of non-empty queue purge - last inserted value should be returned"
    assert (
        queue.empty() is True
    ), "After purge - queue must be empty if there is no external producer"


def test_get_from_queue_when_non_empty_queue_given_with_callback() -> None:
    # given
    successful_reads = []

    def on_successful_read() -> None:
        successful_reads.append(1)

    queue = Queue()
    queue.put(1)
    queue.put(2)
    queue.put(3)

    # when
    _ = get_from_queue(queue=queue, on_successful_read=on_successful_read, purge=True)

    # then
    assert (
        queue.empty() is True
    ), "After purge - queue must be empty if there is no external producer"
    assert (
        len(successful_reads) == 3
    ), "Callback should be called each time result is taken out from queue"


def test_discover_source_properties_when_local_file_given(
    local_video_path: str,
) -> None:
    # given
    video = CV2VideoFrameProducer(local_video_path)

    # when
    result = video.discover_source_properties()

    # then
    assert result.is_file is True, "Path refers to video file, not stream"
    assert result.total_frames == 431, "This video has 431 frames in total"
    assert result.height == 240, "Video height is 240"
    assert result.width == 426, "Video height is 426"
    assert abs(result.fps - 30.0) < 1e-5, "Video file FPS is around 30"


def test_video_source_throwing_error_when_invalid_video_reference_given() -> None:
    # given
    source = VideoSource.init(video_reference="invalid")

    # when
    with pytest.raises(SourceConnectionError):
        source.start()


def test_video_source_describe_source_when_stream_consumption_not_yet_started() -> None:
    # given
    source = VideoSource.init(video_reference="invalid", source_id=2)

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
        source_id=2,
    ), "Source description must denote NOT_STARTED state and invalid source reference"


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
        source_id=None,
    ), "Source description must denote error regarding to connection to invalid source"


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
        assert (
            result.source_properties.is_file is True
        ), "Video file given to VideoSource, not stream"
        assert (
            result.source_reference == local_video_path
        ), "Source reference must match passed video path"
        assert (
            result.state is StreamState.RUNNING
        ), "After reading first frame, before consumption of all frames in default mode, RUNNING is expected state"
        assert (
            result.buffer_filling_strategy is BufferFillingStrategy.WAIT
        ), "Default strategy for filling buffer for video files is WAIT"
        assert (
            result.buffer_consumption_strategy is BufferConsumptionStrategy.LAZY
        ), "Default strategy for consuming buffer for video files is WAIT"

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
        source.read_frame()
        source.pause()
        source._video_consumer._timestamp_created = None  # simulate stream
        source.resume()
        last_frame_before_resume = source.read_frame()
        source.pause()
        time.sleep(pause_resume_delay)
        timestamp_of_resume = datetime.now()
        source.resume()

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
        ).total_seconds() >= pause_resume_delay, "Between first frame decoded after resume and last before pause must a break - at least as long as in between of .pause() and .resume() operations"
    finally:
        tear_down_source(source=source)


@pytest.mark.timeout(90)
@pytest.mark.slow
def test_pausing_video_file(local_video_path: str) -> None:
    # given
    source = VideoSource.init(video_reference=local_video_path)
    pause_resume_delay = 0.2

    try:
        # when
        source.start()
        last_frame_before_resume = source.read_frame()
        source.pause()
        time.sleep(pause_resume_delay)
        source.resume()

        while True:
            frame = source.read_frame()
            if frame.frame_id != last_frame_before_resume.frame_id:
                first_frame_after_resume = frame
                break
            last_frame_before_resume = frame

        # then
        assert (
            first_frame_after_resume.frame_id - last_frame_before_resume.frame_id
        ) == 1, "Pausing and resuming video file must not result in missing frames"
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
        source.read_frame()
        source.pause()
        source._video_consumer._timestamp_created = None  # simulate stream
        source.resume()
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
        assert (
            frame_after_restart.frame_id == last_id_before_restart + 1
        ), "Next frame after restart has next consecutive id after the last one before restart"
        assert (
            frame_after_restart.frame_timestamp - restart_timestamp
        ).total_seconds() > 0, (
            "First frame after restart cannot be decoded faster than restart happens"
        )
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
        source.read_frame()
        source.pause()
        source._video_consumer._timestamp_created = None  # simulate stream
        source.resume()
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
        ).total_seconds() > 0, (
            "First frame after restart cannot be decoded faster than restart happens"
        )
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
        source.read_frame()
        source.pause()
        source._video_consumer._timestamp_created = None  # simulate stream
        source.resume()
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
        assert (
            frame_after_restart.frame_id == last_id_before_restart + 1
        ), "Next frame after restart has next consecutive id after the last one before restart"
        assert (
            frame_after_restart.frame_timestamp - restart_timestamp
        ).total_seconds() > 0, (
            "First frame after restart cannot be decoded faster than restart happens"
        )
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
def test_terminate_running_stream_succeeds_with_buffer_purging(
    local_video_path: str,
) -> None:
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
    source.terminate(purge_frames_buffer=True)
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
        assert (
            0 <= frames_consumed <= 431
        ), "Video has 431 frames, and that's maximum amount that should be processed, some frames may be dropped"
    finally:
        tear_down_source(source=source)


@pytest.mark.timeout(90)
@pytest.mark.slow
def test_consumption_of_video_file_with_desired_fps_succeeds(
    local_video_path: str,
) -> None:
    # given
    source = VideoSource.init(
        video_reference=local_video_path,
        desired_fps=10,
    )

    try:
        # when
        frames_consumed = 0
        source.start()
        for _ in source:
            frames_consumed += 1

        # then
        assert frames_consumed <= 150, (
            "Video has 431 frames at 30fps, at max we should process 144 frames, "
            "with slight randomness possible"
        )
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
        buffer=buffer,
        cause="some",
        status_update_handlers=[handle_status_updates],
    )

    # then
    assert (
        len(updates) == 0
    ), "No updates should be emitted once there was nothing to be dropped"


def test_drop_single_frame_from_buffer_when_buffer_has_video_frame() -> None:
    # given
    buffer = Queue()
    updates = []
    frame_timestamp = datetime.now()
    video_frame = VideoFrame(
        image=np.zeros((128, 128, 3), dtype=np.uint8),
        frame_timestamp=frame_timestamp,
        frame_id=37,
        source_id=3,
    )
    buffer.put(video_frame)

    def handle_status_updates(status_update: StatusUpdate) -> None:
        updates.append(status_update)

    # when
    drop_single_frame_from_buffer(
        buffer=buffer,
        cause="some",
        status_update_handlers=[handle_status_updates],
    )

    # then
    assert len(updates) == 1, "Exactly one frame can be dropped - emitting one update"
    assert updates[0].payload == {
        "frame_timestamp": frame_timestamp,
        "frame_id": 37,
        "source_id": 3,
        "cause": "some",
    }, "Dropped frames details must match content of the buffer"
    assert (
        updates[0].severity is UpdateSeverity.DEBUG
    ), "Severity of DROP event is DEBUG"
    assert updates[0].event_type == "FRAME_DROPPED", "Event type must match"


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
        source_id=38,
    )

    # then
    assert result is False, "Success status on failure must be False"
    assert len(fps_monitor.all_timestamps) == 1, "FPS monitor tick must not be emitted"
    assert (
        buffer.empty() is True
    ), "Nothing can be pushed to buffer once frame decoding failed"


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
        source_id=3,
    )

    # then
    assert result is True, "Success status on decoding must be denoted"
    assert (
        len(fps_monitor.all_timestamps) == 2
    ), "FPS monitor tick must be emitted on success"
    assert buffer.get_nowait() == VideoFrame(
        image=image,
        frame_id=1,
        frame_timestamp=frame_timestamp,
        source_id=3,
    ), "Decoded frame must be saved into buffer"


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
    video.discover_source_properties.return_value = source_properties
    buffer = Queue()

    # when
    consumer.reset(source_properties=source_properties)
    result = consumer.consume_frame(
        video=video,
        declared_source_fps=source_properties.fps,
        is_source_video_file=source_properties.is_file,
        buffer=buffer,
        frames_buffering_allowed=True,
    )

    # then
    assert (
        result is False
    ), "Failure status must be denoted, once grabbing frame is unsuccessful"
    assert buffer.empty() is True, "On failure, nothing can be populated into buffer"


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
    video.discover_source_properties.return_value = source_properties
    buffer = Queue()

    # when
    consumer.reset(source_properties=source_properties)
    result = consumer.consume_frame(
        video=video,
        declared_source_fps=source_properties.fps,
        is_source_video_file=source_properties.is_file,
        buffer=buffer,
        frames_buffering_allowed=False,
    )

    # then
    assert (
        result is True
    ), "Success status must be denoted, once grabbing frame is successful"
    assert (
        buffer.empty() is True
    ), "Despite success in frame grabbing, frames buffering is not allowed, so buffer must not be populated with frame"


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
    video.discover_source_properties.return_value = source_properties
    buffer = Queue()

    # when
    consumer.reset(source_properties=source_properties)
    result = consumer.consume_frame(
        video=video,
        declared_source_fps=source_properties.fps,
        is_source_video_file=source_properties.is_file,
        buffer=buffer,
        frames_buffering_allowed=True,
    )

    # then
    assert (
        result is False
    ), "Failure status must be denoted, once grabbing frame is successful but decoding failed"
    assert (
        buffer.empty() is True
    ), "On decoding failure, nothing can be populated into buffer"


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
    video.discover_source_properties.return_value = source_properties
    buffer = Queue()
    buffer.put(VideoFrame(image=image, frame_timestamp=datetime.now(), frame_id=-2))

    # when
    consumer.reset(source_properties=source_properties)
    result = consumer.consume_frame(
        video=video,
        declared_source_fps=source_properties.fps,
        is_source_video_file=source_properties.is_file,
        buffer=buffer,
        frames_buffering_allowed=True,
    )

    # then
    assert result is True, "Operation status must be denoted"
    assert (
        buffer.get_nowait().frame_id == -2
    ), "Previously injected video frame must be possible to recover from buffer"
    buffered_result = buffer.get_nowait()
    assert (
        buffered_result.frame_id == 1
    ), "Newly decoded frame must be possible to be received from the buffer"
    assert (
        buffered_result.image is image
    ), "Newly decoded frame must be possible to be received from the buffer"


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
    video.discover_source_properties.return_value = source_properties
    buffer = Queue(maxsize=1)
    buffer.put(VideoFrame(image=image, frame_timestamp=datetime.now(), frame_id=-2))

    # when
    consumer.reset(source_properties=source_properties)
    result = consumer.consume_frame(
        video=video,
        declared_source_fps=source_properties.fps,
        is_source_video_file=source_properties.is_file,
        buffer=buffer,
        frames_buffering_allowed=True,
    )

    # then
    assert result is True, "Operation status must be denoted"
    assert (
        buffer.get_nowait().frame_id == -2
    ), "Buffer must contain old frame, as due to strategy, new one should be dropped"


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
    video.discover_source_properties.return_value = source_properties
    buffer = Queue(maxsize=2)
    buffer.put(VideoFrame(image=image, frame_timestamp=datetime.now(), frame_id=-2))
    buffer.put(VideoFrame(image=image, frame_timestamp=datetime.now(), frame_id=-1))

    # when
    consumer.reset(source_properties=source_properties)
    result = consumer.consume_frame(
        video=video,
        declared_source_fps=source_properties.fps,
        is_source_video_file=source_properties.is_file,
        buffer=buffer,
        frames_buffering_allowed=True,
    )

    # then
    assert result is True, "Operation status must be denoted"
    assert (
        buffer.get_nowait().frame_id == -1
    ), "Latest frame, with id=-2 must be dropped once new arrives"
    buffered_result = buffer.get_nowait()
    assert (
        buffered_result.frame_id == 1
    ), "Newly decoded frame must be possible to be received from the buffer"
    assert (
        buffered_result.image is image
    ), "Newly decoded frame must be possible to be received from the buffer"
    assert (
        buffer.empty() is True
    ), "Only 2 frames can be in the buffer, so after both being consumed - buffer is empty"


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
    video.discover_source_properties.return_value = source_properties
    buffer = Queue(maxsize=2)
    buffer.put(VideoFrame(image=image, frame_timestamp=datetime.now(), frame_id=-2))
    buffer.put(VideoFrame(image=image, frame_timestamp=datetime.now(), frame_id=-1))

    # when
    consumer.reset(source_properties=source_properties)
    result = consumer.consume_frame(
        video=video,
        declared_source_fps=source_properties.fps,
        is_source_video_file=source_properties.is_file,
        buffer=buffer,
        frames_buffering_allowed=True,
    )

    # then
    assert result is True, "Operation status must be denoted"
    assert (
        buffer.get_nowait().frame_id == -1
    ), "Latest frame, with id=-2 must be dropped once new arrives and ADAPTIVE strategy does not block it"
    buffered_result = buffer.get_nowait()
    assert (
        buffered_result.frame_id == 1
    ), "Newly decoded frame must be possible to be received from the buffer"
    assert (
        buffered_result.image is image
    ), "Newly decoded frame must be possible to be received from the buffer"
    assert (
        buffer.empty() is True
    ), "Only 2 frames can be in the buffer, so after both being consumed - buffer is empty"


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
    video.discover_source_properties.return_value = source_properties
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
            is_source_video_file=source_properties.is_file,
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
            is_source_video_file=source_properties.is_file,
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
    assert results == [True] * 8, "All of operation should succeed"
    assert (
        len(buffer_content) == 3
    ), "During processing, 3 buffered frames should be consumed"
    assert (
        buffer_content[0].frame_id == 1
    ), "First frame in the buffer is to be the newly decoded one with id 1, as adaptive strategy should not reach the readiness for rejection"
    assert (
        buffer_content[1].frame_id == 2
    ), "Second frame in the buffer is to be the newly decoded one with id 1, as adaptive strategy should not reach the readiness for rejection"
    assert (
        buffer_content[2].frame_id == 7
    ), "After second frame being processed, next 4 (due to `maximum_adaptive_frames_dropped_in_row` parameter) should be rejected as consumer lags, lettiong frame 7 do be emitted"
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
    video.discover_source_properties.return_value = source_properties
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
            is_source_video_file=None,
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
            is_source_video_file=None,
            buffer=buffer,
            frames_buffering_allowed=True,
        )
        results.append(result)
        buffer_content.append(buffer.get_nowait())
        consumer.notify_frame_consumed()
        time.sleep(0.01)

    # then
    # As stream FPS is not announced - we cannot reject any frame
    assert results == [True] * 4, "All of operation should succeed"
    assert (
        len(buffer_content) == 4
    ), "We should be able to grab all new frames from buffer"
    assert (
        buffer_content[0].frame_id == 1
    ), "Adaptive strategy should not prevent natural flow of decoding, hence frame 1 is first"
    assert (
        buffer_content[1].frame_id == 2
    ), "Adaptive strategy should not prevent natural flow of decoding, hence frame 2 is second"
    assert (
        buffer_content[2].frame_id == 3
    ), "Adaptive strategy should not prevent natural flow of decoding, hence frame 3 is third"
    assert (
        buffer_content[3].frame_id == 4
    ), "Adaptive strategy should not prevent natural flow of decoding, hence frame 4 is fourth"
    assert buffer.empty() is True, "Everything should be consumed from buffer"


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
    video.discover_source_properties.return_value = source_properties
    buffer = Queue()

    # when
    results = []
    buffer_content = []
    consumer.reset(source_properties=source_properties)
    for _ in range(3):
        result = consumer.consume_frame(
            video=video,
            declared_source_fps=None,
            is_source_video_file=None,
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
            is_source_video_file=None,
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
    assert results == [True] * 103, "All of operation should succeed"
    assert (
        len(buffer_content) < 103
    ), "With delay in stream consumption, not all frames can be processed as adaptive strategy taking into account reader pace should trigger decoding prevention"
    assert buffer.empty() is True, "Everything should be consumed from buffer"


def test_get_fps_if_tick_happens_now_when_monitor_has_no_ticks_registered() -> None:
    # given
    monitor = sv.FPSMonitor()

    # when
    result = get_fps_if_tick_happens_now(fps_monitor=monitor)

    # then
    assert (
        abs(result) < 1e-5
    ), "Once no ticks was emitted previously, 0 should be reported"


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
    # 100ms per two tics, so 20fps
    assert (
        abs(result - 20) < 1e-5
    ), "Two ticks happened in 109ms, so 20 ticks per second expected"


def assembly_dummy_source_properties(is_file: bool, fps: float) -> SourceProperties:
    return SourceProperties(
        width=128,
        height=128,
        total_frames=-1 if is_file else 10,
        is_file=is_file,
        fps=fps,
    )


@patch("cv2.VideoCapture")
def test_source_properties_initialized_on_video_source_using_string_values(
    mock_video_capture,
) -> None:
    mock_video = MagicMock()
    mock_video.retrieve.return_value = (False, None)
    mock_video_capture.return_value = mock_video

    # given
    source = VideoSource.init(
        video_reference="",
        video_source_properties={"frame_width": 1281, "frame_height": 721, "fps": 32.0},
    )

    # when
    source.start()

    # then
    source._video.stream.set.assert_has_calls(
        [
            call(cv2.CAP_PROP_FRAME_WIDTH, 1281),
            call(cv2.CAP_PROP_FRAME_HEIGHT, 721),
            call(cv2.CAP_PROP_FPS, 32.0),
        ],
        any_order=True,
    )
