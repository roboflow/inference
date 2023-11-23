import time
from datetime import datetime
from queue import Queue
from threading import Thread

import cv2
import supervision as sv
import pytest

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
