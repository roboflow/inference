"""Unit tests for WebRTC session lifecycle management."""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import requests

from inference_sdk.config import WEBRTC_EVENT_LOOP_SHUTDOWN_TIMEOUT
from inference_sdk.webrtc.session import SessionState, VideoMetadata, WebRTCSession

THREAD_TIMEOUT = 5


class AsyncPeerConnectionStub:
    """Async peer connection with deterministic close controls."""

    def __init__(self, release=None, error=None):
        self.close_calls = 0
        self.close_started = threading.Event()
        self.release = release
        self.error = error

    async def close(self):
        self.close_calls += 1
        self.close_started.set()
        if self.release is not None:
            await self.release.wait()
        if self.error is not None:
            raise self.error

    def on(self, event):
        """Register an event callback."""

        def decorator(callback):
            return callback

        return decorator

    def createDataChannel(self, label):  # noqa: N802
        """Create a data channel stub."""
        return DataChannelStub()


class DataChannelStub:
    """Data channel with decorator-style event registration."""

    def on(self, event):
        """Register an event callback."""

        def decorator(callback):
            return callback

        return decorator


class AsyncSourceStub:
    """Async stream source that records cleanup calls."""

    def __init__(self, configure_error=None, cleanup_error=None):
        self.cleanup_calls = 0
        self.cleanup_started = threading.Event()
        self.configure_error = configure_error
        self.cleanup_error = cleanup_error

    async def configure_peer_connection(self, peer_connection):
        """Configure a peer connection or raise the requested error."""
        if self.configure_error is not None:
            raise self.configure_error

    async def cleanup(self):
        self.cleanup_calls += 1
        self.cleanup_started.set()
        if self.cleanup_error is not None:
            raise self.cleanup_error


def stop_leaked_session_loop(session):
    """Stop a loop leaked by the parent implementation after observation."""
    loop = session._loop
    loop_thread = session._loop_thread
    if loop is not None and not loop.is_closed():
        if loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        if loop_thread is not None and loop_thread.is_alive():
            loop_thread.join(timeout=THREAD_TIMEOUT)
        if not loop.is_closed():
            loop.close()
    session._state = SessionState.CLOSED


@pytest.fixture
def mock_session():
    """Create a mock WebRTCSession instance without actually initializing WebRTC."""
    with patch("inference_sdk.webrtc.session._check_webrtc_dependencies"):
        session = WebRTCSession(
            api_url="http://localhost:9001",
            api_key="test_key",
            source=MagicMock(),
            image_input_name="image",
            workflow_config={},
            stream_config=MagicMock(),
        )
    return session


@pytest.fixture
def running_session(mock_session):
    """Attach a real background asyncio loop to a started session."""
    loop = asyncio.new_event_loop()
    loop_started = threading.Event()

    def run_loop():
        asyncio.set_event_loop(loop)
        loop_started.set()
        loop.run_forever()
        loop.close()

    loop_thread = threading.Thread(target=run_loop, daemon=True)
    peer_connection = AsyncPeerConnectionStub()
    source = AsyncSourceStub()

    mock_session._loop = loop
    mock_session._loop_thread = loop_thread
    mock_session._pc = peer_connection
    mock_session._source = source
    mock_session._state = SessionState.STARTED
    loop_thread.start()
    assert loop_started.wait(timeout=THREAD_TIMEOUT)

    yield mock_session, peer_connection, source

    if loop_thread.is_alive():
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(timeout=THREAD_TIMEOUT)


class TestSessionLifecycle:
    """Tests for session lifecycle (creation, starting, closing)."""

    def test_session_starts_in_not_started_state(self, mock_session):
        """Test that session is created in not_started state."""
        assert mock_session._state == SessionState.NOT_STARTED

    def test_close_is_idempotent(self, mock_session):
        """Test that close() can be called multiple times safely."""
        mock_session._state = SessionState.STARTED  # Simulate started state
        mock_session.close()
        assert mock_session._state == SessionState.CLOSED

        # Second call should be a no-op
        mock_session.close()
        assert mock_session._state == SessionState.CLOSED

    def test_ensure_started_changes_state(self, mock_session):
        """Test that _ensure_started() transitions from not_started to started."""
        with patch.object(mock_session, "_init_connection"):
            assert mock_session._state == SessionState.NOT_STARTED
            mock_session._ensure_started()
            assert mock_session._state == SessionState.STARTED
        mock_session.close()

    def test_ensure_started_is_idempotent(self, mock_session):
        """Test that _ensure_started() can be called multiple times."""
        with patch.object(mock_session, "_init_connection") as mock_init:
            mock_session._ensure_started()
            mock_session._ensure_started()
            mock_session._ensure_started()

            # _init_connection should only be called once
            assert mock_init.call_count == 1
        mock_session.close()

    def test_ensure_started_raises_on_closed_session(self, mock_session):
        """Test that _ensure_started() raises error if session is closed."""
        mock_session._state = SessionState.CLOSED

        with pytest.raises(RuntimeError, match="Cannot use closed WebRTCSession"):
            mock_session._ensure_started()

    def test_successful_start_preserves_normal_close(self, mock_session):
        """Test normal startup and cleanup with the owned event-loop thread."""
        source = AsyncSourceStub()
        mock_session._source = source

        async def initialize():
            return None

        with patch.object(mock_session, "_init", initialize):
            mock_session._ensure_started()

        try:
            assert mock_session._state == SessionState.STARTED
            assert mock_session._loop.is_running()
            assert mock_session._loop_thread.is_alive()
            assert mock_session._startup_task is not None
            assert mock_session._startup_task.done()
        finally:
            mock_session.close()

        assert source.cleanup_calls == 1
        assert mock_session._startup_task is None
        assert mock_session._state == SessionState.CLOSED
        assert mock_session._loop.is_closed()
        assert not mock_session._loop_thread.is_alive()

    def test_cleanup_waits_for_startup_cancellation_before_closing_peer(
        self, mock_session
    ):
        """Test that startup cannot publish a peer after cleanup reads it."""
        peer_connection = AsyncPeerConnectionStub()
        source = AsyncSourceStub()
        mock_session._source = source

        async def exercise_cleanup():
            async def publish_peer_during_cancellation():
                mock_session._startup_task = asyncio.current_task()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    mock_session._pc = peer_connection

            startup_task = asyncio.create_task(publish_peer_during_cancellation())
            await asyncio.sleep(0)
            await mock_session._cleanup_async_resources()
            await startup_task

        asyncio.run(exercise_cleanup())

        assert peer_connection.close_calls == 1
        assert source.cleanup_calls == 1
        assert mock_session._startup_task is None

    def test_failure_before_peer_creation_closes_startup_resources(self, mock_session):
        """Test cleanup when initialization fails before creating a peer."""
        source = AsyncSourceStub()
        mock_session._source = source

        async def fail_before_peer_creation():
            raise RuntimeError("startup failed before peer creation")

        observed = None
        try:
            with patch.object(mock_session, "_init", fail_before_peer_creation):
                with pytest.raises(
                    RuntimeError, match="startup failed before peer creation"
                ):
                    mock_session._ensure_started()

            observed = {
                "state": mock_session._state,
                "loop_closed": mock_session._loop.is_closed(),
                "thread_alive": mock_session._loop_thread.is_alive(),
                "source_cleanup_calls": source.cleanup_calls,
            }
        finally:
            stop_leaked_session_loop(mock_session)

        assert observed == {
            "state": SessionState.CLOSED,
            "loop_closed": True,
            "thread_alive": False,
            "source_cleanup_calls": 1,
        }

    def test_failure_after_peer_creation_closes_partial_resources(self, mock_session):
        """Test cleanup when initialization fails after creating a peer."""
        peer_connection = AsyncPeerConnectionStub()
        source = AsyncSourceStub(
            configure_error=RuntimeError("source configuration failed")
        )
        mock_session._source = source

        observed = None
        try:
            with (
                patch("inference_sdk.webrtc.session._check_webrtc_dependencies"),
                patch("aiortc.RTCPeerConnection", return_value=peer_connection),
                patch.object(
                    mock_session,
                    "_get_turn_config",
                    new=AsyncMock(return_value=None),
                ),
            ):
                with pytest.raises(RuntimeError, match="source configuration failed"):
                    mock_session._ensure_started()

            observed = {
                "state": mock_session._state,
                "published_peer": mock_session._pc is peer_connection,
                "peer_close_calls": peer_connection.close_calls,
                "source_cleanup_calls": source.cleanup_calls,
                "loop_closed": mock_session._loop.is_closed(),
                "thread_alive": mock_session._loop_thread.is_alive(),
            }
        finally:
            stop_leaked_session_loop(mock_session)

        assert observed == {
            "state": SessionState.CLOSED,
            "published_peer": True,
            "peer_close_calls": 1,
            "source_cleanup_calls": 1,
            "loop_closed": True,
            "thread_alive": False,
        }

    @pytest.mark.parametrize(
        ("status_code", "expected_message"),
        [
            (404, "WebRTC endpoint not found"),
            (500, "Failed to initialize WebRTC session (HTTP 500)"),
        ],
    )
    def test_http_startup_failure_preserves_error_and_closes_session(
        self, mock_session, status_code, expected_message
    ):
        """Test that HTTP diagnostics and their cause survive startup cleanup."""
        response = MagicMock(status_code=status_code, text="server response")
        http_error = requests.exceptions.HTTPError("request failed", response=response)
        source = AsyncSourceStub()
        mock_session._source = source

        async def fail_with_http_error():
            raise http_error

        with patch.object(mock_session, "_init", fail_with_http_error):
            with pytest.raises(RuntimeError) as raised:
                mock_session._ensure_started()

        assert expected_message in str(raised.value)
        assert raised.value.__cause__ is http_error
        assert mock_session._state == SessionState.CLOSED
        assert source.cleanup_calls == 1
        assert mock_session._loop.is_closed()
        assert not mock_session._loop_thread.is_alive()
        with pytest.raises(RuntimeError, match="Cannot use closed WebRTCSession"):
            mock_session._ensure_started()

    def test_concurrent_startup_failure_initializes_once(self, mock_session):
        """Test that concurrent first callers share one terminal startup attempt."""
        initialization_started = threading.Event()
        release_initialization = threading.Event()
        mock_session._source = None

        def fail_initialization():
            initialization_started.set()
            if not release_initialization.wait(timeout=THREAD_TIMEOUT):
                raise RuntimeError("startup test timed out")
            raise RuntimeError("startup failed")

        with patch.object(
            mock_session, "_init_connection", side_effect=fail_initialization
        ) as initialize:
            with ThreadPoolExecutor(max_workers=2) as executor:
                first = executor.submit(mock_session._ensure_started)
                assert initialization_started.wait(timeout=THREAD_TIMEOUT)
                second = executor.submit(mock_session._ensure_started)
                release_initialization.set()
                errors = [
                    first.exception(timeout=THREAD_TIMEOUT),
                    second.exception(timeout=THREAD_TIMEOUT),
                ]

        assert initialize.call_count == 1
        assert {str(error) for error in errors} == {
            "startup failed",
            "Cannot use closed WebRTCSession",
        }
        assert mock_session._state == SessionState.CLOSED

    def test_startup_interrupt_waits_for_startup_before_cleanup(self, mock_session):
        """Test that teardown waits for interrupted startup to finish."""

        class StartupInterrupt(BaseException):
            pass

        interrupt = StartupInterrupt("interrupted")
        initialization_blocked = threading.Event()
        cancellation_requested = threading.Event()
        cleanup_submitted = threading.Event()
        release_initialization = threading.Event()
        peer_created = threading.Event()
        coordination_errors = []
        peer_connection = AsyncPeerConnectionStub()
        source = AsyncSourceStub()
        mock_session._source = source
        original_submit = asyncio.run_coroutine_threadsafe

        class InterruptingFuture:
            def __init__(self, future):
                self._future = future

            def result(self):
                if not initialization_blocked.wait(timeout=THREAD_TIMEOUT):
                    raise RuntimeError("startup did not reach its blocking section")
                raise interrupt

            def done(self):
                return self._future.done()

            def cancel(self):
                cancellation_requested.set()
                return self._future.cancel()

        submission_count = 0

        def submit(coroutine, loop):
            nonlocal submission_count
            submission_count += 1
            future = original_submit(coroutine, loop)
            if submission_count == 1:
                return InterruptingFuture(future)
            cleanup_submitted.set()
            return future

        async def initialize():
            initialization_blocked.set()
            if not release_initialization.wait(timeout=THREAD_TIMEOUT):
                raise RuntimeError("startup test timed out")
            mock_session._pc = peer_connection
            peer_created.set()
            await asyncio.sleep(0)

        def release_after_cleanup_starts():
            if not cancellation_requested.wait(timeout=THREAD_TIMEOUT):
                coordination_errors.append("startup cancellation was not requested")
            if not cleanup_submitted.wait(timeout=THREAD_TIMEOUT):
                coordination_errors.append("cleanup was not submitted")
            release_initialization.set()

        coordinator = threading.Thread(target=release_after_cleanup_starts, daemon=True)
        coordinator.start()

        with (
            patch(
                "inference_sdk.webrtc.session.asyncio.run_coroutine_threadsafe",
                side_effect=submit,
            ),
            patch.object(mock_session, "_init", initialize),
        ):
            with pytest.raises(StartupInterrupt) as raised:
                mock_session._ensure_started()

        coordinator.join(timeout=THREAD_TIMEOUT)

        assert raised.value is interrupt
        assert not coordinator.is_alive()
        assert coordination_errors == []
        assert cancellation_requested.is_set()
        assert cleanup_submitted.is_set()
        assert peer_created.is_set()
        assert peer_connection.close_calls == 1
        assert source.cleanup_calls == 1
        assert mock_session._startup_task is None
        assert mock_session._state == SessionState.CLOSED
        assert mock_session._loop.is_closed()
        assert not mock_session._loop_thread.is_alive()

    def test_thread_start_failure_closes_unstarted_loop(self, mock_session):
        """Test that a loop is closed when its owner thread cannot start."""
        loop = asyncio.new_event_loop()
        mock_session._source = None

        with (
            patch(
                "inference_sdk.webrtc.session.asyncio.new_event_loop",
                return_value=loop,
            ),
            patch(
                "inference_sdk.webrtc.session.threading.Thread.start",
                side_effect=RuntimeError("thread start failed"),
            ),
        ):
            with pytest.raises(RuntimeError, match="thread start failed"):
                mock_session._ensure_started()

        assert loop.is_closed()
        assert mock_session._loop is None
        assert mock_session._loop_thread is None
        assert mock_session._state == SessionState.CLOSED
        assert mock_session._close_done.is_set()

    def test_submission_failure_closes_unsubmitted_coroutine(self, mock_session):
        """Test that failed scheduling does not leave an unawaited coroutine."""
        captured_coroutines = []
        mock_session._source = None

        def reject_submission(coroutine, loop):
            captured_coroutines.append(coroutine)
            raise RuntimeError("submission failed")

        with patch(
            "inference_sdk.webrtc.session.asyncio.run_coroutine_threadsafe",
            side_effect=reject_submission,
        ):
            with pytest.raises(RuntimeError, match="submission failed"):
                mock_session._ensure_started()

        assert len(captured_coroutines) == 1
        assert captured_coroutines[0].cr_frame is None
        assert mock_session._state == SessionState.CLOSED
        assert mock_session._loop.is_closed()
        assert not mock_session._loop_thread.is_alive()

    def test_startup_failure_cancels_and_drains_pending_tasks(self, mock_session):
        """Test that loop shutdown awaits tasks cancelled after startup failure."""
        task_cancelled = threading.Event()
        source = AsyncSourceStub()
        mock_session._source = source

        async def fail_with_pending_task():
            async def wait_forever():
                try:
                    await asyncio.Event().wait()
                finally:
                    task_cancelled.set()

            asyncio.create_task(wait_forever())
            await asyncio.sleep(0)
            raise RuntimeError("startup failed with pending task")

        with patch.object(mock_session, "_init", fail_with_pending_task):
            with pytest.raises(RuntimeError, match="startup failed with pending task"):
                mock_session._ensure_started()

        assert task_cancelled.is_set()
        assert source.cleanup_calls == 1
        assert mock_session._loop.is_closed()
        assert not mock_session._loop_thread.is_alive()

    def test_cleanup_cancellation_does_not_mask_startup_failure(self, mock_session):
        """Test that cleanup cancellation preserves the original startup error."""
        startup_error = RuntimeError("startup failed")
        source = AsyncSourceStub(cleanup_error=asyncio.CancelledError())
        mock_session._source = source

        async def fail_initialization():
            raise startup_error

        with (
            patch.object(mock_session, "_init", fail_initialization),
            patch("inference_sdk.webrtc.session.logger.exception") as log_exception,
        ):
            with pytest.raises(RuntimeError, match="startup failed") as raised:
                mock_session._ensure_started()

        assert raised.value.__cause__ is startup_error
        assert source.cleanup_calls == 1
        assert mock_session._loop.is_closed()
        assert not mock_session._loop_thread.is_alive()
        log_exception.assert_called_once_with(
            "Failed to clean up WebRTC session after startup failure"
        )

    def test_loop_runner_closes_loop_when_shutdown_steps_fail(self, mock_session):
        """Test that drain failures cannot bypass event-loop close."""
        loop = MagicMock(spec=asyncio.AbstractEventLoop)
        loop.shutdown_asyncgens = MagicMock(side_effect=RuntimeError("shutdown failed"))

        class ImmediateThread:
            def __init__(self, target, args, daemon):
                self._target = target
                self._args = args
                self._alive = False

            def start(self):
                self._alive = True
                try:
                    self._target(*self._args)
                finally:
                    self._alive = False

            def is_alive(self):
                return self._alive

        with (
            patch(
                "inference_sdk.webrtc.session.asyncio.new_event_loop",
                return_value=loop,
            ),
            patch(
                "inference_sdk.webrtc.session.threading.Thread",
                ImmediateThread,
            ),
            patch(
                "inference_sdk.webrtc.session.asyncio.set_event_loop"
            ) as set_event_loop,
            patch(
                "inference_sdk.webrtc.session.asyncio.all_tasks",
                side_effect=RuntimeError("drain failed"),
            ),
            patch(
                "inference_sdk.webrtc.session.asyncio.run_coroutine_threadsafe",
                side_effect=RuntimeError("submission failed"),
            ),
            patch("inference_sdk.webrtc.session.logger.exception") as log_exception,
        ):
            with pytest.raises(RuntimeError, match="submission failed"):
                mock_session._init_connection()

        loop.close.assert_called_once_with()
        assert set_event_loop.call_args_list == [
            ((loop,),),
            ((None,),),
        ]
        assert [call.args[0] for call in log_exception.call_args_list] == [
            "Failed to drain WebRTC event-loop tasks",
            "Failed to shut down WebRTC async generators",
        ]


class TestRunMethod:
    """Tests for run() method and exception handling."""

    def test_run_auto_starts_session(self, mock_session):
        """Test that run() automatically starts the session."""
        with patch.object(mock_session, "_ensure_started") as mock_ensure:
            # Put a frame and immediately close
            @mock_session.on_frame
            def handler(frame, metadata):
                mock_session.close()

            test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
            test_metadata = VideoMetadata(frame_id=1, received_at=datetime.now())
            mock_session._video_queue.put((test_frame, test_metadata))
            mock_session._state = SessionState.STARTED

            mock_session.run()

            # Should have called _ensure_started
            mock_ensure.assert_called_once()

    def test_run_stops_when_close_called(self, mock_session):
        """Test that run() stops when close() is called from handler."""
        frame_count = []

        @mock_session.on_frame
        def count_frames(frame, metadata):
            frame_count.append(1)
            if len(frame_count) >= 2:
                mock_session.close()

        # Put multiple frames in queue (use put_nowait to avoid blocking on full queue)
        for i in range(5):
            test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
            test_metadata = VideoMetadata(frame_id=i, received_at=datetime.now())
            mock_session._video_queue.put_nowait((test_frame, test_metadata))

        # Mock state as started
        mock_session._state = SessionState.STARTED

        mock_session.run()

        # Should have stopped after 2 frames (when close() was called)
        assert len(frame_count) == 2

    def test_run_handles_handler_exceptions_gracefully(self, mock_session):
        """Test that exceptions in handlers don't crash run()."""
        handler1_calls = []
        handler2_calls = []

        @mock_session.on_frame
        def failing_handler(frame, metadata):
            handler1_calls.append(True)
            raise ValueError("Handler error")

        @mock_session.on_frame
        def working_handler(frame, metadata):
            handler2_calls.append(True)
            mock_session.close()

        # Put a frame in queue
        test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        test_metadata = VideoMetadata(frame_id=1, received_at=datetime.now())
        mock_session._video_queue.put((test_frame, test_metadata))

        mock_session._state = SessionState.STARTED

        # Run should not raise despite first handler failing
        mock_session.run()

        # Both handlers should have been called
        assert len(handler1_calls) == 1
        assert len(handler2_calls) == 1

    def test_run_closes_session_on_exception(self, mock_session):
        """Test that run() closes session if exception occurs."""
        mock_session._state = SessionState.STARTED

        # Mock video() to raise an exception
        def raise_exception():
            raise RuntimeError("Test error")
            yield  # Never reached

        with patch.object(mock_session, "video", return_value=raise_exception()):
            with patch.object(mock_session, "close") as mock_close:
                with pytest.raises(RuntimeError, match="Test error"):
                    mock_session.run()

                # Should have called close()
                mock_close.assert_called_once()
        mock_session.close()

    def test_run_closes_session_on_keyboard_interrupt(self, mock_session):
        """Test that run() closes session on Ctrl+C."""
        mock_session._state = SessionState.STARTED

        # Mock video() to raise KeyboardInterrupt
        def raise_interrupt():
            raise KeyboardInterrupt()
            yield  # Never reached

        with patch.object(mock_session, "video", return_value=raise_interrupt()):
            with patch.object(mock_session, "close") as mock_close:
                with pytest.raises(KeyboardInterrupt):
                    mock_session.run()

                # Should have called close()
                mock_close.assert_called_once()
        mock_session.close()


class TestDecorators:
    """Tests for decorator registration."""

    def test_on_frame_registration(self, mock_session):
        """Test that on_frame decorator registers handler."""
        handler_called = []

        @mock_session.on_frame
        def process_frame(frame, metadata):
            handler_called.append((frame, metadata))

        assert len(mock_session._frame_handlers) == 1
        assert mock_session._frame_handlers[0] == process_frame

    def test_on_frame_multiple_handlers(self, mock_session):
        """Test registering multiple frame handlers."""

        @mock_session.on_frame
        def handler1(frame, metadata):
            pass

        @mock_session.on_frame
        def handler2(frame, metadata):
            pass

        assert len(mock_session._frame_handlers) == 2
        assert handler1 in mock_session._frame_handlers
        assert handler2 in mock_session._frame_handlers

    def test_on_data_global_handler(self, mock_session):
        """Test registering global data handler."""

        @mock_session.on_data()
        def handle_data(data, metadata):
            pass

        assert mock_session._data_global_handler == handle_data

    def test_on_data_field_specific_handler(self, mock_session):
        """Test registering field-specific data handler."""

        @mock_session.on_data("predictions")
        def handle_predictions(value, metadata):
            pass

        assert "predictions" in mock_session._data_field_handlers
        assert handle_predictions in mock_session._data_field_handlers["predictions"]

    def test_on_error_registers_handler(self, mock_session):
        """Test registering per-frame error handler."""

        @mock_session.on_error
        def handle_err(errors, metadata):
            pass

        assert handle_err in mock_session._error_handlers

    def test_parse_video_metadata_attaches_errors(self, mock_session):
        """Errors passed alongside metadata are surfaced on VideoMetadata."""
        md = mock_session._parse_video_metadata(
            {"frame_id": 7, "received_at": datetime.now().isoformat()},
            errors=["workflow block X failed"],
        )
        assert md is not None
        assert md.frame_id == 7
        assert md.errors == ["workflow block X failed"]

    def test_video_metadata_default_errors_is_empty(self):
        """VideoMetadata constructed without errors defaults to an empty list."""
        md = VideoMetadata(frame_id=1, received_at=datetime.now())
        assert md.errors == []


class TestVideoStream:
    """Tests for video stream iterator."""

    def test_video_auto_starts_session(self, mock_session):
        """Test that video() automatically starts the session."""
        with patch.object(mock_session, "_ensure_started") as mock_ensure:
            # Put a frame and end signal
            test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
            test_metadata = VideoMetadata(frame_id=1, received_at=datetime.now())
            mock_session._video_queue.put((test_frame, test_metadata))
            mock_session._video_queue.put(None)  # End stream

            # Iterate (should auto-start)
            list(mock_session.video())

            # Should have called _ensure_started
            mock_ensure.assert_called_once()

    def test_video_yields_frame_tuples(self, mock_session):
        """Test that video() yields (frame, metadata) tuples."""
        # Put test frames in queue
        test_frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        test_metadata1 = VideoMetadata(frame_id=1, received_at=datetime.now())

        test_frame2 = np.ones((100, 100, 3), dtype=np.uint8)
        test_metadata2 = VideoMetadata(frame_id=2, received_at=datetime.now())

        mock_session._video_queue.put((test_frame1, test_metadata1))
        mock_session._video_queue.put((test_frame2, test_metadata2))
        mock_session._video_queue.put(None)  # End stream

        # Mock _ensure_started
        with patch.object(mock_session, "_ensure_started"):
            # Iterate and collect
            frames = []
            metadatas = []
            for frame, metadata in mock_session.video():
                frames.append(frame)
                metadatas.append(metadata)

            assert len(frames) == 2
            assert np.array_equal(frames[0], test_frame1)
            assert np.array_equal(frames[1], test_frame2)

            assert len(metadatas) == 2
            assert metadatas[0].frame_id == 1
            assert metadatas[1].frame_id == 2


class TestWaitMethod:
    """Tests for wait() method."""

    def test_wait_auto_starts_session(self, mock_session):
        """Test that wait() automatically starts the session."""
        with patch.object(mock_session, "_ensure_started") as mock_ensure:
            # Put a frame and end signal
            test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
            test_metadata = VideoMetadata(frame_id=1, received_at=datetime.now())

            mock_session._video_queue.put((test_frame, test_metadata))
            mock_session._video_queue.put(None)  # End stream

            # Should not raise and should consume all frames
            mock_session.wait()

            # Should have called _ensure_started
            mock_ensure.assert_called_once()

    def test_wait_blocks_until_stream_ends(self, mock_session):
        """Test that wait() blocks until None is received."""
        with patch.object(mock_session, "_ensure_started"):
            # Put frames in queue
            test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
            test_metadata = VideoMetadata(frame_id=1, received_at=datetime.now())

            mock_session._video_queue.put((test_frame, test_metadata))
            mock_session._video_queue.put(None)  # End stream

            # Should not raise and should consume all frames
            mock_session.wait()

    def test_wait_timeout(self, mock_session):
        """Test that wait() raises TimeoutError on timeout."""
        with patch.object(mock_session, "_ensure_started"):
            # Put a frame but no end signal
            test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
            test_metadata = VideoMetadata(frame_id=1, received_at=datetime.now())
            mock_session._video_queue.put((test_frame, test_metadata))

            with pytest.raises(TimeoutError, match="timed out"):
                mock_session.wait(timeout=0.1)


class TestCloseMethod:
    """Tests for close() method."""

    def test_close_can_be_called_from_handler(self, mock_session):
        """Test that close() can be called from within a frame handler."""
        calls = []

        @mock_session.on_frame
        def handler(frame, metadata):
            calls.append(1)
            mock_session.close()

        # Put frames in queue (use put_nowait to avoid blocking on full queue)
        for i in range(5):
            test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
            test_metadata = VideoMetadata(frame_id=i, received_at=datetime.now())
            mock_session._video_queue.put_nowait((test_frame, test_metadata))

        mock_session._state = SessionState.STARTED
        mock_session.run()

        # Should have stopped after first frame
        assert len(calls) == 1
        assert mock_session._state == SessionState.CLOSED

    def test_close_from_event_loop_returns_and_cleans_up(self, running_session):
        """Test that close() does not block its own event loop."""
        session, peer_connection, source = running_session
        close_returned = threading.Event()

        def close_from_event_loop():
            session.close()
            close_returned.set()

        session._loop.call_soon_threadsafe(close_from_event_loop)

        assert close_returned.wait(timeout=THREAD_TIMEOUT)
        assert session._close_done.wait(timeout=THREAD_TIMEOUT)
        assert peer_connection.close_calls == 1
        assert source.cleanup_calls == 1
        assert not session._loop_thread.is_alive()

    def test_close_from_event_loop_does_not_wait_for_state_lock(self, running_session):
        """Test that callback shutdown can return while startup owns the state lock."""
        session, peer_connection, source = running_session
        close_returned = threading.Event()

        def close_from_event_loop():
            session.close()
            close_returned.set()

        session._state_lock.acquire()

        try:
            session._loop.call_soon_threadsafe(close_from_event_loop)
            assert close_returned.wait(timeout=THREAD_TIMEOUT)
        finally:
            session._state_lock.release()

        assert session._close_done.wait(timeout=THREAD_TIMEOUT)
        assert peer_connection.close_calls == 1
        assert source.cleanup_calls == 1

    def test_run_waits_for_event_loop_close_to_finish(self, running_session):
        """Test that context-manager exit waits for callback-initiated cleanup."""
        session, peer_connection, source = running_session
        release_peer_close = asyncio.Event()
        peer_connection.release = release_peer_close
        video_waiting = threading.Event()
        run_finished = threading.Event()
        run_errors = []
        original_get = session._video_queue.get

        def get_video_frame():
            video_waiting.set()
            return original_get()

        def run_session():
            try:
                session.run()
            except Exception as error:
                run_errors.append(error)
            finally:
                run_finished.set()

        with patch.object(session._video_queue, "get", side_effect=get_video_frame):
            run_thread = threading.Thread(target=run_session, daemon=True)
            run_thread.start()
            try:
                assert video_waiting.wait(timeout=THREAD_TIMEOUT)

                session._loop.call_soon_threadsafe(session.close)
                assert peer_connection.close_started.wait(timeout=THREAD_TIMEOUT)
                assert not run_finished.is_set()

                session._loop.call_soon_threadsafe(release_peer_close.set)
                assert run_finished.wait(timeout=THREAD_TIMEOUT)
            finally:
                if session._loop.is_running():
                    session._loop.call_soon_threadsafe(session.close)
                    session._loop.call_soon_threadsafe(release_peer_close.set)
                run_thread.join(timeout=THREAD_TIMEOUT)

        assert not run_thread.is_alive()
        assert not run_errors
        assert peer_connection.close_calls == 1
        assert source.cleanup_calls == 1

    def test_event_loop_close_returns_during_external_cleanup(self, running_session):
        """Test that a loop-thread duplicate never waits for an external owner."""
        session, peer_connection, source = running_session
        release_peer_close = asyncio.Event()
        peer_connection.release = release_peer_close
        external_close_finished = threading.Event()
        loop_close_returned = threading.Event()

        def close_externally():
            session.close()
            external_close_finished.set()

        def close_from_event_loop():
            session.close()
            loop_close_returned.set()

        external_thread = threading.Thread(target=close_externally, daemon=True)
        external_thread.start()
        try:
            assert peer_connection.close_started.wait(timeout=THREAD_TIMEOUT)

            session._loop.call_soon_threadsafe(close_from_event_loop)
            assert loop_close_returned.wait(timeout=THREAD_TIMEOUT)

            session._loop.call_soon_threadsafe(release_peer_close.set)
            assert external_close_finished.wait(timeout=THREAD_TIMEOUT)
        finally:
            if session._loop.is_running():
                session._loop.call_soon_threadsafe(release_peer_close.set)
            external_thread.join(timeout=THREAD_TIMEOUT)

        assert not external_thread.is_alive()
        assert peer_connection.close_calls == 1
        assert source.cleanup_calls == 1

    def test_close_failure_still_cleans_source_and_completes(self, running_session):
        """Test that later cleanup runs when peer connection shutdown fails."""
        session, peer_connection, source = running_session
        peer_connection.error = RuntimeError("peer close failed")

        with pytest.raises(RuntimeError, match="peer close failed"):
            session.close()

        assert source.cleanup_calls == 1
        assert session._close_done.is_set()
        assert not session._loop_thread.is_alive()

    def test_background_close_failure_is_logged(self, running_session):
        """Test that callback cleanup failures are logged after the callback returns."""
        session, peer_connection, source = running_session
        peer_connection.error = RuntimeError("peer close failed")
        close_returned = threading.Event()
        error_logged = threading.Event()

        def close_from_event_loop():
            session.close()
            close_returned.set()

        with patch(
            "inference_sdk.webrtc.session.logger.exception",
            side_effect=lambda *args, **kwargs: error_logged.set(),
        ) as log_exception:
            session._loop.call_soon_threadsafe(close_from_event_loop)
            assert close_returned.wait(timeout=THREAD_TIMEOUT)
            assert session._close_done.wait(timeout=THREAD_TIMEOUT)
            assert error_logged.wait(timeout=THREAD_TIMEOUT)

        assert source.cleanup_calls == 1
        log_exception.assert_called_once_with("Failed to close WebRTC session")

    def test_background_close_can_retry_when_thread_start_fails(self, running_session):
        """Test that a failed helper start does not strand later close callers."""
        session, peer_connection, source = running_session
        close_returned = threading.Event()
        close_errors = []

        def close_from_event_loop():
            try:
                session.close()
            except RuntimeError as error:
                close_errors.append(error)
            finally:
                close_returned.set()

        with patch(
            "inference_sdk.webrtc.session.threading.Thread.start",
            side_effect=RuntimeError("thread start failed"),
        ):
            session._loop.call_soon_threadsafe(close_from_event_loop)
            assert close_returned.wait(timeout=THREAD_TIMEOUT)

        session.close()

        assert str(close_errors[0]) == "thread start failed"
        assert peer_connection.close_calls == 1
        assert source.cleanup_calls == 1

    def test_close_cleans_resources_when_video_queue_is_full(self, running_session):
        """Test cleanup when the end sentinel cannot be added to a full queue."""
        session, peer_connection, source = running_session
        while not session._video_queue.full():
            session._video_queue.put_nowait(MagicMock())

        session.close()

        assert peer_connection.close_calls == 1
        assert source.cleanup_calls == 1

    def test_close_without_peer_connection_still_cleans_source(self, running_session):
        """Test source cleanup when connection setup did not assign a peer."""
        session, _, source = running_session
        session._pc = None

        session.close()

        assert source.cleanup_calls == 1
        assert session._close_done.is_set()
        assert not session._loop_thread.is_alive()

    def test_close_warns_when_event_loop_thread_does_not_stop(self, mock_session):
        """Test that a timed-out event-loop join is visible to callers."""
        loop = MagicMock(spec=asyncio.AbstractEventLoop)
        loop.is_closed.return_value = False
        loop_thread = MagicMock(spec=threading.Thread)
        loop_thread.is_alive.return_value = True
        mock_session._loop = loop
        mock_session._loop_thread = loop_thread
        mock_session._pc = None
        mock_session._source = None
        mock_session._state = SessionState.STARTED

        with patch("inference_sdk.webrtc.session.logger.warning") as log_warning:
            mock_session.close()

        loop.call_soon_threadsafe.assert_called_once_with(loop.stop)
        loop_thread.join.assert_called_once_with(
            timeout=WEBRTC_EVENT_LOOP_SHUTDOWN_TIMEOUT
        )
        log_warning.assert_called_once_with(
            "WebRTC event loop thread did not stop "
            f"within {WEBRTC_EVENT_LOOP_SHUTDOWN_TIMEOUT}s"
        )
