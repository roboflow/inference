"""
Tests for Issue #685: Inference Pipeline cannot be terminated once initial connect request to camera failed

This test suite verifies that video sources can be terminated from ALL states,
including edge cases like failed camera connections that previously got stuck.
"""
import pytest
from unittest.mock import MagicMock, patch

from inference.core.interfaces.camera.video_source import (
    VideoSource,
    StreamState,
    TERMINATE_ELIGIBLE_STATES,
)
from inference.core.interfaces.camera.exceptions import StreamOperationNotAllowedError


class TestTerminationFromAllStates:
    """Test that termination works from all possible states."""

    def test_terminate_eligible_states_includes_all_critical_states(self):
        """Verify TERMINATE_ELIGIBLE_STATES includes states where connection can fail."""
        # These are the critical states for issue #685
        critical_states = {
            StreamState.NOT_STARTED,      # Before any connection attempt
            StreamState.INITIALISING,     # During connection (where failures happen!)
            StreamState.TERMINATING,      # Already terminating
            StreamState.ERROR,            # After failure
        }

        for state in critical_states:
            assert state in TERMINATE_ELIGIBLE_STATES, (
                f"{state} must be in TERMINATE_ELIGIBLE_STATES to allow cleanup "
                f"of failed connections (Issue #685)"
            )

    def test_all_stream_states_covered(self):
        """Verify we can always terminate, regardless of state."""
        all_states = set(StreamState)

        # These are the ONLY states where termination doesn't make sense
        # (there are none - termination should always be possible)
        never_terminate = set()

        # All other states should allow termination
        should_allow_termination = all_states - never_terminate

        assert TERMINATE_ELIGIBLE_STATES >= should_allow_termination, (
            "Some states cannot be terminated! Missing: "
            f"{should_allow_termination - TERMINATE_ELIGIBLE_STATES}"
        )

    @patch('inference.core.interfaces.camera.video_source.build_hw_producer')
    def test_terminate_from_not_started_state(self, mock_producer):
        """Test termination from NOT_STARTED state (Issue #685 regression test)."""
        # given
        mock_producer.return_value = MagicMock()
        source = VideoSource(video_reference="rtsp://invalid")

        # when - terminate immediately without starting
        assert source.get_state() == StreamState.NOT_STARTED

        # then - should not raise StreamOperationNotAllowedError
        try:
            source.terminate()
            success = True
        except StreamOperationNotAllowedError:
            success = False

        assert success, "Should be able to terminate from NOT_STARTED state"
        assert source.get_state() == StreamState.ENDED

    @patch('inference.core.interfaces.camera.video_source.build_hw_producer')
    def test_terminate_from_initialising_state(self, mock_producer):
        """Test termination from INITIALISING state (main Issue #685 scenario)."""
        # given - mock a producer that hangs during initialization
        producer_mock = MagicMock()
        producer_mock.isOpened.return_value = True

        # Simulate hanging during property discovery
        def slow_discover():
            import time
            time.sleep(0.5)  # Simulate slow/hanging connection
            raise RuntimeError("Connection failed")

        producer_mock.discover_source_properties.side_effect = slow_discover
        mock_producer.return_value = producer_mock

        source = VideoSource(video_reference="rtsp://slow-camera")

        # when - start (which will hang in INITIALISING) then terminate from another thread
        import threading

        start_thread = threading.Thread(target=lambda: source.start())
        start_thread.start()

        # Wait a bit for start to enter INITIALISING state
        import time
        time.sleep(0.1)

        # State should be INITIALISING during the connection attempt
        state_during_init = source.get_state()

        # Try to terminate while initializing (this is the bug!)
        try:
            source.terminate()
            success = True
            error = None
        except StreamOperationNotAllowedError as e:
            success = False
            error = str(e)

        start_thread.join(timeout=2.0)

        # then
        assert success, (
            f"Should be able to terminate from {state_during_init} state. "
            f"Error: {error}. This is the core Issue #685 bug!"
        )

    @patch('inference.core.interfaces.camera.video_source.build_hw_producer')
    def test_terminate_from_error_state_after_connection_failure(self, mock_producer):
        """Test termination from ERROR state after connection failure."""
        # given - mock a producer that fails to connect
        producer_mock = MagicMock()
        producer_mock.isOpened.return_value = False  # Connection failed

        mock_producer.return_value = producer_mock

        source = VideoSource(video_reference="rtsp://invalid-camera")

        # when - try to start (will fail and go to ERROR state)
        try:
            source.start()
        except:
            pass  # Expected to fail

        state_after_failure = source.get_state()

        # then - should be able to terminate from ERROR state
        try:
            source.terminate()
            success = True
        except StreamOperationNotAllowedError:
            success = False

        assert success, (
            f"Should be able to terminate from {state_after_failure} state "
            "after connection failure"
        )

    @patch('inference.core.interfaces.camera.video_source.build_hw_producer')
    def test_terminate_from_terminating_state_is_idempotent(self, mock_producer):
        """Test that calling terminate() while already terminating doesn't error."""
        # given
        producer_mock = MagicMock()
        producer_mock.isOpened.return_value = True
        producer_mock.discover_source_properties.return_value = MagicMock(
            width=640, height=480, fps=30.0, total_frames=0, is_file=False
        )
        producer_mock.grab.return_value = (True, MagicMock())

        mock_producer.return_value = producer_mock

        source = VideoSource(video_reference="rtsp://camera")
        source.start()

        # when - call terminate twice (second call during TERMINATING state)
        import threading

        def terminate_slow():
            import time
            time.sleep(0.1)  # Make termination slow
            source._terminate(wait_on_frames_consumption=False, purge_frames_buffer=True)

        # Start first termination
        source._state = StreamState.TERMINATING

        # Try to terminate again while in TERMINATING state
        try:
            source.terminate()
            success = True
        except StreamOperationNotAllowedError:
            success = False

        # then
        assert success, "Should be able to call terminate() even if already TERMINATING"


class TestIssue685RegressionScenario:
    """Full regression test for Issue #685 scenario."""

    @patch('inference.core.interfaces.camera.video_source.build_hw_producer')
    def test_issue_685_full_scenario(self, mock_producer):
        """
        Reproduce exact Issue #685 scenario:
        1. User calls start_inference_pipeline_with_workflow with invalid camera
        2. Connection fails
        3. User tries to terminate_inference_pipeline
        4. BEFORE FIX: Gets StreamOperationNotAllowedError
        5. AFTER FIX: Termination succeeds
        """
        # given - simulate invalid camera that fails to connect
        producer_mock = MagicMock()
        producer_mock.isOpened.return_value = False  # Connection fails
        mock_producer.return_value = producer_mock

        source = VideoSource(video_reference="rtsp://192.168.1.999:554/invalid")

        # when - simulate user's workflow
        # Step 1: Try to start with invalid camera
        connection_succeeded = False
        try:
            source.start()
            connection_succeeded = True
        except Exception:
            pass  # Connection failed as expected

        # Step 2: Connection failed, user wants to clean up
        state_after_failure = source.get_state()

        # Step 3: User tries to terminate (this is where the bug was!)
        termination_error = None
        try:
            source.terminate()
            termination_succeeded = True
        except StreamOperationNotAllowedError as e:
            termination_succeeded = False
            termination_error = str(e)

        # then - verify fix works
        assert not connection_succeeded, "Connection should have failed"
        assert termination_succeeded, (
            f"REGRESSION: Issue #685 not fixed! "
            f"Cannot terminate from state {state_after_failure}. "
            f"Error: {termination_error}"
        )
        assert source.get_state() == StreamState.ENDED, (
            "After termination, state should be ENDED"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
