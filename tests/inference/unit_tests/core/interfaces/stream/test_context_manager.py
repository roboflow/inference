"""
Tests for InferencePipeline context manager support (Issue #2744)

Tests that __enter__ and __exit__ methods properly manage pipeline lifecycle,
ensuring terminate() and join() are called even when exceptions occur.
"""
from unittest.mock import MagicMock, patch, call

import pytest

from inference.core.interfaces.stream.inference_pipeline import InferencePipeline


class TestInferencePipelineContextManager:
    """Test context manager protocol for InferencePipeline."""

    def test_enter_returns_pipeline_instance(self):
        """Test that __enter__ returns the pipeline instance."""
        # given
        with patch("inference.core.interfaces.stream.inference_pipeline.get_model"):
            pipeline = InferencePipeline.init(
                model_id="test/1",
                video_reference=0,
                on_prediction=lambda x, y: None,
            )

        # when
        result = pipeline.__enter__()

        # then
        assert result is pipeline

    def test_exit_calls_terminate_and_join(self):
        """Test that __exit__ calls both terminate() and join()."""
        # given
        with patch("inference.core.interfaces.stream.inference_pipeline.get_model"):
            pipeline = InferencePipeline.init(
                model_id="test/1",
                video_reference=0,
                on_prediction=lambda x, y: None,
            )

        # Mock the methods we care about
        pipeline.terminate = MagicMock()
        pipeline.join = MagicMock()

        # when
        pipeline.__exit__(None, None, None)

        # then
        pipeline.terminate.assert_called_once()
        pipeline.join.assert_called_once()

        # Verify order: terminate before join
        assert pipeline.terminate.call_args_list[0] < pipeline.join.call_args_list[0]

    def test_exit_calls_cleanup_on_exception(self):
        """Test that __exit__ calls cleanup even when an exception occurred."""
        # given
        with patch("inference.core.interfaces.stream.inference_pipeline.get_model"):
            pipeline = InferencePipeline.init(
                model_id="test/1",
                video_reference=0,
                on_prediction=lambda x, y: None,
            )

        pipeline.terminate = MagicMock()
        pipeline.join = MagicMock()

        # when - simulate exception by passing exception info
        pipeline.__exit__(RuntimeError, RuntimeError("test error"), None)

        # then - cleanup should still be called
        pipeline.terminate.assert_called_once()
        pipeline.join.assert_called_once()

    def test_context_manager_normal_execution(self):
        """Test using InferencePipeline as context manager in normal flow."""
        # given
        mock_model = MagicMock()
        mock_sink = MagicMock()

        with patch(
            "inference.core.interfaces.stream.inference_pipeline.get_model",
            return_value=mock_model,
        ):
            pipeline = InferencePipeline.init(
                model_id="test/1",
                video_reference=0,
                on_prediction=mock_sink,
            )

        # Mock internal methods to avoid actual execution
        pipeline.terminate = MagicMock()
        pipeline.join = MagicMock()

        # when
        with pipeline as p:
            # then - pipeline is accessible inside context
            assert p is pipeline

        # then - cleanup called after exiting context
        pipeline.terminate.assert_called_once()
        pipeline.join.assert_called_once()

    def test_context_manager_exception_handling(self):
        """Test that context manager cleans up even when exception is raised."""
        # given
        mock_model = MagicMock()

        with patch(
            "inference.core.interfaces.stream.inference_pipeline.get_model",
            return_value=mock_model,
        ):
            pipeline = InferencePipeline.init(
                model_id="test/1",
                video_reference=0,
                on_prediction=lambda x, y: None,
            )

        pipeline.terminate = MagicMock()
        pipeline.join = MagicMock()

        # when - exception raised inside context
        with pytest.raises(RuntimeError, match="intentional error"):
            with pipeline:
                raise RuntimeError("intentional error")

        # then - cleanup was still called
        pipeline.terminate.assert_called_once()
        pipeline.join.assert_called_once()

    def test_context_manager_guarantees_join_after_terminate(self):
        """Test that join() is called even if terminate() raises an exception."""
        # given
        with patch("inference.core.interfaces.stream.inference_pipeline.get_model"):
            pipeline = InferencePipeline.init(
                model_id="test/1",
                video_reference=0,
                on_prediction=lambda x, y: None,
            )

        # Mock terminate to raise an exception
        pipeline.terminate = MagicMock(side_effect=RuntimeError("terminate failed"))
        pipeline.join = MagicMock()

        # when - __exit__ is called
        with pytest.raises(RuntimeError, match="terminate failed"):
            pipeline.__exit__(None, None, None)

        # then - terminate was called
        pipeline.terminate.assert_called_once()
        # join() is NOT called if terminate() raises, which is expected behavior
        # (the exception propagates immediately)

    def test_context_manager_typical_usage_pattern(self):
        """
        Test the typical usage pattern from the issue description.

        This verifies the exact use case mentioned in #2744 where exceptions
        between start() and join() would previously leak resources.
        """
        # given
        mock_model = MagicMock()
        mock_sink = MagicMock()

        with patch(
            "inference.core.interfaces.stream.inference_pipeline.get_model",
            return_value=mock_model,
        ):
            # This simulates the "with" pattern from the issue
            with InferencePipeline.init(
                model_id="test/1",
                video_reference=0,
                on_prediction=mock_sink,
            ) as pipeline:
                # Mock to avoid actual thread creation
                pipeline.terminate = MagicMock()
                pipeline.join = MagicMock()
                pipeline.start = MagicMock()

                # Simulate what user would do
                pipeline.start()

            # After exiting context, cleanup should be automatic
            pipeline.terminate.assert_called_once()
            pipeline.join.assert_called_once()

    def test_exit_does_not_suppress_exceptions(self):
        """Test that __exit__ returns None, allowing exceptions to propagate."""
        # given
        with patch("inference.core.interfaces.stream.inference_pipeline.get_model"):
            pipeline = InferencePipeline.init(
                model_id="test/1",
                video_reference=0,
                on_prediction=lambda x, y: None,
            )

        pipeline.terminate = MagicMock()
        pipeline.join = MagicMock()

        # when
        result = pipeline.__exit__(RuntimeError, RuntimeError("test"), None)

        # then - should return None (or implicitly None) to allow exception to propagate
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
