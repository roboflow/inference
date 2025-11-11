"""Unit tests for WebRTC datachannel functionality."""

import json
from datetime import datetime
from typing import Any, Callable, Optional
from unittest.mock import MagicMock

import pytest

from inference_sdk.webrtc.session import VideoMetadata, _DataChannel


def create_mock_channel_and_bind(channel: _DataChannel) -> tuple[MagicMock, Callable]:
    """Create a mock RTCDataChannel and bind to it, returning the message handler.

    Args:
        channel: The _DataChannel instance to bind

    Returns:
        Tuple of (mock_channel, message_handler)
    """
    mock_channel = MagicMock()
    message_handler = None

    def capture_handler(event):
        def decorator(fn):
            nonlocal message_handler
            message_handler = fn
            return fn
        return decorator

    mock_channel.on = capture_handler
    channel.bind(mock_channel)
    return mock_channel, message_handler


class TestVideoMetadata:
    """Tests for VideoMetadata dataclass."""

    def test_video_metadata_creation(self):
        """Test creating VideoMetadata instance."""
        received_at = datetime(2025, 11, 11, 18, 5, 44, 965372)
        metadata = VideoMetadata(
            frame_id=408,
            received_at=received_at,
            pts=1764000,
            time_base=1.1111111111111112e-05,
            declared_fps=15.0,
            measured_fps=None,
        )

        assert metadata.frame_id == 408
        assert metadata.received_at == received_at
        assert metadata.pts == 1764000
        assert metadata.time_base == 1.1111111111111112e-05
        assert metadata.declared_fps == 15.0
        assert metadata.measured_fps is None

    def test_video_metadata_optional_fields(self):
        """Test VideoMetadata with only required fields."""
        received_at = datetime.now()
        metadata = VideoMetadata(frame_id=1, received_at=received_at)

        assert metadata.frame_id == 1
        assert metadata.received_at == received_at
        assert metadata.pts is None
        assert metadata.time_base is None
        assert metadata.declared_fps is None
        assert metadata.measured_fps is None


class TestDataChannel:
    """Tests for _DataChannel class."""

    def test_global_handler_registration(self):
        """Test registering a global handler."""
        channel = _DataChannel()

        @channel.on_data()
        def handler(msg: str):
            pass

        assert channel._global_handler == handler

    def test_field_handler_registration(self):
        """Test registering field-specific handlers."""
        channel = _DataChannel()

        @channel.on_data("predictions")
        def handler(value: Any):
            pass

        assert "predictions" in channel._field_handlers
        assert handler in channel._field_handlers["predictions"]

    def test_multiple_field_handlers(self):
        """Test multiple handlers for the same field."""
        channel = _DataChannel()

        @channel.on_data("predictions")
        def handler1(value: Any):
            pass

        @channel.on_data("predictions")
        def handler2(value: Any):
            pass

        assert len(channel._field_handlers["predictions"]) == 2
        assert handler1 in channel._field_handlers["predictions"]
        assert handler2 in channel._field_handlers["predictions"]

    def test_parse_new_message_format_with_metadata(self):
        """Test parsing new message format with video_metadata."""
        channel = _DataChannel()
        received_values = []
        received_metadata = []

        @channel.on_data("property_definition")
        def handler(value: int, metadata: VideoMetadata):
            received_values.append(value)
            received_metadata.append(metadata)

        # Bind to mock datachannel
        _, message_handler = create_mock_channel_and_bind(channel)

        # Send message in new format
        message = json.dumps({
            "serialized_output_data": {"property_definition": 42},
            "video_metadata": {
                "frame_id": 408,
                "received_at": "2025-11-11T18:05:44.965372",
                "pts": 1764000,
                "time_base": 1.1111111111111112e-05,
                "declared_fps": 15.0,
                "measured_fps": None,
            },
            "errors": [],
        })

        message_handler(message)

        assert len(received_values) == 1
        assert received_values[0] == 42
        assert len(received_metadata) == 1
        assert received_metadata[0].frame_id == 408
        assert received_metadata[0].declared_fps == 15.0

    def test_handler_without_metadata_parameter(self):
        """Test that handler works without metadata parameter (backward compatible)."""
        channel = _DataChannel()
        received_values = []

        @channel.on_data("property_definition")
        def handler(value: int):
            received_values.append(value)

        # Bind to mock datachannel
        _, message_handler = create_mock_channel_and_bind(channel)

        # Send message in new format
        message = json.dumps({
            "serialized_output_data": {"property_definition": 42},
            "video_metadata": {
                "frame_id": 408,
                "received_at": "2025-11-11T18:05:44.965372",
            },
            "errors": [],
        })

        message_handler(message)

        assert len(received_values) == 1
        assert received_values[0] == 42

    def test_multiple_fields_in_message(self):
        """Test handling message with multiple fields in serialized_output_data."""
        channel = _DataChannel()
        received_predictions = []
        received_detections = []

        @channel.on_data("predictions")
        def handler1(value: dict, metadata: VideoMetadata):
            received_predictions.append((value, metadata))

        @channel.on_data("detections")
        def handler2(value: list, metadata: VideoMetadata):
            received_detections.append((value, metadata))

        # Bind to mock datachannel
        _, message_handler = create_mock_channel_and_bind(channel)

        # Send message with multiple fields
        message = json.dumps({
            "serialized_output_data": {
                "predictions": {"class": "dog", "confidence": 0.95},
                "detections": [{"x": 100, "y": 200}],
            },
            "video_metadata": {
                "frame_id": 1,
                "received_at": "2025-11-11T18:05:44.965372",
            },
            "errors": [],
        })

        message_handler(message)

        assert len(received_predictions) == 1
        assert received_predictions[0][0] == {"class": "dog", "confidence": 0.95}
        assert received_predictions[0][1].frame_id == 1

        assert len(received_detections) == 1
        assert received_detections[0][0] == [{"x": 100, "y": 200}]
        assert received_detections[0][1].frame_id == 1

    def test_global_handler_receives_full_dict(self):
        """Test that global handler receives full serialized_output_data dict + metadata."""
        channel = _DataChannel()
        received_data = []
        received_metadata = []

        @channel.on_data()
        def handler(data: dict, metadata: VideoMetadata):
            received_data.append(data)
            received_metadata.append(metadata)

        # Bind to mock datachannel
        _, message_handler = create_mock_channel_and_bind(channel)

        # Send message
        message = json.dumps({
            "serialized_output_data": {"test": "value", "count": 42},
            "video_metadata": {"frame_id": 1, "received_at": "2025-11-11T18:05:44.965372"},
        })

        message_handler(message)

        assert len(received_data) == 1
        assert received_data[0] == {"test": "value", "count": 42}
        assert len(received_metadata) == 1
        assert received_metadata[0].frame_id == 1

    def test_handler_error_handling(self):
        """Test that errors in handlers don't crash the message processing."""
        channel = _DataChannel()

        @channel.on_data("field1")
        def handler1(value: Any, metadata: VideoMetadata):
            raise ValueError("Handler 1 error")

        @channel.on_data("field2")
        def handler2(value: Any, metadata: VideoMetadata):
            pass  # This should still be called

        # Bind to mock datachannel
        _, message_handler = create_mock_channel_and_bind(channel)

        # Send message - should not raise
        message = json.dumps({
            "serialized_output_data": {
                "field1": "value1",
                "field2": "value2",
            },
            "video_metadata": {
                "frame_id": 1,
                "received_at": "2025-11-11T18:05:44.965372",
            },
        })

        # Should not raise despite handler1 error
        message_handler(message)

    def test_malformed_video_metadata(self):
        """Test handling malformed video_metadata gracefully."""
        channel = _DataChannel()
        received_values = []
        received_metadata = []

        @channel.on_data("test")
        def handler(value: Any, metadata: Optional[VideoMetadata]):
            received_values.append(value)
            received_metadata.append(metadata)

        # Bind to mock datachannel
        _, message_handler = create_mock_channel_and_bind(channel)

        # Send message with malformed metadata (missing required field)
        message = json.dumps({
            "serialized_output_data": {"test": "value"},
            "video_metadata": {
                # Missing frame_id and received_at
                "pts": 1000,
            },
        })

        message_handler(message)

        # Handler should still be called, but with None metadata
        assert len(received_values) == 1
        assert received_values[0] == "value"

    def test_non_dict_serialized_output_data(self):
        """Test handling non-dict serialized_output_data."""
        channel = _DataChannel()
        received_values = []

        @channel.on_data("predictions")
        def handler(value: Any):
            received_values.append(value)

        # Bind to mock datachannel
        _, message_handler = create_mock_channel_and_bind(channel)

        # Send message with non-dict serialized_output_data
        message = json.dumps({
            "serialized_output_data": "just a string",
            "video_metadata": {
                "frame_id": 1,
                "received_at": "2025-11-11T18:05:44.965372",
            },
        })

        message_handler(message)

        # Field handler should not be called for non-dict data
        assert len(received_values) == 0

    def test_message_without_video_metadata(self):
        """Test handling messages without video_metadata field."""
        channel = _DataChannel()
        received_values = []

        @channel.on_data("test")
        def handler(value: Any, metadata: Optional[VideoMetadata]):
            received_values.append(value)

        # Bind to mock datachannel
        _, message_handler = create_mock_channel_and_bind(channel)

        # Send message without video_metadata
        message = json.dumps({
            "serialized_output_data": {"test": "value"},
        })

        message_handler(message)

        # Handler should still be called
        assert len(received_values) == 1
        assert received_values[0] == "value"

    def test_invoke_handler_with_metadata(self):
        """Test _invoke_handler with metadata."""
        channel = _DataChannel()
        received_args = []

        def handler(value: int, metadata: VideoMetadata):
            received_args.append((value, metadata))

        metadata = VideoMetadata(
            frame_id=1,
            received_at=datetime.now(),
        )

        channel._invoke_handler(handler, 42, metadata)

        assert len(received_args) == 1
        assert received_args[0][0] == 42
        assert received_args[0][1] == metadata

    def test_invoke_handler_without_metadata(self):
        """Test _invoke_handler without metadata parameter."""
        channel = _DataChannel()
        received_args = []

        def handler(value: int):
            received_args.append(value)

        metadata = VideoMetadata(
            frame_id=1,
            received_at=datetime.now(),
        )

        channel._invoke_handler(handler, 42, metadata)

        assert len(received_args) == 1
        assert received_args[0] == 42

    def test_invoke_handler_with_none_metadata(self):
        """Test _invoke_handler when metadata is None."""
        channel = _DataChannel()
        received_args = []

        def handler(value: int):
            received_args.append(value)

        channel._invoke_handler(handler, 42, None)

        assert len(received_args) == 1
        assert received_args[0] == 42
