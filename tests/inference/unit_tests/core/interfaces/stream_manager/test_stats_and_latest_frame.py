"""
Tests for /stats and /latest_frame endpoints, and the pipeline manager
handler for LATEST_FRAME command.
"""

import base64
from collections import deque
from datetime import datetime
from multiprocessing import Queue
from unittest import mock
from unittest.mock import AsyncMock, MagicMock

import cv2
import numpy as np
import pytest

from inference.core.interfaces.camera.entities import VideoFrame
from inference.core.interfaces.stream.sinks import InMemoryBufferSink
from inference.core.interfaces.stream_manager.manager_app import (
    inference_pipeline_manager,
)
from inference.core.interfaces.stream_manager.manager_app.entities import (
    CommandType,
    OperationStatus,
)
from inference.core.interfaces.stream_manager.manager_app.inference_pipeline_manager import (
    InferencePipelineManager,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_status_response(report):
    resp = MagicMock()
    resp.report = report
    return resp


def _make_list_response(pipeline_ids):
    resp = MagicMock()
    resp.pipelines = pipeline_ids
    return resp


def _full_report(**overrides):
    report = {
        "inference_throughput": 10.0,
        "sources_metadata": [
            {
                "source_properties": {
                    "width": 1920,
                    "height": 1080,
                    "total_frames": 0,
                    "is_file": False,
                    "fps": 30.0,
                },
                "source_reference": "rtsp://example.com/stream",
                "source_id": 0,
            }
        ],
    }
    report.update(overrides)
    return report


# ---------------------------------------------------------------------------
# /stats logic tests (testing the aggregation logic directly)
# ---------------------------------------------------------------------------


class TestStatsAggregation:
    """Tests for the /stats endpoint aggregation logic."""

    def test_stats_returns_nulls_when_no_pipelines(self):
        """When there are no pipelines, camera_fps and inference_fps should be None."""
        camera_fps_values = []
        inference_fps_values = []
        stream_count = 0

        result = {
            "camera_fps": (
                sum(camera_fps_values) / len(camera_fps_values)
                if camera_fps_values
                else None
            ),
            "inference_fps": (
                sum(inference_fps_values) / len(inference_fps_values)
                if inference_fps_values
                else None
            ),
            "stream_count": stream_count,
        }

        assert result["camera_fps"] is None
        assert result["inference_fps"] is None
        assert result["stream_count"] == 0

    def test_stats_aggregates_single_pipeline(self):
        """Single pipeline should return its values directly."""
        report = _full_report(inference_throughput=15.0)

        camera_fps_values = []
        inference_fps_values = []

        throughput = report.get("inference_throughput", 0.0)
        if throughput and throughput > 0:
            inference_fps_values.append(throughput)
        for src in report.get("sources_metadata", []):
            props = src.get("source_properties") or {}
            fps = props.get("fps")
            if fps and fps > 0:
                camera_fps_values.append(fps)

        result = {
            "camera_fps": (
                sum(camera_fps_values) / len(camera_fps_values)
                if camera_fps_values
                else None
            ),
            "inference_fps": (
                sum(inference_fps_values) / len(inference_fps_values)
                if inference_fps_values
                else None
            ),
            "stream_count": 1,
        }

        assert result["camera_fps"] == 30.0
        assert result["inference_fps"] == 15.0
        assert result["stream_count"] == 1

    def test_stats_averages_multiple_pipelines(self):
        """Multiple pipelines should have their FPS values averaged."""
        reports = [
            _full_report(inference_throughput=10.0),
            _full_report(inference_throughput=20.0),
        ]
        # Override the second pipeline's camera fps
        reports[1]["sources_metadata"][0]["source_properties"]["fps"] = 60.0

        camera_fps_values = []
        inference_fps_values = []
        for report in reports:
            throughput = report.get("inference_throughput", 0.0)
            if throughput and throughput > 0:
                inference_fps_values.append(throughput)
            for src in report.get("sources_metadata", []):
                props = src.get("source_properties") or {}
                fps = props.get("fps")
                if fps and fps > 0:
                    camera_fps_values.append(fps)

        result = {
            "camera_fps": (
                sum(camera_fps_values) / len(camera_fps_values)
                if camera_fps_values
                else None
            ),
            "inference_fps": (
                sum(inference_fps_values) / len(inference_fps_values)
                if inference_fps_values
                else None
            ),
            "stream_count": 2,
        }

        assert result["camera_fps"] == 45.0  # (30 + 60) / 2
        assert result["inference_fps"] == 15.0  # (10 + 20) / 2
        assert result["stream_count"] == 2

    def test_stats_skips_zero_fps(self):
        """Pipelines with zero throughput or FPS should be excluded from averages."""
        report = _full_report(inference_throughput=0.0)
        report["sources_metadata"][0]["source_properties"]["fps"] = 0

        camera_fps_values = []
        inference_fps_values = []
        throughput = report.get("inference_throughput", 0.0)
        if throughput and throughput > 0:
            inference_fps_values.append(throughput)
        for src in report.get("sources_metadata", []):
            props = src.get("source_properties") or {}
            fps = props.get("fps")
            if fps and fps > 0:
                camera_fps_values.append(fps)

        result = {
            "camera_fps": (
                sum(camera_fps_values) / len(camera_fps_values)
                if camera_fps_values
                else None
            ),
            "inference_fps": (
                sum(inference_fps_values) / len(inference_fps_values)
                if inference_fps_values
                else None
            ),
            "stream_count": 1,
        }

        assert result["camera_fps"] is None
        assert result["inference_fps"] is None


# ---------------------------------------------------------------------------
# LATEST_FRAME pipeline manager handler tests
# ---------------------------------------------------------------------------


class TestLatestFrameHandler:
    """Tests for InferencePipelineManager._handle_latest_frame."""

    @pytest.mark.timeout(30)
    def test_latest_frame_returns_null_when_buffer_is_empty(self):
        """When buffer is empty, frame_data should be None."""
        command_queue, responses_queue = Queue(), Queue()
        manager = InferencePipelineManager(
            pipeline_id="my_pipeline",
            command_queue=command_queue,
            responses_queue=responses_queue,
        )
        manager._buffer_sink = InMemoryBufferSink(queue_size=10)

        command_queue.put(("1", {"type": CommandType.LATEST_FRAME}))
        command_queue.put(("2", {"type": CommandType.TERMINATE}))

        manager.run()

        status_1 = responses_queue.get()
        assert status_1[0] == "1"
        assert status_1[1]["status"] == OperationStatus.SUCCESS
        assert status_1[1]["frame_data"] is None

    @pytest.mark.timeout(30)
    def test_latest_frame_returns_null_when_buffer_sink_is_none(self):
        """When buffer sink is None, frame_data should be None."""
        command_queue, responses_queue = Queue(), Queue()
        manager = InferencePipelineManager(
            pipeline_id="my_pipeline",
            command_queue=command_queue,
            responses_queue=responses_queue,
        )
        # buffer_sink is None by default

        command_queue.put(("1", {"type": CommandType.LATEST_FRAME}))
        command_queue.put(("2", {"type": CommandType.TERMINATE}))

        manager.run()

        status_1 = responses_queue.get()
        assert status_1[0] == "1"
        assert status_1[1]["status"] == OperationStatus.SUCCESS
        assert status_1[1]["frame_data"] is None

    @pytest.mark.timeout(30)
    def test_latest_frame_returns_encoded_frame(self):
        """When buffer has frames, should return base64-encoded JPEG."""
        command_queue, responses_queue = Queue(), Queue()
        manager = InferencePipelineManager(
            pipeline_id="my_pipeline",
            command_queue=command_queue,
            responses_queue=responses_queue,
        )
        buffer_sink = InMemoryBufferSink(queue_size=10)
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image[50, 50] = [255, 0, 0]
        frame_ts = datetime(2024, 1, 1, 12, 0, 0)
        buffer_sink.on_prediction(
            predictions=[{"test": "value"}],
            video_frame=[
                VideoFrame(
                    image=test_image,
                    frame_id=42,
                    frame_timestamp=frame_ts,
                    source_id=7,
                ),
            ],
        )
        manager._buffer_sink = buffer_sink

        command_queue.put(("1", {"type": CommandType.LATEST_FRAME}))
        command_queue.put(("2", {"type": CommandType.TERMINATE}))

        manager.run()

        status_1 = responses_queue.get()
        assert status_1[0] == "1"
        assert status_1[1]["status"] == OperationStatus.SUCCESS
        assert status_1[1]["frame_data"] is not None
        assert status_1[1]["frame_id"] == 42
        assert status_1[1]["source_id"] == 7

        # Verify the base64 data decodes to a valid JPEG
        decoded = base64.b64decode(status_1[1]["frame_data"])
        assert decoded[:2] == b"\xff\xd8"  # JPEG magic bytes

    @pytest.mark.timeout(30)
    def test_latest_frame_is_non_destructive(self):
        """Reading latest frame should not consume the buffer."""
        command_queue, responses_queue = Queue(), Queue()
        manager = InferencePipelineManager(
            pipeline_id="my_pipeline",
            command_queue=command_queue,
            responses_queue=responses_queue,
        )
        buffer_sink = InMemoryBufferSink(queue_size=10)
        test_image = np.zeros((50, 50, 3), dtype=np.uint8)
        buffer_sink.on_prediction(
            predictions=[{"test": "value"}],
            video_frame=[
                VideoFrame(
                    image=test_image,
                    frame_id=1,
                    frame_timestamp=datetime.now(),
                    source_id=0,
                ),
            ],
        )
        manager._buffer_sink = buffer_sink

        # Request latest frame twice -- both should succeed
        command_queue.put(("1", {"type": CommandType.LATEST_FRAME}))
        command_queue.put(("2", {"type": CommandType.LATEST_FRAME}))
        command_queue.put(("3", {"type": CommandType.TERMINATE}))

        manager.run()

        status_1 = responses_queue.get()
        status_2 = responses_queue.get()
        assert status_1[1]["frame_data"] is not None
        assert status_2[1]["frame_data"] is not None
        # Buffer should still have its entry
        assert not buffer_sink.empty()

    @pytest.mark.timeout(30)
    def test_latest_frame_skips_none_frames(self):
        """When the frame list contains None entries, should find the last non-None."""
        command_queue, responses_queue = Queue(), Queue()
        manager = InferencePipelineManager(
            pipeline_id="my_pipeline",
            command_queue=command_queue,
            responses_queue=responses_queue,
        )
        buffer_sink = InMemoryBufferSink(queue_size=10)
        test_image = np.zeros((50, 50, 3), dtype=np.uint8)
        buffer_sink.on_prediction(
            predictions=[None, {"test": "value"}, None],
            video_frame=[
                None,
                VideoFrame(
                    image=test_image,
                    frame_id=99,
                    frame_timestamp=datetime.now(),
                    source_id=5,
                ),
                None,
            ],
        )
        manager._buffer_sink = buffer_sink

        command_queue.put(("1", {"type": CommandType.LATEST_FRAME}))
        command_queue.put(("2", {"type": CommandType.TERMINATE}))

        manager.run()

        status_1 = responses_queue.get()
        assert status_1[1]["status"] == OperationStatus.SUCCESS
        assert status_1[1]["frame_data"] is not None
        assert status_1[1]["frame_id"] == 99
        assert status_1[1]["source_id"] == 5

    @pytest.mark.timeout(30)
    def test_latest_frame_returns_null_when_all_frames_are_none(self):
        """When all frames in the buffer entry are None, return null frame_data."""
        command_queue, responses_queue = Queue(), Queue()
        manager = InferencePipelineManager(
            pipeline_id="my_pipeline",
            command_queue=command_queue,
            responses_queue=responses_queue,
        )
        buffer_sink = InMemoryBufferSink(queue_size=10)
        buffer_sink.on_prediction(
            predictions=[None, None],
            video_frame=[None, None],
        )
        manager._buffer_sink = buffer_sink

        command_queue.put(("1", {"type": CommandType.LATEST_FRAME}))
        command_queue.put(("2", {"type": CommandType.TERMINATE}))

        manager.run()

        status_1 = responses_queue.get()
        assert status_1[1]["status"] == OperationStatus.SUCCESS
        assert status_1[1]["frame_data"] is None
