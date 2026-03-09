import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inference.core.managers.prometheus import CustomCollector


@pytest.fixture
def collector():
    model_manager = MagicMock()
    model_manager.models.return_value = []
    c = CustomCollector(model_manager)
    return c


def _make_status_response(report):
    resp = MagicMock()
    resp.report = report
    return resp


def _make_list_response(pipeline_ids):
    resp = MagicMock()
    resp.pipelines = pipeline_ids
    return resp


def _full_report(**overrides):
    """Build a realistic pipeline report dict with sensible defaults."""
    report = {
        "inference_throughput": 10.0,
        "latency_reports": [
            {
                "source_id": 0,
                "frame_decoding_latency": 0.02,
                "inference_latency": 0.04,
                "e2e_latency": 0.06,
            }
        ],
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


class TestGetStreamMetrics:
    def test_returns_empty_when_client_is_none(self, collector):
        assert collector.stream_manager_client is None
        assert collector.get_stream_metrics() == {}

    def test_returns_metrics_for_single_pipeline(self, collector):
        client = AsyncMock()
        client.list_pipelines.return_value = _make_list_response(["pipe-1"])
        client.get_status.return_value = _make_status_response(
            _full_report(
                inference_throughput=25.0,
                latency_reports=[
                    {
                        "source_id": 0,
                        "frame_decoding_latency": 0.01,
                        "inference_latency": 0.05,
                        "e2e_latency": 0.08,
                    },
                    {
                        "source_id": 1,
                        "frame_decoding_latency": 0.03,
                        "inference_latency": 0.07,
                        "e2e_latency": 0.12,
                    },
                ],
            )
        )
        collector.stream_manager_client = client

        result = collector.get_stream_metrics()

        assert "pipe-1" in result
        m = result["pipe-1"]
        assert m["inference_throughput"] == 25.0
        assert m["camera_fps"] == 30.0
        assert m["frame_decoding_latency"] == pytest.approx(0.02)
        assert m["inference_latency"] == pytest.approx(0.06)
        assert m["e2e_latency"] == pytest.approx(0.10)

    def test_returns_metrics_for_multiple_pipelines(self, collector):
        client = AsyncMock()
        client.list_pipelines.return_value = _make_list_response(
            ["pipe-a", "pipe-b"]
        )
        client.get_status.side_effect = [
            _make_status_response(
                _full_report(inference_throughput=10.0)
            ),
            _make_status_response(
                _full_report(inference_throughput=20.0)
            ),
        ]
        collector.stream_manager_client = client

        result = collector.get_stream_metrics()

        assert len(result) == 2
        assert result["pipe-a"]["inference_throughput"] == 10.0
        assert result["pipe-b"]["inference_throughput"] == 20.0

    def test_returns_empty_when_no_active_pipelines(self, collector):
        client = AsyncMock()
        client.list_pipelines.return_value = _make_list_response([])
        collector.stream_manager_client = client

        result = collector.get_stream_metrics()

        assert result == {}
        client.get_status.assert_not_called()

    def test_returns_empty_on_connectivity_error(self, collector):
        from inference.core.interfaces.stream_manager.api.errors import (
            ConnectivityError,
        )

        client = AsyncMock()
        client.list_pipelines.side_effect = ConnectivityError(
            private_message="connection refused"
        )
        collector.stream_manager_client = client

        assert collector.get_stream_metrics() == {}

    def test_returns_empty_when_get_status_fails_mid_iteration(self, collector):
        """If get_status raises for one pipeline, the whole fetch fails gracefully."""
        client = AsyncMock()
        client.list_pipelines.return_value = _make_list_response(
            ["ok-pipe", "bad-pipe"]
        )
        client.get_status.side_effect = [
            _make_status_response(_full_report()),
            RuntimeError("stream manager unavailable"),
        ]
        collector.stream_manager_client = client

        assert collector.get_stream_metrics() == {}

    def test_defaults_when_report_missing_keys(self, collector):
        """Report dict with no latency_reports, inference_throughput, or sources_metadata."""
        client = AsyncMock()
        client.list_pipelines.return_value = _make_list_response(["sparse"])
        client.get_status.return_value = _make_status_response({})
        collector.stream_manager_client = client

        result = collector.get_stream_metrics()

        m = result["sparse"]
        assert m["inference_throughput"] == 0.0
        assert m["camera_fps"] == 0.0
        assert m["frame_decoding_latency"] == 0.0
        assert m["inference_latency"] == 0.0
        assert m["e2e_latency"] == 0.0

    def test_runtime_error_fallback_uses_thread_pool(self, collector):
        """When asyncio.run() raises RuntimeError (already in event loop),
        the ThreadPoolExecutor fallback is used."""
        client = AsyncMock()
        client.list_pipelines.return_value = _make_list_response(["fb-pipe"])
        client.get_status.return_value = _make_status_response(_full_report())
        collector.stream_manager_client = client

        expected = {"fb-pipe": {"inference_throughput": 10.0, "camera_fps": 30.0, "frame_decoding_latency": 0.0, "inference_latency": 0.0, "e2e_latency": 0.0}}

        mock_future = MagicMock()
        mock_future.result.return_value = expected

        mock_pool = MagicMock()
        mock_pool.__enter__ = MagicMock(return_value=mock_pool)
        mock_pool.__exit__ = MagicMock(return_value=False)
        mock_pool.submit.return_value = mock_future

        def raise_and_close(coro):
            coro.close()
            raise RuntimeError("cannot be called from a running event loop")

        with patch(
            "inference.core.managers.prometheus.asyncio.run",
            side_effect=raise_and_close,
        ), patch(
            "inference.core.managers.prometheus.concurrent.futures.ThreadPoolExecutor",
            return_value=mock_pool,
        ) as mock_tpe_cls:
            result = collector.get_stream_metrics()

        mock_tpe_cls.assert_called_once_with(max_workers=1)
        mock_pool.submit.assert_called_once()
        submitted_fn = mock_pool.submit.call_args[0][0]
        assert callable(submitted_fn)
        submitted_coro = mock_pool.submit.call_args[0][1]
        submitted_coro.close()
        assert result == expected

    def test_camera_fps_averaged_across_sources(self, collector):
        """Camera FPS is averaged across multiple video sources."""
        client = AsyncMock()
        client.list_pipelines.return_value = _make_list_response(["multi-src"])
        client.get_status.return_value = _make_status_response(
            _full_report(
                sources_metadata=[
                    {"source_properties": {"fps": 30.0}, "source_id": 0},
                    {"source_properties": {"fps": 15.0}, "source_id": 1},
                ]
            )
        )
        collector.stream_manager_client = client

        result = collector.get_stream_metrics()
        assert result["multi-src"]["camera_fps"] == pytest.approx(22.5)


class TestAverageLatencyField:
    def test_handles_none_values(self):
        reports = [
            {"frame_decoding_latency": None},
            {"frame_decoding_latency": 0.04},
        ]
        assert CustomCollector._average_latency_field(
            reports, "frame_decoding_latency"
        ) == pytest.approx(0.04)

    def test_handles_empty_list(self):
        assert CustomCollector._average_latency_field([], "inference_latency") == 0.0

    def test_handles_all_none(self):
        reports = [
            {"e2e_latency": None},
            {"e2e_latency": None},
        ]
        assert CustomCollector._average_latency_field(reports, "e2e_latency") == 0.0

    def test_handles_missing_key(self):
        reports = [{"other_field": 1.0}]
        assert CustomCollector._average_latency_field(
            reports, "inference_latency"
        ) == 0.0

    def test_computes_correct_average(self):
        reports = [
            {"lat": 0.1},
            {"lat": 0.2},
            {"lat": 0.3},
        ]
        assert CustomCollector._average_latency_field(reports, "lat") == pytest.approx(
            0.2
        )


class TestAverageSourceFps:
    def test_single_source(self):
        metadata = [{"source_properties": {"fps": 30.0}}]
        assert CustomCollector._average_source_fps(metadata) == 30.0

    def test_multiple_sources(self):
        metadata = [
            {"source_properties": {"fps": 30.0}},
            {"source_properties": {"fps": 15.0}},
        ]
        assert CustomCollector._average_source_fps(metadata) == pytest.approx(22.5)

    def test_empty_list(self):
        assert CustomCollector._average_source_fps([]) == 0.0

    def test_missing_source_properties(self):
        metadata = [{"source_id": 0}]
        assert CustomCollector._average_source_fps(metadata) == 0.0

    def test_null_source_properties(self):
        metadata = [{"source_properties": None}]
        assert CustomCollector._average_source_fps(metadata) == 0.0

    def test_missing_fps_key(self):
        metadata = [{"source_properties": {"width": 1920}}]
        assert CustomCollector._average_source_fps(metadata) == 0.0

    def test_zero_fps_excluded(self):
        metadata = [
            {"source_properties": {"fps": 0}},
            {"source_properties": {"fps": 30.0}},
        ]
        assert CustomCollector._average_source_fps(metadata) == 30.0

    def test_negative_fps_excluded(self):
        metadata = [
            {"source_properties": {"fps": -1.0}},
            {"source_properties": {"fps": 25.0}},
        ]
        assert CustomCollector._average_source_fps(metadata) == 25.0


class TestCollectYieldsStreamMetrics:
    def test_collect_includes_stream_gauges_with_correct_values(self, collector):
        client = AsyncMock()
        client.list_pipelines.return_value = _make_list_response(["my-pipe"])
        client.get_status.return_value = _make_status_response(_full_report())
        collector.stream_manager_client = client

        metric_families = list(collector.collect())
        by_name = {mf.name: mf for mf in metric_families}

        # Verify sanitized name (hyphen -> underscore)
        assert "inference_pipeline_inference_fps_my_pipe" in by_name
        assert "inference_pipeline_camera_fps_my_pipe" in by_name
        assert "inference_pipeline_frame_decoding_latency_my_pipe" in by_name
        assert "inference_pipeline_inference_latency_my_pipe" in by_name
        assert "inference_pipeline_e2e_latency_my_pipe" in by_name
        assert "inference_pipeline_active_streams" in by_name

        # Verify actual gauge values
        assert by_name["inference_pipeline_inference_fps_my_pipe"].samples[0].value == 10.0
        assert by_name["inference_pipeline_camera_fps_my_pipe"].samples[0].value == 30.0
        assert by_name["inference_pipeline_frame_decoding_latency_my_pipe"].samples[0].value == pytest.approx(0.02)
        assert by_name["inference_pipeline_inference_latency_my_pipe"].samples[0].value == pytest.approx(0.04)
        assert by_name["inference_pipeline_e2e_latency_my_pipe"].samples[0].value == pytest.approx(0.06)
        assert by_name["inference_pipeline_active_streams"].samples[0].value == 1

    def test_collect_yields_zero_active_streams_when_no_client(self, collector):
        metric_families = list(collector.collect())
        active = next(
            mf for mf in metric_families
            if mf.name == "inference_pipeline_active_streams"
        )
        assert active.samples[0].value == 0

    def test_collect_sanitizes_pipeline_ids_with_special_chars(self, collector):
        """Pipeline IDs with dots, slashes, and UUIDs get sanitized for Prometheus."""
        client = AsyncMock()
        client.list_pipelines.return_value = _make_list_response(
            ["my/pipeline.v2"]
        )
        client.get_status.return_value = _make_status_response(_full_report())
        collector.stream_manager_client = client

        metric_families = list(collector.collect())
        names = [mf.name for mf in metric_families]

        # "my/pipeline.v2" -> "my_pipeline_v2"
        assert "inference_pipeline_inference_fps_my_pipeline_v2" in names
        assert "inference_pipeline_camera_fps_my_pipeline_v2" in names

    def test_collect_multiple_pipelines_yields_correct_active_count(self, collector):
        client = AsyncMock()
        client.list_pipelines.return_value = _make_list_response(
            ["p1", "p2", "p3"]
        )
        client.get_status.return_value = _make_status_response(_full_report())
        collector.stream_manager_client = client

        metric_families = list(collector.collect())
        active = next(
            mf for mf in metric_families
            if mf.name == "inference_pipeline_active_streams"
        )
        assert active.samples[0].value == 3

        # Each pipeline should have its own set of gauges
        fps_metrics = [
            mf for mf in metric_families
            if mf.name.startswith("inference_pipeline_inference_fps_")
        ]
        assert len(fps_metrics) == 3

    def test_collect_still_yields_model_metrics_when_stream_fails(self, collector):
        """Stream failure should not prevent model metrics from being yielded."""
        from inference.core.interfaces.stream_manager.api.errors import (
            ConnectivityError,
        )

        client = AsyncMock()
        client.list_pipelines.side_effect = ConnectivityError(
            private_message="down"
        )
        collector.stream_manager_client = client

        metric_families = list(collector.collect())
        names = [mf.name for mf in metric_families]

        assert "num_inferences_total" in names
        assert "avg_inference_time_total" in names
        assert "num_errors_total" in names
        active = next(
            mf for mf in metric_families
            if mf.name == "inference_pipeline_active_streams"
        )
        assert active.samples[0].value == 0
