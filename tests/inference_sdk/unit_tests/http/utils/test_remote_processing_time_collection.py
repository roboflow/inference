from unittest.mock import MagicMock

import pytest
from requests import Response

from inference_sdk.config import (
    PROCESSING_TIME_HEADER,
    RemoteProcessingTimeCollector,
    remote_processing_times,
)
from inference_sdk.http.utils.executors import (
    _collect_remote_processing_times,
    _extract_model_id_from_request_data,
)
from inference_sdk.http.utils.request_building import RequestData


def _make_response(processing_time: str = None, status_code: int = 200) -> Response:
    response = Response()
    response.status_code = status_code
    if processing_time is not None:
        response.headers[PROCESSING_TIME_HEADER] = processing_time
    return response


def _make_request_data(
    url: str = "https://some.com/infer/object_detection",
    model_id: str = None,
) -> RequestData:
    payload = {}
    if model_id is not None:
        payload["model_id"] = model_id
    return RequestData(
        url=url,
        request_elements=1,
        headers=None,
        data=None,
        parameters=None,
        payload=payload if payload else None,
        image_scaling_factors=[None],
    )


class TestExtractModelIdFromRequestData:
    def test_extracts_model_id_from_payload(self) -> None:
        # given
        request_data = _make_request_data(model_id="coco/3")

        # when
        result = _extract_model_id_from_request_data(request_data)

        # then
        assert result == "coco/3"

    def test_falls_back_to_url_path_when_no_model_id_in_payload(self) -> None:
        # given
        request_data = _make_request_data(
            url="https://localhost:9001/infer/object_detection"
        )

        # when
        result = _extract_model_id_from_request_data(request_data)

        # then
        assert result == "infer/object_detection"

    def test_falls_back_to_url_path_when_no_payload(self) -> None:
        # given
        request_data = RequestData(
            url="https://localhost:9001/clip/embed_image",
            request_elements=1,
            headers=None,
            data=None,
            parameters=None,
            payload=None,
            image_scaling_factors=[None],
        )

        # when
        result = _extract_model_id_from_request_data(request_data)

        # then
        assert result == "clip/embed_image"


class TestCollectRemoteProcessingTimes:
    def test_no_collection_when_contextvar_not_set(self) -> None:
        # given
        token = remote_processing_times.set(None)
        responses = [_make_response("0.5")]
        requests_data = [_make_request_data(model_id="m1")]

        try:
            # when
            _collect_remote_processing_times(responses, requests_data)
        finally:
            remote_processing_times.reset(token)

        # then - no error raised, nothing collected

    def test_collects_processing_time_with_model_id(self) -> None:
        # given
        collector = RemoteProcessingTimeCollector()
        token = remote_processing_times.set(collector)
        responses = [_make_response("0.523")]
        requests_data = [_make_request_data(model_id="coco/1")]

        try:
            # when
            _collect_remote_processing_times(responses, requests_data)
        finally:
            remote_processing_times.reset(token)

        # then
        entries = collector.drain()
        assert len(entries) == 1
        assert entries[0] == ("coco/1", 0.523)

    def test_collects_multiple_responses(self) -> None:
        # given
        collector = RemoteProcessingTimeCollector()
        token = remote_processing_times.set(collector)
        responses = [
            _make_response("0.5"),
            _make_response("0.3"),
            _make_response("0.2"),
        ]
        requests_data = [
            _make_request_data(model_id="m1"),
            _make_request_data(model_id="m2"),
            _make_request_data(model_id="m1"),
        ]

        try:
            # when
            _collect_remote_processing_times(responses, requests_data)
        finally:
            remote_processing_times.reset(token)

        # then
        entries = collector.drain()
        assert len(entries) == 3
        total = sum(t for _, t in entries)
        assert abs(total - 1.0) < 1e-9

    def test_skips_responses_without_processing_time_header(self) -> None:
        # given
        collector = RemoteProcessingTimeCollector()
        token = remote_processing_times.set(collector)
        responses = [
            _make_response("0.5"),
            _make_response(None),  # no header
            _make_response("0.3"),
        ]
        requests_data = [
            _make_request_data(model_id="m1"),
            _make_request_data(model_id="m2"),
            _make_request_data(model_id="m3"),
        ]

        try:
            # when
            _collect_remote_processing_times(responses, requests_data)
        finally:
            remote_processing_times.reset(token)

        # then
        entries = collector.drain()
        assert len(entries) == 2

    def test_skips_malformed_processing_time_header(self) -> None:
        # given
        collector = RemoteProcessingTimeCollector()
        token = remote_processing_times.set(collector)
        responses = [
            _make_response("not_a_number"),
            _make_response("0.5"),
        ]
        requests_data = [
            _make_request_data(model_id="m1"),
            _make_request_data(model_id="m2"),
        ]

        try:
            # when
            _collect_remote_processing_times(responses, requests_data)
        finally:
            remote_processing_times.reset(token)

        # then
        entries = collector.drain()
        assert len(entries) == 1
        assert entries[0] == ("m2", 0.5)

    def test_handles_more_responses_than_request_data(self, caplog) -> None:
        # given
        collector = RemoteProcessingTimeCollector()
        token = remote_processing_times.set(collector)
        responses = [_make_response("0.5"), _make_response("0.3")]
        requests_data = [_make_request_data(model_id="m1")]

        try:
            # when
            with caplog.at_level("WARNING"):
                _collect_remote_processing_times(responses, requests_data)
        finally:
            remote_processing_times.reset(token)

        # then - only the paired entry is collected, extra response is skipped
        entries = collector.drain()
        assert len(entries) == 1
        assert entries[0][0] == "m1"
        assert "does not match" in caplog.text
