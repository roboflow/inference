import requests
from requests import Response

from inference_sdk.config import (
    PROCESSING_TIME_HEADER,
    RemoteProcessingTimeCollector,
    remote_processing_times,
)
from inference_sdk.http.client import _collect_processing_time_from_response


def _make_response(processing_time: str = None) -> Response:
    response = Response()
    response.status_code = 200
    if processing_time is not None:
        response.headers[PROCESSING_TIME_HEADER] = processing_time
    return response


class TestCollectProcessingTimeFromResponse:
    def test_no_collection_when_contextvar_not_set(self) -> None:
        # given
        token = remote_processing_times.set(None)
        response = _make_response("0.5")

        try:
            # when
            _collect_processing_time_from_response(response, model_id="clip")
        finally:
            remote_processing_times.reset(token)

        # then - no error raised

    def test_collects_with_model_id(self) -> None:
        # given
        collector = RemoteProcessingTimeCollector()
        token = remote_processing_times.set(collector)
        response = _make_response("0.42")

        try:
            # when
            _collect_processing_time_from_response(response, model_id="clip")
        finally:
            remote_processing_times.reset(token)

        # then
        entries = collector.get_entries()
        assert len(entries) == 1
        assert entries[0] == ("clip", 0.42)

    def test_skips_when_no_header(self) -> None:
        # given
        collector = RemoteProcessingTimeCollector()
        token = remote_processing_times.set(collector)
        response = _make_response(None)

        try:
            # when
            _collect_processing_time_from_response(response, model_id="clip")
        finally:
            remote_processing_times.reset(token)

        # then
        assert collector.has_data() is False

    def test_skips_malformed_header(self) -> None:
        # given
        collector = RemoteProcessingTimeCollector()
        token = remote_processing_times.set(collector)
        response = _make_response("invalid")

        try:
            # when
            _collect_processing_time_from_response(response, model_id="clip")
        finally:
            remote_processing_times.reset(token)

        # then
        assert collector.has_data() is False

    def test_default_model_id(self) -> None:
        # given
        collector = RemoteProcessingTimeCollector()
        token = remote_processing_times.set(collector)
        response = _make_response("0.1")

        try:
            # when
            _collect_processing_time_from_response(response)
        finally:
            remote_processing_times.reset(token)

        # then
        entries = collector.get_entries()
        assert entries[0][0] == "unknown"
