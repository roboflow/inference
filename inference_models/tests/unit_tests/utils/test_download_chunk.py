import os
from unittest.mock import MagicMock, patch

import pytest
import requests
from requests.exceptions import ChunkedEncodingError

from inference_models.errors import RangeRequestNotSupportedError, RetryError
from inference_models.utils.download import download_chunk
from inference_models.utils.file_system import pre_allocate_file, remove_file_if_exists


def _context_manager_for_response(response: MagicMock) -> MagicMock:
    cm = MagicMock()
    cm.__enter__.return_value = response
    cm.__exit__.return_value = False
    return cm


def _response_206(body: bytes, content_length: int | None = None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 206
    if content_length is None:
        content_length = len(body)
    resp.headers = {"Content-Length": str(content_length)}
    resp.iter_content = MagicMock(side_effect=lambda chunk_size: iter([body]))
    resp.raise_for_status = MagicMock()
    return resp


@pytest.fixture
def no_chunk_sleep():
    with patch(
        "inference_models.utils.download._chunk_download_backoff_sleep",
        autospec=True,
    ) as mock_sleep:
        yield mock_sleep


@pytest.fixture
def target_file(tmp_path: str) -> str:
    path = os.path.join(tmp_path, "chunk.bin")
    pre_allocate_file(path=path, file_size=128)
    yield path
    remove_file_if_exists(path=path)


def test_download_chunk_writes_full_range(no_chunk_sleep, target_file: str) -> None:
    body = b"hello"
    resp = _response_206(body, content_length=len(body))
    with patch("inference_models.utils.download.requests.get") as get_mock:
        get_mock.return_value = _context_manager_for_response(resp)
        download_chunk(
            url="https://example.test/file",
            start=0,
            end=4,
            target_path=target_file,
            response_codes_to_retry={503},
            max_attempts=5,
            file_chunk=64,
        )
    get_mock.assert_called_once()
    call_kw = get_mock.call_args.kwargs
    assert call_kw["headers"] == {"Range": "bytes=0-4"}
    with open(target_file, "rb") as f:
        f.seek(0)
        assert f.read(5) == body


def test_download_chunk_retries_on_retryable_status_then_succeeds(
    no_chunk_sleep, target_file: str
) -> None:
    body = b"abcde"
    bad = MagicMock()
    bad.status_code = 503
    bad.headers = {}
    bad.raise_for_status = MagicMock()
    bad.iter_content = MagicMock()

    good = _response_206(body, content_length=len(body))

    with patch("inference_models.utils.download.requests.get") as get_mock:
        get_mock.side_effect = [
            _context_manager_for_response(bad),
            _context_manager_for_response(good),
        ]
        download_chunk(
            url="https://example.test/file",
            start=0,
            end=4,
            target_path=target_file,
            response_codes_to_retry={503},
            max_attempts=5,
            file_chunk=64,
        )
    assert get_mock.call_count == 2
    assert get_mock.call_args_list[0].kwargs["headers"] == {"Range": "bytes=0-4"}
    assert get_mock.call_args_list[1].kwargs["headers"] == {"Range": "bytes=0-4"}
    with open(target_file, "rb") as f:
        assert f.read(5) == body


def test_download_chunk_resumes_after_partial_download_error(
    no_chunk_sleep, target_file: str
) -> None:
    first = MagicMock()
    first.status_code = 206
    first.headers = {"Content-Length": "5"}
    first.raise_for_status = MagicMock()

    def iter_then_error(chunk_size: int):
        yield b"ab"
        raise ChunkedEncodingError("incomplete chunked read")

    first.iter_content = MagicMock(side_effect=iter_then_error)

    second = _response_206(b"cde", content_length=3)

    with patch("inference_models.utils.download.requests.get") as get_mock:
        get_mock.side_effect = [
            _context_manager_for_response(first),
            _context_manager_for_response(second),
        ]
        download_chunk(
            url="https://example.test/file",
            start=0,
            end=4,
            target_path=target_file,
            response_codes_to_retry={503},
            max_attempts=5,
            file_chunk=64,
        )
    assert get_mock.call_count == 2
    assert get_mock.call_args_list[0].kwargs["headers"] == {"Range": "bytes=0-4"}
    assert get_mock.call_args_list[1].kwargs["headers"] == {"Range": "bytes=2-4"}
    with open(target_file, "rb") as f:
        assert f.read(5) == b"abcde"


def test_download_chunk_resumes_after_short_body_vs_content_length(
    no_chunk_sleep, target_file: str
) -> None:
    """Server claims Content-Length 5 but first body is shorter; second request completes span."""
    first = MagicMock()
    first.status_code = 206
    first.headers = {"Content-Length": "5"}
    first.iter_content = MagicMock(side_effect=lambda chunk_size: iter([b"ab"]))
    first.raise_for_status = MagicMock()

    second = _response_206(b"cde", content_length=3)

    with patch("inference_models.utils.download.requests.get") as get_mock:
        get_mock.side_effect = [
            _context_manager_for_response(first),
            _context_manager_for_response(second),
        ]
        download_chunk(
            url="https://example.test/file",
            start=0,
            end=4,
            target_path=target_file,
            response_codes_to_retry={503},
            max_attempts=5,
            file_chunk=64,
        )
    assert get_mock.call_count == 2
    assert get_mock.call_args_list[0].kwargs["headers"] == {"Range": "bytes=0-4"}
    assert get_mock.call_args_list[1].kwargs["headers"] == {"Range": "bytes=2-4"}
    with open(target_file, "rb") as f:
        assert f.read(5) == b"abcde"


def test_download_chunk_raises_range_request_not_supported_on_200(
    no_chunk_sleep, target_file: str
) -> None:
    resp = MagicMock()
    resp.status_code = 200
    resp.headers = {"Content-Length": "5"}
    resp.iter_content = MagicMock()
    resp.raise_for_status = MagicMock()

    with patch("inference_models.utils.download.requests.get") as get_mock:
        get_mock.return_value = _context_manager_for_response(resp)
        with pytest.raises(RangeRequestNotSupportedError) as exc_info:
            download_chunk(
                url="https://example.test/file",
                start=0,
                end=4,
                target_path=target_file,
                response_codes_to_retry={503},
                max_attempts=5,
                file_chunk=64,
            )
    assert "returned 200 instead of 206" in str(exc_info.value)


def test_download_chunk_raises_retry_error_after_connectivity_exhausted(
    no_chunk_sleep, target_file: str
) -> None:
    with patch("inference_models.utils.download.requests.get") as get_mock:
        get_mock.side_effect = requests.Timeout("boom")
        with pytest.raises(RetryError, match="Connectivity error"):
            download_chunk(
                url="https://example.test/file",
                start=0,
                end=4,
                target_path=target_file,
                response_codes_to_retry={503},
                max_attempts=3,
                file_chunk=64,
            )
    assert get_mock.call_count == 3
