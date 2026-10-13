"""Unit tests of ``v2/s3_sink@v1``.

Uploads go to a real boto3 S3 client wrapped in ``botocore.stub.Stubber``:
the stubber answers before any request is signed or sent, so no endpoint is
contacted. Credentials are invented test values.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import boto3
import pytest
from botocore.exceptions import EndpointConnectionError
from botocore.stub import Stubber
from roboflow_workflows.v2.blocks.sinks.s3 import (
    MISSING_CREDENTIALS_MESSAGE,
    S3SinkBlock,
)
from roboflow_workflows.v2.data.secrets import Secret, SecretUnavailableError

KEY_ID = "AKIATESTKEYID000"
SECRET_KEY = "test-SECRET-access-key-9c1e"
CSV_ROW_1 = "name,count\na,1\n"
CSV_ROW_2 = "name,count\nb,2"


class Clock:
    """Wall clock that moves one second per reading."""

    def __init__(self) -> None:
        self.now = datetime(2026, 1, 2, 3, 4, 5, 678, tzinfo=timezone.utc)

    def __call__(self, tz: Any) -> datetime:
        reading = self.now
        self.now += timedelta(seconds=1)
        return reading.astimezone(tz)


class StubbedS3:
    """Client factory over one stubbed boto3 client; records its calls."""

    def __init__(self) -> None:
        self.client = boto3.client(
            "s3",
            aws_access_key_id="factory-default",
            aws_secret_access_key="factory-default",
            region_name="us-east-1",
        )
        self.stubber = Stubber(self.client)
        self.stubber.activate()
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, **credentials: Any) -> Any:
        self.calls.append(credentials)
        return self.client

    def expect_put(self, key: str, body: str, content_type: str) -> None:
        self.stubber.add_response(
            "put_object",
            {},
            {
                "Bucket": "results",
                "Key": key,
                "Body": body.encode(),
                "ContentType": content_type,
            },
        )

    def expect_error(self, code: str, status: int = 400) -> None:
        self.stubber.add_client_error(
            "put_object", service_error_code=code, http_status_code=status
        )


def _local(stamp_seconds: int = 5) -> str:
    moment = datetime(2026, 1, 2, 3, 4, stamp_seconds, 678, tzinfo=timezone.utc)
    return moment.astimezone().strftime("%Y_%m_%d_%H_%M_%S_%f")


def _block(s3: Any, **resources: Any) -> S3SinkBlock:
    resources.setdefault("wall_clock", Clock())
    resources.setdefault("retry_sleep", lambda seconds: None)
    block = S3SinkBlock(s3_client_factory=s3, **resources)

    return block


def _upload(block: S3SinkBlock, content: str, **params: Any) -> Dict[str, Any]:
    arguments: Dict[str, Any] = {
        "file_type": "csv",
        "output_mode": "append_log",
        "bucket_name": "results",
        "s3_prefix": "logs/",
        "file_name_prefix": "run",
        "max_entries_per_file": 1024,
        "aws_access_key_id": KEY_ID,
        "aws_secret_access_key": SECRET_KEY,
        "aws_region": None,
    }
    arguments.update(params)
    result = block.run(content=content, **arguments)

    return result


def _saved(key: str) -> Dict[str, Any]:
    return {
        "error_status": False,
        "message": f"Data saved successfully to s3://results/{key}",
    }


def test_separate_files_uploads_one_object_per_call_with_v1_key() -> None:
    s3 = StubbedS3()
    first, second = f"logs/run_{_local(5)}.txt", f"logs/run_{_local(6)}.txt"
    s3.expect_put(first, "hello", "text/plain")
    s3.expect_put(second, "world", "text/plain")
    block = _block(s3)

    results = [
        _upload(block, text, output_mode="separate_files", file_type="txt")
        for text in ("hello", "world")
    ]

    assert results == [_saved(first), _saved(second)]
    s3.stubber.assert_no_pending_responses()
    assert s3.calls[0] == {
        "aws_access_key_id": KEY_ID,
        "aws_secret_access_key": SECRET_KEY,
        "aws_region": None,
    }


def test_empty_prefix_puts_the_object_at_the_bucket_root() -> None:
    s3 = StubbedS3()
    s3.expect_put(f"run_{_local()}.json", "{}", "application/json")

    result = _upload(
        _block(s3), "{}", output_mode="separate_files", file_type="json", s3_prefix=""
    )

    assert result == _saved(f"run_{_local()}.json")


def test_append_log_reuploads_the_whole_buffer_and_drops_repeated_csv_header() -> None:
    s3 = StubbedS3()
    key = f"logs/run_{_local()}.csv"
    s3.expect_put(key, "name,count\na,1\n", "text/csv")
    s3.expect_put(key, "name,count\na,1\nb,2\n", "text/csv")
    block = _block(s3)

    results = [_upload(block, row) for row in (CSV_ROW_1, CSV_ROW_2)]

    assert results == [_saved(key), _saved(key)]
    s3.stubber.assert_no_pending_responses()


def test_append_log_starts_a_new_key_after_max_entries() -> None:
    s3 = StubbedS3()
    first, second = f"logs/run_{_local(5)}.csv", f"logs/run_{_local(6)}.csv"
    s3.expect_put(first, "name,count\na,1\n", "text/csv")
    s3.expect_put(second, "name,count\nb,2\n", "text/csv")
    block = _block(s3)

    results = [
        _upload(block, row, max_entries_per_file=1) for row in (CSV_ROW_1, CSV_ROW_2)
    ]

    assert results == [_saved(first), _saved(second)]


def test_append_log_json_becomes_jsonl_lines() -> None:
    s3 = StubbedS3()
    key = f"logs/run_{_local()}.jsonl"
    s3.expect_put(key, '{"a": 1}\n', "application/x-ndjson")
    s3.expect_put(key, '{"a": 1}\n{"b": [1, 2]}\n', "application/x-ndjson")
    block = _block(s3)

    results = [
        _upload(block, text, file_type="json")
        for text in ('{\n  "a": 1\n}', '{"b": [1,\n 2]}')
    ]

    assert results == [_saved(key), _saved(key)]


def test_invalid_json_is_reported_without_upload() -> None:
    s3 = StubbedS3()

    result = _upload(_block(s3), "{not json", file_type="json")

    assert result == {"error_status": True, "message": "Invalid JSON content"}
    s3.stubber.assert_no_pending_responses()


def test_changed_bucket_starts_a_new_key() -> None:
    # V1 kept the first key, even when the bucket changed.
    s3 = StubbedS3()
    first, second = f"logs/run_{_local(5)}.csv", f"logs/run_{_local(6)}.csv"
    s3.expect_put(first, "name,count\na,1\n", "text/csv")
    s3.stubber.add_response(
        "put_object",
        {},
        {
            "Bucket": "other",
            "Key": second,
            "Body": b"name,count\nb,2\n",
            "ContentType": "text/csv",
        },
    )
    block = _block(s3)

    _upload(block, CSV_ROW_1)
    result = _upload(block, CSV_ROW_2, bucket_name="other")

    assert result == {
        "error_status": False,
        "message": f"Data saved successfully to s3://other/{second}",
    }


def test_throttling_is_retried_with_doubling_delays() -> None:
    s3 = StubbedS3()
    key = f"logs/run_{_local()}.txt"
    s3.expect_error("SlowDown", 503)
    s3.expect_error("InternalError", 500)
    s3.expect_put(key, "x", "text/plain")
    delays: List[float] = []
    block = _block(s3, retry_sleep=delays.append)

    result = _upload(block, "x", output_mode="separate_files", file_type="txt")

    assert result == _saved(key)
    assert delays == [1.0, 2.0]


def test_retries_stop_after_three_and_report_the_last_error() -> None:
    s3 = StubbedS3()
    for _ in range(4):
        s3.expect_error("SlowDown", 503)
    delays: List[float] = []

    result = _upload(
        _block(s3, retry_sleep=delays.append),
        "x",
        output_mode="separate_files",
        file_type="txt",
    )

    assert result["error_status"] is True
    assert "SlowDown" in result["message"]
    assert delays == [1.0, 2.0, 4.0]


# Independent expected codes: V1 core_steps/sinks/s3/v1.py:45-52 in r9,
# not the V2 retry-classification constant.
@pytest.mark.parametrize(
    "code",
    [
        "InvalidAccessKeyId",
        "AccessDenied",
        "NoSuchBucket",
        "InvalidBucketName",
        "SignatureDoesNotMatch",
        "AuthFailure",
    ],
)
def test_auth_and_bucket_errors_are_not_retried(code: str) -> None:
    s3 = StubbedS3()
    s3.expect_error(code, 403)
    delays: List[float] = []

    result = _upload(_block(s3, retry_sleep=delays.append), CSV_ROW_1)

    assert result["error_status"] is True
    assert code in result["message"]
    s3.stubber.assert_no_pending_responses()
    assert delays == []


class _ScriptedS3Client:
    """Raise the supplied errors on successive uploads, then succeed."""

    def __init__(self, *errors: Exception) -> None:
        self.errors = list(errors)
        self.attempts = 0

    def put_object(self, **request: Any) -> None:
        self.attempts += 1
        if self.errors:
            raise self.errors.pop(0)


@pytest.mark.parametrize("failures", [3, 4])
def test_connection_errors_retry_three_times_with_one_two_four_second_delays(
    failures: int,
) -> None:
    error = EndpointConnectionError(endpoint_url="https://s3.test")
    client = _ScriptedS3Client(*([error] * failures))
    delays: List[float] = []
    block = _block(lambda **credentials: client, retry_sleep=delays.append)

    result = _upload(block, "x", output_mode="separate_files", file_type="txt")

    assert client.attempts == 4
    assert delays == [1.0, 2.0, 4.0]
    if failures == 3:
        assert result == _saved(f"logs/run_{_local()}.txt")
    else:
        assert result["error_status"] is True
        assert result["message"] == str(error)


def test_unexpected_errors_are_not_retried() -> None:
    client = _ScriptedS3Client(RuntimeError("unexpected upload failure"))
    delays: List[float] = []
    block = _block(lambda **credentials: client, retry_sleep=delays.append)

    result = _upload(block, "x", output_mode="separate_files", file_type="txt")

    assert result == {"error_status": True, "message": "unexpected upload failure"}
    assert client.attempts == 1
    assert delays == []


class LeakyClient:
    """Fails with the credentials in its error text, as a library might."""

    def put_object(self, **request: Any) -> None:
        raise RuntimeError(f"denied for {KEY_ID}:{SECRET_KEY}")


def test_failure_messages_are_scrubbed_of_credentials() -> None:
    block = _block(lambda **credentials: LeakyClient())

    result = _upload(block, CSV_ROW_1)

    assert result == {
        "error_status": True,
        "message": (
            "denied for <secret aws_access_key_id>:<secret aws_secret_access_key>"
        ),
    }


def test_client_factory_error_fails_the_step_without_its_text() -> None:
    def factory(**credentials: Any) -> Any:
        raise ValueError(f"bad key {SECRET_KEY}")

    with pytest.raises(RuntimeError) as raised:
        _upload(_block(factory), CSV_ROW_1)

    assert SECRET_KEY not in str(raised.value)
    assert raised.value.__cause__ is None


def test_secret_credentials_are_revealed_to_the_client_factory() -> None:
    s3 = StubbedS3()
    s3.expect_put(f"logs/run_{_local()}.csv", "name,count\na,1\n", "text/csv")

    _upload(
        _block(s3),
        CSV_ROW_1,
        aws_access_key_id=Secret(KEY_ID, name="AWS_KEY"),
        aws_secret_access_key=Secret(SECRET_KEY, name="AWS_SECRET"),
        aws_region="eu-west-1",
    )

    assert s3.calls == [
        {
            "aws_access_key_id": KEY_ID,
            "aws_secret_access_key": SECRET_KEY,
            "aws_region": "eu-west-1",
        }
    ]


def test_unavailable_secret_fails_the_step_before_any_client() -> None:
    s3 = StubbedS3()

    with pytest.raises(SecretUnavailableError):
        _upload(
            _block(s3),
            CSV_ROW_1,
            aws_secret_access_key=Secret.unavailable("AWS_SECRET"),
        )

    assert s3.calls == []


@pytest.mark.parametrize(
    "key_id, secret_key",
    [
        (None, SECRET_KEY),
        (KEY_ID, None),
        ("", SECRET_KEY),
        (KEY_ID, Secret("", name="s")),
    ],
)
def test_missing_credentials_fail_the_step(
    key_id: Optional[Any], secret_key: Optional[Any]
) -> None:
    s3 = StubbedS3()

    with pytest.raises(ValueError, match="ambient AWS credentials are not used"):
        _upload(
            _block(s3),
            CSV_ROW_1,
            aws_access_key_id=key_id,
            aws_secret_access_key=secret_key,
        )

    assert s3.calls == []
    assert "ambient" in MISSING_CREDENTIALS_MESSAGE


def test_disabled_sinks_upload_nothing_and_need_no_credentials() -> None:
    s3 = StubbedS3()

    result = _upload(
        _block(s3, disable_sinks=True),
        CSV_ROW_1,
        aws_access_key_id=None,
        aws_secret_access_key=None,
    )

    assert result == {
        "error_status": False,
        "message": "Sink was disabled by workflow execution policy",
    }
    assert s3.calls == []


def test_utf8_content_is_uploaded_as_utf8() -> None:
    s3 = StubbedS3()
    key = f"logs/run_{_local()}.txt"
    s3.expect_put(key, "zażółć\n", "text/plain")

    result = _upload(_block(s3), "zażółć", file_type="txt")

    assert result == _saved(key)
