import json
from unittest.mock import MagicMock, call, patch

import pytest
from botocore.exceptions import ClientError
from pydantic import ValidationError

from inference.core.workflows.core_steps.sinks.s3.v1 import (
    MAX_UPLOAD_RETRIES,
    NON_RETRYABLE_CLIENT_ERROR_CODES,
    BlockManifest,
    S3SinkBlockV1,
    deduct_csv_header,
    dump_json_inline,
    generate_s3_key,
    upload_content_to_s3,
)


# ---------------------------------------------------------------------------
# Manifest validation
# ---------------------------------------------------------------------------


def test_manifest_parsing_when_input_is_valid() -> None:
    raw_manifest = {
        "type": "roboflow_core/s3_sink@v1",
        "name": "s3_sink",
        "content": "$steps.csv_formatter.csv_content",
        "file_type": "csv",
        "output_mode": "separate_files",
        "bucket_name": "my-bucket",
        "s3_prefix": "logs/detections",
        "file_name_prefix": "run",
        "max_entries_per_file": 512,
    }

    result = BlockManifest.model_validate(raw_manifest)

    assert result.bucket_name == "my-bucket"
    assert result.s3_prefix == "logs/detections"
    assert result.file_name_prefix == "run"
    assert result.max_entries_per_file == 512
    assert result.aws_access_key_id is None
    assert result.aws_secret_access_key is None
    assert result.aws_region is None


def test_manifest_parsing_with_selector_credentials() -> None:
    raw_manifest = {
        "type": "roboflow_core/s3_sink@v1",
        "name": "s3_sink",
        "content": "$steps.formatter.output",
        "file_type": "json",
        "output_mode": "append_log",
        "bucket_name": "$inputs.bucket",
        "s3_prefix": "",
        "file_name_prefix": "workflow_output",
        "max_entries_per_file": "$inputs.max_entries",
        "aws_access_key_id": "$steps.secrets.aws_key_id",
        "aws_secret_access_key": "$steps.secrets.aws_secret",
        "aws_region": "us-east-1",
    }

    result = BlockManifest.model_validate(raw_manifest)

    assert result.aws_access_key_id == "$steps.secrets.aws_key_id"
    assert result.aws_secret_access_key == "$steps.secrets.aws_secret"
    assert result.aws_region == "us-east-1"
    assert result.max_entries_per_file == "$inputs.max_entries"


def test_manifest_parsing_rejects_max_entries_per_file_of_zero() -> None:
    raw_manifest = {
        "type": "roboflow_core/s3_sink@v1",
        "name": "s3_sink",
        "content": "$steps.formatter.output",
        "file_type": "csv",
        "output_mode": "separate_files",
        "bucket_name": "my-bucket",
        "max_entries_per_file": 0,
    }

    with pytest.raises(ValidationError):
        BlockManifest.model_validate(raw_manifest)


def test_manifest_parsing_rejects_negative_max_entries_per_file() -> None:
    raw_manifest = {
        "type": "roboflow_core/s3_sink@v1",
        "name": "s3_sink",
        "content": "$steps.formatter.output",
        "file_type": "csv",
        "output_mode": "separate_files",
        "bucket_name": "my-bucket",
        "max_entries_per_file": -5,
    }

    with pytest.raises(ValidationError):
        BlockManifest.model_validate(raw_manifest)


def test_manifest_parsing_rejects_invalid_file_type() -> None:
    raw_manifest = {
        "type": "roboflow_core/s3_sink@v1",
        "name": "s3_sink",
        "content": "$steps.formatter.output",
        "file_type": "xml",
        "output_mode": "separate_files",
        "bucket_name": "my-bucket",
    }

    with pytest.raises(ValidationError):
        BlockManifest.model_validate(raw_manifest)


def test_manifest_parsing_rejects_invalid_output_mode() -> None:
    raw_manifest = {
        "type": "roboflow_core/s3_sink@v1",
        "name": "s3_sink",
        "content": "$steps.formatter.output",
        "file_type": "csv",
        "output_mode": "overwrite",
        "bucket_name": "my-bucket",
    }

    with pytest.raises(ValidationError):
        BlockManifest.model_validate(raw_manifest)


# ---------------------------------------------------------------------------
# generate_s3_key
# ---------------------------------------------------------------------------


def test_generate_s3_key_with_prefix() -> None:
    key = generate_s3_key(s3_prefix="logs/detections", file_name_prefix="run", file_type="csv")

    assert key.startswith("logs/detections/run_")
    assert key.endswith(".csv")


def test_generate_s3_key_without_prefix() -> None:
    key = generate_s3_key(s3_prefix="", file_name_prefix="output", file_type="txt")

    assert key.startswith("output_")
    assert key.endswith(".txt")
    assert "/" not in key


def test_generate_s3_key_normalizes_trailing_slash_in_prefix() -> None:
    key_with_slash = generate_s3_key(s3_prefix="logs/", file_name_prefix="run", file_type="csv")
    key_without_slash = generate_s3_key(s3_prefix="logs", file_name_prefix="run", file_type="csv")

    assert key_with_slash.startswith("logs/run_")
    assert key_without_slash.startswith("logs/run_")


def test_generate_s3_key_produces_unique_keys() -> None:
    key1 = generate_s3_key(s3_prefix="", file_name_prefix="run", file_type="csv")
    key2 = generate_s3_key(s3_prefix="", file_name_prefix="run", file_type="csv")

    # Timestamps include microseconds so two sequential calls should differ
    assert key1 != key2


# ---------------------------------------------------------------------------
# upload_content_to_s3
# ---------------------------------------------------------------------------


def test_upload_content_to_s3_returns_success_on_ok() -> None:
    mock_s3 = MagicMock()

    result = upload_content_to_s3(
        s3_client=mock_s3,
        bucket_name="my-bucket",
        s3_key="logs/output.txt",
        content="hello world",
        content_type="text/plain",
    )

    assert result["error_status"] is False
    assert "my-bucket" in result["message"]
    assert "logs/output.txt" in result["message"]
    mock_s3.put_object.assert_called_once_with(
        Bucket="my-bucket",
        Key="logs/output.txt",
        Body=b"hello world",
        ContentType="text/plain",
    )


def test_upload_content_to_s3_returns_error_on_client_error() -> None:
    mock_s3 = MagicMock()
    mock_s3.put_object.side_effect = ClientError(
        error_response={"Error": {"Code": "InvalidAccessKeyId", "Message": "The key does not exist."}},
        operation_name="PutObject",
    )

    result = upload_content_to_s3(
        s3_client=mock_s3,
        bucket_name="my-bucket",
        s3_key="logs/output.txt",
        content="hello world",
        content_type="text/plain",
    )

    assert result["error_status"] is True
    assert "InvalidAccessKeyId" in result["message"]


def test_upload_content_to_s3_returns_error_on_access_denied() -> None:
    mock_s3 = MagicMock()
    mock_s3.put_object.side_effect = ClientError(
        error_response={"Error": {"Code": "AccessDenied", "Message": "Access denied."}},
        operation_name="PutObject",
    )

    result = upload_content_to_s3(
        s3_client=mock_s3,
        bucket_name="my-bucket",
        s3_key="logs/output.txt",
        content="hello world",
        content_type="text/plain",
    )

    assert result["error_status"] is True
    assert "AccessDenied" in result["message"]


def test_upload_content_to_s3_returns_error_on_unexpected_exception() -> None:
    mock_s3 = MagicMock()
    mock_s3.put_object.side_effect = RuntimeError("connection reset")

    result = upload_content_to_s3(
        s3_client=mock_s3,
        bucket_name="my-bucket",
        s3_key="logs/output.txt",
        content="hello world",
        content_type="text/plain",
    )

    assert result["error_status"] is True
    assert "connection reset" in result["message"]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def test_deduct_csv_header_removes_first_line() -> None:
    content = "col_a,col_b\n1,2\n3,4\n"
    result = deduct_csv_header(content)
    assert result == "1,2\n3,4\n"


def test_dump_json_inline_serialises_to_single_line() -> None:
    content = json.dumps({"a": 1, "b": [1, 2, 3]}, indent=2)
    result = dump_json_inline(content)
    assert "\n" not in result
    assert json.loads(result) == {"a": 1, "b": [1, 2, 3]}


def test_dump_json_inline_raises_on_invalid_json() -> None:
    with pytest.raises(Exception):
        dump_json_inline("not valid json")


# ---------------------------------------------------------------------------
# S3SinkBlockV1 — separate_files mode
# ---------------------------------------------------------------------------


def _make_block_with_mock_s3():
    """Return (block, mock_s3_client) with create_s3_client patched."""
    mock_s3 = MagicMock()
    block = S3SinkBlockV1()
    return block, mock_s3


def _run_block(block, mock_s3, **kwargs):
    defaults = dict(
        content="hello",
        file_type="txt",
        output_mode="separate_files",
        bucket_name="my-bucket",
        s3_prefix="",
        file_name_prefix="output",
        max_entries_per_file=1024,
    )
    defaults.update(kwargs)
    with patch(
        "inference.core.workflows.core_steps.sinks.s3.v1.create_s3_client",
        return_value=mock_s3,
    ):
        return block.run(**defaults)


def test_separate_files_txt_uploads_unique_objects_per_call() -> None:
    block, mock_s3 = _make_block_with_mock_s3()

    result1 = _run_block(block, mock_s3, content="content-1", file_type="txt")
    result2 = _run_block(block, mock_s3, content="content-2", file_type="txt")

    assert result1["error_status"] is False
    assert result2["error_status"] is False
    assert mock_s3.put_object.call_count == 2

    call_kwargs_1 = mock_s3.put_object.call_args_list[0].kwargs
    call_kwargs_2 = mock_s3.put_object.call_args_list[1].kwargs
    # Each upload must use a distinct S3 key
    assert call_kwargs_1["Key"] != call_kwargs_2["Key"]
    assert call_kwargs_1["Body"] == b"content-1"
    assert call_kwargs_2["Body"] == b"content-2"


def test_separate_files_uses_correct_content_type_for_csv() -> None:
    block, mock_s3 = _make_block_with_mock_s3()

    _run_block(block, mock_s3, content="a,b\n1,2\n", file_type="csv")

    call_kwargs = mock_s3.put_object.call_args.kwargs
    assert call_kwargs["ContentType"] == "text/csv"
    assert call_kwargs["Key"].endswith(".csv")


def test_separate_files_uses_correct_content_type_for_json() -> None:
    block, mock_s3 = _make_block_with_mock_s3()

    _run_block(block, mock_s3, content=json.dumps({"x": 1}), file_type="json")

    call_kwargs = mock_s3.put_object.call_args.kwargs
    assert call_kwargs["ContentType"] == "application/json"
    assert call_kwargs["Key"].endswith(".json")


def test_separate_files_uses_correct_content_type_for_txt() -> None:
    block, mock_s3 = _make_block_with_mock_s3()

    _run_block(block, mock_s3, content="hello", file_type="txt")

    call_kwargs = mock_s3.put_object.call_args.kwargs
    assert call_kwargs["ContentType"] == "text/plain"
    assert call_kwargs["Key"].endswith(".txt")


def test_separate_files_includes_s3_prefix_in_key() -> None:
    block, mock_s3 = _make_block_with_mock_s3()

    _run_block(block, mock_s3, s3_prefix="logs/detections", file_name_prefix="run")

    call_kwargs = mock_s3.put_object.call_args.kwargs
    assert call_kwargs["Key"].startswith("logs/detections/run_")


def test_separate_files_propagates_s3_error() -> None:
    block, mock_s3 = _make_block_with_mock_s3()
    mock_s3.put_object.side_effect = ClientError(
        error_response={"Error": {"Code": "InvalidAccessKeyId", "Message": "Bad key."}},
        operation_name="PutObject",
    )

    result = _run_block(block, mock_s3, content="data")

    assert result["error_status"] is True
    assert "InvalidAccessKeyId" in result["message"]


# ---------------------------------------------------------------------------
# S3SinkBlockV1 — append_log mode
# ---------------------------------------------------------------------------


def test_append_log_txt_accumulates_entries_in_single_object() -> None:
    block, mock_s3 = _make_block_with_mock_s3()

    for i in range(3):
        result = _run_block(
            block, mock_s3,
            content=f"content-{i}",
            file_type="txt",
            output_mode="append_log",
            max_entries_per_file=10,
        )
        assert result["error_status"] is False

    assert mock_s3.put_object.call_count == 3
    # All three calls must use the same key (same S3 object, growing content)
    keys = [c.kwargs["Key"] for c in mock_s3.put_object.call_args_list]
    assert len(set(keys)) == 1, "All entries before rotation should share one key"

    # Final upload has the full accumulated content
    final_body = mock_s3.put_object.call_args_list[-1].kwargs["Body"]
    assert final_body == b"content-0\ncontent-1\ncontent-2\n"


def test_append_log_txt_rotates_to_new_key_after_limit() -> None:
    block, mock_s3 = _make_block_with_mock_s3()

    for i in range(5):
        _run_block(
            block, mock_s3,
            content=f"content-{i}",
            file_type="txt",
            output_mode="append_log",
            max_entries_per_file=3,
        )

    assert mock_s3.put_object.call_count == 5
    keys = [c.kwargs["Key"] for c in mock_s3.put_object.call_args_list]
    # Entries 0-2 share the first key; entries 3-4 share the second key
    assert keys[0] == keys[1] == keys[2]
    assert keys[3] == keys[4]
    assert keys[0] != keys[3], "Rotated object must use a new key"

    # Final content of the first object
    body_at_rotation = mock_s3.put_object.call_args_list[2].kwargs["Body"]
    assert body_at_rotation == b"content-0\ncontent-1\ncontent-2\n"

    # Final content of the second object
    final_body = mock_s3.put_object.call_args_list[4].kwargs["Body"]
    assert final_body == b"content-3\ncontent-4\n"


def test_append_log_json_converts_to_jsonl() -> None:
    block, mock_s3 = _make_block_with_mock_s3()

    for i in range(2):
        result = _run_block(
            block, mock_s3,
            content=json.dumps({"data": i}),
            file_type="json",
            output_mode="append_log",
            max_entries_per_file=10,
        )
        assert result["error_status"] is False

    call_kwargs = mock_s3.put_object.call_args.kwargs
    assert call_kwargs["Key"].endswith(".jsonl")
    assert call_kwargs["ContentType"] == "application/x-ndjson"

    lines = call_kwargs["Body"].decode().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"data": 0}
    assert json.loads(lines[1]) == {"data": 1}


def test_append_log_csv_strips_header_on_subsequent_entries() -> None:
    block, mock_s3 = _make_block_with_mock_s3()

    _run_block(
        block, mock_s3,
        content="col_a,col_b\n0,a\n",
        file_type="csv",
        output_mode="append_log",
        max_entries_per_file=10,
    )
    _run_block(
        block, mock_s3,
        content="col_a,col_b\n1,b\n",
        file_type="csv",
        output_mode="append_log",
        max_entries_per_file=10,
    )

    final_body = mock_s3.put_object.call_args.kwargs["Body"].decode()
    # Header appears only once (from the first write)
    assert final_body.count("col_a,col_b") == 1
    assert "0,a" in final_body
    assert "1,b" in final_body


def test_append_log_invalid_json_returns_error_without_uploading() -> None:
    block, mock_s3 = _make_block_with_mock_s3()

    result = _run_block(
        block, mock_s3,
        content="not valid json",
        file_type="json",
        output_mode="append_log",
        max_entries_per_file=10,
    )

    assert result["error_status"] is True
    assert "Invalid JSON" in result["message"]
    mock_s3.put_object.assert_not_called()


def test_append_log_surfaces_credential_error_on_first_call() -> None:
    """Credential errors must be reported immediately, not deferred until rotation."""
    block, mock_s3 = _make_block_with_mock_s3()
    mock_s3.put_object.side_effect = ClientError(
        error_response={"Error": {"Code": "InvalidAccessKeyId", "Message": "Bad key."}},
        operation_name="PutObject",
    )

    result = _run_block(
        block, mock_s3,
        content="content-0",
        file_type="txt",
        output_mode="append_log",
        max_entries_per_file=1024,
    )

    assert result["error_status"] is True
    assert "InvalidAccessKeyId" in result["message"]


def test_append_log_surfaces_access_denied_error_on_first_call() -> None:
    block, mock_s3 = _make_block_with_mock_s3()
    mock_s3.put_object.side_effect = ClientError(
        error_response={"Error": {"Code": "AccessDenied", "Message": "Access denied."}},
        operation_name="PutObject",
    )

    result = _run_block(
        block, mock_s3,
        content="content-0",
        file_type="txt",
        output_mode="append_log",
        max_entries_per_file=1024,
    )

    assert result["error_status"] is True
    assert "AccessDenied" in result["message"]


def test_append_log_uses_correct_bucket_name() -> None:
    block, mock_s3 = _make_block_with_mock_s3()

    _run_block(
        block, mock_s3,
        content="data",
        file_type="txt",
        output_mode="append_log",
        bucket_name="production-bucket",
        max_entries_per_file=10,
    )

    call_kwargs = mock_s3.put_object.call_args.kwargs
    assert call_kwargs["Bucket"] == "production-bucket"


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------


def _make_retryable_client_error(code: str) -> ClientError:
    return ClientError(
        error_response={"Error": {"Code": code, "Message": "error"}},
        operation_name="PutObject",
    )


@patch("inference.core.workflows.core_steps.sinks.s3.v1.time.sleep")
def test_upload_retries_on_retryable_client_error_then_succeeds(mock_sleep) -> None:
    mock_s3 = MagicMock()
    # Fail twice with a retryable error, succeed on the third attempt
    mock_s3.put_object.side_effect = [
        _make_retryable_client_error("SlowDown"),
        _make_retryable_client_error("InternalError"),
        None,  # success
    ]

    result = upload_content_to_s3(
        s3_client=mock_s3,
        bucket_name="my-bucket",
        s3_key="output.txt",
        content="data",
        content_type="text/plain",
    )

    assert result["error_status"] is False
    assert mock_s3.put_object.call_count == 3
    assert mock_sleep.call_count == 2


@patch("inference.core.workflows.core_steps.sinks.s3.v1.time.sleep")
def test_upload_retries_on_botocore_error_then_succeeds(mock_sleep) -> None:
    from botocore.exceptions import EndpointConnectionError

    mock_s3 = MagicMock()
    mock_s3.put_object.side_effect = [
        EndpointConnectionError(endpoint_url="https://s3.amazonaws.com"),
        None,  # success on second attempt
    ]

    result = upload_content_to_s3(
        s3_client=mock_s3,
        bucket_name="my-bucket",
        s3_key="output.txt",
        content="data",
        content_type="text/plain",
    )

    assert result["error_status"] is False
    assert mock_s3.put_object.call_count == 2
    assert mock_sleep.call_count == 1


@patch("inference.core.workflows.core_steps.sinks.s3.v1.time.sleep")
def test_upload_returns_error_after_all_retries_exhausted(mock_sleep) -> None:
    mock_s3 = MagicMock()
    mock_s3.put_object.side_effect = _make_retryable_client_error("SlowDown")

    result = upload_content_to_s3(
        s3_client=mock_s3,
        bucket_name="my-bucket",
        s3_key="output.txt",
        content="data",
        content_type="text/plain",
    )

    assert result["error_status"] is True
    assert mock_s3.put_object.call_count == 1 + MAX_UPLOAD_RETRIES
    assert mock_sleep.call_count == MAX_UPLOAD_RETRIES


@patch("inference.core.workflows.core_steps.sinks.s3.v1.time.sleep")
def test_upload_does_not_retry_non_retryable_errors(mock_sleep) -> None:
    for error_code in NON_RETRYABLE_CLIENT_ERROR_CODES:
        mock_s3 = MagicMock()
        mock_s3.put_object.side_effect = _make_retryable_client_error(error_code)

        result = upload_content_to_s3(
            s3_client=mock_s3,
            bucket_name="my-bucket",
            s3_key="output.txt",
            content="data",
            content_type="text/plain",
        )

        assert result["error_status"] is True, f"Expected error for {error_code}"
        assert mock_s3.put_object.call_count == 1, f"Expected no retry for {error_code}"
        mock_sleep.assert_not_called()


@patch("inference.core.workflows.core_steps.sinks.s3.v1.time.sleep")
def test_upload_retry_uses_exponential_backoff(mock_sleep) -> None:
    mock_s3 = MagicMock()
    mock_s3.put_object.side_effect = _make_retryable_client_error("SlowDown")

    upload_content_to_s3(
        s3_client=mock_s3,
        bucket_name="my-bucket",
        s3_key="output.txt",
        content="data",
        content_type="text/plain",
    )

    sleep_delays = [c.args[0] for c in mock_sleep.call_args_list]
    # Each delay must be strictly larger than the previous (exponential growth)
    assert all(
        sleep_delays[i] < sleep_delays[i + 1] for i in range(len(sleep_delays) - 1)
    ), f"Expected exponential backoff, got delays: {sleep_delays}"
