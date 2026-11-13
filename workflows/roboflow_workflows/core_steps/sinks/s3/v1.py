import atexit
import json
import logging
from datetime import datetime
from threading import Lock
from time import monotonic, sleep
from typing import Any, List, Literal, Optional, Type, Union
from weakref import WeakSet

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from pydantic import ConfigDict, Field, field_validator
from roboflow_workflows.core_steps.common.entities import StepExecutionMode
from roboflow_workflows.core_steps.common.workload_presets import (
    S3_APPEND_BUFFER_RESTRICTION,
)
from roboflow_workflows.core_steps.sinks.noop import disabled_sink_response
from roboflow_workflows.execution_engine.entities.base import OutputDefinition
from roboflow_workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    INTEGER_KIND,
    SECRET_KIND,
    STRING_KIND,
    Selector,
)
from roboflow_workflows.execution_engine.entities.workload import (
    Discovery,
    RuntimeRestriction,
    WorkOperation,
)
from roboflow_workflows.prototypes.block import (
    BlockResult,
    DependentResource,
    Runtime,
    Severity,
    WorkflowBlock,
    WorkflowBlockManifest,
    actual_restrictions_of,
)
from roboflow_workflows.utils.lru_cache import LRUCache

CONTENT_TYPES = {
    "csv": "text/csv",
    "json": "application/json",
    "jsonl": "application/x-ndjson",
    "txt": "text/plain",
}

# Errors that indicate a configuration or auth problem — retrying will not help.
NON_RETRYABLE_CLIENT_ERROR_CODES = {
    "InvalidAccessKeyId",
    "AccessDenied",
    "NoSuchBucket",
    "InvalidBucketName",
    "SignatureDoesNotMatch",
    "AuthFailure",
}

MAX_UPLOAD_RETRIES = 3  # additional attempts after the initial try
UPLOAD_RETRY_INITIAL_DELAY = 1.0  # seconds; doubles on each subsequent retry
DEFAULT_UPLOAD_INTERVAL_SECONDS = 30
MAX_CACHED_S3_CLIENTS = 32

LONG_DESCRIPTION = """
Save workflow data directly to an AWS S3 bucket, supporting CSV, JSON, and text file formats with configurable output modes for aggregating multiple entries into single objects or saving each entry as a separate S3 object.

## How This Block Works

This block uploads string content from workflow steps to S3 objects. The block:

1. Takes string content (from formatters, predictions, or other string-producing blocks) and S3 configuration as input
2. Connects to AWS S3 using the credentials supplied by the workflow (the server's ambient AWS credentials are never used)
3. Selects the appropriate upload strategy based on `output_mode`:
   - **Separate Files Mode**: Creates a new S3 object for each input, generating unique keys with timestamps
   - **Append Log Mode**: Buffers content in memory under a single S3 key and uploads the accumulated object on its first entry, at most every `upload_interval_seconds` after that, when `max_entries_per_file` is reached, and when the block is torn down
4. For **separate files mode**: Generates a unique S3 key from the prefix, file name prefix, file type, and a timestamp, then uploads the content directly
5. For **append log mode**:
   - Buffers content entries in memory under a single S3 key
   - Applies format-specific handling for appending:
     - **CSV**: Removes the header row from subsequent appends (CSV content must include headers on first write)
     - **JSON**: Converts to JSONL (JSON Lines) format, parsing and re-serializing each JSON document to fit on a single line
     - **TXT**: Appends content directly with newlines
   - Uploads the buffer on the first entry of each object, so credential and permission errors surface on the first run
   - Uploads it again once `upload_interval_seconds` have passed since the last successful upload; entries that arrive sooner are only buffered
   - Tracks entry count and uploads the full buffer as a complete S3 object when `max_entries_per_file` is reached, then starts a fresh buffer with a new key
   - Uploads any remaining buffered data when the block is torn down
6. Returns error status and messages indicating save success or failure

The block supports two storage strategies: separate files mode creates individual timestamped S3 objects per input (useful for organizing outputs by execution), while append log mode accumulates entries in memory and rewrites the accumulated S3 object periodically and on rotation (useful for time-series logging with controlled upload frequency). S3 key names include timestamps (format: `YYYY_MM_DD_HH_MM_SS_microseconds`) for unique keys and chronological ordering.

## AWS Credentials

Credentials can be supplied in two ways:
1. **Workflow inputs** — declare `aws_access_key_id` and `aws_secret_access_key` as workflow inputs of kind `parameter` and connect them to the corresponding fields. This keeps credentials out of the workflow definition and allows them to be supplied at runtime.
2. **Secrets provider block** — connect the credential fields to the output of an `Environment Secrets Store` block, which reads values from server-side environment variables without embedding them in the workflow. Note: this is only available on self-hosted `inference` servers and cannot be used on the Roboflow hosted platform.

## S3 Key Structure

The final S3 key is composed of:
```
{s3_prefix}/{file_name_prefix}_{timestamp}.{extension}
```
For example, with `s3_prefix="logs/detections"`, `file_name_prefix="run"`, and `file_type="csv"`:
```
logs/detections/run_2024_10_18_14_09_57_622297.csv
```
If `s3_prefix` is empty, the key starts directly with the file name.

## Note on Append Log Mode

S3 objects cannot be appended to, so each upload in append log mode rewrites the whole object with every entry buffered so far. To keep that cost bounded, entries are buffered in memory and the object is uploaded only:
- On the first entry of each object, which also validates credentials and permissions right away
- On the first run after `upload_interval_seconds` (default 30) have passed since the last successful upload
- When the `max_entries_per_file` limit is reached (object rotation)
- When the block is torn down: when the workflow releases it (for example at the end of an HTTP request, or when a stream manager pipeline is terminated) or when the Python process exits normally

An entry that does not trigger an upload is reported as buffered and becomes visible in S3 with the next upload. Buffered entries are lost if the process crashes or is killed before that upload. A failed upload is retried on the next run, so errors keep surfacing until an upload succeeds. Set `upload_interval_seconds` to 0 to upload on every run, or use `separate_files` mode if every entry must be in S3 as soon as the step completes.

## Common Use Cases

- **Cloud Data Logging**: Upload detection results, metrics, or workflow outputs directly to S3 for durable cloud storage and downstream processing
- **Data Pipeline Integration**: Export formatted CSV or JSONL files to S3 for consumption by data pipelines, analytics tools, or ML training jobs
- **Batch Result Archival**: Store individual inference results as separate S3 objects organized by timestamp and prefix
- **Time-Series Collection**: Aggregate workflow outputs into batched JSONL or CSV files in S3 for cost-efficient log storage
- **Cross-Service Integration**: Write data to S3 to trigger Lambda functions, feed SQS queues, or integrate with other AWS services
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "S3 Sink",
            "version": "v1",
            "short_description": "Upload data to an AWS S3 bucket.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "sink",
            "ui_manifest": {
                "section": "data_storage",
                "icon": "fal fa-cloud-upload",
                "blockPriority": 4,
            },
        }
    )
    type: Literal["roboflow_core/s3_sink@v1"]
    content: Selector(kind=[STRING_KIND]) = Field(
        description="String content to upload to S3. This should be formatted data from other workflow blocks (e.g., CSV content from CSV Formatter, JSON strings, or plain text). The content format should match the specified file_type. For CSV files in append_log mode, content must include header rows on the first write.",
        examples=["$steps.csv_formatter.csv_content"],
    )
    file_type: Literal["csv", "json", "txt"] = Field(
        default="csv",
        description="Type of file to create: 'csv' (CSV format), 'json' (JSON format, or JSONL in append_log mode), or 'txt' (plain text). In append_log mode, JSON files are stored as .jsonl (JSON Lines) format with one JSON object per line.",
        examples=["csv"],
    )
    output_mode: Literal["append_log", "separate_files"] = Field(
        description="Upload strategy: 'append_log' buffers multiple entries in a single S3 object, uploaded periodically (see upload_interval_seconds) and when the entry limit is reached (useful for batched logging), or 'separate_files' uploads each input as a new S3 object with a unique timestamp-based key (useful for per-execution outputs).",
        examples=["append_log"],
        json_schema_extra={
            "values_metadata": {
                "append_log": {
                    "name": "Append Log",
                    "description": "Buffers multiple entries in a single S3 object, uploaded periodically and on rotation",
                },
                "separate_files": {
                    "name": "Separate File",
                    "description": "Uploads a new S3 object for each input",
                },
            }
        },
    )
    bucket_name: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="Name of the target S3 bucket. Can be a static string or a selector resolving to a string at runtime.",
        examples=["my-inference-results"],
    )
    s3_prefix: Union[Selector(kind=[STRING_KIND]), str] = Field(
        default="",
        description="S3 key prefix (folder path) where objects will be stored. Trailing slashes are normalized automatically. Combined with file_name_prefix and a timestamp to form the full object key. Example: 'logs/detections' produces keys like 'logs/detections/workflow_output_2024_10_18_14_09_57_622297.csv'.",
        examples=["logs/detections"],
        json_schema_extra={
            "always_visible": True,
        },
    )
    file_name_prefix: Union[Selector(kind=[STRING_KIND]), str] = Field(
        default="workflow_output",
        description="Prefix used to generate S3 object names. Combined with a timestamp (format: YYYY_MM_DD_HH_MM_SS_microseconds) and file extension to create unique keys like 'workflow_output_2024_10_18_14_09_57_622297.csv'.",
        examples=["my_output"],
        json_schema_extra={
            "always_visible": True,
        },
    )
    max_entries_per_file: Union[int, Selector(kind=[STRING_KIND])] = Field(
        default=1024,
        description="Maximum number of buffered entries before uploading to S3 and starting a new object in append_log mode. When this limit is reached, the accumulated buffer is uploaded as a complete S3 object and a new buffer starts with a fresh key. Only applies when output_mode is 'append_log'. Must be at least 1.",
        examples=[1024],
        json_schema_extra={
            "relevant_for": {
                "output_mode": {
                    "values": ["append_log"],
                    "required": True,
                },
            }
        },
    )
    upload_interval_seconds: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        default=DEFAULT_UPLOAD_INTERVAL_SECONDS,
        description="Minimum number of seconds between uploads of the current S3 object in append_log mode. The first entry of each object, the entry that reaches max_entries_per_file, and block teardown always upload; any other entry uploads only once this many seconds have passed since the last successful upload, and stays buffered in memory until then. Each upload rewrites the whole object, so a longer interval uploads fewer bytes but delays when entries become visible in S3 and loses more entries if the process crashes. Set to 0 to upload on every run. Only applies when output_mode is 'append_log'.",
        examples=[30, "$inputs.upload_interval_seconds"],
        json_schema_extra={
            "relevant_for": {
                "output_mode": {
                    "values": ["append_log"],
                    "required": False,
                },
            }
        },
    )
    aws_access_key_id: Optional[
        Union[Selector(kind=[SECRET_KIND, STRING_KIND]), str]
    ] = Field(
        default=None,
        description="AWS access key ID for authentication. Required, and must be set together with `aws_secret_access_key`. Recommended: connect this to an Environment Secrets Store block rather than hardcoding.",
        examples=["$steps.secrets.aws_access_key_id"],
    )
    aws_secret_access_key: Optional[
        Union[Selector(kind=[SECRET_KIND, STRING_KIND]), str]
    ] = Field(
        default=None,
        description="AWS secret access key for authentication. Required, and must be set together with `aws_access_key_id`. Recommended: connect this to an Environment Secrets Store block rather than hardcoding.",
        examples=["$steps.secrets.aws_secret_access_key"],
    )
    aws_region: Optional[Union[Selector(kind=[STRING_KIND]), str]] = Field(
        default=None,
        description="AWS region where the bucket is located (e.g., 'us-east-1'). If not provided, boto3's default region is used (AWS_DEFAULT_REGION environment variable or ~/.aws/config).",
        examples=["us-east-1"],
    )

    @field_validator("max_entries_per_file")
    @classmethod
    def ensure_max_entries_per_file_is_correct(cls, value: Any) -> Any:
        if isinstance(value, int) and value < 1:
            raise ValueError("`max_entries_per_file` cannot be lower than 1.")
        return value

    @field_validator("upload_interval_seconds")
    @classmethod
    def ensure_upload_interval_seconds_is_correct(cls, value: Any) -> Any:
        if isinstance(value, int) and value < 0:
            raise ValueError("`upload_interval_seconds` cannot be lower than 0.")
        return value

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="error_status", kind=[BOOLEAN_KIND]),
            OutputDefinition(name="message", kind=[STRING_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"

    @classmethod
    def get_restrictions(cls) -> List[RuntimeRestriction]:
        """Return the legacy editor restrictions of this block.

        Returns:
            Restrictions for the workflow editor. Each shares its code with
            the same caveat in ``get_actual_restrictions()``.
        """
        restriction = RuntimeRestriction(
            code="s3_append_buffer_resets_on_stateless_http",
            severity=Severity.SOFT,
            note=(
                "Append-log mode buffers entries in process memory before "
                "uploading the accumulated object to S3. With remote step "
                "execution on stateless or multi-replica HTTP runtimes, "
                "successive requests may be served by different worker "
                "processes, so append-log objects can reset or split across "
                "workers. Use separate_files mode, or local step execution in "
                "an InferencePipeline when each entry must be captured in a "
                "single ordered log."
            ),
            applies_to_runtimes=[
                Runtime.HOSTED_SERVERLESS,
                Runtime.DEDICATED_DEPLOYMENT,
            ],
            applies_to_step_execution_modes=[StepExecutionMode.REMOTE],
        )
        return [restriction]

    def discover_work_operations(self) -> List[WorkOperation]:
        return [WorkOperation.STORAGE_WRITE, WorkOperation.EXTERNAL_REQUEST]

    def get_actual_restrictions(
        self, *, ignore_environment_restrictions: bool = False
    ) -> Discovery[RuntimeRestriction]:
        # The append-log buffer only exists in append_log mode; separate_files
        # uploads each entry on its own and carries no such caveat.
        # `output_mode` is a plain Literal (no Selector in its union), so it is
        # always known here - `test_output_mode_cannot_hold_a_selector` pins
        # that, and turns a future widening of the field into a failure rather
        # than a silent wrong answer.
        declared: List[RuntimeRestriction] = []
        if self.output_mode == "append_log":
            declared = [S3_APPEND_BUFFER_RESTRICTION]
        return actual_restrictions_of(
            declared=declared,
            node_id=f"$steps.{getattr(self, 'name', '')}",
            ignore_environment_restrictions=ignore_environment_restrictions,
        )

    def discover_dependent_resources(self) -> List[DependentResource]:
        return []


class S3SinkBlockV1(WorkflowBlock):

    def __init__(self, disable_sinks: bool = False):
        self._buffer: List[str] = []
        self._entries_in_buffer: int = 0
        self._entries_uploaded: int = 0
        self._current_key: Optional[str] = None
        self._last_upload_time: Optional[float] = None
        self._s3_client = None
        self._bucket_name: Optional[str] = None
        self._content_type: Optional[str] = None
        # close() can run from the atexit hook while a pipeline thread is in run()
        self._lock = Lock()
        self._disable_sinks = disable_sinks
        _live_blocks.add(self)

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["disable_sinks"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def close(self) -> None:
        """Upload the append_log entries that are not in S3 yet.

        Called when the block is torn down: by ``__del__`` when the workflow
        releases the block, and by an ``atexit`` hook for blocks still alive at
        interpreter exit. Does nothing when every entry is already uploaded. A
        failed upload is only logged, as no run is left to report it to.
        """
        with self._lock:
            self._upload_pending_entries()

    def __del__(self):
        try:
            self.close()
        except Exception as error:
            logging.warning(f"Could not upload buffered S3 sink entries: {error}")

    def run(
        self,
        content: str,
        file_type: Literal["csv", "json", "txt"],
        output_mode: Literal["append_log", "separate_files"],
        bucket_name: str,
        s3_prefix: str,
        file_name_prefix: str,
        max_entries_per_file: int,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_region: Optional[str] = None,
        upload_interval_seconds: int = DEFAULT_UPLOAD_INTERVAL_SECONDS,
    ) -> BlockResult:
        if self._disable_sinks:
            return disabled_sink_response()
        s3_client = get_s3_client(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_region=aws_region,
        )

        if output_mode == "separate_files":
            return self._upload_separate_file(
                s3_client=s3_client,
                content=content,
                file_type=file_type,
                bucket_name=bucket_name,
                s3_prefix=s3_prefix,
                file_name_prefix=file_name_prefix,
            )
        return self._append_to_buffer(
            s3_client=s3_client,
            content=content,
            file_type=file_type,
            bucket_name=bucket_name,
            s3_prefix=s3_prefix,
            file_name_prefix=file_name_prefix,
            max_entries_per_file=max_entries_per_file,
            upload_interval_seconds=upload_interval_seconds,
        )

    def _upload_separate_file(
        self,
        s3_client,
        content: str,
        file_type: Literal["csv", "json", "txt"],
        bucket_name: str,
        s3_prefix: str,
        file_name_prefix: str,
    ) -> BlockResult:
        s3_key = generate_s3_key(
            s3_prefix=s3_prefix,
            file_name_prefix=file_name_prefix,
            file_type=file_type,
        )
        content_type = CONTENT_TYPES.get(file_type, "text/plain")
        return upload_content_to_s3(
            s3_client=s3_client,
            bucket_name=bucket_name,
            s3_key=s3_key,
            content=content,
            content_type=content_type,
        )

    def _append_to_buffer(
        self,
        s3_client,
        content: str,
        file_type: Literal["csv", "json", "txt"],
        bucket_name: str,
        s3_prefix: str,
        file_name_prefix: str,
        max_entries_per_file: int,
        upload_interval_seconds: int,
    ) -> BlockResult:
        extension = file_type if file_type != "json" else "jsonl"
        content_type = CONTENT_TYPES.get(extension, "text/plain")

        if file_type == "json":
            try:
                content = dump_json_inline(content=content)
            except Exception as error:
                logging.warning(
                    f"Could not process JSON content in append mode: {error}"
                )
                return {"error_status": True, "message": "Invalid JSON content"}

        with self._lock:
            needs_rotation = (
                self._current_key is not None
                and self._entries_in_buffer >= max_entries_per_file
            )
            is_first_entry = self._current_key is None

            if needs_rotation or is_first_entry:
                # Entries are left here when the upload of the full object failed:
                # try once more before they are dropped.
                self._upload_pending_entries()
                self._buffer = []
                self._entries_in_buffer = 0
                self._entries_uploaded = 0
                self._last_upload_time = None
                self._current_key = generate_s3_key(
                    s3_prefix=s3_prefix,
                    file_name_prefix=file_name_prefix,
                    file_type=extension,
                )
            elif file_type == "csv":
                content = deduct_csv_header(content=content)

            if not content.endswith("\n"):
                content = f"{content}\n"
            self._buffer.append(content)
            self._entries_in_buffer += 1
            self._s3_client = s3_client
            self._bucket_name = bucket_name
            self._content_type = content_type

            # The first entry of each object uploads right away, so credential and
            # permission errors surface on the first run. The interval counts from
            # the last successful upload, so after a failure the next run retries.
            upload_is_due = (
                self._last_upload_time is None
                or self._entries_in_buffer >= max_entries_per_file
                or monotonic() - self._last_upload_time >= upload_interval_seconds
            )
            if not upload_is_due:
                return {
                    "error_status": False,
                    "message": (
                        "Data buffered for upload to "
                        f"s3://{bucket_name}/{self._current_key}"
                    ),
                }
            result = self._upload_buffer()

        return result

    def _upload_pending_entries(self) -> None:
        pending_entries = self._entries_in_buffer - self._entries_uploaded
        if pending_entries == 0:
            return None

        result = self._upload_buffer()
        if result["error_status"]:
            logging.warning(
                f"Could not upload {pending_entries} buffered entries to "
                f"s3://{self._bucket_name}/{self._current_key}: {result['message']}"
            )

    def _upload_buffer(self) -> BlockResult:
        result = upload_content_to_s3(
            s3_client=self._s3_client,
            bucket_name=self._bucket_name,
            s3_key=self._current_key,
            content="".join(self._buffer),
            content_type=self._content_type,
        )
        if not result["error_status"]:
            self._entries_uploaded = self._entries_in_buffer
            self._last_upload_time = monotonic()

        return result


# A block still alive at interpreter exit is only finalized after botocore is torn
# down, too late to upload, so this hook uploads what those blocks still buffer.
_live_blocks: "WeakSet[S3SinkBlockV1]" = WeakSet()


@atexit.register
def _close_live_blocks() -> None:
    for block in list(_live_blocks):
        block.close()


# Clients are thread-safe and keep their connection pool, so every block instance
# shares one per credentials and region. They are created under the lock because
# the default boto3 session is not thread-safe.
_s3_clients = LRUCache(capacity=MAX_CACHED_S3_CLIENTS)
_s3_clients_lock = Lock()


def get_s3_client(
    aws_access_key_id: Optional[str],
    aws_secret_access_key: Optional[str],
    aws_region: Optional[str],
):
    """Return the S3 client shared by all runs that use these credentials.

    Clients are created on first use and kept in a bounded LRU cache keyed by
    credentials and region, so runs reuse a client and its open connections
    instead of creating a client, and a new TLS connection, per run.

    Args:
        aws_access_key_id: AWS access key ID supplied by the workflow.
        aws_secret_access_key: AWS secret access key supplied by the workflow.
        aws_region: Region of the bucket, or None for boto3's default region.

    Returns:
        The boto3 S3 client for these credentials and region.

    Raises:
        ValueError: If either credential is missing, before the cache is consulted.
    """
    _ensure_credentials_are_provided(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )

    cache_key = (aws_access_key_id, aws_secret_access_key, aws_region)
    with _s3_clients_lock:
        s3_client = _s3_clients.get(cache_key)
        if s3_client is None:
            s3_client = create_s3_client(
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_region=aws_region,
            )
            _s3_clients.set(cache_key, s3_client)

    return s3_client


def create_s3_client(
    aws_access_key_id: Optional[str],
    aws_secret_access_key: Optional[str],
    aws_region: Optional[str],
):
    _ensure_credentials_are_provided(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )
    kwargs = {
        "aws_access_key_id": aws_access_key_id,
        "aws_secret_access_key": aws_secret_access_key,
    }
    if aws_region:
        kwargs["region_name"] = aws_region
    return boto3.client("s3", **kwargs)


def _ensure_credentials_are_provided(
    aws_access_key_id: Optional[str],
    aws_secret_access_key: Optional[str],
) -> None:
    if not aws_access_key_id or not aws_secret_access_key:
        raise ValueError(
            "S3 sink requires `aws_access_key_id` and `aws_secret_access_key` to be "
            "provided by the workflow. The server's ambient AWS credentials are not used."
        )


def generate_s3_key(s3_prefix: str, file_name_prefix: str, file_type: str) -> str:
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    file_name = f"{file_name_prefix.rstrip('/')}_{timestamp}.{file_type}"
    if s3_prefix:
        prefix = s3_prefix.rstrip("/")
        return f"{prefix}/{file_name}"
    return file_name


def upload_content_to_s3(
    s3_client,
    bucket_name: str,
    s3_key: str,
    content: str,
    content_type: str,
) -> dict:
    content_bytes = content.encode("utf-8")
    last_error: Optional[Exception] = None
    for attempt in range(1 + MAX_UPLOAD_RETRIES):
        if attempt > 0:
            delay = UPLOAD_RETRY_INITIAL_DELAY * (2 ** (attempt - 1))
            logging.warning(
                f"Retrying S3 upload to s3://{bucket_name}/{s3_key} "
                f"(attempt {attempt + 1}/{1 + MAX_UPLOAD_RETRIES}) after {delay:.1f}s..."
            )
            sleep(delay)
        try:
            s3_client.put_object(
                Bucket=bucket_name,
                Key=s3_key,
                Body=content_bytes,
                ContentType=content_type,
            )
            return {
                "error_status": False,
                "message": f"Data saved successfully to s3://{bucket_name}/{s3_key}",
            }
        except ClientError as error:
            error_code = error.response["Error"]["Code"]
            if error_code in NON_RETRYABLE_CLIENT_ERROR_CODES:
                logging.warning(f"Non-retryable S3 error ({error_code}): {error}")
                return {"error_status": True, "message": str(error)}
            logging.warning(
                f"Retryable S3 error on attempt {attempt + 1}/{1 + MAX_UPLOAD_RETRIES} "
                f"({error_code}): {error}"
            )
            last_error = error
        except BotoCoreError as error:
            logging.warning(
                f"S3 connection error on attempt {attempt + 1}/{1 + MAX_UPLOAD_RETRIES}: {error}"
            )
            last_error = error
        except Exception as error:
            logging.warning(f"Unexpected error uploading to S3: {error}")
            return {"error_status": True, "message": str(error)}
    return {"error_status": True, "message": str(last_error)}


def deduct_csv_header(content: str) -> str:
    return "\n".join(list(content.split("\n"))[1:])


def dump_json_inline(content: str) -> str:
    parsed_content = json.loads(content)
    return json.dumps(parsed_content)
