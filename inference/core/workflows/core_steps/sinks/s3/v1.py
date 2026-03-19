import json
import logging
from datetime import datetime
from typing import Any, List, Literal, Optional, Type, Union

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from pydantic import ConfigDict, Field, field_validator

from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    SECRET_KIND,
    STRING_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

CONTENT_TYPES = {
    "csv": "text/csv",
    "json": "application/json",
    "jsonl": "application/x-ndjson",
    "txt": "text/plain",
}

LONG_DESCRIPTION = """
Save workflow data directly to an AWS S3 bucket, supporting CSV, JSON, and text file formats with configurable output modes for aggregating multiple entries into single objects or saving each entry as a separate S3 object.

## How This Block Works

This block uploads string content from workflow steps to S3 objects. The block:

1. Takes string content (from formatters, predictions, or other string-producing blocks) and S3 configuration as input
2. Connects to AWS S3 using the provided credentials (or the default AWS credential chain if none are supplied)
3. Selects the appropriate upload strategy based on `output_mode`:
   - **Separate Files Mode**: Creates a new S3 object for each input, generating unique keys with timestamps
   - **Append Log Mode**: Buffers content in memory, uploading a complete object when `max_entries_per_file` is reached or when the block is destroyed
4. For **separate files mode**: Generates a unique S3 key from the prefix, file name prefix, file type, and a timestamp, then uploads the content directly
5. For **append log mode**:
   - Buffers content entries in memory under a single S3 key
   - Applies format-specific handling for appending:
     - **CSV**: Removes the header row from subsequent appends (CSV content must include headers on first write)
     - **JSON**: Converts to JSONL (JSON Lines) format, parsing and re-serializing each JSON document to fit on a single line
     - **TXT**: Appends content directly with newlines
   - Tracks entry count and uploads the full buffer as a complete S3 object when `max_entries_per_file` is reached, then starts a fresh buffer with a new key
   - Uploads any remaining buffered data when the block is destroyed
6. Returns error status and messages indicating save success or failure

The block supports two storage strategies: separate files mode creates individual timestamped S3 objects per input (useful for organizing outputs by execution), while append log mode accumulates entries in memory and writes them as complete S3 objects on rotation (useful for time-series logging with controlled upload frequency). S3 key names include timestamps (format: `YYYY_MM_DD_HH_MM_SS_microseconds`) for unique keys and chronological ordering.

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

In append log mode, data is buffered in memory and only uploaded to S3 when:
- The `max_entries_per_file` limit is reached (object rotation), or
- The block instance is destroyed at workflow teardown

This means data may not be immediately visible in S3 after each step execution. Use `separate_files` mode if immediate S3 visibility is required.

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
        description="Upload strategy: 'append_log' buffers multiple entries and uploads them as a single S3 object when the entry limit is reached (useful for batched logging), or 'separate_files' uploads each input as a new S3 object with a unique timestamp-based key (useful for per-execution outputs).",
        examples=["append_log"],
        json_schema_extra={
            "values_metadata": {
                "append_log": {
                    "name": "Append Log",
                    "description": "Buffers multiple entries and uploads as a single S3 object on rotation",
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
    aws_access_key_id: Optional[Union[Selector(kind=[SECRET_KIND, STRING_KIND]), str]] = Field(
        default=None,
        description="AWS access key ID for authentication. If not provided, boto3's default credential chain is used (environment variables, ~/.aws/credentials, or IAM role). Recommended: connect this to an Environment Secrets Store block rather than hardcoding.",
        examples=["$steps.secrets.aws_access_key_id"],
    )
    aws_secret_access_key: Optional[Union[Selector(kind=[SECRET_KIND, STRING_KIND]), str]] = Field(
        default=None,
        description="AWS secret access key for authentication. If not provided, boto3's default credential chain is used. Recommended: connect this to an Environment Secrets Store block rather than hardcoding.",
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

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="error_status", kind=[BOOLEAN_KIND]),
            OutputDefinition(name="message", kind=[STRING_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class S3SinkBlockV1(WorkflowBlock):

    def __init__(self):
        self._buffer: List[str] = []
        self._entries_in_buffer: int = 0
        self._current_key: Optional[str] = None

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

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
    ) -> BlockResult:
        s3_client = create_s3_client(
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
    ) -> BlockResult:
        extension = file_type if file_type != "json" else "jsonl"
        content_type = CONTENT_TYPES.get(extension, "text/plain")

        if file_type == "json":
            try:
                content = dump_json_inline(content=content)
            except Exception as error:
                logging.warning(f"Could not process JSON content in append mode: {error}")
                return {"error_status": True, "message": "Invalid JSON content"}

        needs_rotation = (
            self._current_key is not None
            and self._entries_in_buffer >= max_entries_per_file
        )
        is_first_entry = self._current_key is None

        if needs_rotation or is_first_entry:
            self._buffer = []
            self._entries_in_buffer = 0
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

        # Always upload the full accumulated buffer so that every run() call
        # validates credentials and permissions, and errors are surfaced immediately
        # rather than only when the buffer rotation limit is reached.
        full_content = "".join(self._buffer)
        return upload_content_to_s3(
            s3_client=s3_client,
            bucket_name=bucket_name,
            s3_key=self._current_key,
            content=full_content,
            content_type=content_type,
        )


def create_s3_client(
    aws_access_key_id: Optional[str],
    aws_secret_access_key: Optional[str],
    aws_region: Optional[str],
):
    kwargs = {}
    if aws_access_key_id:
        kwargs["aws_access_key_id"] = aws_access_key_id
    if aws_secret_access_key:
        kwargs["aws_secret_access_key"] = aws_secret_access_key
    if aws_region:
        kwargs["region_name"] = aws_region
    return boto3.client("s3", **kwargs)


def generate_s3_key(s3_prefix: str, file_name_prefix: str, file_type: str) -> str:
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    file_name = f"{file_name_prefix}_{timestamp}.{file_type}"
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
    try:
        content_bytes = content.encode("utf-8")
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
    except (BotoCoreError, ClientError) as error:
        logging.warning(f"Could not upload to S3: {error}")
        return {"error_status": True, "message": str(error)}
    except Exception as error:
        logging.warning(f"Unexpected error uploading to S3: {error}")
        return {"error_status": True, "message": str(error)}


def deduct_csv_header(content: str) -> str:
    return "\n".join(list(content.split("\n"))[1:])


def dump_json_inline(content: str) -> str:
    parsed_content = json.loads(content)
    return json.dumps(parsed_content)
