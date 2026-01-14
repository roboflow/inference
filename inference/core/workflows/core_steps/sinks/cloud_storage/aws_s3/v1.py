import base64
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

from fastapi import BackgroundTasks
from pydantic import ConfigDict, Field

from inference.core.workflows.core_steps.common.query_language.entities.operations import (
    AllOperationsType,
)
from inference.core.workflows.core_steps.common.query_language.operations.core import (
    build_operations_chain,
)
from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    BYTES_KIND,
    INTEGER_KIND,
    SECRET_KIND,
    STRING_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
The **AWS S3 Sink** block enables uploading data from your Workflow to Amazon S3 buckets.

## How This Block Works

This block uploads content (text, JSON, binary data, or images) to an Amazon S3 bucket. It supports:

* Direct upload of string or binary content
* Automatic content type detection
* Optional data transformations using UQL operations
* Asynchronous (fire-and-forget) execution for better workflow performance
* Cooldown mechanism to prevent excessive uploads

### Authentication

You need to provide AWS credentials:

* `aws_access_key_id`: Your AWS access key
* `aws_secret_access_key`: Your AWS secret key
* `aws_region`: The AWS region where your bucket is located (e.g., "us-east-1")
* `aws_session_token`: (Optional) Session token for temporary credentials

### Uploading Data

Specify the target location:

* `bucket`: The S3 bucket name
* `object_key`: The path/key for the object in S3 (e.g., "folder/subfolder/file.json")

The content can be transformed before upload using `content_operations`:

```
content_operations = [{"type": "ConvertImageToJPEG"}]
```

### Cooldown

The block accepts `cooldown_seconds` (which **defaults to `5` seconds**) to prevent unintended bursts of
uploads. Please adjust it according to your needs, setting `0` indicates no cooldown.

During the cooldown period, consecutive runs of the step will cause `throttling_status` output to be set `True`
and no upload will be performed.

### Async Execution

Configure the `fire_and_forget` property. Set it to True if you want the upload to occur in the background,
allowing the Workflow to proceed without waiting. In this case, `error_status` will always be `False`, so we
**recommend setting `fire_and_forget=False` for debugging purposes**.

## Common Use Cases

- Save workflow results (predictions, analytics) to S3 for long-term storage
- Archive processed images or frames to cloud storage
- Export CSV/JSON reports to S3 for downstream processing
- Create data pipelines that store workflow outputs for ML training
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "AWS S3 Upload",
            "version": "v1",
            "short_description": "Upload data to Amazon S3 bucket.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "sink",
            "ui_manifest": {
                "section": "cloud_storage",
                "icon": "fab fa-aws",
                "blockPriority": 1,
            },
        }
    )
    type: Literal["roboflow_core/aws_s3_sink@v1"]

    aws_access_key_id: Union[str, Selector(kind=[STRING_KIND, SECRET_KIND])] = Field(
        description="AWS Access Key ID for authentication.",
        private=True,
        examples=["$inputs.aws_access_key_id"],
    )
    aws_secret_access_key: Union[str, Selector(kind=[STRING_KIND, SECRET_KIND])] = (
        Field(
            description="AWS Secret Access Key for authentication.",
            private=True,
            examples=["$inputs.aws_secret_access_key"],
        )
    )
    aws_session_token: Optional[
        Union[str, Selector(kind=[STRING_KIND, SECRET_KIND])]
    ] = Field(
        default=None,
        description="AWS Session Token for temporary credentials (optional).",
        private=True,
        examples=["$inputs.aws_session_token"],
    )
    aws_region: Union[str, Selector(kind=[STRING_KIND])] = Field(
        default="us-east-1",
        description="AWS region where the S3 bucket is located.",
        examples=["us-east-1", "eu-west-1"],
    )
    bucket: Union[str, Selector(kind=[STRING_KIND])] = Field(
        description="Name of the S3 bucket to upload to.",
        examples=["my-bucket", "$inputs.bucket_name"],
    )
    object_key: Union[str, Selector(kind=[STRING_KIND])] = Field(
        description="The key (path) for the object in S3. Can include folder structure like 'folder/subfolder/file.json'.",
        examples=["results/output.json", "$inputs.object_key"],
    )
    content: Union[str, Selector(kind=[STRING_KIND, BYTES_KIND])] = Field(
        description="The content to upload. Can be a string, JSON, or binary data.",
        examples=["$steps.csv_formatter.csv_content", "$inputs.data"],
    )
    content_operations: List[AllOperationsType] = Field(
        description="UQL operations to transform content before upload (e.g., convert image to JPEG).",
        default_factory=list,
        examples=[[{"type": "ConvertImageToJPEG"}]],
    )
    content_type: Union[str, Selector(kind=[STRING_KIND])] = Field(
        default="application/octet-stream",
        description="MIME type of the content being uploaded.",
        examples=["application/json", "text/csv", "image/jpeg"],
    )
    metadata: Dict[str, Union[str, Selector(kind=[STRING_KIND])]] = Field(
        default_factory=dict,
        description="Optional metadata to attach to the S3 object.",
        examples=[{"source": "workflow", "version": "1.0"}],
    )
    fire_and_forget: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=True,
        description="If True, upload runs in background without waiting for completion.",
        examples=[True, False],
    )
    disable_sink: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=False,
        description="If True, the upload is skipped entirely.",
        examples=[False, "$inputs.disable_s3_upload"],
    )
    cooldown_seconds: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        default=5,
        description="Minimum seconds between consecutive uploads. Set to 0 for no cooldown.",
        json_schema_extra={"always_visible": True},
        examples=[5, 0],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="error_status", kind=[BOOLEAN_KIND]),
            OutputDefinition(name="throttling_status", kind=[BOOLEAN_KIND]),
            OutputDefinition(name="message", kind=[STRING_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class AWSS3SinkBlockV1(WorkflowBlock):
    def __init__(
        self,
        background_tasks: Optional[BackgroundTasks],
        thread_pool_executor: Optional[ThreadPoolExecutor],
    ):
        self._background_tasks = background_tasks
        self._thread_pool_executor = thread_pool_executor
        self._last_upload_time: Optional[datetime] = None

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["background_tasks", "thread_pool_executor"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        aws_session_token: Optional[str],
        aws_region: str,
        bucket: str,
        object_key: str,
        content: Any,
        content_operations: List[AllOperationsType],
        content_type: str,
        metadata: Dict[str, str],
        fire_and_forget: bool,
        disable_sink: bool,
        cooldown_seconds: int,
    ) -> BlockResult:
        if disable_sink:
            return {
                "error_status": False,
                "throttling_status": False,
                "message": "Sink disabled by parameter `disable_sink`",
            }

        seconds_since_last_upload = cooldown_seconds
        if self._last_upload_time is not None:
            seconds_since_last_upload = (
                datetime.now() - self._last_upload_time
            ).total_seconds()

        if seconds_since_last_upload < cooldown_seconds:
            logging.info("Activated `roboflow_core/aws_s3_sink@v1` cooldown.")
            return {
                "error_status": False,
                "throttling_status": True,
                "message": "Sink cooldown applies",
            }

        # Apply content operations if specified
        if content_operations:
            operations_chain = build_operations_chain(operations=content_operations)
            content = operations_chain(content, global_parameters={})

        # Convert content to bytes if needed
        if isinstance(content, str):
            content = content.encode("utf-8")
        elif isinstance(content, dict) or isinstance(content, list):
            import json

            content = json.dumps(content).encode("utf-8")

        upload_handler = partial(
            upload_to_s3,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            aws_region=aws_region,
            bucket=bucket,
            object_key=object_key,
            content=content,
            content_type=content_type,
            metadata=metadata,
        )

        self._last_upload_time = datetime.now()

        if fire_and_forget and self._background_tasks:
            self._background_tasks.add_task(upload_handler)
            return {
                "error_status": False,
                "throttling_status": False,
                "message": "Upload initiated in background task",
            }

        if fire_and_forget and self._thread_pool_executor:
            self._thread_pool_executor.submit(upload_handler)
            return {
                "error_status": False,
                "throttling_status": False,
                "message": "Upload initiated in background task",
            }

        error_status, message = upload_handler()
        return {
            "error_status": error_status,
            "throttling_status": False,
            "message": message,
        }


def upload_to_s3(
    aws_access_key_id: str,
    aws_secret_access_key: str,
    aws_session_token: Optional[str],
    aws_region: str,
    bucket: str,
    object_key: str,
    content: bytes,
    content_type: str,
    metadata: Dict[str, str],
) -> Tuple[bool, str]:
    try:
        import boto3
        from botocore.config import Config

        config = Config(
            region_name=aws_region,
            retries={"max_attempts": 3, "mode": "standard"},
        )

        session_kwargs = {
            "aws_access_key_id": aws_access_key_id,
            "aws_secret_access_key": aws_secret_access_key,
        }
        if aws_session_token:
            session_kwargs["aws_session_token"] = aws_session_token

        client = boto3.client("s3", config=config, **session_kwargs)

        put_kwargs = {
            "Bucket": bucket,
            "Key": object_key,
            "Body": content,
            "ContentType": content_type,
        }
        if metadata:
            put_kwargs["Metadata"] = metadata

        client.put_object(**put_kwargs)
        return False, f"Successfully uploaded to s3://{bucket}/{object_key}"

    except ImportError:
        logging.error("boto3 is required for AWS S3 uploads. Install with: pip install boto3")
        return True, "boto3 library not installed"
    except Exception as error:
        logging.warning(f"Failed to upload to S3: {str(error)}")
        return True, f"Upload failed: {str(error)}"
