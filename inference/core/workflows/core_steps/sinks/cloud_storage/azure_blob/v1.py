import base64
import hashlib
import hmac
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

import requests
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
The **Azure Blob Storage Sink** block enables uploading data from your Workflow to Azure Blob Storage containers.

## How This Block Works

This block uploads content (text, JSON, binary data, or images) to an Azure Blob Storage container using the REST API. It supports:

* Direct upload of string or binary content
* Two authentication methods: SAS Token or Account Key
* Optional data transformations using UQL operations
* Asynchronous (fire-and-forget) execution for better workflow performance
* Cooldown mechanism to prevent excessive uploads

### Authentication

You can authenticate using one of two methods:

**Method 1: SAS Token (Recommended)**
* `sas_token`: A Shared Access Signature token with write permissions

**Method 2: Account Key**
* `account_key`: Your storage account access key

### Uploading Data

Specify the target location:

* `account_name`: Your Azure Storage account name
* `container_name`: The blob container name
* `blob_name`: The name/path for the blob (e.g., "folder/file.json")

The content can be transformed before upload using `content_operations`:

```
content_operations = [{"type": "ConvertImageToJPEG"}]
```

### Cooldown

The block accepts `cooldown_seconds` (which **defaults to `5` seconds**) to prevent unintended bursts of
uploads. Please adjust it according to your needs, setting `0` indicates no cooldown.

### Async Execution

Configure the `fire_and_forget` property. Set it to True if you want the upload to occur in the background,
allowing the Workflow to proceed without waiting.

## Common Use Cases

- Save workflow results (predictions, analytics) to Azure Blob Storage
- Archive processed images or frames to cloud storage
- Export CSV/JSON reports to Azure for downstream processing
- Integrate with Azure-based data pipelines and analytics
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Azure Blob Storage Upload",
            "version": "v1",
            "short_description": "Upload data to Azure Blob Storage container.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "sink",
            "ui_manifest": {
                "section": "cloud_storage",
                "icon": "fab fa-microsoft",
                "blockPriority": 2,
            },
        }
    )
    type: Literal["roboflow_core/azure_blob_sink@v1"]

    account_name: Union[str, Selector(kind=[STRING_KIND])] = Field(
        description="Azure Storage account name.",
        examples=["mystorageaccount", "$inputs.account_name"],
    )
    container_name: Union[str, Selector(kind=[STRING_KIND])] = Field(
        description="Name of the blob container.",
        examples=["my-container", "$inputs.container_name"],
    )
    blob_name: Union[str, Selector(kind=[STRING_KIND])] = Field(
        description="Name/path of the blob. Can include folder structure like 'folder/file.json'.",
        examples=["results/output.json", "$inputs.blob_name"],
    )
    sas_token: Optional[Union[str, Selector(kind=[STRING_KIND, SECRET_KIND])]] = Field(
        default=None,
        description="SAS token for authentication (include the leading '?'). Preferred over account_key.",
        private=True,
        examples=["$inputs.sas_token"],
    )
    account_key: Optional[Union[str, Selector(kind=[STRING_KIND, SECRET_KIND])]] = (
        Field(
            default=None,
            description="Storage account access key. Used if sas_token is not provided.",
            private=True,
            examples=["$inputs.account_key"],
        )
    )
    content: Union[str, Selector(kind=[STRING_KIND, BYTES_KIND])] = Field(
        description="The content to upload. Can be a string, JSON, or binary data.",
        examples=["$steps.csv_formatter.csv_content", "$inputs.data"],
    )
    content_operations: List[AllOperationsType] = Field(
        description="UQL operations to transform content before upload.",
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
        description="Optional metadata to attach to the blob. Keys are prefixed with 'x-ms-meta-'.",
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
        examples=[False, "$inputs.disable_azure_upload"],
    )
    cooldown_seconds: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        default=5,
        description="Minimum seconds between consecutive uploads. Set to 0 for no cooldown.",
        json_schema_extra={"always_visible": True},
        examples=[5, 0],
    )
    timeout: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        default=30,
        description="Request timeout in seconds.",
        examples=[30, 60],
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


class AzureBlobSinkBlockV1(WorkflowBlock):
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
        account_name: str,
        container_name: str,
        blob_name: str,
        sas_token: Optional[str],
        account_key: Optional[str],
        content: Any,
        content_operations: List[AllOperationsType],
        content_type: str,
        metadata: Dict[str, str],
        fire_and_forget: bool,
        disable_sink: bool,
        cooldown_seconds: int,
        timeout: int,
    ) -> BlockResult:
        if disable_sink:
            return {
                "error_status": False,
                "throttling_status": False,
                "message": "Sink disabled by parameter `disable_sink`",
            }

        if not sas_token and not account_key:
            return {
                "error_status": True,
                "throttling_status": False,
                "message": "Either sas_token or account_key must be provided",
            }

        seconds_since_last_upload = cooldown_seconds
        if self._last_upload_time is not None:
            seconds_since_last_upload = (
                datetime.now() - self._last_upload_time
            ).total_seconds()

        if seconds_since_last_upload < cooldown_seconds:
            logging.info("Activated `roboflow_core/azure_blob_sink@v1` cooldown.")
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
            upload_to_azure_blob,
            account_name=account_name,
            container_name=container_name,
            blob_name=blob_name,
            sas_token=sas_token,
            account_key=account_key,
            content=content,
            content_type=content_type,
            metadata=metadata,
            timeout=timeout,
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


def generate_azure_auth_header(
    account_name: str,
    account_key: str,
    method: str,
    content_length: int,
    content_type: str,
    blob_type: str,
    date_str: str,
    canonicalized_resource: str,
    metadata_headers: str = "",
) -> str:
    """Generate the Authorization header using Shared Key authentication."""
    # String to sign format for Blob service
    # https://docs.microsoft.com/en-us/rest/api/storageservices/authorize-with-shared-key
    string_to_sign = (
        f"{method}\n"  # HTTP verb
        f"\n"  # Content-Encoding
        f"\n"  # Content-Language
        f"{content_length}\n"  # Content-Length
        f"\n"  # Content-MD5
        f"{content_type}\n"  # Content-Type
        f"\n"  # Date
        f"\n"  # If-Modified-Since
        f"\n"  # If-Match
        f"\n"  # If-None-Match
        f"\n"  # If-Unmodified-Since
        f"\n"  # Range
        f"x-ms-blob-type:{blob_type}\n"  # Canonicalized headers
        f"x-ms-date:{date_str}\n"
        f"{metadata_headers}"
        f"x-ms-version:2021-06-08\n"
        f"{canonicalized_resource}"
    )

    # Sign the string with the account key
    key_bytes = base64.b64decode(account_key)
    signature = base64.b64encode(
        hmac.new(key_bytes, string_to_sign.encode("utf-8"), hashlib.sha256).digest()
    ).decode("utf-8")

    return f"SharedKey {account_name}:{signature}"


def upload_to_azure_blob(
    account_name: str,
    container_name: str,
    blob_name: str,
    sas_token: Optional[str],
    account_key: Optional[str],
    content: bytes,
    content_type: str,
    metadata: Dict[str, str],
    timeout: int,
) -> Tuple[bool, str]:
    try:
        # Build the URL
        base_url = f"https://{account_name}.blob.core.windows.net/{container_name}/{blob_name}"

        if sas_token:
            # SAS token authentication - just append to URL
            if not sas_token.startswith("?"):
                sas_token = "?" + sas_token
            url = base_url + sas_token
            headers = {
                "x-ms-blob-type": "BlockBlob",
                "Content-Type": content_type,
                "Content-Length": str(len(content)),
            }
        else:
            # Shared Key authentication
            url = base_url
            date_str = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")
            canonicalized_resource = f"/{account_name}/{container_name}/{blob_name}"

            # Build metadata headers string for signature
            metadata_headers = ""
            if metadata:
                sorted_metadata = sorted(metadata.items())
                metadata_headers = "".join(
                    f"x-ms-meta-{k.lower()}:{v}\n" for k, v in sorted_metadata
                )

            auth_header = generate_azure_auth_header(
                account_name=account_name,
                account_key=account_key,
                method="PUT",
                content_length=len(content),
                content_type=content_type,
                blob_type="BlockBlob",
                date_str=date_str,
                canonicalized_resource=canonicalized_resource,
                metadata_headers=metadata_headers,
            )

            headers = {
                "x-ms-blob-type": "BlockBlob",
                "x-ms-date": date_str,
                "x-ms-version": "2021-06-08",
                "Content-Type": content_type,
                "Content-Length": str(len(content)),
                "Authorization": auth_header,
            }

        # Add metadata headers
        for key, value in metadata.items():
            headers[f"x-ms-meta-{key}"] = value

        response = requests.put(url, headers=headers, data=content, timeout=timeout)
        response.raise_for_status()

        return (
            False,
            f"Successfully uploaded to https://{account_name}.blob.core.windows.net/{container_name}/{blob_name}",
        )

    except requests.exceptions.RequestException as error:
        logging.warning(f"Failed to upload to Azure Blob Storage: {str(error)}")
        return True, f"Upload failed: {str(error)}"
    except Exception as error:
        logging.warning(f"Failed to upload to Azure Blob Storage: {str(error)}")
        return True, f"Upload failed: {str(error)}"
