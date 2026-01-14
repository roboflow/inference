import base64
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
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
The **GCP Cloud Storage Sink** block enables uploading data from your Workflow to Google Cloud Storage buckets.

## How This Block Works

This block uploads content (text, JSON, binary data, or images) to a GCP Cloud Storage bucket using the JSON API. It supports:

* Direct upload of string or binary content
* Two authentication methods: OAuth2 access token or Service Account JSON key
* Optional data transformations using UQL operations
* Asynchronous (fire-and-forget) execution for better workflow performance
* Cooldown mechanism to prevent excessive uploads

### Authentication

You can authenticate using one of two methods:

**Method 1: OAuth2 Access Token**
* `access_token`: A valid OAuth2 access token with storage write permissions

**Method 2: Service Account JSON Key**
* `service_account_json`: The full JSON content of a service account key file
* Requires the `cryptography` library for JWT signing

To generate an access token from service account, you can use:
```bash
gcloud auth print-access-token --impersonate-service-account=YOUR_SERVICE_ACCOUNT
```

### Uploading Data

Specify the target location:

* `bucket`: The GCS bucket name
* `object_name`: The name/path for the object (e.g., "folder/file.json")

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

- Save workflow results (predictions, analytics) to GCS for long-term storage
- Archive processed images or frames to cloud storage
- Export CSV/JSON reports to GCS for BigQuery or other GCP services
- Integrate with GCP-based ML pipelines and Vertex AI
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "GCP Cloud Storage Upload",
            "version": "v1",
            "short_description": "Upload data to Google Cloud Storage bucket.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "sink",
            "ui_manifest": {
                "section": "cloud_storage",
                "icon": "fab fa-google",
                "blockPriority": 3,
            },
        }
    )
    type: Literal["roboflow_core/gcp_gcs_sink@v1"]

    bucket: Union[str, Selector(kind=[STRING_KIND])] = Field(
        description="Name of the GCS bucket to upload to.",
        examples=["my-bucket", "$inputs.bucket_name"],
    )
    object_name: Union[str, Selector(kind=[STRING_KIND])] = Field(
        description="Name/path for the object in GCS. Can include folder structure like 'folder/file.json'.",
        examples=["results/output.json", "$inputs.object_name"],
    )
    access_token: Optional[Union[str, Selector(kind=[STRING_KIND, SECRET_KIND])]] = (
        Field(
            default=None,
            description="OAuth2 access token with storage write permissions. Preferred method.",
            private=True,
            examples=["$inputs.gcp_access_token"],
        )
    )
    service_account_json: Optional[
        Union[str, Selector(kind=[STRING_KIND, SECRET_KIND])]
    ] = Field(
        default=None,
        description="Full JSON content of a service account key file. Requires cryptography library.",
        private=True,
        examples=["$inputs.service_account_json"],
    )
    project_id: Optional[Union[str, Selector(kind=[STRING_KIND])]] = Field(
        default=None,
        description="GCP Project ID. Required if using service_account_json and not specified in the JSON.",
        examples=["my-project", "$inputs.project_id"],
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
        description="Optional metadata to attach to the object.",
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
        examples=[False, "$inputs.disable_gcs_upload"],
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


class GCPGCSSinkBlockV1(WorkflowBlock):
    def __init__(
        self,
        background_tasks: Optional[BackgroundTasks],
        thread_pool_executor: Optional[ThreadPoolExecutor],
    ):
        self._background_tasks = background_tasks
        self._thread_pool_executor = thread_pool_executor
        self._last_upload_time: Optional[datetime] = None
        self._cached_token: Optional[str] = None
        self._token_expiry: Optional[float] = None

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["background_tasks", "thread_pool_executor"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        bucket: str,
        object_name: str,
        access_token: Optional[str],
        service_account_json: Optional[str],
        project_id: Optional[str],
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

        if not access_token and not service_account_json:
            return {
                "error_status": True,
                "throttling_status": False,
                "message": "Either access_token or service_account_json must be provided",
            }

        seconds_since_last_upload = cooldown_seconds
        if self._last_upload_time is not None:
            seconds_since_last_upload = (
                datetime.now() - self._last_upload_time
            ).total_seconds()

        if seconds_since_last_upload < cooldown_seconds:
            logging.info("Activated `roboflow_core/gcp_gcs_sink@v1` cooldown.")
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
            content = json.dumps(content).encode("utf-8")

        # Get or generate access token
        token = access_token
        if not token and service_account_json:
            # Check if we have a valid cached token
            if (
                self._cached_token
                and self._token_expiry
                and time.time() < self._token_expiry - 60
            ):
                token = self._cached_token
            else:
                try:
                    token, expiry = generate_access_token_from_service_account(
                        service_account_json
                    )
                    self._cached_token = token
                    self._token_expiry = expiry
                except Exception as e:
                    return {
                        "error_status": True,
                        "throttling_status": False,
                        "message": f"Failed to generate access token: {str(e)}",
                    }

        upload_handler = partial(
            upload_to_gcs,
            bucket=bucket,
            object_name=object_name,
            access_token=token,
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


def generate_access_token_from_service_account(
    service_account_json: str,
) -> Tuple[str, float]:
    """Generate an OAuth2 access token from a service account JSON key."""
    try:
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding
    except ImportError:
        raise ImportError(
            "cryptography library is required for service account authentication. "
            "Install with: pip install cryptography"
        )

    # Parse the service account JSON
    if isinstance(service_account_json, str):
        sa_info = json.loads(service_account_json)
    else:
        sa_info = service_account_json

    client_email = sa_info["client_email"]
    private_key_pem = sa_info["private_key"]

    # Create JWT
    now = int(time.time())
    expiry = now + 3600  # 1 hour

    header = {"alg": "RS256", "typ": "JWT"}

    payload = {
        "iss": client_email,
        "sub": client_email,
        "aud": "https://oauth2.googleapis.com/token",
        "iat": now,
        "exp": expiry,
        "scope": "https://www.googleapis.com/auth/devstorage.read_write",
    }

    # Encode header and payload
    header_b64 = (
        base64.urlsafe_b64encode(json.dumps(header).encode()).rstrip(b"=").decode()
    )
    payload_b64 = (
        base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
    )
    signing_input = f"{header_b64}.{payload_b64}"

    # Sign with private key
    private_key = serialization.load_pem_private_key(
        private_key_pem.encode(), password=None
    )
    signature = private_key.sign(
        signing_input.encode(),
        padding.PKCS1v15(),
        hashes.SHA256(),
    )
    signature_b64 = base64.urlsafe_b64encode(signature).rstrip(b"=").decode()

    jwt_token = f"{signing_input}.{signature_b64}"

    # Exchange JWT for access token
    response = requests.post(
        "https://oauth2.googleapis.com/token",
        data={
            "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
            "assertion": jwt_token,
        },
        timeout=30,
    )
    response.raise_for_status()

    token_data = response.json()
    return token_data["access_token"], expiry


def upload_to_gcs(
    bucket: str,
    object_name: str,
    access_token: str,
    content: bytes,
    content_type: str,
    metadata: Dict[str, str],
    timeout: int,
) -> Tuple[bool, str]:
    try:
        # URL encode the object name for the API
        import urllib.parse

        encoded_object_name = urllib.parse.quote(object_name, safe="")

        # Use the JSON API upload endpoint
        url = f"https://storage.googleapis.com/upload/storage/v1/b/{bucket}/o?uploadType=media&name={encoded_object_name}"

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": content_type,
        }

        response = requests.post(
            url, headers=headers, data=content, timeout=timeout
        )
        response.raise_for_status()

        # If metadata is provided, update the object metadata
        if metadata:
            metadata_url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/{encoded_object_name}"
            metadata_body = {"metadata": metadata}
            metadata_response = requests.patch(
                metadata_url,
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json",
                },
                json=metadata_body,
                timeout=timeout,
            )
            metadata_response.raise_for_status()

        return False, f"Successfully uploaded to gs://{bucket}/{object_name}"

    except requests.exceptions.RequestException as error:
        logging.warning(f"Failed to upload to GCS: {str(error)}")
        return True, f"Upload failed: {str(error)}"
    except Exception as error:
        logging.warning(f"Failed to upload to GCS: {str(error)}")
        return True, f"Upload failed: {str(error)}"
