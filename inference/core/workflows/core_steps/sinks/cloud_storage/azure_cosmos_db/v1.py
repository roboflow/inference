import base64
import hashlib
import hmac
import json
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union
from urllib.parse import quote

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
    DICTIONARY_KIND,
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
The **Azure Cosmos DB Sink** block enables storing workflow data in Azure Cosmos DB using the SQL API.

## How This Block Works

This block creates or upserts documents in Azure Cosmos DB using the REST API. It supports:

* Creating new documents
* Upserting documents (create if not exists, update if exists)
* Automatic partition key handling
* Optional data transformations using UQL operations
* Asynchronous (fire-and-forget) execution for better workflow performance
* Cooldown mechanism to prevent excessive writes

### Authentication

You need to provide Azure Cosmos DB credentials:

* `account_endpoint`: Your Cosmos DB account endpoint (e.g., "https://myaccount.documents.azure.com:443/")
* `master_key`: Your Cosmos DB master key (primary or secondary)

### Document Structure

Each document requires:

* `database`: The database name
* `container`: The container name
* `document`: The document data (JSON object)
* `partition_key_value`: The partition key value for the document

The document must include an `id` field. If not provided, a UUID will be automatically generated.

### Operations

The block supports two operations:

* `create`: Creates a new document. Fails if document with same ID exists.
* `upsert`: Creates or replaces the document.

### Data Transformation

Transform data before saving using `document_operations`:

```
document_operations = {
    "predictions": [{"type": "DetectionsPropertyExtract", "property_name": "class_name"}]
}
```

### Cooldown

The block accepts `cooldown_seconds` (which **defaults to `5` seconds**) to prevent unintended bursts of
writes. Please adjust it according to your needs, setting `0` indicates no cooldown.

### Async Execution

Configure the `fire_and_forget` property. Set it to True if you want the write to occur in the background,
allowing the Workflow to proceed without waiting.

## Common Use Cases

- Store workflow predictions and analytics in Cosmos DB for querying
- Create audit logs of workflow executions
- Build real-time dashboards from workflow results
- Store time-series data from video processing
- Integrate with Azure Functions and other Azure services
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Azure Cosmos DB",
            "version": "v1",
            "short_description": "Store data in Azure Cosmos DB using SQL API.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "sink",
            "ui_manifest": {
                "section": "cloud_storage",
                "icon": "fab fa-microsoft",
                "blockPriority": 4,
            },
        }
    )
    type: Literal["roboflow_core/azure_cosmos_db_sink@v1"]

    account_endpoint: Union[str, Selector(kind=[STRING_KIND])] = Field(
        description="Azure Cosmos DB account endpoint URL.",
        examples=[
            "https://myaccount.documents.azure.com:443/",
            "$inputs.cosmos_endpoint",
        ],
    )
    master_key: Union[str, Selector(kind=[STRING_KIND, SECRET_KIND])] = Field(
        description="Azure Cosmos DB master key (primary or secondary).",
        private=True,
        examples=["$inputs.cosmos_master_key"],
    )
    database: Union[str, Selector(kind=[STRING_KIND])] = Field(
        description="Name of the Cosmos DB database.",
        examples=["my-database", "$inputs.database_name"],
    )
    container: Union[str, Selector(kind=[STRING_KIND])] = Field(
        description="Name of the Cosmos DB container.",
        examples=["my-container", "$inputs.container_name"],
    )
    partition_key_value: Union[str, Selector(kind=[STRING_KIND])] = Field(
        description="The partition key value for the document. Must match a field value in the document.",
        examples=["category1", "$inputs.partition_key"],
    )
    document: Union[Dict[str, Any], Selector(kind=[DICTIONARY_KIND, STRING_KIND])] = (
        Field(
            description="The document to store. Can be a dictionary or JSON string. Must include an 'id' field or one will be auto-generated.",
            examples=[{"id": "doc1", "type": "prediction", "data": "$steps.model.predictions"}],
        )
    )
    document_operations: Dict[str, List[AllOperationsType]] = Field(
        description="UQL operations to transform document fields before saving.",
        default_factory=dict,
        examples=[
            {
                "predictions": [
                    {"type": "DetectionsPropertyExtract", "property_name": "class_name"}
                ]
            }
        ],
    )
    operation: Literal["create", "upsert"] = Field(
        default="upsert",
        description="Operation type: 'create' (fails if exists) or 'upsert' (create or replace).",
        examples=["upsert", "create"],
    )
    fire_and_forget: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=True,
        description="If True, write runs in background without waiting for completion.",
        examples=[True, False],
    )
    disable_sink: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=False,
        description="If True, the write is skipped entirely.",
        examples=[False, "$inputs.disable_cosmos_write"],
    )
    cooldown_seconds: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        default=5,
        description="Minimum seconds between consecutive writes. Set to 0 for no cooldown.",
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
            OutputDefinition(name="document_id", kind=[STRING_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class AzureCosmosDBSinkBlockV1(WorkflowBlock):
    def __init__(
        self,
        background_tasks: Optional[BackgroundTasks],
        thread_pool_executor: Optional[ThreadPoolExecutor],
    ):
        self._background_tasks = background_tasks
        self._thread_pool_executor = thread_pool_executor
        self._last_write_time: Optional[datetime] = None

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["background_tasks", "thread_pool_executor"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        account_endpoint: str,
        master_key: str,
        database: str,
        container: str,
        partition_key_value: str,
        document: Union[Dict[str, Any], str],
        document_operations: Dict[str, List[AllOperationsType]],
        operation: Literal["create", "upsert"],
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
                "document_id": None,
            }

        seconds_since_last_write = cooldown_seconds
        if self._last_write_time is not None:
            seconds_since_last_write = (
                datetime.now() - self._last_write_time
            ).total_seconds()

        if seconds_since_last_write < cooldown_seconds:
            logging.info("Activated `roboflow_core/azure_cosmos_db_sink@v1` cooldown.")
            return {
                "error_status": False,
                "throttling_status": True,
                "message": "Sink cooldown applies",
                "document_id": None,
            }

        # Parse document if it's a string
        if isinstance(document, str):
            try:
                document = json.loads(document)
            except json.JSONDecodeError as e:
                return {
                    "error_status": True,
                    "throttling_status": False,
                    "message": f"Invalid JSON document: {str(e)}",
                    "document_id": None,
                }

        # Make a copy to avoid modifying the original
        document = dict(document)

        # Apply document operations if specified
        if document_operations:
            for field_name, operations in document_operations.items():
                if field_name in document and operations:
                    operations_chain = build_operations_chain(operations=operations)
                    document[field_name] = operations_chain(
                        document[field_name], global_parameters={}
                    )

        # Ensure document has an ID
        if "id" not in document:
            document["id"] = str(uuid.uuid4())

        document_id = document["id"]

        write_handler = partial(
            write_to_cosmos_db,
            account_endpoint=account_endpoint,
            master_key=master_key,
            database=database,
            container=container,
            partition_key_value=partition_key_value,
            document=document,
            operation=operation,
            timeout=timeout,
        )

        self._last_write_time = datetime.now()

        if fire_and_forget and self._background_tasks:
            self._background_tasks.add_task(write_handler)
            return {
                "error_status": False,
                "throttling_status": False,
                "message": "Write initiated in background task",
                "document_id": document_id,
            }

        if fire_and_forget and self._thread_pool_executor:
            self._thread_pool_executor.submit(write_handler)
            return {
                "error_status": False,
                "throttling_status": False,
                "message": "Write initiated in background task",
                "document_id": document_id,
            }

        error_status, message = write_handler()
        return {
            "error_status": error_status,
            "throttling_status": False,
            "message": message,
            "document_id": document_id if not error_status else None,
        }


def generate_cosmos_auth_header(
    master_key: str,
    verb: str,
    resource_type: str,
    resource_link: str,
    date_str: str,
) -> str:
    """Generate the Authorization header for Cosmos DB REST API."""
    # Create the string to sign
    # Format: {verb}\n{resourceType}\n{resourceLink}\n{date}\n\n
    text = f"{verb.lower()}\n{resource_type.lower()}\n{resource_link}\n{date_str.lower()}\n\n"

    # Decode the master key from base64
    key = base64.b64decode(master_key)

    # Create HMAC-SHA256 signature
    signature = base64.b64encode(
        hmac.new(key, text.encode("utf-8"), hashlib.sha256).digest()
    ).decode("utf-8")

    # Create the authorization token
    # Format: type={type}&ver={version}&sig={signature}
    auth_token = f"type=master&ver=1.0&sig={signature}"

    # URL encode the token
    return quote(auth_token, safe="")


def write_to_cosmos_db(
    account_endpoint: str,
    master_key: str,
    database: str,
    container: str,
    partition_key_value: str,
    document: Dict[str, Any],
    operation: Literal["create", "upsert"],
    timeout: int,
) -> Tuple[bool, str]:
    try:
        # Ensure endpoint has proper format
        if not account_endpoint.endswith("/"):
            account_endpoint = account_endpoint + "/"

        # Build the resource link and URL
        resource_link = f"dbs/{database}/colls/{container}"
        url = f"{account_endpoint}{resource_link}/docs"

        # Generate date string in RFC 1123 format
        date_str = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")

        # Generate authorization header
        auth_header = generate_cosmos_auth_header(
            master_key=master_key,
            verb="POST",
            resource_type="docs",
            resource_link=resource_link,
            date_str=date_str,
        )

        # Prepare headers
        headers = {
            "Authorization": auth_header,
            "x-ms-date": date_str,
            "x-ms-version": "2018-12-31",
            "Content-Type": "application/json",
            "x-ms-documentdb-partitionkey": json.dumps([partition_key_value]),
        }

        # Add upsert header if operation is upsert
        if operation == "upsert":
            headers["x-ms-documentdb-is-upsert"] = "true"

        # Make the request
        response = requests.post(
            url, headers=headers, json=document, timeout=timeout
        )

        # Check for success (201 Created or 200 OK for upsert)
        if response.status_code in [200, 201]:
            return (
                False,
                f"Successfully wrote document '{document.get('id')}' to {database}/{container}",
            )
        else:
            error_message = response.text
            try:
                error_json = response.json()
                error_message = error_json.get("message", response.text)
            except Exception:
                pass
            logging.warning(
                f"Cosmos DB write failed with status {response.status_code}: {error_message}"
            )
            return True, f"Write failed: {error_message}"

    except requests.exceptions.RequestException as error:
        logging.warning(f"Failed to write to Cosmos DB: {str(error)}")
        return True, f"Write failed: {str(error)}"
    except Exception as error:
        logging.warning(f"Failed to write to Cosmos DB: {str(error)}")
        return True, f"Write failed: {str(error)}"
