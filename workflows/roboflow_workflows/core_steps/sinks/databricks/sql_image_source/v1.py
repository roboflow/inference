"""Databricks SQL Image Source block.

Queries a Databricks SQL warehouse and turns image bytes stored in a table
column into Workflow images, one output element per returned row.
"""

import base64
import datetime
import decimal
import io
import logging
import re
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple, Type, Union
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image
from pydantic import ConfigDict, Field
from roboflow_workflows.environment import (
    ALLOW_WEBHOOK_WORKFLOWS_SINK_TO_NON_GLOBAL_ADDRESSES,
    GCP_SERVERLESS,
    LAMBDA,
)
from roboflow_workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    OutputDefinition,
    WorkflowImageData,
)
from roboflow_workflows.execution_engine.entities.types import (
    DICTIONARY_KIND,
    IMAGE_KIND,
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
    AirGappedAvailability,
    BlockResult,
    DependentResource,
    WorkflowBlock,
    WorkflowBlockManifest,
)
from roboflow_workflows.utils.url_input import (
    URLAddressNotAllowedError,
    resolve_and_validate_ips,
)

logger = logging.getLogger(__name__)

# Only Databricks-operated workspace domains are accepted as a destination.
# Combined with the connector's TLS certificate verification against the
# hostname, this keeps the outbound request pointed at Databricks even if DNS
# answers change between resolution and connect.
ALLOWED_HOST_SUFFIXES = (
    ".cloud.databricks.com",
    ".gcp.databricks.com",
    ".azuredatabricks.net",
)
HOSTNAME_PATTERN = re.compile(
    r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?(\.[a-z0-9]([a-z0-9-]*[a-z0-9])?)+$"
)
# SQL warehouse (`/sql/1.0/warehouses/<id>`, legacy `/sql/1.0/endpoints/<id>`)
# or all-purpose cluster (`/sql/protocolv1/o/<org>/<cluster-id>`) paths.
HTTP_PATH_PATTERN = re.compile(
    r"^/sql/(1\.0/(warehouses|endpoints)/[A-Za-z0-9]{1,64}"
    r"|protocolv1/o/[0-9]{1,32}/[A-Za-z0-9-]{1,64})$"
)
DATABRICKS_HTTPS_PORT = 443
SOCKET_TIMEOUT_SECONDS = 60
FETCH_BATCH_SIZE = 32
MAX_ROWS = 1_000
MAX_IMAGE_PAYLOAD_BYTES = 32 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
MAX_TOTAL_DECODED_BYTES = 1024 * 1024 * 1024

SHORT_DESCRIPTION = (
    "Query a Databricks SQL warehouse and turn image bytes stored in a table "
    "column into Workflow images."
)

LONG_DESCRIPTION = """
Run a SQL statement against a Databricks SQL warehouse and emit every returned row
as a Workflow image, so that images living in a Delta table can flow into any
downstream block (models, filters, the **Roboflow Dataset Upload** sink, ...).

## How This Block Works

1. Validates the destination: `server_hostname` must be a Databricks workspace
   host (`*.cloud.databricks.com`, `*.gcp.databricks.com`,
   `*.azuredatabricks.net`) and `http_path` a SQL warehouse or cluster path.
2. Opens a connection to the SQL warehouse using the personal access token
   (`access_token`).
3. Executes `sql_statement` and fetches at most `max_rows` rows, in batches.
4. Reads the image payload from `image_column` (a `BINARY` column, or a base64
   string) and decodes it with OpenCV.
5. Emits one element per row:
    * `images` - the decoded image,
    * `image_names` - value of `image_name_column` (or a generated UUID),
    * `metadata` - the remaining non-binary columns of the row as a
      JSON-friendly dictionary.

Rows whose image cannot be decoded, or that exceed the size limits below, are
skipped when `skip_undecodable` is `True` (default) and raise an error otherwise.

The block takes no image input, so it runs once per Workflow execution and its
outputs are lists with one element per row; downstream blocks that accept
batches receive the rows one by one.

## Limits

* at most 1000 rows per run (`max_rows`),
* at most 32 MiB of encoded bytes per image,
* at most 25 megapixels per image (checked from the image header before decoding),
* at most 1 GiB of decoded pixels per run; the step fails once the budget is
  exhausted, so lower `max_rows` or add a `LIMIT` to the query for large images.

## Security

The host is resolved before connecting. On the Roboflow hosted platform the
resolved address must be public; on self-hosted servers the operator's
`ALLOW_WEBHOOK_WORKFLOWS_SINK_TO_NON_GLOBAL_ADDRESSES` setting applies (set it to
`false` to refuse private destinations, for example when the workspace is reached
through Private Link). The connector always verifies the TLS certificate of the
Databricks host.

## Requirements

* The `databricks-sql-connector` package must be installed next to the
  Workflows runtime (`pip install "roboflow-workflows[databricks]"`); every
  Roboflow inference server image ships it.
* The token needs `SELECT` on the queried table and `CAN USE` on the warehouse.
  Store it as a Workflow **input** or **secret** (`$inputs.databricks_token`),
  never inline in the Workflow definition.

## Typical use

**Databricks SQL Image Source -> Roboflow Dataset Upload** (with the upload
block's `image_name` wired to `$steps.<this step>.image_names`) replicates
https://roboflow.com/blog/upload-images-from-databricks-to-roboflow inside a
Workflow. Put a model block in between to pre-label the images.

Example query for a table with an `id` key, a `bytes` image column and two
metadata columns:

```
SELECT id, bytes, raccoon_name, weight
FROM catalog.schema.images
LIMIT 100
```
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Databricks SQL Image Source",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "transformation",
            "ui_manifest": {
                "section": "data_storage",
                "icon": "fal fa-database",
                "blockPriority": 5,
            },
        }
    )
    type: Literal["roboflow_core/databricks_sql_image_source@v1"]
    server_hostname: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="Databricks workspace host without scheme, e.g. "
        "`dbc-a1b2c3d4-e5f6.cloud.databricks.com` or "
        "`adb-1234567890123456.7.azuredatabricks.net`. Only Databricks workspace "
        "domains are accepted.",
        examples=["dbc-a1b2c3d4-e5f6.cloud.databricks.com", "$inputs.databricks_host"],
    )
    http_path: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="HTTP path of the SQL warehouse (`/sql/1.0/warehouses/<id>`) or "
        "cluster (`/sql/protocolv1/o/<org>/<cluster-id>`), from its Connection "
        "details tab.",
        examples=[
            "/sql/1.0/warehouses/abcdef1234567890",
            "$inputs.databricks_http_path",
        ],
    )
    access_token: Union[Selector(kind=[SECRET_KIND, STRING_KIND]), str] = Field(
        description="Databricks personal access token with read access to the table. "
        "Reference a Workflow input or secret rather than pasting the value.",
        examples=["$inputs.databricks_token"],
        json_schema_extra={"private": True},
    )
    sql_statement: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="SELECT statement returning the image column plus any metadata columns.",
        examples=[
            "SELECT id, bytes, raccoon_name, weight FROM catalog.schema.images LIMIT 50",
            "$inputs.sql",
        ],
    )
    image_column: str = Field(
        default="bytes",
        description="Name of the column holding the encoded image (BINARY, or a base64 string).",
        examples=["bytes", "content"],
    )
    image_name_column: Optional[str] = Field(
        default=None,
        description="Optional column whose value becomes the image name (e.g. a primary key). "
        "A UUID is generated when unset.",
        examples=["id", "path"],
    )
    max_rows: int = Field(
        default=100,
        ge=1,
        le=MAX_ROWS,
        description="Cap on rows fetched per run (at most 1000), applied on top of any "
        "LIMIT in the SQL. Decoded pixels are also capped at 1 GiB per run.",
        examples=[100, 500],
    )
    skip_undecodable: bool = Field(
        default=True,
        description="Skip rows whose image bytes cannot be decoded or exceed the per-image "
        "size limits, instead of failing the step.",
        examples=[True, False],
    )

    @classmethod
    def get_output_dimensionality_offset(cls) -> int:
        return 1

    @classmethod
    def get_air_gapped_availability(cls) -> AirGappedAvailability:
        return AirGappedAvailability(available=False, reason="requires_internet")

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="images", kind=[IMAGE_KIND]),
            OutputDefinition(name="image_names", kind=[STRING_KIND]),
            OutputDefinition(name="metadata", kind=[DICTIONARY_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"

    def discover_work_operations(self) -> List[WorkOperation]:
        # One warehouse round-trip, then every row's bytes are decoded.
        return [
            WorkOperation.EXTERNAL_REQUEST,
            WorkOperation.STORAGE_READ,
            WorkOperation.IMAGE_ENCODING,
        ]

    def get_actual_restrictions(
        self, *, ignore_environment_restrictions: bool = False
    ) -> Discovery[RuntimeRestriction]:
        # Reads a remote warehouse only; no local storage, cooldown or
        # environment access is involved.
        return Discovery[RuntimeRestriction](
            items=[], complete=True, unknown_reasons=[]
        )

    def discover_dependent_resources(self) -> List[DependentResource]:
        # Databricks is not a Roboflow platform resource.
        return []


class DatabricksSQLImageSourceBlockV1(WorkflowBlock):
    """Block emitting rows of a Databricks SQL query as Workflow images."""

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        server_hostname: str,
        http_path: str,
        access_token: str,
        sql_statement: str,
        image_column: str,
        image_name_column: Optional[str],
        max_rows: int,
        skip_undecodable: bool,
    ) -> BlockResult:
        validate_destination(server_hostname, http_path=http_path)
        max_rows = min(max_rows, MAX_ROWS)

        results: List[Dict[str, Any]] = []
        decoded_bytes_total = 0
        rows = iter_rows(
            sql_statement,
            server_hostname=server_hostname,
            http_path=http_path,
            access_token=access_token,
            max_rows=max_rows,
        )
        for index, row in enumerate(rows):
            if image_column not in row:
                raise ValueError(
                    f"Column `{image_column}` not present in query result. "
                    f"Available columns: {sorted(row.keys())}"
                )
            try:
                numpy_image = decode_image(row[image_column])
            except Exception as error:
                if skip_undecodable:
                    logger.warning(
                        "roboflow_core/databricks_sql_image_source@v1: skipping row %s - %s",
                        index,
                        error,
                    )
                    continue
                raise ValueError(
                    f"Could not decode image in row {index} column `{image_column}`: {error}"
                ) from error
            decoded_bytes_total += int(numpy_image.nbytes)
            if decoded_bytes_total > MAX_TOTAL_DECODED_BYTES:
                raise ValueError(
                    f"Decoded images exceed the {MAX_TOTAL_DECODED_BYTES} byte budget "
                    f"after {index} rows; lower `max_rows` or add a LIMIT to the query."
                )
            name = resolve_image_name(row, image_name_column=image_name_column)
            metadata = {
                key: to_json_safe(value)
                for key, value in row.items()
                if key != image_column and not _is_binary(value)
            }
            image = WorkflowImageData(
                parent_metadata=ImageParentMetadata(parent_id=f"databricks.{name}"),
                numpy_image=numpy_image,
            )
            results.append({"images": image, "image_names": name, "metadata": metadata})

        return results


def validate_destination(server_hostname: str, *, http_path: str) -> List[str]:
    """Check that the connection target is a Databricks workspace.

    The host must be a bare DNS name (no scheme, port, path or userinfo and no
    IP literal) under one of ``ALLOWED_HOST_SUFFIXES``; the path must be a SQL
    warehouse or cluster HTTP path. The host is then resolved and every
    answer checked: on the Roboflow hosted platform only public addresses are
    accepted, elsewhere the operator's
    ``ALLOW_WEBHOOK_WORKFLOWS_SINK_TO_NON_GLOBAL_ADDRESSES`` setting decides.

    Args:
        server_hostname: Value of the ``server_hostname`` manifest field.
        http_path: Value of the ``http_path`` manifest field.

    Returns:
        The resolved IP addresses of the host.

    Raises:
        ValueError: If the host or path is malformed, points outside Databricks,
            cannot be resolved, or resolves to an address the runtime refuses.
    """
    if not isinstance(server_hostname, str) or not isinstance(http_path, str):
        raise ValueError("`server_hostname` and `http_path` must be strings")

    host = server_hostname.strip().lower().rstrip(".")
    if not host or len(host) > 253 or not HOSTNAME_PATTERN.match(host):
        raise ValueError(
            "`server_hostname` must be a bare Databricks workspace hostname such as "
            "`dbc-a1b2c3d4-e5f6.cloud.databricks.com` (no scheme, port or path)."
        )
    if not host.endswith(ALLOWED_HOST_SUFFIXES):
        raise ValueError(
            f"`server_hostname` `{host}` is not a Databricks workspace host; allowed "
            f"domains: {', '.join('*' + suffix for suffix in ALLOWED_HOST_SUFFIXES)}."
        )
    if not HTTP_PATH_PATTERN.match(http_path):
        raise ValueError(
            "`http_path` must look like `/sql/1.0/warehouses/<id>` or "
            "`/sql/protocolv1/o/<org>/<cluster-id>`."
        )

    allow_non_global = ALLOW_WEBHOOK_WORKFLOWS_SINK_TO_NON_GLOBAL_ADDRESSES and not (
        GCP_SERVERLESS or LAMBDA
    )
    try:
        resolved_ips = resolve_and_validate_ips(
            host=host,
            port=DATABRICKS_HTTPS_PORT,
            allow_non_global_addresses=allow_non_global,
        )
    except URLAddressNotAllowedError as error:
        raise ValueError(
            f"`server_hostname` `{host}` resolves to an address this runtime does "
            f"not allow: {error}"
        ) from error
    except Exception as error:
        raise ValueError(f"Could not resolve `server_hostname` `{host}`.") from error

    return resolved_ips


def iter_rows(
    sql_statement: str,
    *,
    server_hostname: str,
    http_path: str,
    access_token: str,
    max_rows: int,
) -> Iterator[Dict[str, Any]]:
    """Execute a statement on a Databricks SQL warehouse and yield rows as dicts.

    Rows are fetched in batches of ``FETCH_BATCH_SIZE`` so that at most one
    batch of raw payloads is held besides the decoded images. The
    ``databricks-sql-connector`` import is deferred so that the Workflows
    runtime loads without the package; the block then fails at run time with
    an actionable message.

    Args:
        sql_statement: SQL statement to execute.
        server_hostname: Databricks workspace host, without scheme.
        http_path: HTTP path of the SQL warehouse.
        access_token: Databricks personal access token.
        max_rows: Maximum number of rows to yield.

    Yields:
        One dictionary per fetched row, keyed by column name.

    Raises:
        RuntimeError: If ``databricks-sql-connector`` is not installed.
    """
    try:
        from databricks import sql as databricks_sql
    except ImportError as error:
        raise RuntimeError(
            "`roboflow_core/databricks_sql_image_source@v1` requires the "
            "`databricks-sql-connector` package. Install it in the environment running "
            'Workflows (pip install "roboflow-workflows[databricks]" or '
            "pip install databricks-sql-connector)."
        ) from error

    with databricks_sql.connect(
        server_hostname=server_hostname,
        http_path=http_path,
        access_token=access_token,
        _socket_timeout=SOCKET_TIMEOUT_SECONDS,
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(sql_statement)
            yielded = 0
            while yielded < max_rows:
                batch = cursor.fetchmany(min(FETCH_BATCH_SIZE, max_rows - yielded))
                if not batch:
                    break
                for row in batch:
                    yield row.asDict()
                    yielded += 1


def decode_image(payload: Any) -> np.ndarray:
    """Decode an image payload read from a table column into a BGR array.

    The encoded size is capped at ``MAX_IMAGE_PAYLOAD_BYTES`` and the
    dimensions, read from the image header before any pixel is decoded, at
    ``MAX_IMAGE_PIXELS``.

    Args:
        payload: Encoded image as ``bytes``, ``bytearray``, ``memoryview`` or a
            base64 ``str``.

    Returns:
        Decoded image as an OpenCV BGR ``np.ndarray``.

    Raises:
        ValueError: If the payload is ``NULL``, of an unsupported type, over
            the size limits, or cannot be decoded as an image.
    """
    if payload is None:
        raise ValueError("image payload is NULL")

    if isinstance(payload, memoryview):
        payload = payload.tobytes()
    elif isinstance(payload, bytearray):
        payload = bytes(payload)
    elif isinstance(payload, str):
        if len(payload) > MAX_IMAGE_PAYLOAD_BYTES * 4 // 3 + 4:
            raise ValueError(f"image payload exceeds {MAX_IMAGE_PAYLOAD_BYTES} bytes")
        payload = base64.b64decode(payload)
    if not isinstance(payload, bytes):
        raise ValueError(f"unsupported image payload type: {type(payload).__name__}")
    if len(payload) > MAX_IMAGE_PAYLOAD_BYTES:
        raise ValueError(
            f"image payload of {len(payload)} bytes exceeds {MAX_IMAGE_PAYLOAD_BYTES} bytes"
        )

    width, height = read_image_dimensions(payload)
    if width * height > MAX_IMAGE_PIXELS:
        raise ValueError(
            f"image of {width}x{height} pixels exceeds {MAX_IMAGE_PIXELS} pixels"
        )

    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("could not decode bytes as image")

    return image


def read_image_dimensions(payload: bytes) -> Tuple[int, int]:
    """Read ``(width, height)`` from an encoded image header without decoding pixels.

    Args:
        payload: Encoded image bytes.

    Returns:
        Tuple of width and height in pixels.

    Raises:
        ValueError: If the bytes are not a recognised image.
    """
    try:
        with Image.open(io.BytesIO(payload)) as image:
            width, height = image.size
    except Exception as error:
        raise ValueError("could not read image header") from error

    return width, height


def resolve_image_name(row: Dict[str, Any], *, image_name_column: Optional[str]) -> str:
    """Pick the image name for a row.

    Args:
        row: Row returned by the query, keyed by column name.
        image_name_column: Column whose value becomes the name, or ``None``.

    Returns:
        The stringified column value when present, otherwise a fresh UUID.
    """
    if image_name_column and row.get(image_name_column) is not None:
        return str(row[image_name_column])

    generated_name = str(uuid4())
    return generated_name


def to_json_safe(value: Any) -> Any:
    """Convert connector-returned values into JSON-serialisable equivalents.

    Args:
        value: Any value returned by the SQL connector.

    Returns:
        ``Decimal`` as ``float``, dates and times in ISO 8601, containers
        converted recursively with binary members dropped, and all other
        values unchanged.
    """
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, (datetime.date, datetime.datetime, datetime.time)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): to_json_safe(v) for k, v in value.items() if not _is_binary(v)}
    if isinstance(value, (list, tuple)):
        return [to_json_safe(v) for v in value if not _is_binary(v)]

    return value


def _is_binary(value: Any) -> bool:
    return isinstance(value, (bytes, bytearray, memoryview))
