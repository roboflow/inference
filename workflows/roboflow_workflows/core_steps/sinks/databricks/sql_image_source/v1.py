"""Databricks SQL Image Source block.

Queries a Databricks SQL warehouse and turns image bytes stored in a table
column into Workflow images, one output element per returned row.
"""

import base64
import datetime
import decimal
import logging
from typing import Any, Dict, List, Literal, Optional, Type, Union
from uuid import uuid4

import cv2
import numpy as np
from pydantic import ConfigDict, Field
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

logger = logging.getLogger(__name__)

SHORT_DESCRIPTION = (
    "Query a Databricks SQL warehouse and turn image bytes stored in a table "
    "column into Workflow images."
)

LONG_DESCRIPTION = """
Run a SQL statement against a Databricks SQL warehouse and emit every returned row
as a Workflow image, so that images living in a Delta table can flow into any
downstream block (models, filters, the **Roboflow Dataset Upload** sink, ...).

## How This Block Works

1. Opens a connection to the SQL warehouse using `server_hostname`, `http_path`
   and a personal access token (`access_token`).
2. Executes `sql_statement` and fetches at most `max_rows` rows.
3. Reads the image payload from `image_column` (a `BINARY` column, or a base64
   string) and decodes it with OpenCV.
4. Emits one element per row:
    * `images` - the decoded image,
    * `image_names` - value of `image_name_column` (or a generated UUID),
    * `metadata` - all other columns of the row as a JSON-friendly dictionary.

Rows whose image cannot be decoded are skipped when `skip_undecodable` is `True`
(default) and raise an error otherwise.

The block takes no image input, so it runs once per Workflow execution and its
outputs are lists with one element per row; downstream blocks that accept
batches receive the rows one by one.

## Requirements

* The `databricks-sql-connector` package must be installed next to the
  Workflows runtime (`pip install "roboflow-workflows[databricks]"`); the
  Roboflow inference server images ship it.
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
        "`adb-1234567890123456.7.azuredatabricks.net`.",
        examples=["dbc-a1b2c3d4-e5f6.cloud.databricks.com", "$inputs.databricks_host"],
    )
    http_path: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="HTTP path of the SQL warehouse, from its Connection details tab.",
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
        le=10_000,
        description="Safety cap on rows fetched per run, applied on top of any LIMIT in the SQL.",
        examples=[100, 1000],
    )
    skip_undecodable: bool = Field(
        default=True,
        description="Skip rows whose image bytes cannot be decoded instead of failing the step.",
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
        rows = fetch_rows(
            sql_statement,
            server_hostname=server_hostname,
            http_path=http_path,
            access_token=access_token,
            max_rows=max_rows,
        )

        results: List[Dict[str, Any]] = []
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
            name = resolve_image_name(row, image_name_column=image_name_column)
            metadata = {
                key: to_json_safe(value)
                for key, value in row.items()
                if key != image_column
            }
            image = WorkflowImageData(
                parent_metadata=ImageParentMetadata(parent_id=f"databricks.{name}"),
                numpy_image=numpy_image,
            )
            results.append({"images": image, "image_names": name, "metadata": metadata})

        return results


def fetch_rows(
    sql_statement: str,
    *,
    server_hostname: str,
    http_path: str,
    access_token: str,
    max_rows: int,
) -> List[Dict[str, Any]]:
    """Execute a statement on a Databricks SQL warehouse and return rows as dicts.

    The ``databricks-sql-connector`` import is deferred so that the Workflows
    runtime loads without the package; the block then fails at run time with
    an actionable message.

    Args:
        sql_statement: SQL statement to execute.
        server_hostname: Databricks workspace host, without scheme.
        http_path: HTTP path of the SQL warehouse.
        access_token: Databricks personal access token.
        max_rows: Maximum number of rows to fetch from the cursor.

    Returns:
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
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(sql_statement)
            fetched = cursor.fetchmany(max_rows)

    rows = [row.asDict() for row in fetched]
    return rows


def decode_image(payload: Any) -> np.ndarray:
    """Decode an image payload read from a table column into a BGR array.

    Args:
        payload: Encoded image as ``bytes``, ``bytearray``, ``memoryview`` or a
            base64 ``str``.

    Returns:
        Decoded image as an OpenCV BGR ``np.ndarray``.

    Raises:
        ValueError: If the payload is ``NULL``, of an unsupported type, or
            cannot be decoded as an image.
    """
    if payload is None:
        raise ValueError("image payload is NULL")

    if isinstance(payload, memoryview):
        payload = payload.tobytes()
    elif isinstance(payload, bytearray):
        payload = bytes(payload)
    elif isinstance(payload, str):
        payload = base64.b64decode(payload)
    if not isinstance(payload, bytes):
        raise ValueError(f"unsupported image payload type: {type(payload).__name__}")

    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("could not decode bytes as image")

    return image


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
        ``bytes``-like values as base64 strings, ``Decimal`` as ``float``,
        dates and times in ISO 8601, containers converted recursively, and all
        other values unchanged.
    """
    if isinstance(value, (bytes, bytearray, memoryview)):
        return base64.b64encode(bytes(value)).decode("ascii")
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, (datetime.date, datetime.datetime, datetime.time)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_safe(v) for v in value]

    return value
