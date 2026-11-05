"""Databricks SQL Image Source block.

Queries a Databricks SQL warehouse and turns image bytes stored in a table
column into Workflow images, one output element per returned row.
"""

import base64
import contextlib
import datetime
import decimal
import io
import json
import logging
import re
import threading
import time
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple, Type, Union
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image
from pydantic import ConfigDict, Field, field_validator
from roboflow_workflows.core_steps.integrations.databricks.availability import (
    CONNECTOR_INSTALL_HINT,
    DISABLED_HINT,
    block_enabled,
    connector_available,
    databricks_unavailability_reason,
    running_on_hosted_platform,
)
from roboflow_workflows.core_steps.integrations.databricks.sql_image_source.transport import (
    PinnedDestination,
    install_pinned_transport,
    pinned_destination,
)
from roboflow_workflows.environment import (
    ALLOW_DATABRICKS_WORKFLOWS_SOURCE_TO_NON_GLOBAL_ADDRESSES,
    DATABRICKS_SQL_IMAGE_SOURCE_MAX_IMAGE_BYTES,
    DATABRICKS_SQL_IMAGE_SOURCE_MAX_IMAGE_PIXELS,
    DATABRICKS_SQL_IMAGE_SOURCE_MAX_RUN_BYTES,
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
    Runtime,
    RuntimeRestriction,
    Severity,
    WorkOperation,
)
from roboflow_workflows.prototypes.block import (
    AirGappedAvailability,
    BlockResult,
    DependentResource,
    WorkflowBlock,
    WorkflowBlockManifest,
    actual_restrictions_of,
    is_workflow_selector,
)
from roboflow_workflows.utils.url_input import (
    URLAddressNotAllowedError,
    address_is_global,
    resolve_and_validate_ips,
)

logger = logging.getLogger(__name__)

# Only Databricks-operated workspace domains are accepted as a destination. The
# connector then connects to the validated IP address with the hostname kept
# for TLS SNI, certificate verification and the Host header (see transport.py).
ALLOWED_HOST_SUFFIXES = (
    ".cloud.databricks.com",
    ".gcp.databricks.com",
    ".azuredatabricks.net",
)
HOSTNAME_PATTERN = re.compile(
    r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?(\.[a-z0-9]([a-z0-9-]*[a-z0-9])?)+$"
)
# SQL warehouse (`/sql/1.0/warehouses/<id>`, legacy `/sql/1.0/endpoints/<id>`)
# or all-purpose cluster (`/sql/protocolv1/o/<org>/<cluster-id>`) paths. The
# cluster form runs customer code on the receiving end and is refused on the
# hosted platform.
WAREHOUSE_PATH_PATTERN = re.compile(
    r"/sql/1\.0/(warehouses|endpoints)/[A-Za-z0-9]{1,64}"
)
CLUSTER_PATH_PATTERN = re.compile(r"/sql/protocolv1/o/[0-9]{1,32}/[A-Za-z0-9-]{1,64}")
DATABRICKS_HTTPS_PORT = 443
# Deadline model. `timeout_seconds` is the overall deadline for connecting,
# executing / polling, fetching and decoding. When it expires the watchdog
# cancels the statement and closes the connection; the blocked request in the
# step's thread then fails within one socket timeout plus one bounded retry
# delay, which is the cleanup interval after which the step has returned and
# released its execution slot.
SOCKET_TIMEOUT_SECONDS = 30
RETRY_ATTEMPTS = 3
RETRY_DELAY_MAX_SECONDS = 5
CLEANUP_GRACE_SECONDS = SOCKET_TIMEOUT_SECONDS + RETRY_DELAY_MAX_SECONDS
# Connector buffering. `arraysize` rows / `buffer_size_bytes` bytes bound the
# inline result chunk the warehouse returns per fetch, so the connector holds at
# most one chunk (plus its lz4-decompressed copy) ahead of the row loop.
FETCH_BATCH_SIZE = 8
MIN_RESULT_BUFFER_BYTES = 1024 * 1024
MAX_ROWS = 1_000
MAX_TIMEOUT_SECONDS = 600
ALLOWED_IMAGE_FORMATS = ("JPEG", "PNG", "WEBP")
# Budgets apply per block invocation (not per process): the operator sets them
# against worker memory and concurrency, see the block documentation.
MAX_IMAGE_PAYLOAD_BYTES = DATABRICKS_SQL_IMAGE_SOURCE_MAX_IMAGE_BYTES
MAX_IMAGE_PIXELS = DATABRICKS_SQL_IMAGE_SOURCE_MAX_IMAGE_PIXELS
MAX_RUN_BYTES = DATABRICKS_SQL_IMAGE_SOURCE_MAX_RUN_BYTES
MAX_IMAGE_NAME_CHARS = 255
MAX_METADATA_VALUE_CHARS = 4_096
MAX_METADATA_BYTES_PER_ROW = 64 * 1024
MAX_METADATA_DEPTH = 4
MAX_METADATA_ITEMS = 256
IMAGE_CHANNELS = 3
# CA bundle handed to the connector (``None`` = system trust store). A hook for
# the real-transport test harness and for self-hosted servers behind a private
# CA; never relaxes verification.
TLS_TRUSTED_CA_FILE: Optional[str] = None

HOSTED_PLATFORM_RESTRICTION = RuntimeRestriction(
    code="unavailable_on_hosted_platform",
    severity=Severity.HARD,
    note=(
        "The Databricks SQL Image Source is disabled by default on the Roboflow "
        "hosted serverless platform: the operator enables it with "
        "DATABRICKS_SQL_IMAGE_SOURCE_ENABLED=true (and removes `databricks` from "
        "WORKFLOW_DISABLED_BLOCK_PATTERNS where set)."
    ),
    applies_to_runtimes=[Runtime.HOSTED_SERVERLESS],
)
VIDEO_PIPELINE_RESTRICTION = RuntimeRestriction(
    code="queries_warehouse_on_every_run",
    severity=Severity.SOFT,
    note=(
        "Each run executes the SQL statement again. In a video pipeline that is "
        "one warehouse query per frame; gate the step or run it from a "
        "scheduled Workflow instead."
    ),
    applies_to_runtimes=[Runtime.INFERENCE_PIPELINE],
)

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
   `*.azuredatabricks.net`) and `http_path` a SQL warehouse path (cluster paths
   are accepted on self-hosted servers only). The host is resolved and every
   address must be public unless the operator allows private destinations.
2. Connects to the validated address (the hostname is kept for TLS and the
   `Host` header, so a later DNS change cannot redirect the connection), with
   cloud-fetch result links, telemetry and HTTP redirects disabled.
3. Executes `sql_statement` with a statement timeout and fetches at most
   `max_rows` rows in small batches, under an overall deadline
   (`timeout_seconds`) that cancels the query when it expires.
4. Reads the image payload from `image_column` (a `BINARY` column, or a base64
   string): JPEG, PNG or WEBP only, checked from the header before decoding.
5. Emits one element per row:
    * `images` - the decoded image,
    * `image_names` - value of `image_name_column` (or a generated UUID),
    * `metadata` - the remaining non-binary columns of the row as a
      JSON-friendly dictionary.

Rows whose image cannot be decoded, or that exceed the per-image limits below,
are skipped when `skip_undecodable` is `True` (default) and raise an error
otherwise. If every fetched row is skipped the step fails rather than returning
an empty result.

Each invocation of the step runs the statement once; with a batch-valued
`sql_statement` it runs once per element.

## Limits

Budgets are per block invocation; the operator sizes them against worker memory
and concurrency (defaults in brackets, environment variables in the
Availability section):

* at most 1000 rows per run (`max_rows`), fetched in batches of 8 with the
  connector's inline result chunk bounded to the same rows and to twice the
  per-image byte limit, so the connector never buffers more than one chunk ahead,
* JPEG, PNG or WEBP only; the format and the dimensions are read from the header
  before any pixel is decoded [32 MiB encoded, 25 megapixels per image], and the
  decoded array is checked again,
* one byte budget per run covering encoded bytes, decoded pixels, names and
  metadata, reserved before each image is decoded [1 GiB]; the step fails (never
  skips) when it is exhausted,
* metadata strings are truncated to 4096 characters, nested values to 4 levels
  and 256 items, one row's metadata to 64 KiB, names to 255 characters,
* the whole run is bounded by `timeout_seconds` (default 120 s, at most 600 s):
  connection, execution and polling, fetching and decoding. The statement also
  carries a server-side `STATEMENT_TIMEOUT`, HTTP retries are bounded (3
  attempts, 5 s maximum delay, within the remaining time) and redirects are
  refused. On expiry the statement is cancelled and the connection closed; the
  step returns within one socket timeout (30 s) plus one retry delay after the
  deadline, which is when its execution slot is released.

## Availability and security

* `DATABRICKS_SQL_IMAGE_SOURCE_ENABLED` is the rollout switch. Unset, it is
  `true` on self-hosted servers and `false` on the Roboflow hosted platform
  (`GCP_SERVERLESS` / `LAMBDA`); hosted workers set it explicitly. When it is
  `false` the block is not listed by `/workflows/blocks/describe`, a Workflow
  that references it (saved or submitted directly) is rejected at compile time
  with this reason, and no Databricks connection is attempted. The value is
  read at process start: changing it is a redeployment or restart, not a live
  kill switch. After a rollout, queued jobs that reference the block fail at
  compile time; jobs already running finish or hit their `timeout_seconds`
  deadline (at most 600 s plus the cleanup interval above).
* The connector is an optional extra, installed separately from the switch:
  `pip install "roboflow-workflows[databricks]"`. The Roboflow server images
  built from `docker/` and `inference_server/docker/` ship it. Without it the
  block is likewise unlisted and rejected with an install hint.
* Budgets: `DATABRICKS_SQL_IMAGE_SOURCE_MAX_RUN_BYTES`,
  `DATABRICKS_SQL_IMAGE_SOURCE_MAX_IMAGE_BYTES` and
  `DATABRICKS_SQL_IMAGE_SOURCE_MAX_IMAGE_PIXELS` override the defaults above.
  Size the run budget so that concurrent invocations times (run budget + 2 x
  image bytes of connector buffer + one image's decode temporaries) fit in the
  worker's free memory.
* `WORKFLOW_DISABLED_BLOCK_PATTERNS=databricks` remains a valid operational
  safeguard on multi-tenant workers; the explicit switch is the documented
  rollout control.
* Only Databricks workspace hosts are accepted and the connection is pinned to
  the address validated before connecting (hostname kept for TLS verification
  and the `Host` header), with cloud-fetch result links, telemetry, feature-flag
  lookups and proxies disabled for the duration of the run. Hosts that resolve
  to private or link-local addresses are refused unless
  `ALLOW_DATABRICKS_WORKFLOWS_SOURCE_TO_NON_GLOBAL_ADDRESSES=true` (for example
  Private Link on a self-hosted server; never on the hosted platform).
* The SQL statement runs with whatever rights the token has. Use a token with
  `SELECT` on the table and `CAN USE` on the warehouse only, and store it as a
  Workflow **input** or **secret** (`$inputs.databricks_token`) rather than a
  literal in a shared Workflow definition.

## Typical use

**Databricks SQL Image Source -> Roboflow Dataset Upload** (with the upload
block's `image_name` wired to `$steps.<this step>.image_names`) replicates
https://blog.roboflow.com/upload-images-from-databricks-to-roboflow inside a
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
        description="HTTP path of the SQL warehouse (`/sql/1.0/warehouses/<id>`), from "
        "its Connection details tab. Cluster paths "
        "(`/sql/protocolv1/o/<org>/<cluster-id>`) work on self-hosted servers only.",
        examples=[
            "/sql/1.0/warehouses/abcdef1234567890",
            "$inputs.databricks_http_path",
        ],
    )
    access_token: Union[Selector(kind=[SECRET_KIND, STRING_KIND]), str] = Field(
        description="Databricks personal access token with read access to the table. "
        "Reference it from a Workflow input or secret rather than saving a literal "
        "in a shared Workflow. The statement runs with this token's rights, so use a "
        "read-only token.",
        examples=["$inputs.databricks_token"],
        json_schema_extra={"private": True},
    )
    sql_statement: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="SELECT statement returning the image column plus any metadata "
        "columns. The statement is executed as given, with the token's rights.",
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
        description="Optional column whose value becomes the image name (e.g. a primary key); "
        "it must exist in the result. A UUID is generated when unset or NULL.",
        examples=["id", "path"],
    )
    max_rows: int = Field(
        default=100,
        ge=1,
        le=MAX_ROWS,
        description="Cap on rows fetched per run (at most 1000), applied on top of any "
        "LIMIT in the SQL. The 1 GiB per-run budget is the practical bound for large "
        "images.",
        examples=[100, 500],
    )
    timeout_seconds: int = Field(
        default=120,
        ge=5,
        le=MAX_TIMEOUT_SECONDS,
        description="Overall deadline for the run: connecting, query execution and "
        "polling, fetching and decoding. The statement is cancelled and the "
        "connection closed when it expires.",
        examples=[120, 300],
    )
    skip_undecodable: bool = Field(
        default=True,
        description="Skip rows whose image bytes cannot be decoded or exceed the per-image "
        "limits, instead of failing the step. The step still fails when every row is "
        "skipped.",
        examples=[True, False],
    )

    @field_validator("server_hostname")
    @classmethod
    def _literal_hostname_is_a_databricks_host(cls, value: Any) -> Any:
        if isinstance(value, str) and not is_workflow_selector(value):
            normalise_hostname(value)
        return value

    @field_validator("http_path")
    @classmethod
    def _literal_path_is_a_sql_path(cls, value: Any) -> Any:
        if isinstance(value, str) and not is_workflow_selector(value):
            classify_http_path(value)
        return value

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

    @classmethod
    def get_restrictions(cls) -> List[RuntimeRestriction]:
        restrictions = [VIDEO_PIPELINE_RESTRICTION]
        if not block_enabled():
            restrictions.insert(0, HOSTED_PLATFORM_RESTRICTION)
        return restrictions

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
        restrictions = actual_restrictions_of(
            declared=self.get_restrictions(),
            node_id=f"$steps.{getattr(self, 'name', '')}",
            ignore_environment_restrictions=ignore_environment_restrictions,
        )
        return restrictions

    def discover_dependent_resources(self) -> List[DependentResource]:
        # Databricks is not a Roboflow platform resource.
        return []


class DatabricksSQLImageSourceBlockV1(WorkflowBlock):
    """Block emitting rows of a Databricks SQL query as Workflow images."""

    def __init__(self, disable_sinks: bool = False):
        self._disable_sinks = disable_sinks

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["disable_sinks"]

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
        timeout_seconds: int,
        skip_undecodable: bool,
    ) -> BlockResult:
        ensure_block_enabled()
        if self._disable_sinks:
            raise RuntimeError(
                "`roboflow_core/databricks_sql_image_source@v1` is disabled by the "
                "server's execution policy (disable_sinks): the statement would run "
                "with the token's rights, which the policy forbids."
            )
        from roboflow_workflows.core_steps.integrations.databricks.sql_image_source.isolation import (
            run_isolated,
        )

        deadline = time.monotonic() + float(timeout_seconds)
        request = dict(
            server_hostname=server_hostname,
            http_path=http_path,
            access_token=access_token,
            sql_statement=sql_statement,
            image_column=image_column,
            image_name_column=image_name_column,
            max_rows=max_rows,
            timeout_seconds=timeout_seconds,
            skip_undecodable=skip_undecodable,
        )
        limits = dict(
            max_run_bytes=MAX_RUN_BYTES,
            max_image_bytes=MAX_IMAGE_PAYLOAD_BYTES,
            max_image_pixels=MAX_IMAGE_PIXELS,
            allow_non_global=ALLOW_DATABRICKS_WORKFLOWS_SOURCE_TO_NON_GLOBAL_ADDRESSES,
            hosted=running_on_hosted_platform(),
        )
        results = run_isolated(request, limits=limits, deadline=deadline)
        return results

    def _run_in_process(
        self,
        server_hostname: str,
        http_path: str,
        access_token: str,
        sql_statement: str,
        image_column: str,
        image_name_column: Optional[str],
        max_rows: int,
        timeout_seconds: int,
        skip_undecodable: bool,
        *,
        deadline: Optional[float] = None,
    ) -> BlockResult:
        # Internal worker entry point, also used by focused connector tests.
        ensure_block_enabled()
        if self._disable_sinks:
            raise RuntimeError("Databricks execution is disabled by disable_sinks")
        if deadline is None:
            deadline = time.monotonic() + float(timeout_seconds)
        destination = validate_destination(server_hostname, http_path=http_path)
        _check_deadline(deadline)

        results: List[Dict[str, Any]] = []
        budget = RunBudget(limit=MAX_RUN_BYTES)
        fetched = 0
        skipped: List[str] = []
        rows = iter_rows(
            sql_statement,
            destination=destination,
            http_path=http_path,
            access_token=access_token,
            max_rows=max_rows,
            deadline=deadline,
        )
        with contextlib.closing(rows):
            for index, row in enumerate(rows):
                _check_deadline(deadline)
                fetched += 1
                if image_column not in row:
                    raise ValueError(
                        f"Column `{image_column}` not present in query result. "
                        f"Available columns: {sorted(row.keys())}"
                    )
                if image_name_column and image_name_column not in row:
                    raise ValueError(
                        f"Column `{image_name_column}` (image_name_column) not present "
                        f"in query result. Available columns: {sorted(row.keys())}"
                    )
                try:
                    name = resolve_image_name(row, image_name_column=image_name_column)
                    metadata = build_metadata(row, image_column=image_column)
                    budget.reserve(len(name) + metadata_size(metadata))
                    numpy_image = decode_image(row[image_column], budget=budget)
                except Exception as error:
                    if not skip_undecodable or isinstance(error, RunBudgetExceeded):
                        raise ValueError(
                            f"Could not decode image in row {index} column "
                            f"`{image_column}`: {error}"
                        ) from error
                    skipped.append(f"row {index}: {error}")
                    logger.warning(
                        "roboflow_core/databricks_sql_image_source@v1: skipping %s",
                        skipped[-1],
                    )
                    continue
                image = WorkflowImageData(
                    parent_metadata=ImageParentMetadata(parent_id=f"databricks.{name}"),
                    numpy_image=numpy_image,
                )
                results.append(
                    {"images": image, "image_names": name, "metadata": metadata}
                )

        _check_deadline(deadline)
        if fetched and not results:
            raise ValueError(
                f"All {fetched} fetched rows were skipped; last reason: {skipped[-1]}"
            )
        return results


# ---------------------------------------------------------------------------
# Availability and destination
# ---------------------------------------------------------------------------


def _check_deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise TimeoutError("Databricks query exceeded timeout_seconds")


def ensure_block_enabled() -> None:
    """Refuse to run where the block is not available.

    Raises:
        RuntimeError: If ``DATABRICKS_SQL_IMAGE_SOURCE_ENABLED`` is false or the
            connector package is missing. Raised before any destination lookup
            or connection attempt.
    """
    if not block_enabled():
        raise RuntimeError(
            "`roboflow_core/databricks_sql_image_source@v1` cannot run here: "
            + DISABLED_HINT
        )
    if not connector_available():
        raise RuntimeError(
            "`roboflow_core/databricks_sql_image_source@v1` requires the "
            "`databricks-sql-connector` package: " + CONNECTOR_INSTALL_HINT
        )

    reason = databricks_unavailability_reason()
    if reason is not None:
        raise RuntimeError("Databricks SQL Image Source cannot run here: " + reason)


def normalise_hostname(server_hostname: Any) -> str:
    """Return the lower-cased hostname, or raise if it is not a Databricks host.

    Args:
        server_hostname: Raw manifest value.

    Returns:
        The normalised hostname (lower case, no trailing dot). Whitespace is
        not tolerated: the value is validated as a whole string.

    Raises:
        ValueError: If the value is not a bare DNS name under one of
            ``ALLOWED_HOST_SUFFIXES``.
    """
    if not isinstance(server_hostname, str):
        raise ValueError("`server_hostname` must be a string")

    host = server_hostname.lower().rstrip(".")
    if not host or len(host) > 253 or not HOSTNAME_PATTERN.fullmatch(host):
        raise ValueError(
            "`server_hostname` must be a bare Databricks workspace hostname such as "
            "`dbc-a1b2c3d4-e5f6.cloud.databricks.com` (no scheme, port or path)."
        )
    if not host.endswith(ALLOWED_HOST_SUFFIXES):
        raise ValueError(
            f"`server_hostname` `{host}` is not a Databricks workspace host; allowed "
            f"domains: {', '.join('*' + suffix for suffix in ALLOWED_HOST_SUFFIXES)}."
        )

    return host


def classify_http_path(http_path: Any) -> str:
    """Return ``"warehouse"`` or ``"cluster"`` for a valid HTTP path.

    Args:
        http_path: Raw manifest value.

    Returns:
        The path kind.

    Raises:
        ValueError: If the value is not a SQL warehouse or cluster path.
    """
    if not isinstance(http_path, str):
        raise ValueError("`http_path` must be a string")
    if WAREHOUSE_PATH_PATTERN.fullmatch(http_path):
        return "warehouse"
    if CLUSTER_PATH_PATTERN.fullmatch(http_path):
        return "cluster"

    raise ValueError(
        "`http_path` must look like `/sql/1.0/warehouses/<id>` or "
        "`/sql/protocolv1/o/<org>/<cluster-id>`."
    )


def validate_destination(server_hostname: Any, *, http_path: Any) -> PinnedDestination:
    """Check the connection target and resolve it to a pinned address.

    The host must be a bare DNS name under one of ``ALLOWED_HOST_SUFFIXES``;
    the path must be a SQL warehouse path (cluster paths are refused on the
    hosted platform). The host is resolved and every answer checked: non-global
    addresses are refused unless
    ``ALLOW_DATABRICKS_WORKFLOWS_SOURCE_TO_NON_GLOBAL_ADDRESSES`` is set, and
    never allowed on the hosted platform. The first validated address is the
    one the transport connects to.

    Args:
        server_hostname: Value of the ``server_hostname`` manifest field.
        http_path: Value of the ``http_path`` manifest field.

    Returns:
        The validated hostname and address the connector must connect to.

    Raises:
        ValueError: If the host or path is malformed, points outside Databricks,
            cannot be resolved, or resolves to an address the runtime refuses.
    """
    host = normalise_hostname(server_hostname)
    path_kind = classify_http_path(http_path)
    hosted = running_on_hosted_platform()
    if path_kind == "cluster" and hosted:
        raise ValueError(
            "`http_path` cluster paths (`/sql/protocolv1/...`) are not allowed on the "
            "hosted platform; use a SQL warehouse path."
        )

    allow_non_global = (
        ALLOW_DATABRICKS_WORKFLOWS_SOURCE_TO_NON_GLOBAL_ADDRESSES and not hosted
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

    pinned_ip = _pick_address(resolved_ips, allow_non_global=allow_non_global)
    destination = PinnedDestination(
        hostname=host, ip_address=pinned_ip, port=DATABRICKS_HTTPS_PORT
    )
    return destination


def _pick_address(resolved_ips: List[str], *, allow_non_global: bool) -> str:
    for candidate in resolved_ips:
        if allow_non_global or address_is_global(candidate):
            return candidate
    raise ValueError("no resolved address is allowed by the runtime's egress policy")


# ---------------------------------------------------------------------------
# Query execution
# ---------------------------------------------------------------------------


def connection_kwargs(
    destination: PinnedDestination, *, http_path: str, timeout_seconds: float
) -> Dict[str, Any]:
    """Keyword arguments for ``databricks.sql.connect``.

    Args:
        destination: Validated destination.
        http_path: SQL warehouse or cluster HTTP path.
        timeout_seconds: Remaining run time; bounds sockets and the statement.

    Returns:
        Keyword arguments that disable cloud-fetch result links, telemetry and
        redirects, bound retries (attempts, delay and total duration) and
        sockets to the remaining time, and set a server-side statement timeout.
    """
    statement_timeout = max(1, int(timeout_seconds))
    kwargs = {
        "server_hostname": destination.hostname,
        "http_path": http_path,
        "_port": destination.port,
        "use_cloud_fetch": False,
        "enable_telemetry": False,
        "session_configuration": {"STATEMENT_TIMEOUT": str(statement_timeout)},
        "_socket_timeout": min(SOCKET_TIMEOUT_SECONDS, statement_timeout),
        "_retry_stop_after_attempts_count": RETRY_ATTEMPTS,
        "_retry_stop_after_attempts_duration": statement_timeout,
        "_retry_delay_max": RETRY_DELAY_MAX_SECONDS,
        "_retry_max_redirects": 0,
    }
    if TLS_TRUSTED_CA_FILE is not None:
        kwargs["_tls_trusted_ca_file"] = TLS_TRUSTED_CA_FILE
    return kwargs


def result_buffer_bytes() -> int:
    """Return the inline result chunk size requested from the warehouse.

    Two maximum-size images always fit, so a batch of ``FETCH_BATCH_SIZE`` rows
    is only ever smaller than that, never larger.
    """
    return max(MIN_RESULT_BUFFER_BYTES, 2 * MAX_IMAGE_PAYLOAD_BYTES)


def iter_rows(
    sql_statement: str,
    *,
    destination: PinnedDestination,
    http_path: str,
    access_token: str,
    max_rows: int,
    deadline: float,
) -> Iterator[Dict[str, Any]]:
    """Execute a statement on the pinned warehouse and yield rows as dicts.

    Rows are fetched in batches of ``FETCH_BATCH_SIZE`` and the cursor asks the
    warehouse for inline result chunks of at most that many rows and
    :func:`result_buffer_bytes` bytes, with a server-side row limit of
    ``max_rows``. A watchdog cancels the cursor and closes the connection when
    ``deadline`` passes, and the loop raises ``TimeoutError`` at the next batch
    boundary. Closing the generator early (``contextlib.closing``) closes the
    cursor and the connection.

    Args:
        sql_statement: SQL statement to execute.
        destination: Validated hostname and pinned address.
        http_path: SQL warehouse or cluster HTTP path.
        access_token: Databricks personal access token.
        max_rows: Maximum number of rows to yield.
        deadline: ``time.monotonic()`` value after which the run is aborted.

    Yields:
        One dictionary per fetched row, keyed by column name.

    Raises:
        RuntimeError: If the connector is missing or fails; the token is
            redacted from the message.
        TimeoutError: If ``deadline`` passes.
    """
    try:
        from databricks import sql as databricks_sql
    except ImportError as error:
        raise RuntimeError(
            "`roboflow_core/databricks_sql_image_source@v1` requires the "
            "`databricks-sql-connector` package: " + CONNECTOR_INSTALL_HINT
        ) from error

    install_pinned_transport()
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Databricks query deadline expired before connecting")
    kwargs = connection_kwargs(
        destination, http_path=http_path, timeout_seconds=remaining
    )
    kwargs["access_token"] = access_token
    expired = threading.Event()

    def _redacted(error: BaseException) -> RuntimeError:
        text = str(error).replace(access_token, "***") if access_token else str(error)
        return RuntimeError(f"Databricks query failed: {type(error).__name__}: {text}")

    with pinned_destination(destination):
        try:
            connection = databricks_sql.connect(**kwargs)
        except Exception as error:
            raise _redacted(error) from None
        with connection:
            with connection.cursor(
                arraysize=FETCH_BATCH_SIZE,
                buffer_size_bytes=result_buffer_bytes(),
                row_limit=max_rows,
            ) as cursor:
                watchdog = threading.Timer(
                    max(0.0, deadline - time.monotonic()),
                    _cancel_query,
                    args=(cursor, connection, expired),
                )
                watchdog.daemon = True
                watchdog.start()
                try:
                    try:
                        cursor.execute(sql_statement)
                    except Exception as error:
                        if expired.is_set():
                            raise TimeoutError(
                                "Databricks query exceeded timeout_seconds and was cancelled"
                            ) from None
                        raise _redacted(error) from None
                    yielded = 0
                    while yielded < max_rows:
                        if expired.is_set() or time.monotonic() > deadline:
                            raise TimeoutError(
                                "Databricks query exceeded timeout_seconds and was cancelled"
                            )
                        try:
                            batch = cursor.fetchmany(
                                min(FETCH_BATCH_SIZE, max_rows - yielded)
                            )
                        except Exception as error:
                            if expired.is_set():
                                raise TimeoutError(
                                    "Databricks query exceeded timeout_seconds and was "
                                    "cancelled"
                                ) from None
                            raise _redacted(error) from None
                        if not batch:
                            break
                        for row in batch:
                            yield row.asDict()
                            yielded += 1
                finally:
                    watchdog.cancel()
                    watchdog.join(timeout=0.1)


def _cancel_query(cursor: Any, connection: Any, expired: threading.Event) -> None:
    expired.set()
    for action in (getattr(cursor, "cancel", None), getattr(connection, "close", None)):
        if action is None:
            continue
        try:
            action()
        except Exception:  # best effort; the loop reports the timeout
            logger.debug("databricks watchdog cleanup failed", exc_info=True)


# ---------------------------------------------------------------------------
# Decoding and budget
# ---------------------------------------------------------------------------


class RunBudgetExceeded(ValueError):
    """Raised when a run would exceed ``MAX_RUN_BYTES``."""


class RunBudget:
    """Byte budget shared by every row of one block invocation.

    Encoded payloads, decoded pixels (reserved from the image header before
    any pixel is decoded), names and metadata all count against it. It bounds
    what the step retains and hands downstream; it is not a process-wide RSS
    guarantee (connector buffers and decode temporaries are bounded
    separately, see ``result_buffer_bytes`` and the block documentation).
    """

    def __init__(self, limit: int):
        self.limit = limit
        self.used = 0

    def reserve(self, nbytes: int) -> None:
        if self.used + nbytes > self.limit:
            raise RunBudgetExceeded(
                f"run would exceed the {self.limit} byte budget "
                f"({self.used} used, {nbytes} more requested); lower `max_rows` or "
                "add a LIMIT to the query"
            )
        self.used += nbytes


def decode_image(payload: Any, *, budget: Optional[RunBudget] = None) -> np.ndarray:
    """Decode an image payload read from a table column into a BGR array.

    The encoded size is capped at ``MAX_IMAGE_PAYLOAD_BYTES``, the format must
    be one of ``ALLOWED_IMAGE_FORMATS`` and the dimensions, read from the header
    before any pixel is decoded, are capped at ``MAX_IMAGE_PIXELS``; the decoded
    array is checked again after decoding.

    Args:
        payload: Encoded image as ``bytes``, ``bytearray``, ``memoryview`` or a
            base64 ``str``.
        budget: Optional per-run budget; encoded and decoded bytes are reserved
            from it before decoding.

    Returns:
        Decoded image as an OpenCV BGR ``np.ndarray``.

    Raises:
        ValueError: If the payload is ``NULL``, of an unsupported type or
            format, over the size limits, or cannot be decoded.
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

    image_format, width, height = read_image_header(payload)
    if image_format not in ALLOWED_IMAGE_FORMATS:
        raise ValueError(
            f"image format {image_format} is not allowed; use one of "
            f"{', '.join(ALLOWED_IMAGE_FORMATS)}"
        )
    pixels = width * height
    if pixels > MAX_IMAGE_PIXELS:
        raise ValueError(
            f"image of {width}x{height} pixels exceeds {MAX_IMAGE_PIXELS} pixels"
        )
    if budget is not None:
        budget.reserve(len(payload) + pixels * IMAGE_CHANNELS)

    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("could not decode bytes as image")
    decoded_pixels = int(image.shape[0]) * int(image.shape[1])
    if decoded_pixels > MAX_IMAGE_PIXELS or decoded_pixels > pixels:
        raise ValueError(
            f"decoded image of {image.shape[1]}x{image.shape[0]} pixels does not "
            f"match its header ({width}x{height})"
        )

    return image


def read_image_header(payload: bytes) -> Tuple[str, int, int]:
    """Read ``(format, width, height)`` from an encoded image without decoding pixels.

    Args:
        payload: Encoded image bytes.

    Returns:
        Pillow's format name and the dimensions in pixels.

    Raises:
        ValueError: If the bytes are not a recognised image.
    """
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image_format = str(image.format or "")
            width, height = image.size
    except Exception as error:
        raise ValueError("could not read image header") from error

    return image_format, int(width), int(height)


def resolve_image_name(row: Dict[str, Any], *, image_name_column: Optional[str]) -> str:
    """Pick the image name for a row.

    Args:
        row: Row returned by the query, keyed by column name.
        image_name_column: Column whose value becomes the name, or ``None``.

    Returns:
        The stringified column value when present, otherwise a fresh UUID.

    Raises:
        ValueError: If the name is longer than ``MAX_IMAGE_NAME_CHARS``.
    """
    if image_name_column and row.get(image_name_column) is not None:
        name = str(row[image_name_column])
        if len(name) > MAX_IMAGE_NAME_CHARS:
            raise ValueError(
                f"image name of {len(name)} characters exceeds {MAX_IMAGE_NAME_CHARS}"
            )
        return name

    generated_name = str(uuid4())
    return generated_name


def build_metadata(row: Dict[str, Any], *, image_column: str) -> Dict[str, Any]:
    """Build the per-row metadata: every non-binary column except the image.

    Args:
        row: Row returned by the query, keyed by column name.
        image_column: Name of the image column to leave out.

    Returns:
        JSON-serialisable dictionary; strings are truncated to
        ``MAX_METADATA_VALUE_CHARS`` and nested values capped (see
        :func:`to_json_safe`).

    Raises:
        ValueError: If the row's metadata exceeds ``MAX_METADATA_BYTES_PER_ROW``.
    """
    metadata = {
        key: to_json_safe(value)
        for key, value in row.items()
        if key != image_column and not _is_binary(value)
    }
    size = metadata_size(metadata)
    if size > MAX_METADATA_BYTES_PER_ROW:
        raise ValueError(
            f"row metadata of {size} bytes exceeds {MAX_METADATA_BYTES_PER_ROW} bytes"
        )
    return metadata


def metadata_size(metadata: Dict[str, Any]) -> int:
    """Return the serialised size of ``metadata`` in bytes."""
    return len(json.dumps(metadata, ensure_ascii=False).encode("utf-8"))


def to_json_safe(value: Any, *, depth: int = 0) -> Any:
    """Convert connector-returned values into JSON-serialisable equivalents.

    Args:
        value: Any value returned by the SQL connector.
        depth: Nesting level of ``value``; containers deeper than
            ``MAX_METADATA_DEPTH`` are stringified.

    Returns:
        ``Decimal`` as an exact string, dates and times in ISO 8601, strings
        truncated to ``MAX_METADATA_VALUE_CHARS``, containers converted
        recursively (binary members dropped, at most ``MAX_METADATA_ITEMS``
        members, ``MAX_METADATA_DEPTH`` levels), scalars unchanged and any
        other type as ``str(value)``.
    """
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return (
            value
            if len(value) <= MAX_METADATA_VALUE_CHARS
            else value[:MAX_METADATA_VALUE_CHARS]
        )
    if isinstance(value, decimal.Decimal):
        return str(value)
    if isinstance(value, (datetime.date, datetime.datetime, datetime.time)):
        return value.isoformat()
    if isinstance(value, dict):
        if depth >= MAX_METADATA_DEPTH:
            return str(value)[:MAX_METADATA_VALUE_CHARS]
        items = [(k, v) for k, v in value.items() if not _is_binary(v)]
        return {
            str(k): to_json_safe(v, depth=depth + 1)
            for k, v in items[:MAX_METADATA_ITEMS]
        }
    if isinstance(value, (list, tuple)):
        if depth >= MAX_METADATA_DEPTH:
            return str(value)[:MAX_METADATA_VALUE_CHARS]
        items = [v for v in value if not _is_binary(v)]
        return [to_json_safe(v, depth=depth + 1) for v in items[:MAX_METADATA_ITEMS]]

    return str(value)[:MAX_METADATA_VALUE_CHARS]


def _is_binary(value: Any) -> bool:
    return isinstance(value, (bytes, bytearray, memoryview))
