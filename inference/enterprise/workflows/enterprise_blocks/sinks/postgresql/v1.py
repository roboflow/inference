import ipaddress
import logging
import socket
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Set, Type, Union, get_args

import requests
from fastapi import BackgroundTasks
from pydantic import ConfigDict, Field, TypeAdapter, ValidationError, field_validator

try:
    import psycopg
    from psycopg import sql
except ImportError:
    psycopg = None
    sql = None

from inference.core.env import (
    ALLOW_POSTGRESQL_WORKFLOWS_SINK_TO_NON_GLOBAL_ADDRESSES,
    GCP_SERVERLESS,
    LAMBDA,
    POSTGRESQL_WORKFLOWS_SINK_BLACKLISTED_ADDRESSES,
)
from inference.core.workflows.core_steps.sinks.noop import disabled_sink_response
from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    DICTIONARY_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    SECRET_KIND,
    STRING_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    Runtime,
    RuntimeRestriction,
    Severity,
    WorkflowBlock,
    WorkflowBlockManifest,
)

logger = logging.getLogger(__name__)
SSLMode = Literal["disable", "allow", "prefer", "require", "verify-ca", "verify-full"]
BOOLEAN_ADAPTER = TypeAdapter(bool)
LONG_DESCRIPTION = """
Insert workflow data into an existing PostgreSQL table. Supply one dictionary or a
non-empty list of dictionaries with the same column names. All rows are committed
in one transaction; a failed insert rolls back the batch. Values use native driver
adaptation, including `None` for SQL NULL. Convert nested objects to the desired
representation upstream; the block does not automatically serialize JSON.

Use separate `schema_name` and `table_name` fields, without SQL quoting. For
`analytics.detections`, set schema to `analytics` and table to `detections`.
Names are case-sensitive and safely quoted. Tables are not created automatically.
The database account needs the appropriate schema, INSERT and sequence permissions.

The runtime must have network access to the database. Use a secret input for the
password. An omitted password uses authentication configured in the runtime/libpq.
TLS defaults to `require`; use `verify-full` and configure libpq's trusted root
certificate to verify the server identity. A local non-TLS database requires
explicit `sslmode=disable`.

This sink is not available on the Roboflow hosted platform: when the runtime is
hosted serverless (`GCP_SERVERLESS` or `LAMBDA`) every call fails without opening a
connection.

On self-hosted runtimes the operator may restrict which destinations the sink can
reach. When the server sets `ALLOW_POSTGRESQL_WORKFLOWS_SINK_TO_NON_GLOBAL_ADDRESSES=False`, a
host that resolves to a non-global address (loopback, private, link-local/metadata,
CGNAT, ULA) is rejected and the connection is pinned to the validated public IP so a
second DNS lookup cannot rebind it; Unix-socket paths and multi-host lists are
rejected. The server may also set `POSTGRESQL_WORKFLOWS_SINK_BLACKLISTED_ADDRESSES`
(comma-separated IPs or hostnames, empty by default) to deny specific destinations
regardless of the non-global setting; the raw host and every resolved IP are checked
against it. The default on self-hosted runtimes is permissive (any destination).

Set `fire_and_forget=false` to observe commit success or failure, especially when
streaming. Background mode returns scheduling status, not persistence confirmation;
failures are logged and the queued work is dropped on pipeline shutdown. There are
no automatic retries, upserts, or exactly-once guarantees. A connection failure
during commit can leave the outcome unknown. `connect_timeout` and `statement_timeout`
also bound client-side socket inactivity (via libpq `tcp_user_timeout`), so a silently
dropped connection fails within roughly `connect_timeout + statement_timeout` instead
of hanging until TCP keepalive fires.

Requires the Psycopg binary driver included in the Inference runtime. Self-hosted
servers can enable enterprise blocks with `LOAD_ENTERPRISE_BLOCKS=True`. The block
runs on self-hosted CPU/GPU servers and self-managed streaming pipelines; it is not
available on the Roboflow hosted platform (see above).

## Example

Create the destination table before running the workflow:

```sql
CREATE TABLE public.detections (
    camera_id text NOT NULL,
    label text NOT NULL,
    confidence double precision
);
```

Use the following workflow, supplying `database_host`, `database_password`, and
`row` at execution time through the runtime's supported input/secret delivery path.
Do not save a real password as a literal in a shared workflow.

```json
{
  "version": "1.0",
  "inputs": [
    {"type": "WorkflowParameter", "name": "database_host"},
    {"type": "WorkflowParameter", "name": "database_password"},
    {"type": "WorkflowParameter", "name": "row"}
  ],
  "steps": [{
    "type": "roboflow_core/postgresql_sink@v1",
    "name": "save_detection",
    "host": "$inputs.database_host",
    "port": 5432,
    "database": "production",
    "username": "workflow_writer",
    "password": "$inputs.database_password",
    "schema_name": "public",
    "table_name": "detections",
    "data": "$inputs.row",
    "sslmode": "verify-full",
    "fire_and_forget": false
  }],
  "outputs": [
    {"type": "JsonField", "name": "error_status", "selector": "$steps.save_detection.error_status"},
    {"type": "JsonField", "name": "message", "selector": "$steps.save_detection.message"}
  ]
}
```

For example, `row` can be `{"camera_id": "line-1", "label": "part", "confidence": 0.95}`.
In an inference workflow, wire `data` to an upstream dictionary or list-of-values
output containing rows.
A list of rows must be non-empty and every row must have the same column names.
Column order within a dictionary does not matter. Values retain their native driver
representation; `None` becomes SQL NULL. Nested objects are not automatically
converted to JSON.
"""


# Local copy of the SSRF address primitives from
# ``inference.core.utils.url_input`` (``address_is_global`` and
# ``resolve_and_validate_ips``) so this enterprise sink owns its
# connection-boundary logic without importing from core utils. The util's
# ``URLAddressNotAllowedError`` is renamed here to ``SinkAddressNotAllowedError``
# to avoid two different classes sharing one name. Keep the copied bodies in sync
# with the source. The denylist is layered on top in ``denylisted_reason``
# rather than baked into the copy.
class SinkAddressNotAllowedError(Exception):
    """Raised when a sink host resolves to a destination that is not permitted."""


def address_is_global(address: str) -> bool:
    """Return True only for public, routable unicast addresses.

    ``ipaddress.is_global`` already excludes loopback, private (RFC1918),
    link-local (incl. 169.254.169.254 metadata), CGNAT (100.64/10), ULA
    (fc00::/7), unspecified and reserved ranges, so a single check covers the
    destinations the advisory asks us to block. IPv4-mapped IPv6 is unwrapped so
    ``::ffff:127.0.0.1`` cannot smuggle a loopback target past the check.
    """
    try:
        parsed = ipaddress.ip_address(address)
    except ValueError:
        return False
    if isinstance(parsed, ipaddress.IPv6Address) and parsed.ipv4_mapped is not None:
        parsed = parsed.ipv4_mapped
    return parsed.is_global


def resolve_and_validate_ips(
    host: str,
    port: int,
    allow_non_global_addresses: bool,
) -> List[str]:
    """Resolve ``host`` and, unless non-global is allowed, require every
    resolved IP to be global. Returns the resolved IPs (validated ones first
    would be identical since all must pass).

    Rejecting when *any* resolved address is non-global is deliberately
    conservative: it prevents a rebinding-style response that mixes a global and
    a non-global A-record from later steering the pinned connection to the
    non-global one.
    """
    try:
        addr_infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as error:
        # Unresolvable host is a normal connection failure, not an SSRF block.
        raise requests.exceptions.ConnectionError(
            f"Could not resolve host: {host}"
        ) from error
    resolved_ips = [info[4][0] for info in addr_infos]
    if not resolved_ips:
        raise requests.exceptions.ConnectionError(f"Could not resolve host: {host}")
    if not allow_non_global_addresses:
        for ip in resolved_ips:
            if not address_is_global(ip):
                raise SinkAddressNotAllowedError(
                    f"Host '{host}' resolves to non-global address '{ip}'."
                )
    return resolved_ips


def denylisted_reason(
    host: str,
    resolved_ips: List[str],
    blacklisted_addresses: Optional[Set[str]],
) -> Optional[str]:
    """Denylist screen layered on top of :func:`resolve_and_validate_ips` (kept
    separate from the copied resolver). Returns a failure reason when the raw
    host or any resolved IP is denylisted, else None."""
    if not blacklisted_addresses:
        return None
    if host in blacklisted_addresses:
        return "host is blocked by the sink address denylist"
    for ip in resolved_ips:
        if ip in blacklisted_addresses:
            return f"host '{host}' resolves to a denylisted address '{ip}'"
    return None


def validate_integer(value: Any, name: str, maximum: int) -> None:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer between 1 and {maximum}")


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "PostgreSQL Sink",
            "version": "v1",
            "short_description": "Save data to a PostgreSQL database.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "sink",
            "ui_manifest": {
                "section": "data_storage",
                "icon": "fal fa-database",
                "blockPriority": 3,
                "enterprise_only": True,
            },
        }
    )
    type: Literal["roboflow_core/postgresql_sink@v1"]
    host: Union[Selector(kind=[STRING_KIND]), str] = Field(description="Database host")
    database: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="Database name"
    )
    username: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="Database user"
    )
    password: Optional[Union[Selector(kind=[SECRET_KIND]), str]] = Field(
        default=None, description="Database password", examples=["$inputs.pg_password"]
    )
    port: Union[Selector(kind=[INTEGER_KIND]), int] = Field(
        default=5432, description="Database port"
    )
    schema_name: Union[Selector(kind=[STRING_KIND]), str] = Field(
        default="public", description="Raw schema name, without SQL quoting"
    )
    table_name: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="Raw table name, without schema prefix or SQL quoting"
    )
    data: Union[
        Selector(kind=[DICTIONARY_KIND, LIST_OF_VALUES_KIND]),
        Dict[str, Any],
        List[Dict[str, Any]],
    ] = Field(
        description="One row or a non-empty list of rows with identical column names"
    )
    fire_and_forget: Union[Selector(kind=[BOOLEAN_KIND]), bool] = Field(
        default=True, description="Schedule the insert without waiting for commit"
    )
    sslmode: Union[
        Selector(kind=[STRING_KIND]),
        SSLMode,
    ] = Field(default="require", description="PostgreSQL TLS mode")
    connect_timeout: Union[Selector(kind=[INTEGER_KIND]), int] = Field(
        default=10, description="Connection timeout in seconds"
    )
    statement_timeout: Union[Selector(kind=[INTEGER_KIND]), int] = Field(
        default=10000, description="Server statement timeout in milliseconds"
    )

    @field_validator("port", "connect_timeout", "statement_timeout", mode="before")
    @classmethod
    def validate_numeric_setting(cls, value: Any, info: Any) -> Any:
        if not (isinstance(value, str) and value.startswith("$")):
            maximum = 65535 if info.field_name == "port" else 2147483647
            validate_integer(value, info.field_name, maximum)
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
        return [
            RuntimeRestriction(
                severity=Severity.SOFT,
                note="Use fire_and_forget=false to observe persistence failures and avoid accumulating background writes when the database is slower than the stream.",
                applies_to_runtimes=[Runtime.INFERENCE_PIPELINE],
            ),
        ]


class PostgreSQLSinkBlockV1(WorkflowBlock):
    def __init__(
        self,
        background_tasks: Optional[BackgroundTasks],
        thread_pool_executor: Optional[ThreadPoolExecutor],
        disable_sinks: bool = False,
    ):
        self._background_tasks = background_tasks
        self._thread_pool_executor = thread_pool_executor
        self._disable_sinks = disable_sinks

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["background_tasks", "thread_pool_executor", "disable_sinks"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        host: str,
        database: str,
        username: str,
        table_name: str,
        data: Union[Dict[str, Any], List[Dict[str, Any]]],
        password: Optional[str] = None,
        port: int = 5432,
        schema_name: str = "public",
        fire_and_forget: bool = True,
        sslmode: str = "require",
        connect_timeout: int = 10,
        statement_timeout: int = 10000,
    ) -> BlockResult:
        if self._disable_sinks:
            return disabled_sink_response()
        if GCP_SERVERLESS or LAMBDA:
            # The sink is not available on the Roboflow hosted platform: it must
            # never open an outbound database connection from hosted workers.
            return failure(
                "PostgreSQL sink is not available on the Roboflow hosted platform"
            )
        try:
            fire_and_forget = BOOLEAN_ADAPTER.validate_python(fire_and_forget)
        except ValidationError:
            return failure("fire_and_forget must be a boolean")
        # Validate every input synchronously, before scheduling, so a bad
        # request is reported to the caller instead of silently failing inside a
        # background task that has already returned "scheduled".
        try:
            rows = validate_inputs(
                host=host,
                database=database,
                username=username,
                schema_name=schema_name,
                table_name=table_name,
                data=data,
                port=port,
                sslmode=sslmode,
                connect_timeout=connect_timeout,
                statement_timeout=statement_timeout,
            )
        except ValueError as error:
            return failure(str(error))
        if psycopg is None:
            return failure(
                'PostgreSQL driver unavailable. Install "psycopg[binary]>=3.2,<4" in the runtime.'
            )
        task = partial(
            self._process_data,
            host=host,
            database=database,
            username=username,
            password=password,
            port=port,
            schema_name=schema_name,
            table_name=table_name,
            rows=rows,
            sslmode=sslmode,
            connect_timeout=connect_timeout,
            statement_timeout=statement_timeout,
        )
        if fire_and_forget and self._background_tasks is not None:
            self._background_tasks.add_task(task)
        elif fire_and_forget and self._thread_pool_executor is not None:
            self._thread_pool_executor.submit(task)
        else:
            return task()
        return {"error_status": False, "message": "Data processing scheduled"}

    def _process_data(
        self,
        host: str,
        database: str,
        username: str,
        password: Optional[str],
        port: int,
        schema_name: str,
        table_name: str,
        rows: List[Dict[str, Any]],
        sslmode: str,
        connect_timeout: int,
        statement_timeout: int,
    ) -> Dict[str, Any]:
        # Connection boundary gate. When a policy is active (non-global blocked
        # or a denylist configured) resolve the host, enforce the policy, then
        # pin the socket to the validated IP (host is kept for TLS SNI / cert
        # verification) so a second DNS lookup cannot rebind the connection to an
        # internal target.
        hostaddr: Optional[str] = None
        if connection_policy_active():
            try:
                resolved_ips = resolve_and_validate_ips(
                    host=host,
                    port=port,
                    allow_non_global_addresses=ALLOW_POSTGRESQL_WORKFLOWS_SINK_TO_NON_GLOBAL_ADDRESSES,
                )
            except SinkAddressNotAllowedError as error:
                return failure(str(error))
            except Exception:
                return failure(f"Could not resolve host: {host}")
            reason = denylisted_reason(
                host, resolved_ips, POSTGRESQL_WORKFLOWS_SINK_BLACKLISTED_ADDRESSES
            )
            if reason is not None:
                return failure(reason)
            hostaddr = resolved_ips[0]
        connect_kwargs: Dict[str, Any] = dict(
            host=host,
            port=port,
            dbname=database,
            user=username,
            password=password,
            sslmode=sslmode,
            connect_timeout=connect_timeout,
            # Server-side statement_timeout is useless against a peer that never
            # replies; tcp_user_timeout bounds client-side socket inactivity so a
            # silently dropped connection fails fast instead of pinning a worker
            # until TCP keepalive fires (~2h on Linux defaults).
            tcp_user_timeout=connect_timeout * 1000 + statement_timeout,
            autocommit=False,
        )
        if hostaddr is not None:
            connect_kwargs["hostaddr"] = hostaddr
        phase = "connection"
        try:
            with psycopg.connect(**connect_kwargs) as connection:
                phase = "insert"
                with connection.cursor() as cursor:
                    cursor.execute(
                        "SELECT set_config('statement_timeout', %s, true)",
                        (str(statement_timeout),),
                    )
                    columns = list(rows[0])
                    query = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
                        sql.Identifier(schema_name, table_name),
                        sql.SQL(", ").join(map(sql.Identifier, columns)),
                        sql.SQL(", ").join(sql.Placeholder() for _ in columns),
                    )
                    cursor.executemany(
                        query,
                        [tuple(row[column] for column in columns) for row in rows],
                    )
                phase = "commit"
            return {
                "error_status": False,
                "message": f"Successfully inserted {len(rows)} records",
            }
        except psycopg.Error as error:
            # Server diagnostics can include row contents; do not expose them.
            code = error.sqlstate
            suffix = (
                f" (SQLSTATE {code})"
                if code and len(code) == 5 and code.isascii() and code.isalnum()
                else ""
            )
            category = next(
                (
                    name
                    for name in (
                        "IntegrityError",
                        "DataError",
                        "ProgrammingError",
                        "OperationalError",
                        "InterfaceError",
                        "NotSupportedError",
                        "InternalError",
                    )
                    if isinstance(error, getattr(psycopg, name))
                ),
                "DatabaseError",
            )
            uncertainty = (
                " Commit outcome may be unknown; verify before retrying."
                if phase == "commit"
                and isinstance(
                    error, (psycopg.OperationalError, psycopg.InterfaceError)
                )
                and (not code or code.startswith("08"))
                else ""
            )
            return failure(
                f"PostgreSQL {phase} failed: {category}{suffix}.{uncertainty}"
            )
        except Exception:
            return failure("Unexpected PostgreSQL insert failure")


def failure(message: str) -> Dict[str, Any]:
    logger.error("PostgreSQL Sink: %s", message)
    return {"error_status": True, "message": message}


def validate_inputs(
    host: Any,
    database: Any,
    username: Any,
    schema_name: Any,
    table_name: Any,
    data: Any,
    port: Any,
    sslmode: Any,
    connect_timeout: Any,
    statement_timeout: Any,
) -> List[Dict[str, Any]]:
    """Validate every runtime input up front. Raises ValueError on the first
    problem; returns the normalized list of rows. Network-time checks (host
    resolution / non-global gate) happen later, in _process_data."""
    rows = validate_rows(data)
    for name, value in (
        ("host", host),
        ("database", database),
        ("username", username),
        ("schema_name", schema_name),
        ("table_name", table_name),
    ):
        if not isinstance(value, str) or not value or "\x00" in value:
            raise ValueError(
                f"{name} must be a non-empty string without NUL characters"
            )
    validate_integer(port, "port", 65535)
    validate_integer(connect_timeout, "connect_timeout", 2147483647)
    validate_integer(statement_timeout, "statement_timeout", 2147483647)
    if not isinstance(sslmode, str) or sslmode not in get_args(SSLMode):
        raise ValueError("Unsupported sslmode")
    if connection_policy_active():
        # With a policy active only a single TCP hostname/IP is accepted; a
        # Unix-socket path or a comma-separated multi-host list cannot be safely
        # IP-gated or denylisted.
        if host.startswith("/") or "," in host:
            raise ValueError(
                "host must be a single global TCP hostname or IP address"
            )
    return rows


def connection_policy_active() -> bool:
    """True when the PostgreSQL sink must resolve and screen the destination:
    either non-global addresses are blocked, or a denylist is configured."""
    return (
        not ALLOW_POSTGRESQL_WORKFLOWS_SINK_TO_NON_GLOBAL_ADDRESSES
        or bool(POSTGRESQL_WORKFLOWS_SINK_BLACKLISTED_ADDRESSES)
    )


def validate_rows(data: Any) -> List[Dict[str, Any]]:
    rows = [data] if isinstance(data, dict) else data
    if not isinstance(rows, list) or not rows:
        raise ValueError("Data must be a non-empty dictionary or list of dictionaries")
    if any(not isinstance(row, dict) or not row for row in rows):
        raise ValueError("Each row must be a non-empty dictionary")
    columns = set(rows[0])
    if any(
        not isinstance(column, str) or not column or "\x00" in column
        for column in columns
    ):
        raise ValueError(
            "Column names must be non-empty strings without NUL characters"
        )
    if any(set(row) != columns for row in rows):
        raise ValueError("All rows must have the same column names")
    return rows
