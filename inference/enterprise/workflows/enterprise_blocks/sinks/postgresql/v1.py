import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Type, Union

from fastapi import BackgroundTasks
from pydantic import ConfigDict, Field, field_validator

try:
    import psycopg
    from psycopg import sql
except ImportError:
    psycopg = None
    sql = None

from inference.core.workflows.core_steps.sinks.noop import disabled_sink_response
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
    Runtime,
    RuntimeRestriction,
    Severity,
    WorkflowBlock,
    WorkflowBlockManifest,
)

logger = logging.getLogger(__name__)
SSL_MODES = {"disable", "allow", "prefer", "require", "verify-ca", "verify-full"}
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

Set `fire_and_forget=false` to observe commit success or failure, especially when
streaming. Background mode returns scheduling status, not persistence confirmation;
failures are logged and work can be lost on shutdown. There are no automatic retries,
upserts, or exactly-once guarantees. A connection failure during commit can leave the
outcome unknown. Connection and statement timeouts do not impose a total network deadline.

Requires the Psycopg binary driver included in the Inference runtime. Self-hosted
servers can enable enterprise blocks with `LOAD_ENTERPRISE_BLOCKS=True`. Cloud execution requires a compatible worker image, block registration, and database
connectivity; deploying this block does not automatically update hosted workers.
"""


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
        Selector(kind=[DICTIONARY_KIND]), Dict[str, Any], List[Dict[str, Any]]
    ] = Field(
        description="One row or a non-empty list of rows with identical column names"
    )
    fire_and_forget: Union[Selector(kind=[BOOLEAN_KIND]), bool] = Field(
        default=True, description="Schedule the insert without waiting for commit"
    )
    sslmode: Union[
        Selector(kind=[STRING_KIND]),
        Literal["disable", "allow", "prefer", "require", "verify-ca", "verify-full"],
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
                severity=Severity.HARD,
                note="PostgreSQL Sink is not available in hosted serverless execution.",
                applies_to_runtimes=[Runtime.HOSTED_SERVERLESS],
            ),
            RuntimeRestriction(
                severity=Severity.SOFT,
                note="Writes fail unless the worker can reach and authenticate to the database; background status only confirms scheduling.",
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
        task = partial(
            self._process_data,
            host=host,
            database=database,
            username=username,
            password=password,
            port=port,
            schema_name=schema_name,
            table_name=table_name,
            data=data,
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
        data: Union[Dict[str, Any], List[Dict[str, Any]]],
        sslmode: str,
        connect_timeout: int,
        statement_timeout: int,
    ) -> Dict[str, Any]:
        try:
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
            if not isinstance(sslmode, str) or sslmode not in SSL_MODES:
                raise ValueError("Unsupported sslmode")
        except ValueError as error:
            return failure(str(error))
        if psycopg is None:
            return failure(
                'PostgreSQL driver unavailable. Install "psycopg[binary]>=3.2,<4" in the runtime.'
            )
        try:
            with psycopg.connect(
                host=host,
                port=port,
                dbname=database,
                user=username,
                password=password,
                sslmode=sslmode,
                connect_timeout=connect_timeout,
                autocommit=False,
            ) as connection:
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
            return {
                "error_status": False,
                "message": f"Successfully inserted {len(rows)} records",
            }
        except psycopg.Error as error:
            # Server diagnostics can include row contents; do not expose them.
            code = error.sqlstate
            suffix = (
                f" (SQLSTATE {code})"
                if code and len(code) == 5 and code.isalnum()
                else ""
            )
            return failure(
                f"PostgreSQL operation failed{suffix}. Check connectivity, permissions and column types; commit outcome may be unknown."
            )
        except Exception:
            return failure("Unexpected PostgreSQL insert failure")


def failure(message: str) -> Dict[str, Any]:
    logger.error("PostgreSQL Sink: %s", message)
    return {"error_status": True, "message": message}


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
