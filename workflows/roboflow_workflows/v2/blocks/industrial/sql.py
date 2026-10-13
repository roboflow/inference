"""PostgreSQL and Microsoft SQL Server sinks: insert workflow rows into a table.

Ports of V1 ``roboflow_core/postgresql_sink@v1`` and
``roboflow_core/microsoft_sql_server_sink@v1``::

    disabled by the host?        ─▶ error_status=False, V1's disabled message
    invalid rows or settings?    ─▶ error_status=True; nothing connects
    rows ─▶ SQLInsert(statement, rows)   names quoted, values stay parameters
    password ─▶ Secret; unavailable (e.g. replayed) ─▶ the step fails
    fire_and_forget and sink_effects ─▶ background, else written now
        └─▶ client.insert(target, insert): one transaction; commit, or roll back

Clients. The database work is an ``SQLClient``: ``insert(target, insert)``
writes all rows in one transaction and commits, or rolls back and raises.
A host may pass its own client as the ``postgresql_client`` or
``sql_server_client`` resource, e.g. one that keeps a connection open::

    host owns                          block borrows
    ─────────                          ─────────────
    client = MyPooledClient(...)
    with BackgroundEffects() as effects:
        with plan.create_session(resources={
            "postgresql_client": client, "sink_effects": effects}) as session:
            ...                        client.insert(...) per write; never close()
    client.close()                     # after runs stop and effects drain

Without an injected client, a block uses its default client
(``PsycopgClient`` / ``PyodbcClient``). It opens one connection per write and
closes it in ``finally``; with background effects this whole lifetime stays
inside the effect. Nothing is pooled or kept open between writes, so every
write pays a connection setup; reusing a connection is the injected path.
The drivers are imported at first use, never when the catalogue loads.

Failures and retries. A failed write returns ``error_status=True`` and is
never retried: after a failed commit the rows may already be stored, and the
message then says the outcome may be unknown. In the background the step only
reports ``"Data processing scheduled"``; the result reaches the host's
``BackgroundEffects.failures()``. Messages are scrubbed of the password.
PostgreSQL messages also never carry server text (it can hold row values), as
in V1; SQL Server messages keep the driver text, as in V1.

Differences from V1 (both blocks):

* Background writes need the host's ``sink_effects``; without it a
  fire-and-forget write runs now (V1 also did without a background runner).
* The password is a ``Secret``; one without a value fails the step.
* A parameter of the wrong type or out of range fails compilation, or, when it
  comes from a selector, fails the step (V2 validates selector values like
  literals). V1 returned ``error_status=True`` for these. Row data and names
  are still checked by ``run`` and reported with V1's messages.
* Inline failures are not logged; V1 logged each PostgreSQL failure at ERROR.
  Background failures are logged at WARNING by ``BackgroundEffects``.

PostgreSQL only:

* Destination screening (hosted platform aside) belongs to the default
  client; an injected client applies its own connection policy.
* Screening uses ``roboflow_workflows.utils.url_input``, which also rejects
  multicast addresses (V1's private copy did not).
* A ``%`` in a name is doubled for psycopg's ``%s`` parameter style.

SQL Server only:

* Names are quoted with brackets (V1 put raw text into the statement). The
  table name may be qualified (up to 4 dot-separated parts, each plain,
  ``[bracketed]`` or ``"quoted"``); each column name is one identifier.
* A connection-string value containing ``;``, ``{`` or ``}`` or edge spaces is
  enclosed in braces (V1 inserted it raw).
* Rows are checked before connecting, with the PostgreSQL rules and messages;
  V1 checked after connecting and reported "An unexpected error occurred".
* A missing driver is reported before scheduling, not inside the background.
* A failed commit adds the outcome-unknown note; a failed rollback no longer
  replaces the original error.
* ``data`` also accepts list selectors. ``port`` from a selector may be an int
  or a digit string; a literal ``port`` must be an int.

Not verified: psycopg and pyodbc themselves, FreeTDS's handling of braced
values and of ``?`` inside bracketed names, and any real database. Tests use
stand-in clients and stand-in driver modules.
"""

import copy
import importlib
import re
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Literal,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
)

from pydantic import Field, StrictBool, StrictInt, StrictStr, field_validator
from roboflow_workflows import environment
from roboflow_workflows.execution_engine.entities.workload import (
    Discovery,
    DiscoveryProblem,
    DiscoveryProblemCode,
    Runtime,
    RuntimeRestriction,
    Severity,
    WorkOperation,
    incomplete_discovery,
)
from roboflow_workflows.v2.blocks.sinks.runtime import (
    BACKGROUND_MESSAGE,
    ERROR_STATUS_OUTPUT,
    MESSAGE_OUTPUT,
    BackgroundEffects,
    EffectOutcome,
    deliver,
    disabled_sink_message,
)
from roboflow_workflows.v2.data.kinds import (
    BOOLEAN_KIND,
    DICTIONARY_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
)
from roboflow_workflows.v2.data.secrets import (
    SECRET_KIND,
    Revealed,
    Secret,
    reveal_secrets,
    to_secret,
)
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Ref

POSTGRESQL_BLOCK_TYPE = "v2/postgresql_sink@v1"
SQL_SERVER_BLOCK_TYPE = "v2/microsoft_sql_server_sink@v1"

SCHEDULED_MESSAGE = "Data processing scheduled"
INSERTED_MESSAGE = "Successfully inserted {} records"
UNKNOWN_COMMIT_NOTE = " Commit outcome may be unknown; verify before retrying."
POSTGRESQL_DRIVER_MESSAGE = (
    'PostgreSQL driver unavailable. Install "psycopg[binary]>=3.2,<4" in the runtime.'
)
SQL_SERVER_DRIVER_MESSAGE = (
    "pyodbc package is not installed. Please contact Roboflow's Enterprise "
    "support team for assistance."
)
HOSTED_PLATFORM_MESSAGE = (
    "PostgreSQL sink is not available on the Roboflow hosted platform"
)
POSTGRESQL_UNEXPECTED_MESSAGE = "Unexpected PostgreSQL insert failure"
TABLE_NAME_MESSAGE = (
    "table_name must be a table name of up to 4 dot-separated parts, each a "
    'plain, [bracketed] or "quoted" identifier'
)

SSLMode = Literal["disable", "allow", "prefer", "require", "verify-ca", "verify-full"]
MAX_PORT = 65535
MAX_INT32 = 2147483647
SET_STATEMENT_TIMEOUT = "SELECT set_config('statement_timeout', %s, true)"
SQL_SERVER_SESSION_OPTIONS = (
    "SET ANSI_NULLS ON",
    "SET ANSI_PADDING ON",
    "SET ANSI_WARNINGS ON",
    "SET ARITHABORT ON",
    "SET QUOTED_IDENTIFIER ON",
)
PSYCOPG_CATEGORIES = (
    "IntegrityError",
    "DataError",
    "ProgrammingError",
    "OperationalError",
    "InterfaceError",
    "NotSupportedError",
    "InternalError",
)

HOSTED_PLATFORM_RESTRICTION = RuntimeRestriction(
    code="unavailable_on_hosted_platform",
    severity=Severity.HARD,
    note=(
        "On the Roboflow hosted platform the sink writes nothing and opens no "
        "PostgreSQL connection; an enabled sink returns `error_status=true` on "
        "every run."
    ),
    applies_to_runtimes=[Runtime.HOSTED_SERVERLESS],
)
FIRE_AND_FORGET_RESTRICTION = RuntimeRestriction(
    code="fire_and_forget_hides_persistence_failures",
    severity=Severity.SOFT,
    note=(
        "Use fire_and_forget=false to observe persistence failures and avoid "
        "accumulating background writes when the database is slower than the "
        "stream."
    ),
    applies_to_runtimes=[Runtime.INFERENCE_PIPELINE],
)

_SQL_SERVER_NAME_PART = re.compile(
    r"\[(?:[^\]]|\]\])*\]"  # [bracketed], "]]" escapes "]"
    r'|"(?:[^"]|"")*"'  # "quoted", '""' escapes '"'
    r"|(?:[^\W\d]|[@#])[\w@#$]*"  # plain identifier
)


class SQLSinkError(Exception):
    """A write failure whose message may be reported as the step's message.

    Clients raise it with text that holds no credential; the block still
    scrubs the password from it. Any other exception from a client is
    reported with a generic message.
    """


@dataclass(frozen=True)
class SQLInsert:
    """One parameterized insert of a batch of rows.

    Args:
        statement: SQL text in the driver's parameter style: quoted names and
            one placeholder per column, never a value.
        rows: Parameter values, one tuple per row in column order. A detached
            copy: later changes to workflow payloads do not reach it.
    """

    statement: str
    rows: Tuple[Tuple[Any, ...], ...]


@dataclass(frozen=True)
class PostgreSQLTarget:
    """Where and how a PostgreSQL write connects.

    Args:
        host: Host name or IP address.
        port: TCP port.
        database: Database name.
        username: Database user.
        password: Password, or ``None`` for authentication configured in libpq.
        sslmode: libpq TLS mode.
        connect_timeout: Connection timeout in seconds.
        statement_timeout: Server statement timeout in milliseconds.
    """

    host: str
    port: int
    database: str
    username: str
    password: Optional[Secret]
    sslmode: str
    connect_timeout: int
    statement_timeout: int


@dataclass(frozen=True)
class SQLServerTarget:
    """Where and how a SQL Server write connects.

    Args:
        host: Host name or IP address.
        port: TCP port.
        database: Database name.
        username: SQL Server user; Windows authentication unless both
            ``username`` and a non-empty ``password`` are given.
        password: Password, or ``None``.
    """

    host: str
    port: int
    database: str
    username: Optional[str]
    password: Optional[Secret]


class SQLClient(Protocol):
    """Writes one batch of rows; the caller of the session owns its lifetime."""

    def insert(self, target: Any, insert: SQLInsert) -> None:
        """Run ``insert.statement`` for each row in one transaction and commit.

        Args:
            target: ``PostgreSQLTarget`` or ``SQLServerTarget``.
            insert: The statement and its parameter rows.

        Raises:
            SQLSinkError: With a reportable message, after rolling back.
        """


@dataclass(frozen=True)
class AddressPolicy:
    """V1's destination screening of the default PostgreSQL client.

    Inactive by default. When active, the host must be one TCP host name or
    address; it is resolved, screened, and the connection is pinned to the
    first resolved address so a second DNS answer cannot redirect it.

    Args:
        allow_non_global_addresses: Permit loopback, private and link-local
            destinations.
        blocked_addresses: Host names or IPs always refused.
        allowed_addresses: When non-empty, the only host names or IPs allowed:
            the host itself, or every address it resolves to.
    """

    allow_non_global_addresses: bool = True
    blocked_addresses: FrozenSet[str] = frozenset()
    allowed_addresses: FrozenSet[str] = frozenset()

    @classmethod
    def from_environment(cls) -> "AddressPolicy":
        """Read V1's ``POSTGRESQL_WORKFLOWS_SINK_*`` settings.

        Returns:
            The policy the server operator configured.
        """
        policy = cls(
            allow_non_global_addresses=(
                environment.ALLOW_POSTGRESQL_WORKFLOWS_SINK_TO_NON_GLOBAL_ADDRESSES
            ),
            blocked_addresses=frozenset(
                environment.POSTGRESQL_WORKFLOWS_SINK_BLACKLISTED_ADDRESSES or ()
            ),
            allowed_addresses=frozenset(
                environment.POSTGRESQL_WORKFLOWS_SINK_WHITELISTED_ADDRESSES or ()
            ),
        )

        return policy

    @property
    def active(self) -> bool:
        """Whether destinations are resolved and screened."""
        return (
            not self.allow_non_global_addresses
            or bool(self.blocked_addresses)
            or bool(self.allowed_addresses)
        )

    def check_host(self, host: str) -> None:
        """Refuse host forms that cannot be screened, before scheduling.

        Args:
            host: The step's host.

        Raises:
            SQLSinkError: For a Unix-socket path or a multi-host list.
        """
        if self.active and (host.startswith("/") or "," in host):
            raise SQLSinkError(
                "host must be a single global TCP hostname or IP address"
            )

    def pin(
        self, host: str, *, port: int, resolve: Callable[..., List[str]]
    ) -> Optional[str]:
        """Resolve and screen ``host``; return the address to connect to.

        Args:
            host: The step's host.
            port: The step's port.
            resolve: ``resolve(host, port, allow_non_global_addresses=...)``,
                returning the resolved addresses.

        Returns:
            The first resolved address, or ``None`` when the policy is inactive.

        Raises:
            SQLSinkError: When the destination is refused or unresolvable.
        """
        if not self.active:
            return None

        try:
            addresses = resolve(
                host,
                port,
                allow_non_global_addresses=self.allow_non_global_addresses,
            )
        except SQLSinkError:
            raise
        except Exception:
            raise SQLSinkError(f"Could not resolve host: {host}") from None

        reason = self._refusal(host, addresses)
        if reason is not None:
            raise SQLSinkError(reason)

        return addresses[0]

    def _refusal(self, host: str, addresses: List[str]) -> Optional[str]:
        if host in self.blocked_addresses:
            return "host is blocked by the sink address denylist"
        for address in addresses:
            if address in self.blocked_addresses:
                return f"host '{host}' resolves to a denylisted address '{address}'"
        if not self.allowed_addresses or host in self.allowed_addresses:
            return None
        if addresses and all(
            address in self.allowed_addresses for address in addresses
        ):
            return None

        return "host is not in the sink address allowlist"


def resolve_addresses(
    host: str, port: int, *, allow_non_global_addresses: bool
) -> List[str]:
    """Resolve ``host`` with the shared SSRF screen of ``url_input``.

    Args:
        host: Host name or IP address.
        port: TCP port.
        allow_non_global_addresses: Permit non-global addresses.

    Returns:
        Every resolved address.

    Raises:
        SQLSinkError: When an address is not global and that is not allowed.
    """
    from roboflow_workflows.utils.url_input import (
        URLAddressNotAllowedError,
        resolve_and_validate_ips,
    )

    try:
        addresses = resolve_and_validate_ips(host, port, allow_non_global_addresses)
    except URLAddressNotAllowedError as error:
        raise SQLSinkError(str(error)) from None

    return addresses


def import_driver(module_name: str, *, missing_message: str) -> Any:
    """Import an optional database driver at first use.

    Args:
        module_name: ``psycopg`` or ``pyodbc``.
        missing_message: Message naming what to install.

    Returns:
        The driver module.

    Raises:
        SQLSinkError: With ``missing_message`` when the import fails.
    """
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        raise SQLSinkError(missing_message) from None

    return module


class PsycopgClient:
    """Default PostgreSQL client: one psycopg connection per write.

    Each ``insert`` connects, writes, commits or rolls back, and closes the
    connection in ``finally``. It keeps no connection, thread or pool, so one
    instance may serve any number of steps and threads.

    Args:
        policy: Destination screening; the server's V1 settings by default.
        resolve: Address resolver used when the policy is active.
    """

    def __init__(
        self,
        *,
        policy: Optional[AddressPolicy] = None,
        resolve: Callable[..., List[str]] = resolve_addresses,
    ) -> None:
        self._policy = AddressPolicy.from_environment() if policy is None else policy
        self._resolve = resolve

    def preflight(self, target: PostgreSQLTarget) -> None:
        """Check, before scheduling, what V1 checked before scheduling.

        Args:
            target: The step's connection settings.

        Raises:
            SQLSinkError: For a host form the policy refuses, or no driver.
        """
        self._policy.check_host(target.host)
        import_driver("psycopg", missing_message=POSTGRESQL_DRIVER_MESSAGE)

    def insert(self, target: PostgreSQLTarget, insert: SQLInsert) -> None:
        """Write all rows in one transaction on a new connection.

        Args:
            target: The step's connection settings.
            insert: The statement and its rows.

        Raises:
            SQLSinkError: With V1's message: the phase, the error category
                and SQLSTATE, never server text.
        """
        psycopg = import_driver("psycopg", missing_message=POSTGRESQL_DRIVER_MESSAGE)
        hostaddr = self._policy.pin(
            target.host, port=target.port, resolve=self._resolve
        )
        settings = postgresql_connect_settings(target, hostaddr=hostaddr)

        phase = "connection"
        try:
            connection = psycopg.connect(**settings)
            try:
                phase = "insert"
                with connection.cursor() as cursor:
                    cursor.execute(
                        SET_STATEMENT_TIMEOUT, (str(target.statement_timeout),)
                    )
                    cursor.executemany(insert.statement, insert.rows)
                phase = "commit"
                connection.commit()
            except BaseException:
                _rollback_quietly(connection)
                raise
            finally:
                _close_quietly(connection)
        except psycopg.Error as error:
            message = describe_psycopg_error(error, phase=phase, psycopg=psycopg)
            raise SQLSinkError(message) from None


def postgresql_connect_settings(
    target: PostgreSQLTarget, *, hostaddr: Optional[str]
) -> Dict[str, Any]:
    """Return V1's ``psycopg.connect`` keyword arguments; reveals the password.

    Args:
        target: The step's connection settings.
        hostaddr: Screened address to pin the connection to, if any.

    Returns:
        Keyword arguments, including the plaintext password.
    """
    settings: Dict[str, Any] = {
        "host": target.host,
        "port": target.port,
        "dbname": target.database,
        "user": target.username,
        "password": None if target.password is None else target.password.reveal(),
        "sslmode": target.sslmode,
        "connect_timeout": target.connect_timeout,
        # Bounds client-side inactivity; statement_timeout alone cannot.
        "tcp_user_timeout": target.connect_timeout * 1000 + target.statement_timeout,
        "autocommit": False,
    }
    if hostaddr is not None:
        settings["hostaddr"] = hostaddr

    return settings


def describe_psycopg_error(error: Exception, *, phase: str, psycopg: Any) -> str:
    """Return V1's message for a psycopg error, without server text.

    Args:
        error: The psycopg exception.
        phase: ``connection``, ``insert`` or ``commit``.
        psycopg: The driver module, for its error classes.

    Returns:
        E.g. ``"PostgreSQL insert failed: IntegrityError (SQLSTATE 23505)."``.
    """
    code = getattr(error, "sqlstate", None)
    valid_code = (
        isinstance(code, str) and len(code) == 5 and code.isascii() and code.isalnum()
    )
    suffix = f" (SQLSTATE {code})" if valid_code else ""
    category = next(
        (
            name
            for name in PSYCOPG_CATEGORIES
            if isinstance(error, getattr(psycopg, name))
        ),
        "DatabaseError",
    )
    outcome_unknown = (
        phase == "commit"
        and isinstance(error, (psycopg.OperationalError, psycopg.InterfaceError))
        and (not code or code.startswith("08"))
    )
    note = UNKNOWN_COMMIT_NOTE if outcome_unknown else ""
    message = f"PostgreSQL {phase} failed: {category}{suffix}.{note}"

    return message


class PyodbcClient:
    """Default SQL Server client: one pyodbc FreeTDS connection per write.

    Each ``insert`` connects, sets V1's session options, executes the
    statement once per row, commits or rolls back, and closes in ``finally``.
    It keeps no connection, thread or pool.
    """

    def preflight(self, target: SQLServerTarget) -> None:
        """Report a missing driver before scheduling.

        Args:
            target: The step's connection settings.

        Raises:
            SQLSinkError: When pyodbc cannot be imported.
        """
        import_driver("pyodbc", missing_message=SQL_SERVER_DRIVER_MESSAGE)

    def insert(self, target: SQLServerTarget, insert: SQLInsert) -> None:
        """Write all rows in one transaction on a new connection.

        Args:
            target: The step's connection settings.
            insert: The statement and its rows.

        Raises:
            SQLSinkError: With V1's message, including the driver text.
        """
        pyodbc = import_driver("pyodbc", missing_message=SQL_SERVER_DRIVER_MESSAGE)
        password = None if target.password is None else target.password.reveal()
        connection_string = sql_server_connection_string(target, password=password)

        try:
            connection = pyodbc.connect(connection_string, autocommit=False)
        except pyodbc.Error as error:
            raise SQLSinkError(str(error)) from None

        try:
            _set_session_options(connection, pyodbc=pyodbc)
            _insert_rows(connection, insert, pyodbc=pyodbc)
        finally:
            _close_quietly(connection)


def sql_server_connection_string(
    target: SQLServerTarget, *, password: Optional[str]
) -> str:
    """Return V1's FreeTDS ODBC connection string.

    Args:
        target: The step's connection settings.
        password: The plaintext password, or ``None``.

    Returns:
        ``DRIVER={FreeTDS};SERVER=…;PORT=…;DATABASE=…;`` then ``UID=…;PWD=…``
        or ``Trusted_Connection=yes``.
    """
    parts = [
        "DRIVER={FreeTDS}",
        f"SERVER={odbc_value(target.host)}",
        f"PORT={target.port}",
        f"DATABASE={odbc_value(target.database)}",
    ]
    if target.username and password:
        parts += [f"UID={odbc_value(target.username)}", f"PWD={odbc_value(password)}"]
    else:
        parts.append("Trusted_Connection=yes")
    connection_string = ";".join(parts)

    return connection_string


def odbc_value(value: str) -> str:
    """Brace an ODBC connection-string value only when it needs it.

    Args:
        value: The raw value.

    Returns:
        ``value`` unchanged, or ``{value}`` with ``}`` doubled when it holds
        ``;``, ``{`` or ``}`` or starts or ends with a space.
    """
    if not any(mark in value for mark in ";{}") and value == value.strip():
        return value

    braced = "{" + value.replace("}", "}}") + "}"

    return braced


def _set_session_options(connection: Any, *, pyodbc: Any) -> None:
    cursor = connection.cursor()
    try:
        for option in SQL_SERVER_SESSION_OPTIONS:
            cursor.execute(option)
        connection.commit()
    except pyodbc.Error as error:
        _rollback_quietly(connection)
        raise SQLSinkError(
            f"Failed to set required session parameters: {error}"
        ) from None
    finally:
        _close_quietly(cursor)


def _insert_rows(connection: Any, insert: SQLInsert, *, pyodbc: Any) -> None:
    committing = False
    try:
        cursor = connection.cursor()
        try:
            for row in insert.rows:
                cursor.execute(insert.statement, list(row))
            committing = True
            connection.commit()
        finally:
            _close_quietly(cursor)
    except Exception as error:
        _rollback_quietly(connection)
        if isinstance(error, pyodbc.DataError):
            detail = f"Data conversion error: {error}"
        elif isinstance(error, pyodbc.Error):
            detail = f"Failed to insert data: {error}"
        else:
            detail = str(error)
        note = UNKNOWN_COMMIT_NOTE if committing else ""
        raise SQLSinkError(
            f"Error preparing or executing insert: {detail}{note}"
        ) from None


def _rollback_quietly(connection: Any) -> None:
    # The original error is the one to report; a failed rollback adds nothing.
    try:
        connection.rollback()
    except Exception:
        pass


def _close_quietly(resource: Any) -> None:
    try:
        resource.close()
    except Exception:
        pass


def validate_rows(data: Any) -> List[Dict[str, Any]]:
    """Check V1 PostgreSQL's row rules; used by both blocks.

    Args:
        data: One row as a dict, or a list of them.

    Returns:
        The rows as a list.

    Raises:
        ValueError: With V1's message for the first problem.
    """
    rows = [data] if isinstance(data, dict) else data
    if not isinstance(rows, list) or not rows:
        raise ValueError("Data must be a non-empty dictionary or list of dictionaries")
    if any(not isinstance(row, dict) or not row for row in rows):
        raise ValueError("Each row must be a non-empty dictionary")

    columns = set(rows[0])
    if any(not _is_name(column) for column in columns):
        raise ValueError(
            "Column names must be non-empty strings without NUL characters"
        )
    if any(set(row) != columns for row in rows):
        raise ValueError("All rows must have the same column names")

    return rows


def detached_parameters(
    rows: List[Dict[str, Any]], *, columns: Sequence[str]
) -> Tuple[Tuple[Any, ...], ...]:
    """Return a deep copy of the row values in column order.

    Args:
        rows: Validated rows.
        columns: Column order, normally the first row's keys.

    Returns:
        One tuple per row.
    """
    parameters = tuple(tuple(row[column] for column in columns) for row in rows)
    detached = copy.deepcopy(parameters)

    return detached


def quote_postgresql_identifier(name: str) -> str:
    """Quote one PostgreSQL name as psycopg's ``sql.Identifier`` does.

    The name is enclosed in double quotes with ``"`` doubled; ``%`` is also
    doubled because the statement uses psycopg's ``%s`` parameter style.

    Args:
        name: Raw name, e.g. ``detections``.

    Returns:
        E.g. ``"detections"``.
    """
    quoted = '"' + name.replace('"', '""').replace("%", "%%") + '"'

    return quoted


def postgresql_insert(
    rows: List[Dict[str, Any]], *, schema_name: str, table_name: str
) -> SQLInsert:
    """Build V1's PostgreSQL insert.

    Args:
        rows: Validated rows.
        schema_name: Raw schema name.
        table_name: Raw table name.

    Returns:
        ``INSERT INTO "schema"."table" ("a", "b") VALUES (%s, %s)`` and its rows.
    """
    columns = list(rows[0])
    table = ".".join(
        quote_postgresql_identifier(name) for name in (schema_name, table_name)
    )
    names = ", ".join(quote_postgresql_identifier(column) for column in columns)
    placeholders = ", ".join("%s" for _ in columns)
    insert = SQLInsert(
        statement=f"INSERT INTO {table} ({names}) VALUES ({placeholders})",
        rows=detached_parameters(rows, columns=columns),
    )

    return insert


def quote_sql_server_identifier(name: str) -> str:
    """Quote one SQL Server name with brackets, ``]`` doubled (``QUOTENAME``).

    Args:
        name: Raw name.

    Returns:
        E.g. ``[detections]``.
    """
    quoted = "[" + name.replace("]", "]]") + "]"

    return quoted


def quote_sql_server_table(table_name: str) -> str:
    """Parse a possibly qualified SQL Server table name and re-quote each part.

    Args:
        table_name: E.g. ``detections``, ``dbo.detections`` or
            ``[my db].dbo."detections"``.

    Returns:
        E.g. ``[dbo].[detections]``.

    Raises:
        ValueError: When the text is not 1 to 4 dot-separated identifiers.
    """
    parts = []
    position = 0
    while True:
        match = _SQL_SERVER_NAME_PART.match(table_name, position)
        if match is None:
            raise ValueError(TABLE_NAME_MESSAGE)

        parts.append(_unquote_sql_server_part(match.group()))
        position = match.end()
        if position == len(table_name):
            break
        if table_name[position] != ".":
            raise ValueError(TABLE_NAME_MESSAGE)
        position += 1

    if len(parts) > 4 or not all(_is_name(part) for part in parts):
        raise ValueError(TABLE_NAME_MESSAGE)

    quoted = ".".join(quote_sql_server_identifier(part) for part in parts)

    return quoted


def _unquote_sql_server_part(part: str) -> str:
    if part.startswith("["):
        return part[1:-1].replace("]]", "]")
    if part.startswith('"'):
        return part[1:-1].replace('""', '"')

    return part


def sql_server_insert(rows: List[Dict[str, Any]], *, table_name: str) -> SQLInsert:
    """Build V1's SQL Server insert, with every name bracket-quoted.

    Args:
        rows: Validated rows.
        table_name: Possibly qualified table name.

    Returns:
        ``INSERT INTO [dbo].[t] ([a],[b]) VALUES (?,?)`` and its rows.

    Raises:
        ValueError: For an invalid table name.
    """
    columns = list(rows[0])
    table = quote_sql_server_table(table_name)
    names = ",".join(quote_sql_server_identifier(column) for column in columns)
    placeholders = ",".join("?" for _ in columns)
    insert = SQLInsert(
        statement=f"INSERT INTO {table} ({names}) VALUES ({placeholders})",
        rows=detached_parameters(rows, columns=columns),
    )

    return insert


def _is_name(value: Any) -> bool:
    return isinstance(value, str) and bool(value) and "\x00" not in value


def _require_names(**values: Any) -> None:
    for name, value in values.items():
        if not _is_name(value):
            raise ValueError(
                f"{name} must be a non-empty string without NUL characters"
            )


def _require_port(value: Any) -> int:
    if isinstance(value, str) and value.isascii() and value.isdigit():
        value = int(value)
    if type(value) is not int or not 1 <= value <= MAX_PORT:
        raise ValueError(f"port must be an integer between 1 and {MAX_PORT}")

    return value


def write_rows(
    client: SQLClient,
    target: Any,
    insert: SQLInsert,
    *,
    revealed: Revealed,
    fire_and_forget: bool,
    sink_effects: Optional[BackgroundEffects],
    label: str,
    unexpected: Callable[[Exception], str],
) -> Dict[str, Any]:
    """Write once, now or in the background, and return the step's outputs.

    Args:
        client: The borrowed or default client; never closed here.
        target: Connection settings for the client.
        insert: The statement and its rows.
        revealed: The revealed password, to scrub messages.
        fire_and_forget: The step does not wait when background effects exist.
        sink_effects: The host's background runner, if any.
        label: Block type, for background failures.
        unexpected: Message for an exception that is not ``SQLSinkError``.

    Returns:
        ``error_status`` and ``message``.

    Raises:
        RuntimeError: When ``sink_effects`` is already closed.
    """

    def effect() -> EffectOutcome:
        try:
            client.insert(target, insert)
        except SQLSinkError as error:
            return True, revealed.scrub(str(error))
        except Exception as error:
            return True, revealed.scrub(unexpected(error))

        return False, INSERTED_MESSAGE.format(len(insert.rows))

    error_status, message = deliver(
        effect, fire_and_forget=fire_and_forget, sink_effects=sink_effects, label=label
    )
    if not error_status and message == BACKGROUND_MESSAGE:
        message = SCHEDULED_MESSAGE

    return {"error_status": error_status, "message": message}


def _failure(message: str) -> Dict[str, Any]:
    return {"error_status": True, "message": message}


def _restrictions(
    fire_and_forget: Any, *, always: List[RuntimeRestriction]
) -> Union[List[RuntimeRestriction], Discovery]:
    if isinstance(fire_and_forget, str):
        problem = DiscoveryProblem(
            code=DiscoveryProblemCode.UNRESOLVED_SELECTOR,
            description=(
                f"Field `fire_and_forget` is set by selector `{fire_and_forget}`, "
                "which is only known at run time, so the restrictions are not "
                "fully known."
            ),
            details={
                "declaration": "restrictions",
                "field": "fire_and_forget",
                "selector": fire_and_forget,
            },
        )
        discovery = incomplete_discovery(always, [problem])
        return discovery

    if fire_and_forget:
        return [*always, FIRE_AND_FORGET_RESTRICTION]

    return list(always)


DATA_DESCRIPTION = (
    "One row as a dictionary, or a non-empty list of rows with the same column "
    "names. Values are passed to the driver as parameters; None is SQL NULL."
)
FIRE_AND_FORGET_DESCRIPTION = (
    "Do not wait for the commit when the host provides background effects; "
    "otherwise the rows are written now. Set false to see write failures."
)
PASSWORD_DESCRIPTION = (
    "Database password. Prefer a secret from a secrets store step or a "
    "workflow input of kind secret."
)


class PostgreSQLSinkBlock(Block):
    """Insert rows into an existing PostgreSQL table, all in one transaction.

    Example: ``data={"camera_id": "line-1", "label": "part"}`` with
    ``schema_name="public"`` and ``table_name="detections"`` runs
    ``INSERT INTO "public"."detections" ("camera_id", "label") VALUES (%s, %s)``
    with ``("line-1", "part")``. Names are case-sensitive; tables are not
    created. Pass a ``postgresql_client`` resource to reuse a connection; the
    block borrows it and never closes it. A borrowed client shared by steps or
    sessions must allow concurrent ``insert`` calls.
    """

    type = POSTGRESQL_BLOCK_TYPE
    metadata = {
        "name": "PostgreSQL Sink",
        "port_of": "roboflow_core/postgresql_sink@v1",
        "section": "data_storage",
    }
    outputs = {"error_status": ERROR_STATUS_OUTPUT, "message": MESSAGE_OUTPUT}

    class Params(BlockParams):
        host: StrictStr | Ref(STRING_KIND) = Field(
            description="Database host name or IP address.",
            examples=["db.example.com"],
        )
        database: StrictStr | Ref(STRING_KIND) = Field(
            description="Database name.", examples=["production"]
        )
        username: StrictStr | Ref(STRING_KIND) = Field(
            description="Database user.", examples=["workflow_writer"]
        )
        password: Optional[StrictStr] | Ref(SECRET_KIND, STRING_KIND) = Field(
            default=None,
            description=(
                f"{PASSWORD_DESCRIPTION} None uses authentication configured "
                "in libpq."
            ),
            examples=["$steps.secrets.pg_password"],
        )
        port: StrictInt | Ref(INTEGER_KIND) = Field(
            default=5432,
            ge=1,
            le=MAX_PORT,
            description="Database port.",
            examples=[5432],
        )
        schema_name: StrictStr | Ref(STRING_KIND) = Field(
            default="public",
            description="Raw schema name, without SQL quoting.",
            examples=["public"],
        )
        table_name: StrictStr | Ref(STRING_KIND) = Field(
            description="Raw table name, without schema prefix or SQL quoting.",
            examples=["detections"],
        )
        data: (
            Dict[str, Any]
            | List[Dict[str, Any]]
            | Ref(DICTIONARY_KIND, LIST_OF_VALUES_KIND)
        ) = Field(description=DATA_DESCRIPTION, examples=["$steps.row.output"])
        fire_and_forget: StrictBool | Ref(BOOLEAN_KIND) = Field(
            default=True, description=FIRE_AND_FORGET_DESCRIPTION, examples=[False]
        )
        sslmode: SSLMode | Ref(STRING_KIND) = Field(
            default="require",
            description="libpq TLS mode; verify-full also checks the server name.",
            examples=["verify-full"],
        )
        connect_timeout: StrictInt | Ref(INTEGER_KIND) = Field(
            default=10,
            ge=1,
            le=MAX_INT32,
            description="Connection timeout in seconds.",
            examples=[10],
        )
        statement_timeout: StrictInt | Ref(INTEGER_KIND) = Field(
            default=10000,
            ge=1,
            le=MAX_INT32,
            description="Server statement timeout in milliseconds.",
            examples=[10000],
        )

    def __init__(
        self,
        *,
        postgresql_client: Optional[SQLClient] = None,
        sink_effects: Optional[BackgroundEffects] = None,
        disable_sinks: bool = False,
    ) -> None:
        self._default_client = PsycopgClient() if postgresql_client is None else None
        self._client = (
            self._default_client if postgresql_client is None else postgresql_client
        )
        self._sink_effects = sink_effects
        self._disable_sinks = disable_sinks

    @classmethod
    def discover_work_operations(cls, params: BlockParams) -> List[WorkOperation]:
        """Declare the database write."""
        return [WorkOperation.EXTERNAL_REQUEST]

    @classmethod
    def discover_restrictions(
        cls, params: BlockParams
    ) -> Union[List[RuntimeRestriction], Discovery]:
        """Declare V1's hosted-platform and fire-and-forget restrictions."""
        restrictions = _restrictions(
            params.fire_and_forget, always=[HOSTED_PLATFORM_RESTRICTION]
        )

        return restrictions

    def run(
        self,
        *,
        host: str,
        database: str,
        username: str,
        password: Optional[Union[str, Secret]],
        port: int,
        schema_name: str,
        table_name: str,
        data: Any,
        fire_and_forget: bool,
        sslmode: str,
        connect_timeout: int,
        statement_timeout: int,
    ) -> Dict[str, Any]:
        """Insert the rows unless sinks are disabled.

        Args:
            host: Database host.
            database: Database name.
            username: Database user.
            password: Password or ``None``.
            port: Database port.
            schema_name: Raw schema name.
            table_name: Raw table name.
            data: One row or a list of rows.
            fire_and_forget: Do not wait when background effects exist.
            sslmode: libpq TLS mode.
            connect_timeout: Seconds.
            statement_timeout: Milliseconds.

        Returns:
            ``error_status`` and ``message``.

        Raises:
            SecretUnavailableError: When the password has no value here.
        """
        if self._disable_sinks:
            return {
                "error_status": False,
                "message": disabled_sink_message(by_policy=True),
            }

        if environment.GCP_SERVERLESS or environment.LAMBDA:
            return _failure(HOSTED_PLATFORM_MESSAGE)

        try:
            rows = validate_rows(data)
            _require_names(
                host=host,
                database=database,
                username=username,
                schema_name=schema_name,
                table_name=table_name,
            )
        except ValueError as error:
            return _failure(str(error))

        secret = to_secret(password, name="password")
        revealed = reveal_secrets(secret)
        target = PostgreSQLTarget(
            host=host,
            port=port,
            database=database,
            username=username,
            password=secret,
            sslmode=sslmode,
            connect_timeout=connect_timeout,
            statement_timeout=statement_timeout,
        )
        if self._default_client is not None:
            try:
                self._default_client.preflight(target)
            except SQLSinkError as error:
                return _failure(revealed.scrub(str(error)))

        insert = postgresql_insert(rows, schema_name=schema_name, table_name=table_name)
        outputs = write_rows(
            self._client,
            target,
            insert,
            revealed=revealed,
            fire_and_forget=fire_and_forget,
            sink_effects=self._sink_effects,
            label=POSTGRESQL_BLOCK_TYPE,
            unexpected=lambda error: POSTGRESQL_UNEXPECTED_MESSAGE,
        )

        return outputs


class SQLServerSinkBlock(Block):
    """Insert rows into an existing Microsoft SQL Server table, in one transaction.

    Example: ``data=[{"part": "ok"}, {"part": "defect"}]`` with
    ``table_name="dbo.detections"`` runs
    ``INSERT INTO [dbo].[detections] ([part]) VALUES (?)`` once per row, then
    commits. Without both ``username`` and ``password`` the connection uses
    Windows authentication (``Trusted_Connection=yes``), as in V1. Pass a
    ``sql_server_client`` resource to reuse a connection; the block borrows it
    and never closes it.
    """

    type = SQL_SERVER_BLOCK_TYPE
    metadata = {
        "name": "Microsoft SQL Server Sink",
        "port_of": "roboflow_core/microsoft_sql_server_sink@v1",
        "section": "data_storage",
    }
    outputs = {"error_status": ERROR_STATUS_OUTPUT, "message": MESSAGE_OUTPUT}

    class Params(BlockParams):
        host: StrictStr | Ref(STRING_KIND) = Field(
            description="SQL Server host name or IP address.",
            examples=["192.168.1.100"],
        )
        port: StrictInt | Ref(INTEGER_KIND, STRING_KIND) = Field(
            default=1433,
            description="SQL Server port: an integer or a string of digits.",
            examples=[1433],
        )
        database: StrictStr | Ref(STRING_KIND) = Field(
            description="Target database name.", examples=["production_db"]
        )
        username: Optional[StrictStr] | Ref(STRING_KIND, allow_none=True) = Field(
            default=None,
            description="SQL Server user; omit for Windows authentication.",
            examples=["db_user"],
        )
        password: Optional[StrictStr] | Ref(SECRET_KIND, STRING_KIND) = Field(
            default=None,
            description=PASSWORD_DESCRIPTION,
            examples=["$steps.secrets.sql_password"],
        )
        table_name: StrictStr | Ref(STRING_KIND) = Field(
            description="Table name, optionally qualified, e.g. dbo.detections.",
            examples=["dbo.detections"],
        )
        data: (
            Dict[str, Any]
            | List[Dict[str, Any]]
            | Ref(DICTIONARY_KIND, LIST_OF_VALUES_KIND)
        ) = Field(description=DATA_DESCRIPTION, examples=["$steps.row.output"])
        fire_and_forget: StrictBool | Ref(BOOLEAN_KIND) = Field(
            default=True, description=FIRE_AND_FORGET_DESCRIPTION, examples=[False]
        )

        # Also checks values from selectors: the engine reruns field validators.
        @field_validator("port")
        @classmethod
        def _port_in_range(cls, value: Union[int, str]) -> Union[int, str]:
            if isinstance(value, str) and value.startswith("$"):
                return value

            _require_port(value)

            return value

    def __init__(
        self,
        *,
        sql_server_client: Optional[SQLClient] = None,
        sink_effects: Optional[BackgroundEffects] = None,
        disable_sinks: bool = False,
    ) -> None:
        self._default_client = PyodbcClient() if sql_server_client is None else None
        self._client = (
            self._default_client if sql_server_client is None else sql_server_client
        )
        self._sink_effects = sink_effects
        self._disable_sinks = disable_sinks

    @classmethod
    def discover_work_operations(cls, params: BlockParams) -> List[WorkOperation]:
        """Declare the database write."""
        return [WorkOperation.EXTERNAL_REQUEST]

    @classmethod
    def discover_restrictions(
        cls, params: BlockParams
    ) -> Union[List[RuntimeRestriction], Discovery]:
        """Declare V1's fire-and-forget restriction."""
        restrictions = _restrictions(params.fire_and_forget, always=[])

        return restrictions

    def run(
        self,
        *,
        host: str,
        port: Union[int, str],
        database: str,
        username: Optional[str],
        password: Optional[Union[str, Secret]],
        table_name: str,
        data: Any,
        fire_and_forget: bool,
    ) -> Dict[str, Any]:
        """Insert the rows unless sinks are disabled.

        Args:
            host: SQL Server host.
            port: Port, as an int or digit string.
            database: Database name.
            username: User or ``None``.
            password: Password or ``None``.
            table_name: Possibly qualified table name.
            data: One row or a list of rows.
            fire_and_forget: Do not wait when background effects exist.

        Returns:
            ``error_status`` and ``message``.

        Raises:
            SecretUnavailableError: When the password has no value here.
        """
        if self._disable_sinks:
            return {
                "error_status": False,
                "message": disabled_sink_message(by_policy=True),
            }

        try:
            rows = validate_rows(data)
            _require_names(host=host, database=database, table_name=table_name)
            if username is not None:
                _require_names(username=username)
            port = _require_port(port)
            insert = sql_server_insert(rows, table_name=table_name)
        except ValueError as error:
            return _failure(str(error))

        secret = to_secret(password, name="password")
        revealed = reveal_secrets(secret)
        target = SQLServerTarget(
            host=host, port=port, database=database, username=username, password=secret
        )
        if self._default_client is not None:
            try:
                self._default_client.preflight(target)
            except SQLSinkError as error:
                return _failure(str(error))

        outputs = write_rows(
            self._client,
            target,
            insert,
            revealed=revealed,
            fire_and_forget=fire_and_forget,
            sink_effects=self._sink_effects,
            label=SQL_SERVER_BLOCK_TYPE,
            unexpected=lambda error: f"An unexpected error occurred: {error}",
        )

        return outputs
