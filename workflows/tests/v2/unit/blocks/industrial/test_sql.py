"""Unit tests of the PostgreSQL and SQL Server sinks.

Nothing here connects to a database. ``RecordingClient`` stands in for a
host-injected client; ``fake_psycopg`` and ``fake_pyodbc`` stand in for the
driver modules so the default clients' connection handling can run. They
prove wiring only, not driver or database behaviour.
"""

import sys
import threading
import types
from typing import Any, Dict, List, Optional

import pytest
from roboflow_workflows import environment
from roboflow_workflows.v2.blocks.industrial import sql
from roboflow_workflows.v2.blocks.industrial.sql import (
    AddressPolicy,
    PostgreSQLSinkBlock,
    PostgreSQLTarget,
    PsycopgClient,
    PyodbcClient,
    SQLInsert,
    SQLServerSinkBlock,
    SQLServerTarget,
    SQLSinkError,
    quote_sql_server_table,
)
from roboflow_workflows.v2.blocks.sinks.runtime import BackgroundEffects
from roboflow_workflows.v2.data.secrets import Secret, SecretUnavailableError

PASSWORD = "pw-SENTINEL-41c9"
WAIT = 10.0


class RecordingClient:
    """Stand-in for a host-injected client: records inserts, can fail."""

    def __init__(self, *, error: Optional[Exception] = None) -> None:
        self.error = error
        self.calls: List[Any] = []
        self.closed = False

    def insert(self, target: Any, insert: SQLInsert) -> None:
        self.calls.append((target, insert))
        if self.error is not None:
            raise self.error

    def close(self) -> None:
        self.closed = True


class GatedClient(RecordingClient):
    """Holds each insert until ``release`` is set."""

    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def insert(self, target: Any, insert: SQLInsert) -> None:
        self.started.set()
        assert self.release.wait(WAIT)
        super().insert(target, insert)


def _pg_params(**overrides: Any) -> Dict[str, Any]:
    params = {
        "host": "db.example.com",
        "database": "production",
        "username": "writer",
        "password": PASSWORD,
        "port": 5432,
        "schema_name": "public",
        "table_name": "detections",
        "data": {"camera_id": "line-1", "label": "part"},
        "fire_and_forget": False,
        "sslmode": "require",
        "connect_timeout": 10,
        "statement_timeout": 10000,
    }
    params.update(overrides)

    return params


def _ms_params(**overrides: Any) -> Dict[str, Any]:
    params = {
        "host": "sql.example.com",
        "port": 1433,
        "database": "production_db",
        "username": "db_user",
        "password": PASSWORD,
        "table_name": "dbo.detections",
        "data": {"part": "ok", "score": 0.9},
        "fire_and_forget": False,
    }
    params.update(overrides)

    return params


# PostgreSQL block with an injected client


def test_postgresql_statement_and_parameters_are_separate() -> None:
    # given
    client = RecordingClient()
    block = PostgreSQLSinkBlock(postgresql_client=client)

    # when
    result = block.run(**_pg_params())

    # then
    assert result == {
        "error_status": False,
        "message": "Successfully inserted 1 records",
    }
    target, insert = client.calls[0]
    assert insert.statement == (
        'INSERT INTO "public"."detections" ("camera_id", "label") VALUES (%s, %s)'
    )
    assert insert.rows == (("line-1", "part"),)
    assert target.host == "db.example.com"
    assert target.password.reveal() == PASSWORD
    assert PASSWORD not in repr(target)


def test_postgresql_values_never_enter_the_statement() -> None:
    # given
    client = RecordingClient()
    hostile = "x'); DROP TABLE detections; --"
    rows = [
        {"label": hostile, "note": None, "quote": 'say "hi"'},
        {"note": "second", "quote": "", "label": "O'Brien"},
    ]

    # when
    result = PostgreSQLSinkBlock(postgresql_client=client).run(**_pg_params(data=rows))

    # then
    insert = client.calls[0][1]
    assert result["message"] == "Successfully inserted 2 records"
    assert insert.statement == (
        'INSERT INTO "public"."detections" ("label", "note", "quote") '
        "VALUES (%s, %s, %s)"
    )
    assert insert.rows == ((hostile, None, 'say "hi"'), ("O'Brien", "second", ""))
    assert "DROP" not in insert.statement


def test_postgresql_names_are_quoted_like_psycopg_identifiers() -> None:
    # given
    client = RecordingClient()

    # when
    PostgreSQLSinkBlock(postgresql_client=client).run(
        **_pg_params(
            schema_name='My"Schema',
            table_name="Detections",
            data={'a"b': 1, "pct%": 2},
        )
    )

    # then
    assert client.calls[0][1].statement == (
        'INSERT INTO "My""Schema"."Detections" ("a""b", "pct%%") VALUES (%s, %s)'
    )


@pytest.mark.parametrize(
    "data, message",
    [
        ([], "Data must be a non-empty dictionary or list of dictionaries"),
        ("row", "Data must be a non-empty dictionary or list of dictionaries"),
        ({}, "Each row must be a non-empty dictionary"),
        ([{"a": 1}, 3], "Each row must be a non-empty dictionary"),
        ({"": 1}, "Column names must be non-empty strings without NUL characters"),
        ({"a\x00": 1}, "Column names must be non-empty strings without NUL characters"),
        ([{"a": 1}, {"b": 1}], "All rows must have the same column names"),
    ],
)
def test_malformed_rows_fail_without_touching_the_client(
    data: Any, message: str
) -> None:
    # given
    pg_client, ms_client = RecordingClient(), RecordingClient()

    # when
    pg = PostgreSQLSinkBlock(postgresql_client=pg_client).run(**_pg_params(data=data))
    ms = SQLServerSinkBlock(sql_server_client=ms_client).run(**_ms_params(data=data))

    # then
    assert pg == {"error_status": True, "message": message}
    assert ms == {"error_status": True, "message": message}
    assert pg_client.calls == [] and ms_client.calls == []


@pytest.mark.parametrize("field", ["host", "database", "username", "schema_name"])
def test_postgresql_empty_names_fail_with_v1_message(field: str) -> None:
    # given
    client = RecordingClient()

    # when
    result = PostgreSQLSinkBlock(postgresql_client=client).run(
        **_pg_params(**{field: ""})
    )

    # then
    assert result == {
        "error_status": True,
        "message": f"{field} must be a non-empty string without NUL characters",
    }
    assert client.calls == []


def test_a_client_failure_is_reported_once_and_scrubbed_of_the_password() -> None:
    # given
    client = RecordingClient(error=SQLSinkError(f"login with {PASSWORD} refused"))

    # when
    result = PostgreSQLSinkBlock(postgresql_client=client).run(**_pg_params())

    # then
    assert result == {
        "error_status": True,
        "message": "login with <secret password> refused",
    }
    assert len(client.calls) == 1


def test_an_unexpected_client_error_gets_v1s_generic_message() -> None:
    # given
    client = RecordingClient(error=TimeoutError(f"timed out; password={PASSWORD}"))

    # when
    pg = PostgreSQLSinkBlock(postgresql_client=client).run(**_pg_params())
    ms = SQLServerSinkBlock(sql_server_client=client).run(**_ms_params())

    # then
    assert pg == {
        "error_status": True,
        "message": "Unexpected PostgreSQL insert failure",
    }
    assert ms == {
        "error_status": True,
        "message": "An unexpected error occurred: timed out; password=<secret password>",
    }
    assert len(client.calls) == 2


def test_disabled_sinks_touch_neither_client_secret_nor_driver(monkeypatch) -> None:
    # given
    monkeypatch.delitem(sys.modules, "psycopg", raising=False)
    monkeypatch.delitem(sys.modules, "pyodbc", raising=False)
    client = RecordingClient()
    unavailable = Secret.unavailable("PG_PASSWORD")
    blocks = [
        (PostgreSQLSinkBlock(postgresql_client=client, disable_sinks=True), _pg_params),
        (PostgreSQLSinkBlock(disable_sinks=True), _pg_params),
        (SQLServerSinkBlock(sql_server_client=client, disable_sinks=True), _ms_params),
        (SQLServerSinkBlock(disable_sinks=True), _ms_params),
    ]

    # when
    results = [block.run(**params(password=unavailable)) for block, params in blocks]

    # then
    disabled = {
        "error_status": False,
        "message": "Sink was disabled by workflow execution policy",
    }
    assert results == [disabled] * 4
    assert client.calls == []
    assert "psycopg" not in sys.modules and "pyodbc" not in sys.modules


def test_an_unavailable_secret_fails_the_step_before_any_write() -> None:
    # given
    client = RecordingClient()
    password = Secret.unavailable("PG_PASSWORD")

    # when
    with pytest.raises(SecretUnavailableError):
        PostgreSQLSinkBlock(postgresql_client=client).run(
            **_pg_params(password=password)
        )
    with pytest.raises(SecretUnavailableError):
        SQLServerSinkBlock(sql_server_client=client).run(
            **_ms_params(password=password)
        )

    # then
    assert client.calls == []


def test_a_borrowed_client_is_used_as_is_and_never_closed() -> None:
    # given
    client = RecordingClient()
    block = PostgreSQLSinkBlock(postgresql_client=client)

    # when
    block.run(**_pg_params())
    block.run(**_pg_params(fire_and_forget=True))
    del block

    # then
    assert len(client.calls) == 2
    assert client.closed is False


def test_the_hosted_platform_refuses_without_writing(monkeypatch) -> None:
    # given
    monkeypatch.setattr(environment, "GCP_SERVERLESS", True)
    client = RecordingClient()

    # when
    result = PostgreSQLSinkBlock(postgresql_client=client).run(**_pg_params())

    # then
    assert result == {
        "error_status": True,
        "message": "PostgreSQL sink is not available on the Roboflow hosted platform",
    }
    assert client.calls == []


def test_fire_and_forget_without_sink_effects_writes_now() -> None:
    # given
    client = RecordingClient()

    # when
    result = PostgreSQLSinkBlock(postgresql_client=client).run(
        **_pg_params(fire_and_forget=True)
    )

    # then
    assert result == {
        "error_status": False,
        "message": "Successfully inserted 1 records",
    }
    assert len(client.calls) == 1


def test_background_write_uses_a_detached_copy_and_reports_scheduled() -> None:
    # given
    client = GatedClient()
    row = {"camera_id": "line-1", "label": "part"}
    with BackgroundEffects() as effects:
        block = PostgreSQLSinkBlock(postgresql_client=client, sink_effects=effects)

        # when
        result = block.run(**_pg_params(data=row, fire_and_forget=True))
        assert client.started.wait(WAIT)
        row["label"] = "changed after scheduling"
        client.release.set()
        assert effects.drain(WAIT)

    # then
    assert result == {"error_status": False, "message": "Data processing scheduled"}
    assert client.calls[0][1].rows == (("line-1", "part"),)
    assert effects.stats().completed == 1


def test_a_background_failure_reaches_the_host_scrubbed_and_is_not_retried() -> None:
    # given
    client = RecordingClient(error=SQLSinkError(f"bad password {PASSWORD}"))
    with BackgroundEffects() as effects:
        block = SQLServerSinkBlock(sql_server_client=client, sink_effects=effects)

        # when
        result = block.run(**_ms_params(fire_and_forget=True))
        assert effects.drain(WAIT)

    # then
    assert result == {"error_status": False, "message": "Data processing scheduled"}
    assert [failure.message for failure in effects.failures()] == [
        "bad password <secret password>"
    ]
    assert len(client.calls) == 1


def test_waiting_callers_write_now_even_with_sink_effects() -> None:
    # given
    client = RecordingClient()
    with BackgroundEffects() as effects:
        block = PostgreSQLSinkBlock(postgresql_client=client, sink_effects=effects)

        # when
        result = block.run(**_pg_params(fire_and_forget=False))

    # then
    assert result["message"] == "Successfully inserted 1 records"
    assert effects.stats().submitted == 0


# SQL Server block with an injected client


def test_sql_server_statement_brackets_every_name() -> None:
    # given
    client = RecordingClient()
    rows = [{"part": "ok", "score": 0.9}, {"score": None, "part": "it's ]bad"}]

    # when
    result = SQLServerSinkBlock(sql_server_client=client).run(**_ms_params(data=rows))

    # then
    target, insert = client.calls[0]
    assert result == {
        "error_status": False,
        "message": "Successfully inserted 2 records",
    }
    assert insert.statement == (
        "INSERT INTO [dbo].[detections] ([part],[score]) VALUES (?,?)"
    )
    assert insert.rows == (("ok", 0.9), ("it's ]bad", None))
    assert target == SQLServerTarget(
        host="sql.example.com",
        port=1433,
        database="production_db",
        username="db_user",
        password=target.password,
    )


def test_sql_server_column_names_cannot_inject_sql() -> None:
    # given
    client = RecordingClient()
    hostile = "a]) VALUES (1); DROP TABLE t; --"

    # when
    SQLServerSinkBlock(sql_server_client=client).run(**_ms_params(data={hostile: 1}))

    # then
    assert client.calls[0][1].statement == (
        "INSERT INTO [dbo].[detections] ([a]]) VALUES (1); DROP TABLE t; --]) "
        "VALUES (?)"
    )


@pytest.mark.parametrize(
    "table_name, quoted",
    [
        ("detections", "[detections]"),
        ("dbo.detections", "[dbo].[detections]"),
        ("[my db].dbo.[a]]b]", "[my db].[dbo].[a]]b]"),
        ('"q""t"', '[q"t]'),
        ("srv.db.dbo.#tmp", "[srv].[db].[dbo].[#tmp]"),
    ],
)
def test_sql_server_table_names_are_parsed_and_requoted(
    table_name: str, quoted: str
) -> None:
    assert quote_sql_server_table(table_name) == quoted


@pytest.mark.parametrize(
    "table_name",
    ["dbo.t; DROP TABLE x", "a.b.c.d.e", "db..t", "[]", "1table", "t ", "dbo.[t"],
)
def test_sql_server_rejects_table_names_that_are_not_identifiers(
    table_name: str,
) -> None:
    # given
    client = RecordingClient()

    # when
    result = SQLServerSinkBlock(sql_server_client=client).run(
        **_ms_params(table_name=table_name)
    )

    # then
    assert result == {"error_status": True, "message": sql.TABLE_NAME_MESSAGE}
    assert client.calls == []


@pytest.mark.parametrize("port, expected", [(1433, 1433), ("1433", 1433)])
def test_sql_server_port_may_be_a_digit_string(port: Any, expected: int) -> None:
    # given
    client = RecordingClient()

    # when
    SQLServerSinkBlock(sql_server_client=client).run(**_ms_params(port=port))

    # then
    assert client.calls[0][0].port == expected


# Restriction declarations


@pytest.mark.parametrize(
    "fire_and_forget, codes",
    [
        (
            True,
            ["unavailable_on_hosted_platform", sql.FIRE_AND_FORGET_RESTRICTION.code],
        ),
        (False, ["unavailable_on_hosted_platform"]),
    ],
)
def test_postgresql_restrictions_follow_v1(fire_and_forget: bool, codes: list) -> None:
    # given
    params = PostgreSQLSinkBlock.Params.model_validate(
        {
            "host": "h",
            "database": "d",
            "username": "u",
            "table_name": "t",
            "data": {"a": 1},
            "fire_and_forget": fire_and_forget,
        }
    )

    # when
    restrictions = PostgreSQLSinkBlock.discover_restrictions(params)

    # then
    assert [restriction.code for restriction in restrictions] == codes


# Default PostgreSQL client with a stand-in psycopg module


class _PsycopgError(Exception):
    def __init__(self, text: str = "", *, sqlstate: Optional[str] = None) -> None:
        super().__init__(text)
        self.sqlstate = sqlstate


PSYCOPG_ERRORS = {
    name: type(name, (_PsycopgError,), {}) for name in sql.PSYCOPG_CATEGORIES
}


def _fake_psycopg(*, fail_at: Optional[str] = None, error: Any = None):
    """A stand-in ``psycopg`` module whose connections log every call."""
    module = types.ModuleType("psycopg")
    module.Error = _PsycopgError
    for name, error_class in PSYCOPG_ERRORS.items():
        setattr(module, name, error_class)
    module.log = []
    module.connect_settings = []

    def fail(step: str) -> None:
        if fail_at == step:
            raise error

    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            module.log.append("cursor.close")

        def execute(self, statement, parameters):
            module.log.append(("execute", statement, parameters))

        def executemany(self, statement, rows):
            module.log.append(("executemany", statement, rows))
            fail("insert")

    class Connection:
        def cursor(self):
            return Cursor()

        def commit(self):
            module.log.append("commit")
            fail("commit")

        def rollback(self):
            module.log.append("rollback")

        def close(self):
            module.log.append("close")

    def connect(**settings):
        module.connect_settings.append(settings)
        fail("connection")
        return Connection()

    module.connect = connect

    return module


def _pg_target(**overrides: Any) -> PostgreSQLTarget:
    fields = {
        "host": "db.example.com",
        "port": 5432,
        "database": "production",
        "username": "writer",
        "password": Secret(PASSWORD, name="password"),
        "sslmode": "require",
        "connect_timeout": 10,
        "statement_timeout": 10000,
    }
    fields.update(overrides)

    return PostgreSQLTarget(**fields)


INSERT = SQLInsert(statement="INSERT INTO t (a) VALUES (%s)", rows=((1,), (None,)))


def test_default_postgresql_client_connects_writes_commits_and_closes(
    monkeypatch,
) -> None:
    # given
    driver = _fake_psycopg()
    monkeypatch.setitem(sys.modules, "psycopg", driver)

    # when
    PsycopgClient(policy=AddressPolicy()).insert(_pg_target(), INSERT)

    # then
    assert driver.connect_settings == [
        {
            "host": "db.example.com",
            "port": 5432,
            "dbname": "production",
            "user": "writer",
            "password": PASSWORD,
            "sslmode": "require",
            "connect_timeout": 10,
            "tcp_user_timeout": 20000,
            "autocommit": False,
        }
    ]
    assert driver.log == [
        ("execute", sql.SET_STATEMENT_TIMEOUT, ("10000",)),
        ("executemany", INSERT.statement, INSERT.rows),
        "cursor.close",
        "commit",
        "close",
    ]


@pytest.mark.parametrize(
    "fail_at, error_name, sqlstate, message",
    [
        (
            "insert",
            "IntegrityError",
            "23505",
            "PostgreSQL insert failed: IntegrityError (SQLSTATE 23505).",
        ),
        (
            "commit",
            "OperationalError",
            None,
            "PostgreSQL commit failed: OperationalError."
            " Commit outcome may be unknown; verify before retrying.",
        ),
        (
            "commit",
            "IntegrityError",
            "23503",
            "PostgreSQL commit failed: IntegrityError (SQLSTATE 23503).",
        ),
    ],
)
def test_default_postgresql_client_rolls_back_and_hides_server_text(
    monkeypatch, fail_at: str, error_name: str, sqlstate: Optional[str], message: str
) -> None:
    # given
    error = PSYCOPG_ERRORS[error_name](
        f"Key (label)=(secret-row-value) {PASSWORD}", sqlstate=sqlstate
    )
    driver = _fake_psycopg(fail_at=fail_at, error=error)
    monkeypatch.setitem(sys.modules, "psycopg", driver)

    # when
    with pytest.raises(SQLSinkError) as raised:
        PsycopgClient(policy=AddressPolicy()).insert(_pg_target(), INSERT)

    # then
    assert str(raised.value) == message
    assert raised.value.__cause__ is None and raised.value.__suppress_context__
    assert driver.log[-2:] == ["rollback", "close"]
    assert driver.log.count("commit") == (1 if fail_at == "commit" else 0)


def test_default_postgresql_connection_failure_reports_only_the_category(
    monkeypatch,
) -> None:
    # given
    error = PSYCOPG_ERRORS["OperationalError"](
        f"password {PASSWORD} rejected", sqlstate="28P01"
    )
    driver = _fake_psycopg(fail_at="connection", error=error)
    monkeypatch.setitem(sys.modules, "psycopg", driver)

    # when
    with pytest.raises(SQLSinkError) as raised:
        PsycopgClient(policy=AddressPolicy()).insert(_pg_target(), INSERT)

    # then
    assert str(raised.value) == (
        "PostgreSQL connection failed: OperationalError (SQLSTATE 28P01)."
    )
    assert driver.log == []


def test_default_postgresql_block_reports_a_missing_driver_before_scheduling(
    monkeypatch,
) -> None:
    # given
    monkeypatch.setitem(sys.modules, "psycopg", None)
    with BackgroundEffects() as effects:
        block = PostgreSQLSinkBlock(sink_effects=effects)

        # when
        result = block.run(**_pg_params(fire_and_forget=True))

    # then
    assert result == {"error_status": True, "message": sql.POSTGRESQL_DRIVER_MESSAGE}
    assert '"psycopg[binary]>=3.2,<4"' in sql.POSTGRESQL_DRIVER_MESSAGE
    assert effects.stats().submitted == 0


def test_default_postgresql_block_writes_through_the_default_client(
    monkeypatch,
) -> None:
    # given
    monkeypatch.setattr(
        environment, "ALLOW_POSTGRESQL_WORKFLOWS_SINK_TO_NON_GLOBAL_ADDRESSES", True
    )
    monkeypatch.setattr(
        environment, "POSTGRESQL_WORKFLOWS_SINK_BLACKLISTED_ADDRESSES", None
    )
    monkeypatch.setattr(
        environment, "POSTGRESQL_WORKFLOWS_SINK_WHITELISTED_ADDRESSES", None
    )
    driver = _fake_psycopg()
    monkeypatch.setitem(sys.modules, "psycopg", driver)

    # when
    result = PostgreSQLSinkBlock().run(**_pg_params())

    # then
    assert result == {
        "error_status": False,
        "message": "Successfully inserted 1 records",
    }
    assert driver.log[-2:] == ["commit", "close"]


# Address policy of the default PostgreSQL client


def _resolver(addresses: List[str]):
    calls = []

    def resolve(host, port, *, allow_non_global_addresses):
        calls.append((host, port, allow_non_global_addresses))
        return addresses

    resolve.calls = calls

    return resolve


def test_an_inactive_policy_resolves_nothing() -> None:
    # given
    resolve = _resolver(["203.0.113.7"])

    # when
    pinned = AddressPolicy().pin("db", port=5432, resolve=resolve)

    # then
    assert pinned is None
    assert resolve.calls == []


def test_an_active_policy_pins_the_first_screened_address(monkeypatch) -> None:
    # given
    driver = _fake_psycopg()
    monkeypatch.setitem(sys.modules, "psycopg", driver)
    policy = AddressPolicy(allow_non_global_addresses=False)
    client = PsycopgClient(
        policy=policy, resolve=_resolver(["203.0.113.7", "203.0.113.8"])
    )

    # when
    client.insert(_pg_target(), INSERT)

    # then
    assert driver.connect_settings[0]["hostaddr"] == "203.0.113.7"
    assert driver.connect_settings[0]["host"] == "db.example.com"


@pytest.mark.parametrize(
    "policy, addresses, message",
    [
        (
            AddressPolicy(blocked_addresses=frozenset({"db.example.com"})),
            ["203.0.113.7"],
            "host is blocked by the sink address denylist",
        ),
        (
            AddressPolicy(blocked_addresses=frozenset({"203.0.113.7"})),
            ["203.0.113.7"],
            "host 'db.example.com' resolves to a denylisted address '203.0.113.7'",
        ),
        (
            AddressPolicy(allowed_addresses=frozenset({"203.0.113.7"})),
            ["203.0.113.7", "198.51.100.1"],
            "host is not in the sink address allowlist",
        ),
    ],
)
def test_a_refused_destination_never_connects(
    monkeypatch, policy: AddressPolicy, addresses: List[str], message: str
) -> None:
    # given
    driver = _fake_psycopg()
    monkeypatch.setitem(sys.modules, "psycopg", driver)

    # when
    with pytest.raises(SQLSinkError) as raised:
        PsycopgClient(policy=policy, resolve=_resolver(addresses)).insert(
            _pg_target(), INSERT
        )

    # then
    assert str(raised.value) == message
    assert driver.connect_settings == []


def test_an_unresolvable_host_is_a_plain_failure(monkeypatch) -> None:
    # given
    monkeypatch.setitem(sys.modules, "psycopg", _fake_psycopg())

    def resolve(host, port, *, allow_non_global_addresses):
        raise OSError("no such host")

    client = PsycopgClient(
        policy=AddressPolicy(allow_non_global_addresses=False), resolve=resolve
    )

    # when
    with pytest.raises(SQLSinkError) as raised:
        client.insert(_pg_target(), INSERT)

    # then
    assert str(raised.value) == "Could not resolve host: db.example.com"


def test_the_shared_resolver_refuses_a_loopback_literal() -> None:
    # given: an IP literal; getaddrinfo needs no DNS for it

    # when
    with pytest.raises(SQLSinkError) as raised:
        sql.resolve_addresses("127.0.0.1", 5432, allow_non_global_addresses=False)

    # then
    assert (
        str(raised.value)
        == "Host '127.0.0.1' resolves to non-global address '127.0.0.1'."
    )


@pytest.mark.parametrize("host", ["/var/run/postgresql", "db1,db2"])
def test_an_active_policy_refuses_unscreenable_hosts_before_scheduling(
    monkeypatch, host: str
) -> None:
    # given
    monkeypatch.setattr(
        environment, "ALLOW_POSTGRESQL_WORKFLOWS_SINK_TO_NON_GLOBAL_ADDRESSES", False
    )
    monkeypatch.setitem(sys.modules, "psycopg", _fake_psycopg())

    # when
    result = PostgreSQLSinkBlock().run(**_pg_params(host=host))

    # then
    assert result == {
        "error_status": True,
        "message": "host must be a single global TCP hostname or IP address",
    }


# Default SQL Server client with a stand-in pyodbc module


class _PyodbcError(Exception):
    pass


class _PyodbcDataError(_PyodbcError):
    pass


def _fake_pyodbc(*, fail_at: Optional[str] = None, error: Any = None, row: int = 0):
    """A stand-in ``pyodbc`` module whose connections log every call."""
    module = types.ModuleType("pyodbc")
    module.Error = _PyodbcError
    module.DataError = _PyodbcDataError
    module.log = []
    inserted = []

    class Cursor:
        def execute(self, statement, parameters=None):
            module.log.append(("execute", statement, parameters))
            if statement.startswith("SET") and fail_at == "session":
                raise error
            if parameters is not None:
                inserted.append(parameters)
                if fail_at == "insert" and len(inserted) == row:
                    raise error

        def close(self):
            module.log.append("cursor.close")

    class Connection:
        def cursor(self):
            return Cursor()

        def commit(self):
            module.log.append("commit")
            if fail_at == "commit" and inserted:
                raise error

        def rollback(self):
            module.log.append("rollback")

        def close(self):
            module.log.append("close")

    def connect(connection_string, *, autocommit):
        module.log.append(("connect", connection_string, autocommit))
        if fail_at == "connection":
            raise error
        return Connection()

    module.connect = connect

    return module


def _ms_target(**overrides: Any) -> SQLServerTarget:
    fields = {
        "host": "sql.example.com",
        "port": 1433,
        "database": "production_db",
        "username": "db_user",
        "password": Secret(PASSWORD, name="password"),
    }
    fields.update(overrides)

    return SQLServerTarget(**fields)


MS_INSERT = SQLInsert(statement="INSERT INTO [t] ([a]) VALUES (?)", rows=((1,), (2,)))


def test_default_sql_server_client_follows_v1s_sequence(monkeypatch) -> None:
    # given
    driver = _fake_pyodbc()
    monkeypatch.setitem(sys.modules, "pyodbc", driver)

    # when
    PyodbcClient().insert(_ms_target(), MS_INSERT)

    # then
    connection_string = (
        "DRIVER={FreeTDS};SERVER=sql.example.com;PORT=1433;"
        f"DATABASE=production_db;UID=db_user;PWD={PASSWORD}"
    )
    session = [("execute", option, None) for option in sql.SQL_SERVER_SESSION_OPTIONS]
    assert driver.log == [
        ("connect", connection_string, False),
        *session,
        "commit",
        "cursor.close",
        ("execute", MS_INSERT.statement, [1]),
        ("execute", MS_INSERT.statement, [2]),
        "commit",
        "cursor.close",
        "close",
    ]


@pytest.mark.parametrize(
    "username, password",
    [(None, PASSWORD), ("db_user", None), ("db_user", "")],
)
def test_sql_server_uses_windows_authentication_without_both_credentials(
    username: Optional[str], password: Optional[str]
) -> None:
    # given
    target = _ms_target(
        username=username,
        password=None if password is None else Secret(password, name="password"),
    )

    # when
    connection_string = sql.sql_server_connection_string(target, password=password)

    # then
    assert connection_string.endswith(";DATABASE=production_db;Trusted_Connection=yes")
    assert "PWD" not in connection_string


def test_sql_server_braces_connection_values_that_need_it() -> None:
    # given
    target = _ms_target(database="db;Trusted_Connection=yes")

    # when
    connection_string = sql.sql_server_connection_string(target, password="p}w;x")

    # then
    assert connection_string == (
        "DRIVER={FreeTDS};SERVER=sql.example.com;PORT=1433;"
        "DATABASE={db;Trusted_Connection=yes};UID=db_user;PWD={p}}w;x}"
    )


@pytest.mark.parametrize(
    "fail_at, error, expected",
    [
        (
            "insert",
            _PyodbcDataError(f"bad value; {PASSWORD}"),
            "Error preparing or executing insert: Data conversion error: "
            "bad value; <secret password>",
        ),
        (
            "insert",
            _PyodbcError("constraint violated"),
            "Error preparing or executing insert: Failed to insert data: "
            "constraint violated",
        ),
        (
            "commit",
            _PyodbcError("link lost"),
            "Error preparing or executing insert: Failed to insert data: link lost"
            " Commit outcome may be unknown; verify before retrying.",
        ),
    ],
)
def test_default_sql_server_failures_roll_back_close_and_scrub(
    monkeypatch, fail_at: str, error: Exception, expected: str
) -> None:
    # given
    driver = _fake_pyodbc(fail_at=fail_at, error=error, row=2)
    monkeypatch.setitem(sys.modules, "pyodbc", driver)
    block = SQLServerSinkBlock()

    # when
    result = block.run(**_ms_params(data=[{"a": 1}, {"a": 2}]))

    # then
    assert result == {"error_status": True, "message": expected}
    assert driver.log[-2:] == ["rollback", "close"]
    connects = [entry for entry in driver.log if entry[0] == "connect"]
    assert len(connects) == 1


def test_default_sql_server_connection_failure_keeps_driver_text_scrubbed(
    monkeypatch,
) -> None:
    # given
    error = _PyodbcError(f"[08001] login failed for PWD={PASSWORD}")
    monkeypatch.setitem(
        sys.modules, "pyodbc", _fake_pyodbc(fail_at="connection", error=error)
    )

    # when
    result = SQLServerSinkBlock().run(**_ms_params())

    # then
    assert result == {
        "error_status": True,
        "message": "[08001] login failed for PWD=<secret password>",
    }


def test_default_sql_server_session_option_failure_closes(monkeypatch) -> None:
    # given
    driver = _fake_pyodbc(fail_at="session", error=_PyodbcError("denied"))
    monkeypatch.setitem(sys.modules, "pyodbc", driver)

    # when
    with pytest.raises(SQLSinkError) as raised:
        PyodbcClient().insert(_ms_target(), MS_INSERT)

    # then
    assert str(raised.value) == "Failed to set required session parameters: denied"
    assert driver.log[-3:] == ["rollback", "cursor.close", "close"]


def test_default_sql_server_block_reports_a_missing_driver_before_scheduling(
    monkeypatch,
) -> None:
    # given
    monkeypatch.setitem(sys.modules, "pyodbc", None)
    with BackgroundEffects() as effects:
        block = SQLServerSinkBlock(sink_effects=effects)

        # when
        result = block.run(**_ms_params(fire_and_forget=True))

    # then
    assert result == {"error_status": True, "message": sql.SQL_SERVER_DRIVER_MESSAGE}
    assert effects.stats().submitted == 0
