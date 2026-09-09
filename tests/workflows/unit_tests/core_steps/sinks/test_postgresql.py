import asyncio
from unittest.mock import MagicMock

import psycopg
import pytest
from fastapi import BackgroundTasks
from pydantic import ValidationError

from inference.core.workflows.core_steps.sinks.noop import disabled_sink_response
from inference.enterprise.workflows.enterprise_blocks.sinks.postgresql import v1


def arguments(**overrides):
    values = dict(
        host="localhost",
        database="test",
        username="writer",
        table_name="events",
        data={"label": "part", "score": 0.9},
    )
    values.update(overrides)
    return values


@pytest.fixture
def connect(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr(v1.psycopg, "connect", mock)
    return mock


def test_manifest_defaults_and_integer_selectors():
    manifest = v1.BlockManifest(
        type="roboflow_core/postgresql_sink@v1",
        name="pg",
        **arguments(port="$inputs.port"),
    )
    assert manifest.port == "$inputs.port"
    assert manifest.sslmode == "require"
    assert manifest.statement_timeout == 10000
    assert {output.name for output in manifest.describe_outputs()} == {
        "message",
        "error_status",
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("port", 0),
        ("port", 65536),
        ("connect_timeout", 0),
        ("statement_timeout", -1),
        ("sslmode", "bad"),
    ],
)
def test_invalid_manifest(field, value):
    with pytest.raises(ValidationError):
        v1.BlockManifest(
            type="roboflow_core/postgresql_sink@v1",
            name="pg",
            **arguments(**{field: value}),
        )


@pytest.mark.parametrize(
    "data", [None, [], {}, [{}], "bad", [1], [{"a": 1}, {"b": 2}], {"": 1}, {1: 1}]
)
def test_invalid_rows_do_not_connect(data, connect):
    result = v1.PostgreSQLSinkBlockV1(None, None).run(**arguments(data=data))
    assert result["error_status"]
    connect.assert_not_called()


@pytest.mark.parametrize(
    "field,value",
    [
        ("port", True),
        ("port", "$inputs.port"),
        ("connect_timeout", -1),
        ("statement_timeout", 0),
        ("sslmode", "bad"),
        ("host", ""),
    ],
)
def test_invalid_resolved_settings_do_not_connect(field, value, connect):
    result = v1.PostgreSQLSinkBlockV1(None, None).run(**arguments(**{field: value}))
    assert result["error_status"]
    connect.assert_not_called()


def test_bound_values_and_quoted_identifiers(connect):
    result = v1.PostgreSQLSinkBlockV1(None, None).run(
        **arguments(
            schema_name='my"schema',
            table_name="select",
            data=[{"a": "'); DROP TABLE events; --", "b": None}, {"b": 2, "a": "ok"}],
        )
    )
    cursor = (
        connect.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value
    )
    query, rows = cursor.executemany.call_args.args
    assert (
        query.as_string()
        == 'INSERT INTO "my""schema"."select" ("a", "b") VALUES (%s, %s)'
    )
    assert rows == [("'); DROP TABLE events; --", None), ("ok", 2)]
    assert result == {
        "error_status": False,
        "message": "Successfully inserted 2 records",
    }
    connect.return_value.__exit__.assert_called_once_with(None, None, None)


def test_commit_failure_is_not_success(connect, monkeypatch):
    log = MagicMock()
    monkeypatch.setattr(v1, "logger", log)
    connect.return_value.__exit__.side_effect = psycopg.OperationalError(
        "secret-password"
    )
    result = v1.PostgreSQLSinkBlockV1(None, None).run(**arguments())
    assert result["error_status"]
    assert "Commit outcome may be unknown" in result["message"]
    assert "OperationalError" in result["message"]
    log.error.assert_called_once_with("PostgreSQL Sink: %s", result["message"])
    assert "secret-password" not in result["message"] + str(log.error.call_args)


def test_missing_driver(monkeypatch):
    monkeypatch.setattr(v1, "psycopg", None)
    assert (
        "driver unavailable"
        in v1.PostgreSQLSinkBlockV1(None, None).run(**arguments())["message"]
    )


def test_disabled_sink_precedes_all_processing(monkeypatch, connect):
    tasks, pool = MagicMock(), MagicMock()
    monkeypatch.setattr(v1, "psycopg", None)
    result = v1.PostgreSQLSinkBlockV1(tasks, pool, disable_sinks=True).run(
        **arguments(data=None)
    )
    assert result == disabled_sink_response()
    tasks.add_task.assert_not_called()
    pool.submit.assert_not_called()
    connect.assert_not_called()


def test_background_tasks_take_precedence_and_log_failures(connect, monkeypatch):
    log = MagicMock()
    monkeypatch.setattr(v1, "logger", log)
    tasks, pool = BackgroundTasks(), MagicMock()
    connect.side_effect = psycopg.OperationalError("secret-password")
    result = v1.PostgreSQLSinkBlockV1(tasks, pool).run(**arguments())
    assert result == {"error_status": False, "message": "Data processing scheduled"}
    connect.assert_not_called()
    pool.submit.assert_not_called()
    asyncio.run(tasks())
    assert "PostgreSQL connection failed" in str(log.error.call_args)
    assert "secret-password" not in str(log.error.call_args)


def test_thread_pool_fallback(connect):
    pool = MagicMock()
    result = v1.PostgreSQLSinkBlockV1(None, pool).run(**arguments())
    assert result["message"] == "Data processing scheduled"
    connect.assert_not_called()
    task = pool.submit.call_args.args[0]
    assert not task()["error_status"]


@pytest.mark.parametrize("flag", [False, "false", "False", 0])
def test_synchronous_mode_does_not_schedule(connect, flag):
    tasks, pool = MagicMock(), MagicMock()
    result = v1.PostgreSQLSinkBlockV1(tasks, pool).run(
        **arguments(fire_and_forget=flag)
    )
    assert result["message"] == "Successfully inserted 1 records"
    tasks.add_task.assert_not_called()
    pool.submit.assert_not_called()


def test_enterprise_registration():
    from inference.enterprise.workflows.enterprise_blocks.loader import (
        load_enterprise_blocks,
    )

    assert v1.PostgreSQLSinkBlockV1 in load_enterprise_blocks()


def test_connection_and_transaction_settings(connect):
    v1.PostgreSQLSinkBlockV1(None, None).run(
        **arguments(
            password="secret",
            port=5433,
            sslmode="verify-full",
            connect_timeout=7,
            statement_timeout=1234,
        )
    )
    connect.assert_called_once_with(
        host="localhost",
        port=5433,
        dbname="test",
        user="writer",
        password="secret",
        sslmode="verify-full",
        connect_timeout=7,
        autocommit=False,
    )
    cursor = (
        connect.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value
    )
    cursor.execute.assert_called_once_with(
        "SELECT set_config('statement_timeout', %s, true)", ("1234",)
    )


@pytest.mark.parametrize("value", [None, "sometimes", [], {}])
def test_invalid_boolean_does_not_schedule(value, connect):
    tasks, pool = MagicMock(), MagicMock()
    result = v1.PostgreSQLSinkBlockV1(tasks, pool).run(
        **arguments(fire_and_forget=value)
    )
    assert result["error_status"]
    tasks.add_task.assert_not_called()
    pool.submit.assert_not_called()
    connect.assert_not_called()


@pytest.mark.parametrize(
    "phase,error,category",
    [
        (
            "connection",
            psycopg.OperationalError("sensitive hostname"),
            "OperationalError",
        ),
        ("insert", psycopg.errors.CheckViolation("sensitive row"), "IntegrityError"),
        (
            "commit",
            psycopg.errors.CheckViolation("sensitive deferred constraint"),
            "IntegrityError",
        ),
    ],
)
def test_failure_categories_without_false_commit_uncertainty(
    connect, monkeypatch, phase, error, category
):
    log = MagicMock()
    monkeypatch.setattr(v1, "logger", log)
    if phase == "connection":
        connect.side_effect = error
    elif phase == "insert":
        connect.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value.executemany.side_effect = (
            error
        )
    else:
        connect.return_value.__exit__.side_effect = error
    result = v1.PostgreSQLSinkBlockV1(None, None).run(**arguments())
    assert f"PostgreSQL {phase} failed: {category}" in result["message"]
    assert "unknown" not in result["message"]
    assert "sensitive" not in result["message"] + str(log.error.call_args)


def test_non_ascii_sqlstate_is_not_exposed(connect):
    class InvalidState(psycopg.Error):
        sqlstate = "é1234"

    connect.side_effect = InvalidState("sensitive")
    result = v1.PostgreSQLSinkBlockV1(None, None).run(**arguments())
    assert "SQLSTATE" not in result["message"]


def test_serverless_manifest_contract():
    from inference.core.workflows.prototypes.block import Runtime, Severity

    assert not any(
        restriction.severity == Severity.HARD
        and Runtime.HOSTED_SERVERLESS in restriction.applies_to_runtimes
        for restriction in v1.BlockManifest.get_restrictions()
    )
    assert not v1.BlockManifest.model_json_schema()["ui_manifest"].get(
        "local_only", False
    )
