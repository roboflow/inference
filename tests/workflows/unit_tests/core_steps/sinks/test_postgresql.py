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
        **arguments(port="$inputs.port")
    )
    assert manifest.port == "$inputs.port"
    assert manifest.sslmode == "require"
    assert manifest.statement_timeout == 10000
    assert {output.name for output in manifest.describe_outputs()} == {
        "message",
        "error_status",
    }
    assert manifest.model_json_schema()["ui_manifest"].get("local_only", False) is False


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
            **arguments(**{field: value})
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


def test_commit_failure_is_not_success(connect, caplog):
    connect.return_value.__exit__.side_effect = psycopg.OperationalError(
        "secret-password"
    )
    result = v1.PostgreSQLSinkBlockV1(None, None).run(**arguments())
    assert result["error_status"]
    assert "secret-password" not in result["message"] + caplog.text


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
    assert "PostgreSQL operation failed" in str(log.error.call_args)
    assert "secret-password" not in str(log.error.call_args)


def test_thread_pool_fallback(connect):
    pool = MagicMock()
    result = v1.PostgreSQLSinkBlockV1(None, pool).run(**arguments())
    assert result["message"] == "Data processing scheduled"
    connect.assert_not_called()
    task = pool.submit.call_args.args[0]
    assert not task()["error_status"]


def test_synchronous_mode_does_not_schedule(connect):
    tasks, pool = MagicMock(), MagicMock()
    result = v1.PostgreSQLSinkBlockV1(tasks, pool).run(
        **arguments(fire_and_forget=False)
    )
    assert result["message"] == "Successfully inserted 1 records"
    tasks.add_task.assert_not_called()
    pool.submit.assert_not_called()


def test_enterprise_registration():
    from inference.enterprise.workflows.enterprise_blocks.loader import (
        load_enterprise_blocks,
    )

    assert v1.PostgreSQLSinkBlockV1 in load_enterprise_blocks()
