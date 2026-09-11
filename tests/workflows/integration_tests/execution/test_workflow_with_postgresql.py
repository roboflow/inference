"""Use POSTGRESQL_TEST_DSN pointing at a disposable database with CREATE privileges."""

import os
from typing import Literal
from unittest.mock import MagicMock
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql

from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import LIST_OF_VALUES_KIND
from inference.core.workflows.prototypes.block import (
    WorkflowBlock,
    WorkflowBlockManifest,
)
from inference.enterprise.workflows.enterprise_blocks.sinks.postgresql.v1 import (
    PostgreSQLSinkBlockV1,
)


@pytest.fixture
def database():
    dsn = os.environ.get("POSTGRESQL_TEST_DSN")
    if not dsn:
        pytest.skip("Set POSTGRESQL_TEST_DSN to a disposable PostgreSQL database")
    schema = "inference_test_" + uuid4().hex
    with psycopg.connect(dsn, autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        try:
            connection.execute(
                sql.SQL(
                    'CREATE TABLE {} ("select" text, score integer CHECK (score >= 0))'
                ).format(sql.Identifier(schema, 'event"rows'))
            )
            params = psycopg.conninfo.conninfo_to_dict(dsn)
            kwargs = dict(
                host=params.get("host", "localhost"),
                port=int(params.get("port", 5432)),
                database=params.get("dbname", "postgres"),
                username=params.get("user", "postgres"),
                password=params.get("password"),
                sslmode=params.get("sslmode", "disable"),
                schema_name=schema,
                table_name='event"rows',
                fire_and_forget=False,
            )
            yield connection, kwargs
        finally:
            connection.execute(
                sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema))
            )


def select_rows(connection, kwargs):
    return connection.execute(
        sql.SQL('SELECT "select", score FROM {} ORDER BY score').format(
            sql.Identifier(kwargs["schema_name"], kwargs["table_name"])
        )
    ).fetchall()


def test_real_insert_native_values_and_null(database):
    connection, kwargs = database
    result = PostgreSQLSinkBlockV1(None, None).run(
        **kwargs,
        data=[
            {"select": "'); DROP TABLE anything; --", "score": 1},
            {"score": 2, "select": None},
        ]
    )
    assert not result["error_status"]
    assert select_rows(connection, kwargs) == [
        ("'); DROP TABLE anything; --", 1),
        (None, 2),
    ]


def test_real_batch_rollback(database):
    connection, kwargs = database
    result = PostgreSQLSinkBlockV1(None, None).run(
        **kwargs,
        data=[{"select": "valid", "score": 1}, {"select": "invalid", "score": -1}]
    )
    assert result["error_status"]
    assert "23514" in result["message"]
    assert select_rows(connection, kwargs) == []


def test_statement_timeout_releases_connection(database):
    connection, kwargs = database
    connection.execute("BEGIN")
    connection.execute(
        sql.SQL("LOCK TABLE {} IN ACCESS EXCLUSIVE MODE").format(
            sql.Identifier(kwargs["schema_name"], kwargs["table_name"])
        )
    )
    try:
        result = PostgreSQLSinkBlockV1(None, None).run(
            **kwargs, statement_timeout=100, data={"select": "blocked", "score": 1}
        )
        assert result["error_status"]
        assert "57014" in result["message"]
    finally:
        connection.execute("ROLLBACK")
    assert select_rows(connection, kwargs) == []


@pytest.fixture
def enterprise_blocks(monkeypatch):
    from inference.core import env
    from inference.core.workflows.execution_engine.introspection import blocks_loader
    from inference.core.workflows.execution_engine.v1.compiler.core import (
        COMPILATION_CACHE,
    )

    monkeypatch.setattr(env, "LOAD_ENTERPRISE_BLOCKS", True)
    monkeypatch.setattr(blocks_loader, "LOAD_ENTERPRISE_BLOCKS", True)
    blocks_loader.load_core_workflow_blocks.cache_clear()
    yield
    blocks_loader.load_core_workflow_blocks.cache_clear()
    with COMPILATION_CACHE._cache_lock:
        COMPILATION_CACHE._cache.clear()
        COMPILATION_CACHE._keys_buffer.clear()


def test_workflow_resolves_inputs_and_writes(database, enterprise_blocks):
    connection, kwargs = database
    definition = {
        "version": "1.0",
        "inputs": [
            {"type": "WorkflowParameter", "name": key}
            for key in ["rows", "port", "password"]
        ],
        "steps": [
            {
                "type": "roboflow_core/postgresql_sink@v1",
                "name": "pg",
                **kwargs,
                "data": "$inputs.rows",
                "port": "$inputs.port",
                "password": "$inputs.password",
            }
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "status",
                "selector": "$steps.pg.error_status",
            }
        ],
    }
    engine = ExecutionEngine.init(workflow_definition=definition)
    result = engine.run(
        runtime_parameters={
            "rows": {"select": "workflow", "score": 3},
            "port": kwargs["port"],
            "password": kwargs["password"],
        }
    )
    assert result == [{"status": False}]
    assert select_rows(connection, kwargs) == [("workflow", 3)]


class RowsManifest(WorkflowBlockManifest):
    type: Literal["test/rows@v1"]

    @classmethod
    def describe_outputs(cls):
        return [OutputDefinition(name="rows", kind=[LIST_OF_VALUES_KIND])]


class RowsBlock(WorkflowBlock):
    @classmethod
    def get_manifest(cls):
        return RowsManifest

    def run(self):
        return {
            "rows": [{"select": "upstream", "score": 4}, {"select": "list", "score": 5}]
        }


def test_workflow_upstream_list_and_false_boolean_selector(
    database, enterprise_blocks, monkeypatch
):
    from inference.core.workflows.execution_engine.introspection import blocks_loader

    connection, kwargs = database
    original_load = blocks_loader.load_blocks
    monkeypatch.setattr(
        blocks_loader, "load_blocks", lambda: original_load() + [RowsBlock]
    )
    blocks_loader.load_core_workflow_blocks.cache_clear()
    tasks = MagicMock()
    definition = {
        "version": "1.0",
        "inputs": [{"type": "WorkflowParameter", "name": "background"}],
        "steps": [
            {"type": "test/rows@v1", "name": "source"},
            {
                "type": "roboflow_core/postgresql_sink@v1",
                "name": "pg",
                **kwargs,
                "data": "$steps.source.rows",
                "fire_and_forget": "$inputs.background",
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "status",
                "selector": "$steps.pg.error_status",
            }
        ],
    }
    engine = ExecutionEngine.init(
        workflow_definition=definition,
        init_parameters={"workflows_core.background_tasks": tasks},
    )
    assert engine.run(runtime_parameters={"background": "false"}) == [{"status": False}]
    tasks.add_task.assert_not_called()
    assert select_rows(connection, kwargs) == [("upstream", 4), ("list", 5)]
