"""SQL sinks through the real compiler and session, with stand-in clients.

The blocks are registered in a test-local catalogue until root adds them to
the shared one. Every client is an in-process ``RecordingClient``; no test
opens a database connection::

    inputs / secrets store ─▶ v2/postgresql_sink@v1 ──────────┐
                            ─▶ v2/microsoft_sql_server_sink@v1 ┴▶ RecordingClient
                                        │                        (host-owned, borrowed)
                                        └▶ outputs: error_status, message
"""

from typing import Any, Dict, List, Optional

import pytest
from roboflow_workflows.v2.blocks.industrial.sql import (
    PostgreSQLSinkBlock,
    SQLInsert,
    SQLServerSinkBlock,
    SQLSinkError,
)
from roboflow_workflows.v2.blocks.secrets import (
    SECRET_RECORDING_CODEC,
    EnvironmentSecretsStoreBlock,
    StaticSecrets,
)
from roboflow_workflows.v2.blocks.sinks.runtime import BackgroundEffects
from roboflow_workflows.v2.data.secrets import SECRET_KIND, Secret
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.errors import (
    ParamsValidationError,
    StepExecutionError,
)
from roboflow_workflows.v2.engine.introspection.workload import discover_workload

PASSWORD = "pw-SENTINEL-93be"
CATALOGUE = Catalogue(
    [PostgreSQLSinkBlock, SQLServerSinkBlock, EnvironmentSecretsStoreBlock],
    kinds=[SECRET_KIND],
    codecs=[SECRET_RECORDING_CODEC],
)
OUTPUTS = [
    {
        "type": "JsonField",
        "name": "error_status",
        "selector": "$steps.sink.error_status",
    },
    {"type": "JsonField", "name": "message", "selector": "$steps.sink.message"},
]
INPUTS = [
    {"type": "WorkflowParameter", "name": "row"},
    {"type": "WorkflowParameter", "name": "password", "kind": ["secret"]},
    {"type": "WorkflowParameter", "name": "fire_and_forget", "default_value": False},
]


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


def _postgresql_step(**overrides: Any) -> Dict[str, Any]:
    step = {
        "type": "v2/postgresql_sink@v1",
        "name": "sink",
        "host": "db.example.com",
        "database": "production",
        "username": "writer",
        "password": "$inputs.password",
        "table_name": "detections",
        "data": "$inputs.row",
        "fire_and_forget": "$inputs.fire_and_forget",
    }
    step.update(overrides)

    return step


def _compile(step: Dict[str, Any], *extra_steps: Dict[str, Any]):
    definition = {
        "version": "2.0",
        "inputs": INPUTS,
        "steps": [*extra_steps, step],
        "outputs": OUTPUTS,
    }
    plan = compile_workflow(definition, catalogue=CATALOGUE)

    return plan


def _chain_text(error: Optional[BaseException]) -> str:
    texts = []
    while error is not None:
        texts.append(f"{type(error).__name__}: {error}")
        error = error.__cause__ or error.__context__

    return "\n".join(texts)


def test_a_compiled_postgresql_step_writes_one_parameterized_batch() -> None:
    # given
    plan = _compile(_postgresql_step())
    client = RecordingClient()
    session = plan.create_session(resources={"postgresql_client": client})
    rows = [
        {"label": "part", "confidence": 0.9},
        {"label": "O'Brien", "confidence": None},
    ]

    # when
    result = session.run({"row": rows, "password": PASSWORD})

    # then
    assert result.rows(serialize=True) == [
        {"error_status": False, "message": "Successfully inserted 2 records"}
    ]
    target, insert = client.calls[0]
    assert insert.statement == (
        'INSERT INTO "public"."detections" ("label", "confidence") VALUES (%s, %s)'
    )
    assert insert.rows == (("part", 0.9), ("O'Brien", None))
    assert target.password.name == "input"
    assert target.password.reveal() == PASSWORD


def test_a_secrets_store_password_reaches_the_client_and_failures_are_scrubbed() -> (
    None
):
    # given
    store = {
        "type": "v2/environment_secrets_store@v1",
        "name": "secrets",
        "variables_storing_secrets": ["PG_PASSWORD"],
    }
    plan = _compile(_postgresql_step(password="$steps.secrets.pg_password"), store)
    client = RecordingClient(error=SQLSinkError(f"rejected {PASSWORD}"))
    session = plan.create_session(
        resources={
            "postgresql_client": client,
            "secrets": StaticSecrets({"PG_PASSWORD": PASSWORD}),
        }
    )

    # when
    row = session.run({"row": {"label": "part"}}).rows(serialize=True)[0]

    # then
    assert row == {"error_status": True, "message": "rejected <secret PG_PASSWORD>"}
    assert client.calls[0][0].password.name == "PG_PASSWORD"
    assert len(client.calls) == 1


def test_an_unavailable_secret_fails_the_step_and_writes_nothing() -> None:
    # given
    plan = _compile(_postgresql_step())
    client = RecordingClient()
    session = plan.create_session(resources={"postgresql_client": client})

    # when
    with pytest.raises(StepExecutionError) as raised:
        session.run({"row": {"label": "part"}, "password": Secret.unavailable("PG")})

    # then
    assert "SecretUnavailableError" in _chain_text(raised.value)
    assert client.calls == []


def test_background_writes_finish_when_the_host_closes_its_effects() -> None:
    # given
    plan = _compile(_postgresql_step())
    client = RecordingClient()

    # when
    with BackgroundEffects() as effects:
        session = plan.create_session(
            resources={"postgresql_client": client, "sink_effects": effects}
        )
        rows = [
            session.run(
                {"row": {"n": n}, "password": PASSWORD, "fire_and_forget": True}
            ).rows()[0]
            for n in range(3)
        ]

    # then
    assert rows == [{"error_status": False, "message": "Data processing scheduled"}] * 3
    assert sorted(insert.rows for _, insert in client.calls) == [
        ((0,),),
        ((1,),),
        ((2,),),
    ]
    assert effects.stats().completed == 3


def test_disabled_sinks_resource_skips_the_client() -> None:
    # given
    plan = _compile(_postgresql_step())
    client = RecordingClient()
    session = plan.create_session(
        resources={"postgresql_client": client, "disable_sinks": True}
    )

    # when
    row = session.run({"row": {"label": "part"}, "password": PASSWORD}).rows()[0]

    # then
    assert row == {
        "error_status": False,
        "message": "Sink was disabled by workflow execution policy",
    }
    assert client.calls == []


def test_one_borrowed_client_serves_two_sessions_and_stays_open() -> None:
    # given
    plan = _compile(_postgresql_step())
    client = RecordingClient()

    # when
    for _ in range(2):
        with plan.create_session(resources={"postgresql_client": client}) as session:
            session.run({"row": {"label": "part"}, "password": PASSWORD})

    # then
    assert len(client.calls) == 2
    assert client.closed is False


def test_a_compiled_sql_server_step_writes_bracketed_names() -> None:
    # given
    step = {
        "type": "v2/microsoft_sql_server_sink@v1",
        "name": "sink",
        "host": "sql.example.com",
        "database": "production_db",
        "table_name": "dbo.detections",
        "data": "$inputs.row",
        "fire_and_forget": False,
    }
    plan = _compile(step)
    client = RecordingClient()
    session = plan.create_session(resources={"sql_server_client": client})

    # when
    result = session.run({"row": [{"part": "ok"}, {"part": "defect"}]})

    # then
    assert result.rows() == [
        {"error_status": False, "message": "Successfully inserted 2 records"}
    ]
    target, insert = client.calls[0]
    assert insert.statement == "INSERT INTO [dbo].[detections] ([part]) VALUES (?)"
    assert insert.rows == (("ok",), ("defect",))
    assert (target.username, target.password, target.port) == (None, None, 1433)


def test_a_literal_out_of_range_port_fails_compilation() -> None:
    # when
    with pytest.raises(ParamsValidationError) as raised:
        _compile(_postgresql_step(port=0))

    # then
    assert "port" in str(raised.value)


def test_a_selector_fire_and_forget_leaves_restrictions_incomplete() -> None:
    # given
    plan = _compile(_postgresql_step())

    # when
    restrictions = (
        discover_workload(plan).step("$steps.sink").describe()["restrictions"]
    )

    # then
    assert restrictions["complete"] is False
    assert [item["code"] for item in restrictions["items"]] == [
        "unavailable_on_hosted_platform"
    ]
    assert restrictions["unknown_reasons"][0]["details"]["node_id"] == "$steps.sink"
