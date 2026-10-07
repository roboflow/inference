"""Fire-and-forget declarations of the Microsoft SQL Server, Event Writer and
OPC UA Writer sinks.

All three dispatch their write in the background when ``fire_and_forget`` is
true, so a failed write is not returned. Their actual declarations used to omit
that caveat. They now declare it for a literal ``True``, omit it for a literal
``False``, and report a selector as unknown rather than guessing either way.
The OPC UA Writer keeps its cooldown caveat in every branch. The legacy editor
``get_restrictions()`` stays unchanged.

Manifests are real and validated, and the public report comes from the real
introspection API. No database, event service or OPC UA server is contacted:
the two dispatch tests replace the write with a stub.
"""

from typing import Any, Callable, Dict, List
from unittest.mock import MagicMock

import pytest
import roboflow_workflows.enterprise_blocks.sinks.opc_writer.v1 as opc_writer_module
from fastapi import BackgroundTasks
from roboflow_workflows.core_steps.common.workload_presets import (
    COOLDOWN_ACTUAL_RESTRICTION,
    FIRE_AND_FORGET_RESTRICTION,
)
from roboflow_workflows.enterprise_blocks.sinks.event_writer.v1 import (
    BlockManifest as EventWriterManifest,
)
from roboflow_workflows.enterprise_blocks.sinks.microsoft_sql_server.v1 import (
    BlockManifest as SQLServerManifest,
)
from roboflow_workflows.enterprise_blocks.sinks.microsoft_sql_server.v1 import (
    MicrosoftSQLServerSinkBlockV1,
)
from roboflow_workflows.enterprise_blocks.sinks.opc_writer.v1 import (
    BlockManifest as OPCWriterManifest,
)
from roboflow_workflows.enterprise_blocks.sinks.opc_writer.v1 import (
    OPCWriterSinkBlockV1,
)
from roboflow_workflows.execution_engine.entities.workload import (
    Discovery,
    restriction_metadata_of,
    unresolved_selector_problem,
)
from roboflow_workflows.execution_engine.introspection import blocks_loader
from roboflow_workflows.execution_engine.introspection.workload import (
    describe_workflow_workload,
)

from tests.unit_tests.workload_declaration_helpers import (
    declared_restrictions,
    portable_restrictions_discovery,
)

ENTERPRISE_PLUGIN = "roboflow_workflows.enterprise_blocks.loader"
SQL_SERVER_TYPE = "roboflow_core/microsoft_sql_server_sink@v1"
EVENT_WRITER_TYPE = "roboflow_enterprise/event_writer_sink@v1"
OPC_WRITER_TYPE = "roboflow_enterprise/opc_writer_sink@v1"
FIRE_AND_FORGET_CODE = "fire_and_forget_hides_persistence_failures"
COOLDOWN_CODE = "cooldown_timer_resets_on_stateless_http"
SELECTOR = "$inputs.fire_and_forget"
FIRE_AND_FORGET_VALUES = [True, False, SELECTOR]
STEP_NAME = "sink"


def _sql_server_step(fire_and_forget: Any) -> Dict[str, Any]:
    return {
        "type": SQL_SERVER_TYPE,
        "name": STEP_NAME,
        "host": "db.example.invalid",
        "database": "db",
        "table_name": "events",
        "data": {"value": 1},
        "fire_and_forget": fire_and_forget,
    }


def _event_writer_step(fire_and_forget: Any) -> Dict[str, Any]:
    return {
        "type": EVENT_WRITER_TYPE,
        "name": STEP_NAME,
        "event_ingestion_url": "http://events.example.invalid",
        "event_schema": "custom",
        "output_image": "$inputs.image",
        "custom_value": "anomaly",
        "fire_and_forget": fire_and_forget,
    }


def _opc_writer_step(fire_and_forget: Any) -> Dict[str, Any]:
    return {
        "type": OPC_WRITER_TYPE,
        "name": STEP_NAME,
        "url": "opc.tcp://10.0.0.5:4840",
        "namespace": "http://example.com/ns",
        "object_name": "Line1",
        "variable_name": "Status",
        "value": "ok",
        "fire_and_forget": fire_and_forget,
    }


# (manifest class, step builder, restrictions known independently of the switch)
BLOCKS = {
    SQL_SERVER_TYPE: (SQLServerManifest, _sql_server_step, []),
    EVENT_WRITER_TYPE: (EventWriterManifest, _event_writer_step, []),
    OPC_WRITER_TYPE: (
        OPCWriterManifest,
        _opc_writer_step,
        [COOLDOWN_ACTUAL_RESTRICTION],
    ),
}


def _manifest(block_type: str, fire_and_forget: Any) -> Any:
    manifest_class, build_step, _ = BLOCKS[block_type]
    manifest = manifest_class.model_validate(build_step(fire_and_forget))

    return manifest


def _expected_items(block_type: str, fire_and_forget: Any) -> List[Any]:
    _, _, unconditional = BLOCKS[block_type]
    expected = list(unconditional)
    if fire_and_forget is True:
        expected.append(FIRE_AND_FORGET_RESTRICTION)

    return expected


def _selector_reason(node_id: str) -> Any:
    reason = unresolved_selector_problem(
        node_id=node_id,
        declaration="restrictions",
        field="fire_and_forget",
        selector=SELECTOR,
    )

    return reason


def _workflow_definition(
    build_step: Callable[[Any], Dict[str, Any]], fire_and_forget: Any
) -> Dict[str, Any]:
    return {
        "version": "1.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
            {
                "type": "WorkflowParameter",
                "name": "fire_and_forget",
                "default_value": True,
            },
        ],
        "steps": [build_step(fire_and_forget)],
        "outputs": [
            {
                "type": "JsonField",
                "name": "error_status",
                "selector": f"$steps.{STEP_NAME}.error_status",
            }
        ],
    }


def _describe_with_enterprise_plugin(definition: Dict[str, Any]) -> Any:
    """Describe a definition with the enterprise plugin registered.

    Every loader cache is cleared on the way in AND on the way out, so a cached
    registry does not leak into other modules.
    """
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("WORKFLOWS_PLUGINS", ENTERPRISE_PLUGIN)
        blocks_loader.clear_caches()
        try:
            description = describe_workflow_workload(definition)
        finally:
            blocks_loader.clear_caches()

    return description


# ---------------------------------------------------------------------------
# Validated manifests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fire_and_forget", FIRE_AND_FORGET_VALUES)
@pytest.mark.parametrize("block_type", list(BLOCKS))
def test_the_caveat_follows_the_fire_and_forget_switch(
    block_type: str, fire_and_forget: Any
) -> None:
    manifest = _manifest(block_type, fire_and_forget)
    expected = _expected_items(block_type, fire_and_forget)

    # exact: no hosted-platform gate either, run() has no such guard here
    assert declared_restrictions(manifest) == expected
    discovery = portable_restrictions_discovery(manifest)
    assert discovery.items == [restriction_metadata_of(item) for item in expected]
    if isinstance(fire_and_forget, bool):
        assert discovery.complete is True, "a literal switch is fully knowable"
        assert discovery.unknown_reasons == []
    else:
        # the caveat MAY apply: neither declared nor claimed absent
        assert discovery.complete is False
        assert discovery.unknown_reasons == [_selector_reason("$steps.sink")]


@pytest.mark.parametrize("fire_and_forget", FIRE_AND_FORGET_VALUES)
@pytest.mark.parametrize("block_type", list(BLOCKS))
def test_the_host_view_matches_the_portable_view(
    block_type: str, fire_and_forget: Any
) -> None:
    """The caveat has no configuration predicate, so this host drops nothing."""
    manifest = _manifest(block_type, fire_and_forget)

    host_view = manifest.get_actual_restrictions(ignore_environment_restrictions=False)
    portable_view = manifest.get_actual_restrictions(
        ignore_environment_restrictions=True
    )
    assert isinstance(host_view, Discovery)
    assert host_view == portable_view


@pytest.mark.parametrize("manifest_class", [SQLServerManifest, EventWriterManifest])
def test_legacy_editor_restrictions_stay_unchanged(manifest_class: type) -> None:
    # the OPC UA Writer is pinned in test_industrial_workload_restrictions.py
    assert manifest_class.get_restrictions() == []


# ---------------------------------------------------------------------------
# Public workload report
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fire_and_forget", FIRE_AND_FORGET_VALUES)
@pytest.mark.parametrize("block_type", list(BLOCKS))
def test_the_caveat_through_the_public_api(
    block_type: str, fire_and_forget: Any
) -> None:
    _, build_step, _ = BLOCKS[block_type]
    description = _describe_with_enterprise_plugin(
        _workflow_definition(build_step, fire_and_forget)
    )

    [step] = description.steps
    assert step.node_id == "$steps.sink"
    assert step.block_type == block_type
    restrictions = step.restrictions
    codes = [restriction.code for restriction in restrictions.items]
    expected_codes = [
        item.code for item in _expected_items(block_type, fire_and_forget)
    ]
    assert codes == expected_codes
    if isinstance(fire_and_forget, bool):
        assert restrictions.complete
        assert restrictions.unknown_reasons == []
    else:
        # the input's default_value is True; the declaration must NOT adopt it
        assert not restrictions.complete
        assert FIRE_AND_FORGET_CODE not in codes
        assert restrictions.unknown_reasons == [_selector_reason("$steps.sink")]
    if block_type == OPC_WRITER_TYPE:
        assert COOLDOWN_CODE in codes


# ---------------------------------------------------------------------------
# Dispatch: the premise of the caveat, for the two blocks without such tests
# ---------------------------------------------------------------------------


def test_sql_server_fire_and_forget_returns_before_the_insert_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    insert = MagicMock(return_value={"error_status": True, "message": "failed"})
    monkeypatch.setattr(MicrosoftSQLServerSinkBlockV1, "_process_data", insert)
    tasks = BackgroundTasks()
    block = MicrosoftSQLServerSinkBlockV1(tasks, None)
    arguments = {
        "host": "db.example.invalid",
        "port": 1433,
        "database": "db",
        "table_name": "events",
        "data": {"value": 1},
    }

    scheduled = block.run(**arguments, fire_and_forget=True)
    assert scheduled == {"error_status": False, "message": "Data processing scheduled"}
    insert.assert_not_called()
    assert len(tasks.tasks) == 1

    awaited = block.run(**arguments, fire_and_forget=False)
    assert awaited == {"error_status": True, "message": "failed"}
    insert.assert_called_once()


def test_opc_writer_fire_and_forget_returns_before_the_write_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write = MagicMock(return_value=(True, "failed"))
    monkeypatch.setattr(opc_writer_module, "opc_connect_and_write_value", write)
    tasks = BackgroundTasks()
    arguments = {
        "url": "opc.tcp://10.0.0.5:4840",
        "namespace": "http://example.com/ns",
        "user_name": None,
        "password": None,
        "object_name": "Line1",
        "variable_name": "Status",
        "value": "ok",
        "cooldown_seconds": 0,
    }

    scheduled = OPCWriterSinkBlockV1(tasks, None).run(**arguments, fire_and_forget=True)
    assert scheduled["error_status"] is False
    write.assert_not_called()
    assert len(tasks.tasks) == 1

    awaited = OPCWriterSinkBlockV1(tasks, None).run(**arguments, fire_and_forget=False)
    assert awaited["error_status"] is True
    assert awaited["message"] == "failed"
    write.assert_called_once()
