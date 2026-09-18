"""Workload declarations of the sink blocks.

Sinks are where the portable declaration differs most from the legacy one:
``get_restrictions()`` evaluates this host's flags and returns human notes,
while ``discover_portable_restrictions()`` must return every branch, each with
the flag pinned in its condition, so a caller can answer the question for a
DIFFERENT deployment than the one being asked.
"""

from typing import Any, List

import pytest
import roboflow_workflows.core_steps.sinks.local_file.v1 as local_file_module
from pydantic import ValidationError
from roboflow_workflows.core_steps.sinks.local_file.v1 import (
    BlockManifest as LocalFileManifest,
)
from roboflow_workflows.core_steps.sinks.onvif_movement.v1 import (
    BlockManifest as OnvifManifest,
)
from roboflow_workflows.core_steps.sinks.s3.v1 import BlockManifest as S3Manifest
from roboflow_workflows.core_steps.sinks.webhook.v1 import (
    BlockManifest as WebhookManifest,
)
from roboflow_workflows.enterprise_blocks.sinks.postgresql.v1 import (
    BlockManifest as PostgreSQLManifest,
)
from roboflow_workflows.execution_engine.entities.workload import (
    Discovery,
    Runtime,
    Severity,
    StepExecutionMode,
    WorkOperation,
)
from roboflow_workflows.execution_engine.introspection.workload import (
    describe_workflow_workload,
)
from roboflow_workflows.prototypes.block import COOLDOWN_HTTP_SOFT_PORTABLE_RESTRICTION

LOCAL_STORAGE_FLAG = "ALLOW_WORKFLOW_BLOCKS_ACCESSING_LOCAL_STORAGE"


def _local_file() -> LocalFileManifest:
    return LocalFileManifest(
        type="roboflow_core/local_file_sink@v1",
        name="file_sink",
        content="$steps.formatter.output",
        file_type="csv",
        output_mode="append_log",
        target_directory="/tmp/workflow-output",
        file_name_prefix="results",
    )


def _s3() -> S3Manifest:
    return S3Manifest(
        type="roboflow_core/s3_sink@v1",
        name="s3_sink",
        content="$steps.formatter.output",
        output_mode="append_log",
        bucket_name="bucket",
    )


def _webhook() -> WebhookManifest:
    return WebhookManifest(
        type="roboflow_core/webhook_sink@v1",
        name="webhook",
        url="https://example.com/hook",
        method="POST",
    )


def _onvif() -> OnvifManifest:
    return OnvifManifest(
        type="roboflow_core/onvif_sink@v1",
        name="ptz",
        predictions="$steps.model.predictions",
        camera_ip="192.168.0.10",
        camera_port=80,
        camera_username="user",
        camera_password="secret",
    )


def test_local_file_sink_declares_a_storage_write() -> None:
    assert _local_file().discover_work_operations() == [WorkOperation.STORAGE_WRITE]


def test_local_file_sink_declares_both_branches_of_the_storage_flag() -> None:
    by_code = {
        restriction.code: restriction
        for restriction in _local_file().discover_portable_restrictions()
    }
    assert by_code["local_storage_access_disabled"].severity is Severity.HARD
    assert by_code["local_storage_access_disabled"].when.configuration_equals == {
        LOCAL_STORAGE_FLAG: False
    }
    assert set(by_code["local_storage_access_disabled"].when.runtimes) == {
        Runtime.HOSTED_SERVERLESS,
        Runtime.DEDICATED_DEPLOYMENT,
    }
    volume = by_code["writes_to_deployment_volume_not_retrievable"]
    assert volume.severity is Severity.SOFT
    assert volume.when.runtimes == [Runtime.DEDICATED_DEPLOYMENT]
    assert volume.when.configuration_equals == {LOCAL_STORAGE_FLAG: True}
    ephemeral = by_code["ephemeral_container_disk_loses_writes"]
    assert ephemeral.when.runtimes == [Runtime.HOSTED_SERVERLESS]
    assert ephemeral.when.configuration_equals == {LOCAL_STORAGE_FLAG: True}


def test_local_file_declaration_is_independent_of_this_host_flag(monkeypatch) -> None:
    """The legacy hook branches on the flag; the portable one must not."""
    manifest = _local_file()
    monkeypatch.setattr(local_file_module, LOCAL_STORAGE_FLAG, True, raising=False)
    with_storage = manifest.discover_portable_restrictions()
    legacy_with_storage = LocalFileManifest.get_restrictions()
    monkeypatch.setattr(local_file_module, LOCAL_STORAGE_FLAG, False, raising=False)
    without_storage = manifest.discover_portable_restrictions()
    legacy_without_storage = LocalFileManifest.get_restrictions()
    assert with_storage == without_storage
    # the legacy API keeps its environment-dependent behaviour untouched
    assert legacy_with_storage != legacy_without_storage


def test_s3_sink_declares_storage_and_transport() -> None:
    manifest = _s3()
    assert manifest.discover_work_operations() == [
        WorkOperation.STORAGE_WRITE,
        WorkOperation.EXTERNAL_REQUEST,
    ]
    restrictions = manifest.discover_portable_restrictions()
    assert [restriction.code for restriction in restrictions] == [
        "s3_append_buffer_resets_on_stateless_http"
    ]
    assert restrictions[0].when.step_execution_modes == [StepExecutionMode.REMOTE]


def test_a_notification_sink_declares_the_cooldown_caveat() -> None:
    manifest = _webhook()
    assert manifest.discover_work_operations() == [WorkOperation.EXTERNAL_REQUEST]
    assert manifest.discover_portable_restrictions() == [
        COOLDOWN_HTTP_SOFT_PORTABLE_RESTRICTION
    ]


def test_onvif_sink_declares_the_lan_requirement() -> None:
    manifest = _onvif()
    assert manifest.discover_work_operations() == [WorkOperation.EXTERNAL_REQUEST]
    restrictions = manifest.discover_portable_restrictions()
    assert [restriction.code for restriction in restrictions] == [
        "requires_lan_access_to_device"
    ]
    assert restrictions[0].severity is Severity.HARD
    assert set(restrictions[0].when.runtimes) == {
        Runtime.HOSTED_SERVERLESS,
        Runtime.DEDICATED_DEPLOYMENT,
    }
    # the legacy note and the portable code describe the same two runtimes
    legacy = OnvifManifest.get_restrictions()
    assert len(legacy) == 1
    assert legacy[0].severity is Severity.HARD
    assert set(legacy[0].applies_to_runtimes) == set(restrictions[0].when.runtimes)


def test_a_local_write_is_not_declared_as_an_external_request() -> None:
    """Local storage and network transport are different costs."""
    assert (
        WorkOperation.EXTERNAL_REQUEST not in _local_file().discover_work_operations()
    )


# ---------------------------------------------------------------------------
# Codex round-001 F003 / F004: a caveat that only applies in one MODE must not
# be declared unconditionally, and a mode supplied at run time must not be
# answered with a complete declaration in either direction.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "output_mode, expected",
    [
        ("append_log", ["s3_append_buffer_resets_on_stateless_http"]),
        ("separate_files", []),
    ],
)
def test_s3_append_caveat_follows_the_literal_output_mode(
    output_mode: str, expected: List[str]
) -> None:
    manifest = S3Manifest(
        type="roboflow_core/s3_sink@v1",
        name="s3_sink",
        content="$steps.formatter.output",
        output_mode=output_mode,
        bucket_name="bucket",
    )
    declared = manifest.discover_portable_restrictions()
    assert isinstance(declared, list), "a literal mode is fully knowable"
    assert [restriction.code for restriction in declared] == expected


def test_s3_output_mode_cannot_hold_a_selector() -> None:
    """Why the S3 hook needs no selector branch.

    `output_mode` is a plain `Literal`, so an unresolved value cannot reach the
    hook. If the field is ever widened to accept a selector this test fails and
    the incomplete-discovery branch has to be added, exactly as for
    `fire_and_forget` below.
    """
    with pytest.raises(ValidationError):
        S3Manifest(
            type="roboflow_core/s3_sink@v1",
            name="s3_sink",
            content="$steps.formatter.output",
            output_mode="$inputs.output_mode",
            bucket_name="bucket",
        )


def _postgresql(fire_and_forget: Any) -> PostgreSQLManifest:
    return PostgreSQLManifest(
        type="roboflow_core/postgresql_sink@v1",
        name="database",
        host="example.invalid",
        database="db",
        username="user",
        table_name="events",
        data={"value": 1},
        fire_and_forget=fire_and_forget,
    )


@pytest.mark.parametrize(
    "fire_and_forget, expected",
    [(True, ["fire_and_forget_hides_persistence_failures"]), (False, [])],
)
def test_postgresql_caveat_follows_the_literal_fire_and_forget(
    fire_and_forget: bool, expected: List[str]
) -> None:
    declared = _postgresql(fire_and_forget).discover_portable_restrictions()
    assert isinstance(declared, list), "a literal switch is fully knowable"
    assert [restriction.code for restriction in declared] == expected


def test_postgresql_reports_a_selector_as_unknown_not_as_absence() -> None:
    """A runtime value means the caveat MAY apply.

    Declaring it would be as wrong as declaring its absence, so nothing is
    claimed complete and the reason names the field and the step.
    """
    declared = _postgresql("$inputs.fire_and_forget").discover_portable_restrictions()
    assert isinstance(declared, Discovery)
    assert declared.complete is False
    assert declared.items == []
    assert declared.unknown_reasons == [
        "fire_and_forget_selector_unresolved:$steps.database"
    ]


def test_the_legacy_postgresql_and_s3_declarations_are_unchanged() -> None:
    """`get_restrictions()` is a classmethod and stays mode-blind.

    F003/F004 changed only the portable hooks. The legacy API keeps returning
    its single unconditional note for both blocks, whatever the manifest says.
    """
    for manifest_class, expected_codes in (
        (S3Manifest, 1),
        (PostgreSQLManifest, 1),
    ):
        legacy = manifest_class.get_restrictions()
        assert len(legacy) == expected_codes
        assert legacy[0].note


# Adopted from the Codex round-001 reviewer reproducers, through the public
# introspection API.
@pytest.mark.parametrize("output_mode", ["append_log", "separate_files"])
def test_s3_append_restriction_through_the_public_api(output_mode: str) -> None:
    definition = {
        "version": "1.0",
        "inputs": [
            {
                "type": "WorkflowParameter",
                "name": "content",
                "default_value": "record",
            }
        ],
        "steps": [
            {
                "type": "roboflow_core/s3_sink@v1",
                "name": "sink",
                "content": "$inputs.content",
                "output_mode": output_mode,
                "bucket_name": "test-bucket",
            }
        ],
        "outputs": [],
    }
    restrictions = describe_workflow_workload(definition).steps[0].restrictions
    assert restrictions.complete
    codes = {restriction.code for restriction in restrictions.items}
    assert ("s3_append_buffer_resets_on_stateless_http" in codes) is (
        output_mode == "append_log"
    )


@pytest.mark.parametrize("fire_and_forget", [True, False, "$inputs.fire_and_forget"])
def test_postgresql_restriction_through_the_public_api(fire_and_forget) -> None:
    from roboflow_workflows.execution_engine.introspection import blocks_loader

    definition = {
        "version": "1.0",
        "inputs": [
            {
                "type": "WorkflowParameter",
                "name": "fire_and_forget",
                "default_value": True,
            }
        ],
        "steps": [
            {
                "type": "roboflow_core/postgresql_sink@v1",
                "name": "sink",
                "host": "example.invalid",
                "database": "db",
                "username": "user",
                "table_name": "events",
                "data": {"value": 1},
                "fire_and_forget": fire_and_forget,
            }
        ],
        "outputs": [],
    }
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv(
            "WORKFLOWS_PLUGINS", "roboflow_workflows.enterprise_blocks.loader"
        )
        blocks_loader.clear_caches()
        try:
            restrictions = describe_workflow_workload(definition).steps[0].restrictions
        finally:
            blocks_loader.clear_caches()
    codes = {restriction.code for restriction in restrictions.items}
    if isinstance(fire_and_forget, bool):
        assert restrictions.complete
        assert (
            "fire_and_forget_hides_persistence_failures" in codes
        ) is fire_and_forget
    else:
        # the input's default_value is True; the declaration must NOT adopt it
        assert not restrictions.complete
        assert restrictions.items == []
        assert restrictions.unknown_reasons == [
            "fire_and_forget_selector_unresolved:$steps.sink"
        ]
