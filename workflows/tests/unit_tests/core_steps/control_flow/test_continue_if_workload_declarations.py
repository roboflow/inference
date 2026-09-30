"""ContinueIf declares its ``stop_delay`` grace-period state.

With ``stop_delay > 0`` the block keeps the time of the last successful
condition in its instance (``ContinueIfBlockV1.start_time``) and keeps the
branch open for that long. The grace period is lost wherever that instance is
not preserved between calls - the same failure mode, and therefore the same
restriction code, as a sink cooldown timer. The default ``stop_delay`` of zero
keeps no such state and must not declare it.

The ``stop_delay`` field also admits a selector, but its ``gt=0`` constraint
makes normal schema validation reject both a selector and an explicit zero
(a preexisting inconsistency, out of scope here). The selector case is
therefore exercised on a manifest copied from a validated one with
``model_copy(update=...)``, which skips validation on purpose.
"""

from typing import Any, Dict, Optional

import pytest
from roboflow_workflows.core_steps.common.entities import StepExecutionMode
from roboflow_workflows.core_steps.common.workload_presets import (
    COOLDOWN_ACTUAL_RESTRICTION,
)
from roboflow_workflows.core_steps.flow_control.continue_if.v1 import (
    BlockManifest,
    ContinueIfBlockV1,
)
from roboflow_workflows.execution_engine.entities.workload import (
    Runtime,
    unresolved_selector_problem,
)
from roboflow_workflows.execution_engine.introspection.blocks_loader import (
    _get_restrictions,
)
from roboflow_workflows.execution_engine.introspection.workload import (
    describe_workflow_workload,
)
from roboflow_workflows.execution_engine.v1.compiler.entities import BlockSpecification
from roboflow_workflows.prototypes.block import COOLDOWN_HTTP_SOFT_RESTRICTION

from tests.unit_tests.workload_declaration_helpers import (
    declared_restrictions,
    portable_restrictions_discovery,
)

COOLDOWN_CODE = "cooldown_timer_resets_on_stateless_http"
CONDITION_STATEMENT = {
    "type": "StatementGroup",
    "statements": [
        {
            "type": "BinaryStatement",
            "left_operand": {"type": "DynamicOperand", "operand_name": "left"},
            "comparator": {"type": "(Number) =="},
            "right_operand": {"type": "StaticOperand", "value": 1},
        }
    ],
}
POSITIVE_STOP_DELAYS = [0.001, 1, 5, 30.5]


def _manifest(stop_delay: Optional[float] = None) -> BlockManifest:
    fields: Dict[str, Any] = {
        "type": "roboflow_core/continue_if@v1",
        "name": "gate",
        "condition_statement": CONDITION_STATEMENT,
        "evaluation_parameters": {"left": "$inputs.some"},
        "next_steps": ["$steps.blur"],
    }
    if stop_delay is not None:
        fields["stop_delay"] = stop_delay

    return BlockManifest.model_validate(fields)


def _selector_manifest(selector: str) -> BlockManifest:
    # Constructed on purpose: normal validation rejects a selector here (see
    # the module docstring), yet the hook must still answer conservatively.
    manifest = _manifest().model_copy(update={"stop_delay": selector})

    return manifest


def _workflow(stop_delay: Optional[float] = None) -> dict:
    gate: Dict[str, Any] = {
        "type": "roboflow_core/continue_if@v1",
        "name": "gate",
        "condition_statement": CONDITION_STATEMENT,
        "evaluation_parameters": {"left": "$inputs.some"},
        "next_steps": ["$steps.blur"],
    }
    if stop_delay is not None:
        gate["stop_delay"] = stop_delay

    return {
        "version": "1.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
            {"type": "WorkflowParameter", "name": "some"},
        ],
        "steps": [
            gate,
            {
                "type": "roboflow_core/image_blur@v1",
                "name": "blur",
                "image": "$inputs.image",
            },
        ],
        "outputs": [
            {"type": "JsonField", "name": "blurred", "selector": "$steps.blur.image"}
        ],
    }


def _gate_restrictions(introspection: Any) -> Any:
    matching = [step for step in introspection.steps if step.node_id == "$steps.gate"]
    assert len(matching) == 1

    return matching[0].restrictions


def test_default_stop_delay_declares_no_state() -> None:
    # given
    manifest = _manifest()

    # when
    discovery = portable_restrictions_discovery(manifest)
    host_view = manifest.get_actual_restrictions()

    # then
    assert manifest.stop_delay == 0
    assert discovery.items == []
    assert discovery.complete is True
    assert discovery.unknown_reasons == []
    assert host_view.items == []
    assert host_view.complete is True


@pytest.mark.parametrize("stop_delay", POSITIVE_STOP_DELAYS)
def test_positive_stop_delay_declares_the_timer_state_caveat(
    stop_delay: float,
) -> None:
    # given
    manifest = _manifest(stop_delay)

    # when
    actual = declared_restrictions(manifest)
    discovery = portable_restrictions_discovery(manifest)

    # then
    assert actual == [COOLDOWN_ACTUAL_RESTRICTION]
    assert discovery.complete is True
    assert [item.code for item in discovery.items] == [COOLDOWN_CODE]
    condition = discovery.items[0].when
    assert set(condition.runtimes) == {
        Runtime.HOSTED_SERVERLESS,
        Runtime.DEDICATED_DEPLOYMENT,
    }
    assert condition.step_execution_modes is None
    assert condition.input_modes is None
    assert condition.configuration_equals == {}


@pytest.mark.parametrize("stop_delay", POSITIVE_STOP_DELAYS)
def test_positive_stop_delay_host_view_matches_portable_view(
    stop_delay: float,
) -> None:
    # given
    manifest = _manifest(stop_delay)

    # when
    host_view = manifest.get_actual_restrictions(ignore_environment_restrictions=False)
    portable_view = manifest.get_actual_restrictions(
        ignore_environment_restrictions=True
    )

    # then
    assert host_view.items == portable_view.items == [COOLDOWN_ACTUAL_RESTRICTION]
    assert host_view.complete is True
    assert portable_view.complete is True


def test_selector_stop_delay_declares_the_caveat_and_stays_incomplete() -> None:
    # given
    manifest = _selector_manifest("$inputs.stop_delay")

    # when
    discovery = manifest.get_actual_restrictions(ignore_environment_restrictions=True)

    # then
    assert discovery.items == [COOLDOWN_ACTUAL_RESTRICTION]
    assert discovery.complete is False
    assert discovery.unknown_reasons == [
        unresolved_selector_problem(
            node_id="$steps.gate",
            declaration="restrictions",
            field="stop_delay",
            selector="$inputs.stop_delay",
        )
    ]


def test_legacy_restrictions_are_the_conservative_cooldown_caveat() -> None:
    # when
    legacy = BlockManifest.get_restrictions()
    projection = _get_restrictions(
        BlockSpecification(
            block_source="workflows_core",
            identifier=f"{ContinueIfBlockV1.__module__}.ContinueIfBlockV1",
            block_class=ContinueIfBlockV1,
            manifest_class=BlockManifest,
        )
    )

    # then
    assert len(legacy) == 1
    assert legacy[0] is COOLDOWN_HTTP_SOFT_RESTRICTION
    assert projection == [
        {
            "severity": "soft",
            "note": COOLDOWN_HTTP_SOFT_RESTRICTION.note,
            "applies_to_runtimes": ["hosted_serverless", "dedicated_deployment"],
            "applies_to_step_execution_modes": ["remote"],
        }
    ]


def test_legacy_and_actual_caveats_differ_only_in_step_execution_mode() -> None:
    # given
    legacy = BlockManifest.get_restrictions()[0]
    actual = declared_restrictions(_manifest(5))[0]

    # then
    assert legacy.code == actual.code == COOLDOWN_CODE
    assert legacy.severity is actual.severity
    assert set(legacy.applies_to_runtimes) == set(actual.applies_to_runtimes)
    assert legacy.applies_to_input_modes is None
    assert actual.applies_to_input_modes is None
    assert actual.applies_to_step_execution_modes is None


def test_public_workload_report_has_no_caveat_for_default_stop_delay() -> None:
    # when
    restrictions = _gate_restrictions(describe_workflow_workload(_workflow()))

    # then
    assert restrictions.items == []
    assert restrictions.complete is True


@pytest.mark.parametrize(
    "step_execution_mode", [StepExecutionMode.LOCAL, StepExecutionMode.REMOTE]
)
def test_public_workload_report_caveat_does_not_depend_on_step_execution_mode(
    step_execution_mode: StepExecutionMode,
) -> None:
    # when
    introspection = describe_workflow_workload(
        _workflow(stop_delay=5),
        init_parameters={"workflows_core.step_execution_mode": step_execution_mode},
    )

    # then
    payload = _gate_restrictions(introspection).model_dump(mode="json")
    assert payload["complete"] is True
    assert [item["code"] for item in payload["items"]] == [COOLDOWN_CODE]
    assert payload["items"][0]["severity"] == "soft"
    assert payload["items"][0]["when"] == {
        "type": "restriction_condition_v1",
        "runtimes": ["dedicated_deployment", "hosted_serverless"],
        "step_execution_modes": None,
        "input_modes": None,
        "configuration_equals": {},
    }
