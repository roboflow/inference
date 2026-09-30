"""discover_workload: honest per-step discoveries and their unions."""

import json
from pathlib import Path

import pytest
from roboflow_workflows.execution_engine.entities.workload import (
    DiscoveryProblemCode,
    WorkOperation,
)
from roboflow_workflows.execution_engine.v2.introspection import (
    WorkloadReport,
    discover_workload,
)

from tests.unit_tests.execution_engine.v2.introspection.fixtures import compile_fixture


@pytest.fixture
def sentinel(tmp_path: Path) -> Path:
    return tmp_path / "executed.txt"


@pytest.fixture
def report(sentinel: Path) -> WorkloadReport:
    return discover_workload(compile_fixture(sentinel))


def reasons(discovery) -> list:
    return [(reason.code, reason.details) for reason in discovery.unknown_reasons]


def test_literal_resource_is_complete_and_a_root_selector_stays_unresolved(
    report: WorkloadReport,
) -> None:
    # when: the root step selects its model from an input that has a default
    root = report.step("$steps.detect").resources

    # then: the caller may override the default, so the identity is unknown
    assert [item.identifier for item in root.items] == ["$inputs.model"]
    assert root.complete is False
    assert reasons(root) == [
        (
            DiscoveryProblemCode.UNRESOLVED_SELECTOR,
            {
                "node_id": "$steps.detect",
                "declaration": "resources",
                "field": "identifier",
                "selector": "$inputs.model",
                "resource_type": "roboflow_platform_model",
            },
        )
    ]


def test_nested_literal_binding_reaches_the_hook_as_a_literal(
    report: WorkloadReport,
) -> None:
    # when
    child = report.step("$steps.child/detect")

    # then
    assert [item.identifier for item in child.resources.items] == ["child-model"]
    assert child.resources.complete is True
    assert child.operations.items == [WorkOperation.MODEL_INFERENCE]
    assert child.operations.complete is True
    assert child.restrictions.complete is True and child.restrictions.items == []


def test_missing_declaration_is_unknown_not_absent(report: WorkloadReport) -> None:
    # when
    undeclared = report.step("$steps.undeclared")

    # then
    for domain, discovery in (
        ("resources", undeclared.resources),
        ("operations", undeclared.operations),
        ("restrictions", undeclared.restrictions),
    ):
        assert discovery.items == [] and discovery.complete is False
        assert reasons(discovery) == [
            (
                DiscoveryProblemCode.DECLARATION_UNAVAILABLE,
                {
                    "node_id": "$steps.undeclared",
                    "declaration": domain,
                    "block_type": "demo/undeclared@v1",
                },
            )
        ]


def test_each_failing_hook_is_reported_exactly_and_alone(
    report: WorkloadReport,
) -> None:
    # when
    faulty = report.step("$steps.faulty")

    # then: raising hook
    assert reasons(faulty.operations) == [
        (
            DiscoveryProblemCode.DECLARATION_FAILED,
            {
                "node_id": "$steps.faulty",
                "declaration": "operations",
                "block_type": "demo/faulty@v1",
            },
        )
    ]
    assert "abc123" not in json.dumps(faulty.describe())
    # invalid shape
    assert reasons(faulty.restrictions) == [
        (
            DiscoveryProblemCode.DECLARATION_FAILED,
            {
                "node_id": "$steps.faulty",
                "declaration": "restrictions",
                "block_type": "demo/faulty@v1",
            },
        )
    ]
    # blank identifier: the item is kept, never fabricated
    assert [item.identifier for item in faulty.resources.items] == ["  "]
    assert reasons(faulty.resources) == [
        (
            DiscoveryProblemCode.INVALID_RESOURCE_IDENTIFIER,
            {
                "node_id": "$steps.faulty",
                "declaration": "resources",
                "field": "identifier",
                "resource_type": "storage_bucket",
            },
        )
    ]
    # other steps are unaffected
    assert report.step("$steps.child/detect").operations.complete is True


def test_restrictions_are_projected_to_portable_metadata(
    report: WorkloadReport,
) -> None:
    # when
    restricted = report.step("$steps.restricted").restrictions

    # then
    [restriction] = restricted.items
    assert restriction.code == "needs_gpu"
    assert restriction.when.configuration_equals == {"device": "cpu"}
    assert restricted.complete is True


def test_custom_python_is_unknown_work_with_local_execution_restrictions(
    report: WorkloadReport, sentinel: Path
) -> None:
    # when
    custom = report.step("$steps.custom")

    # then
    assert custom.block_type == "Sentinel"
    assert custom.operations.items == [WorkOperation.CUSTOM_PYTHON]
    assert custom.operations.complete is False
    assert reasons(custom.operations) == [
        (
            DiscoveryProblemCode.CUSTOM_PYTHON_INTERNALS_UNKNOWN,
            {"declaration": "operations", "node_id": "$steps.custom"},
        )
    ]
    codes = {item.code: item for item in custom.restrictions.items}
    assert set(codes) == {
        "custom_python_local_code_disallowed",
        "custom_python_not_sandboxed",
        "custom_python_remote_transport_unavailable",
    }
    assert codes["custom_python_local_code_disallowed"].when.configuration_equals == {
        "allow_local_code": False
    }
    assert custom.restrictions.complete is False
    assert [
        (item.resource_type, item.identifier) for item in custom.resources.items
    ] == [
        ("python_module", "json"),
        ("python_module", "numpy"),
    ]
    assert custom.resources.complete is False
    assert not sentinel.exists()


def test_unions_keep_every_reason_and_inventory_only_literal_resources(
    report: WorkloadReport,
) -> None:
    # then
    inventory = {
        (item.resource_type, item.identifier): item.used_by_steps
        for item in report.resources.items
    }
    assert inventory == {
        ("roboflow_platform_model", "child-model"): ["$steps.child/detect"],
        ("python_module", "numpy"): ["$steps.custom"],
        ("python_module", "json"): ["$steps.custom"],
    }
    assert report.resources.complete is False
    resource_problems = {
        (reason.code, reason.details["node_id"])
        for reason in report.resources.unknown_reasons
    }
    assert (
        DiscoveryProblemCode.UNRESOLVED_SELECTOR,
        "$steps.detect",
    ) in resource_problems
    assert (
        DiscoveryProblemCode.INVALID_RESOURCE_IDENTIFIER,
        "$steps.faulty",
    ) in resource_problems

    assert set(report.operations.items) == {
        WorkOperation.MODEL_INFERENCE,
        WorkOperation.CUSTOM_PYTHON,
    }
    assert report.operations.complete is False
    assert report.restrictions.complete is False
    json.dumps(report.describe())


def test_steps_without_hooks_or_problems_leave_a_complete_union(
    tmp_path: Path,
) -> None:
    # given: a plan of the child workflow alone, whose model is a literal
    from roboflow_workflows.execution_engine.v2 import compile_workflow

    from tests.unit_tests.execution_engine.v2.introspection.fixtures import (
        CATALOGUE,
        CHILD,
    )

    definition = dict(CHILD)
    definition["steps"] = [dict(CHILD["steps"][0], model_id="literal-model")]
    plan = compile_workflow(definition, catalogue=CATALOGUE)

    # when
    report = discover_workload(plan)

    # then
    assert report.resources.complete is True
    assert [item.identifier for item in report.resources.items] == ["literal-model"]
    assert report.operations.complete is True
    assert report.restrictions.complete is True and report.restrictions.items == []


def test_constants_reach_hooks_through_nested_child_inputs() -> None:
    # given
    from tests.unit_tests.execution_engine.v2.introspection.fixtures import (
        compile_deep_fixture,
    )

    # when
    report = discover_workload(compile_deep_fixture())

    # then: literal and child default are known; a root selector stays unknown
    literal = report.step("$steps.literal/inner/detect").resources
    defaulted = report.step("$steps.defaulted/inner/detect").resources
    selected = report.step("$steps.selected/inner/detect").resources
    assert ([i.identifier for i in literal.items], literal.complete) == (
        ["deep-literal"],
        True,
    )
    assert ([i.identifier for i in defaulted.items], defaulted.complete) == (
        ["middle-default"],
        True,
    )
    assert selected.complete is False
    assert reasons(selected) == [
        (
            DiscoveryProblemCode.UNRESOLVED_SELECTOR,
            {
                "node_id": "$steps.selected/inner/detect",
                "declaration": "resources",
                "field": "identifier",
                "selector": "$inputs.model",
                "resource_type": "roboflow_platform_model",
            },
        )
    ]


@pytest.mark.parametrize(
    "binding, identifier",
    [
        pytest.param({"model": "gated-literal"}, "gated-literal", id="literal"),
        pytest.param({}, "gated-default", id="default"),
        pytest.param({"model": "$inputs.model_id"}, None, id="root-selector"),
    ],
)
def test_constants_reach_hooks_through_a_gated_child_output(
    binding, identifier
) -> None:
    # given: a gated child forwards its model id to a parent step (decision 026)
    from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
    from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
    from roboflow_workflows.execution_engine.v2.introspection import (
        describe_workflow,
        discover_connections,
    )

    from tests.unit_tests.execution_engine.v2.compilation.fixture_blocks import Gate
    from tests.unit_tests.execution_engine.v2.introspection.fixtures import CATALOGUE

    child = {
        "version": "2.0",
        "inputs": [
            {
                "type": "WorkflowParameter",
                "name": "model",
                "kind": ["string"],
                "default_value": "gated-default",
            }
        ],
        "steps": [
            {"type": "demo/undeclared@v1", "name": "body", "value": "$inputs.model"}
        ],
        "outputs": [
            {"type": "JsonField", "name": "model", "selector": "$inputs.model"}
        ],
    }
    definition = {
        "version": "2.0",
        "inputs": [
            {"type": "WorkflowBatchInput", "name": "values", "kind": ["float"]},
            {"type": "WorkflowBatchInput", "name": "keep"},
            {"type": "WorkflowParameter", "name": "model_id", "kind": ["string"]},
        ],
        "steps": [
            {
                "type": "test/gate@v1",
                "name": "gate",
                "value": "$inputs.keep",
                "next_steps": ["$steps.child"],
            },
            {
                "type": "roboflow_core/inner_workflow@v1",
                "name": "child",
                "workflow_definition": child,
                "parameter_bindings": binding,
            },
            {
                "type": "demo/detect@v1",
                "name": "detect",
                "image": "$inputs.values",
                "model_id": "$steps.child.model",
            },
        ],
        "outputs": [],
    }
    plan = compile_workflow(
        definition, catalogue=Catalogue.merge(CATALOGUE, Catalogue([Gate]))
    )

    # when
    resources = discover_workload(plan).step("$steps.detect").resources
    connections = discover_connections(plan)
    [child_output] = describe_workflow(plan)["child_outputs"]

    # then: the gate does not make a static identity dynamic
    if identifier is None:
        assert resources.complete is False
    else:
        assert ([item.identifier for item in resources.items], resources.complete) == (
            [identifier],
            True,
        )
    assert child_output["port"] == "$steps.child.model"
    assert child_output["axes"] == ["inputs"], "effective layout of the batch gate"
    assert [gate["controller"] for gate in child_output["gates"]] == ["$steps.gate"]
    assert ("control", "$steps.gate", "$steps.child.model") in {
        (c.kind, c.source, c.target) for c in connections
    }
