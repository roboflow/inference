"""Compiled schedules: direct dependencies and depth-first priorities per run unit.

``CompiledWorkflow.schedule(domain)`` is what a concurrent runtime reads instead
of walking bindings, gates and child boundaries per pulse::

    fork ─▶ left ─▶ left_tail ─┐
         └▶ right ─▶ right_tail ┴▶ join      priorities continue a path first
"""

from dataclasses import FrozenInstanceError, replace
from typing import Dict, List

import pytest
from roboflow_workflows.v2.engine.compilation import (
    CompileOptions,
    compile_workflow,
)
from roboflow_workflows.v2.engine.errors import ContractError, StepPath
from roboflow_workflows.v2.engine.plan import CompiledWorkflow, StepSchedule

from tests.v2.support.compilation.fixture_blocks import (
    CATALOGUE,
    gate,
    nested,
    parameter,
    step,
    workflow,
)
from tests.v2.support.m7_demand.blocks import CATALOGUE as DEMAND_CATALOGUE
from tests.v2.support.m7_demand.blocks import Double, Join, definition
from tests.v2.support.m7_demand.blocks import step as demand_step
from tests.v2.unit.engine.compiler.test_operator_compilation import (
    PAIR,
    compile_definition,
)
from tests.v2.unit.engine.compiler.test_operator_compilation import (
    group as active_group,
)
from tests.v2.unit.engine.compiler.test_operator_compilation import window

VALUE = [parameter("v", kind=["float"])]


def _compile(steps: List[dict], outputs: Dict[str, str]) -> CompiledWorkflow:
    plan = compile_workflow(
        workflow(steps, outputs, inputs=VALUE),
        catalogue=CATALOGUE,
        options=CompileOptions(),
    )

    return plan


def _edges(schedule: StepSchedule) -> Dict[StepPath, List[StepPath]]:
    edges = {
        item.path: [schedule.steps[before].path for before in predecessors]
        for item, predecessors in zip(schedule.steps, schedule.predecessors)
    }

    return edges


def _ranked(schedule: StepSchedule) -> List[str]:
    order = sorted(range(len(schedule.steps)), key=schedule.priorities.__getitem__)
    ranked = ["/".join(schedule.steps[position].path) for position in order]

    return ranked


def _assert_consistent(schedule: StepSchedule) -> None:
    count = len(schedule.steps)
    assert sorted(schedule.priorities) == list(range(count))
    for position, before in enumerate(schedule.predecessors):
        assert all(item < position for item in before)
        assert all(
            schedule.priorities[item] < schedule.priorities[position] for item in before
        )
        for item in before:
            assert position in schedule.successors[item]
    assert sum(map(len, schedule.successors)) == sum(map(len, schedule.predecessors))
    assert schedule.roots == tuple(
        sorted(
            (
                position
                for position, before in enumerate(schedule.predecessors)
                if not before
            ),
            key=schedule.priorities.__getitem__,
        )
    )
    assert dict(schedule.index) == {
        item.path: position for position, item in enumerate(schedule.steps)
    }


def _assert_recorded_match(plan: CompiledWorkflow) -> None:
    edges = _edges(plan.schedule())
    for item in plan.steps:
        assert sorted(item.dependencies) == sorted(edges[item.path]), item.path


FORK_JOIN = [
    step("scale", "fork", value="$inputs.v"),
    step("scale", "left", value="$steps.fork.scaled"),
    step("scale", "right", value="$steps.fork.scaled"),
    step("scale", "left_tail", value="$steps.left.scaled"),
    step("scale", "right_tail", value="$steps.right.scaled"),
    step(
        "merge", "join", values=["$steps.left_tail.scaled", "$steps.right_tail.scaled"]
    ),
]


def test_an_uneven_fork_join_continues_each_path_before_the_next_branch() -> None:
    plan = _compile(FORK_JOIN, {"out": "$steps.join.value"})
    schedule = plan.schedule()

    assert [item.path for item in schedule.steps] == [item.path for item in plan.steps]
    assert _edges(schedule)[("join",)] == [("left_tail",), ("right_tail",)]
    assert _ranked(schedule) == [
        "fork",
        "left",
        "left_tail",
        "right",
        "right_tail",
        "join",
    ]
    assert [schedule.steps[item].path for item in schedule.roots] == [("fork",)]
    fork = schedule.index[("fork",)]
    assert [schedule.steps[item].path for item in schedule.successors[fork]] == [
        ("left",),
        ("right",),
    ]
    _assert_consistent(schedule)
    _assert_recorded_match(plan)


def test_a_chain_keeps_plan_order_as_its_priorities() -> None:
    plan = _compile(
        [
            step("scale", "a", value="$inputs.v"),
            step("scale", "b", value="$steps.a.scaled"),
            step("scale", "c", value="$steps.b.scaled"),
        ],
        {"out": "$steps.c.scaled"},
    )

    assert plan.schedule().priorities == (0, 1, 2)
    assert plan.schedule().predecessors == ((), (0,), (1,))


def test_crossing_dependencies_and_shared_producers_keep_every_direct_edge() -> None:
    plan = _compile(
        [
            step("scale", "x", value="$inputs.v"),
            step("scale", "y", value="$inputs.v"),
            step("merge", "p", values=["$steps.x.scaled", "$steps.y.scaled"]),
            step("merge", "q", values=["$steps.y.scaled", "$steps.x.scaled"]),
            step("scale", "independent", value="$inputs.v"),
        ],
        {
            "p": "$steps.p.value",
            "q": "$steps.q.value",
            "i": "$steps.independent.scaled",
        },
    )
    schedule = plan.schedule()

    assert _edges(schedule)[("p",)] == [("x",), ("y",)]
    assert _edges(schedule)[("q",)] == [("x",), ("y",)]
    assert _edges(schedule)[("independent",)] == []
    assert [schedule.steps[item].path for item in schedule.roots] == [
        ("x",),
        ("y",),
        ("independent",),
    ]
    _assert_consistent(schedule)
    _assert_recorded_match(plan)


def test_gates_and_whole_child_targets_are_control_dependencies() -> None:
    child = workflow(
        [
            step("scale", "inner", value="$inputs.x"),
            step("scale", "deeper", value="$steps.inner.scaled"),
        ],
        {"result": "$steps.deeper.scaled", "forwarded": "$inputs.x"},
        inputs=[parameter("x", kind=["float"])],
    )
    plan = _compile(
        [
            step("scale", "before", value="$inputs.v"),
            gate("decide", "$inputs.v", ["child", "gated"]),
            step("scale", "gated", value="$inputs.v"),
            nested(
                "child",
                workflow_definition=child,
                parameter_bindings={"x": "$steps.before.scaled"},
            ),
            step("scale", "reads_result", value="$steps.child.result"),
            step("scale", "reads_forwarded", value="$steps.child.forwarded"),
            step(
                "merge",
                "join",
                values=["$steps.gated.scaled", "$steps.reads_forwarded.scaled"],
            ),
        ],
        {"result": "$steps.reads_result.scaled", "join": "$steps.join.value"},
    )
    edges = _edges(plan.schedule())

    assert edges[("gated",)] == [("decide",)]
    # A child input is followed to its producer; a gated child step also
    # waits for the controller of its whole child.
    assert edges[("child", "inner")] == [("before",), ("decide",)]
    assert edges[("child", "deeper")] == [("decide",), ("child", "inner")]
    assert edges[("reads_result",)] == [("child", "deeper")]
    # A child output forwarding a child input is gated by the whole child.
    assert edges[("reads_forwarded",)] == [("before",), ("decide",)]
    assert edges[("join",)] == [("gated",), ("reads_forwarded",)]
    _assert_consistent(plan.schedule())
    _assert_recorded_match(plan)


def test_nested_children_pass_values_through_inner_boundaries() -> None:
    leaf = workflow(
        [step("scale", "leaf_step", value="$inputs.y")],
        {"out": "$steps.leaf_step.scaled"},
        inputs=[parameter("y", kind=["float"])],
    )
    middle = workflow(
        [
            nested(
                "leaf", workflow_definition=leaf, parameter_bindings={"y": "$inputs.x"}
            ),
            step("scale", "after_leaf", value="$steps.leaf.out"),
        ],
        {"out": "$steps.after_leaf.scaled"},
        inputs=[parameter("x", kind=["float"])],
    )
    plan = _compile(
        [
            step("scale", "first", value="$inputs.v"),
            nested(
                "middle",
                workflow_definition=middle,
                parameter_bindings={"x": "$steps.first.scaled"},
            ),
            step("scale", "last", value="$steps.middle.out"),
        ],
        {"out": "$steps.last.scaled"},
    )
    edges = _edges(plan.schedule())

    assert edges[("middle", "leaf", "leaf_step")] == [("first",)]
    assert edges[("middle", "after_leaf")] == [("middle", "leaf", "leaf_step")]
    assert edges[("last",)] == [("middle", "after_leaf")]
    _assert_consistent(plan.schedule())
    _assert_recorded_match(plan)


def test_active_routes_hold_their_domain_and_static_steps_only() -> None:
    plan = compile_definition(
        [
            step("scale", "static", value="$inputs.factor"),
            step(
                "scale",
                "on_cam",
                value="$sources.cam.image",
                factor="$steps.static.scaled",
            ),
            step("scale", "on_probe", value="$sources.probe.value"),
            step("scale", "on_pair", value="$operators.pair.left"),
            step("collapse", "on_clip", data="$operators.clip.scaled"),
        ],
        [PAIR, window("clip", {"scaled": "$steps.on_pair.scaled"})],
        [active_group("cams", "$sources.cam.image", out="$steps.on_cam.scaled")],
        inputs=[parameter("factor", default=1.0, kind=["float"])],
    )

    cam = plan.schedule("cam")
    assert [item.path for item in cam.steps] == [("static",), ("on_cam",)]
    assert _edges(cam)[("on_cam",)] == [("static",)]
    assert [item.path for item in plan.schedule("probe").steps] == [
        ("static",),
        ("on_probe",),
    ]
    # Operator ports are pulse inputs of the operator's own domain.
    pair = plan.schedule("pair")
    assert _edges(pair)[("on_pair",)] == []
    assert _edges(plan.schedule("clip"))[("on_clip",)] == []
    for domain in (None,) + plan.domains:
        _assert_consistent(plan.schedule(domain))
        if domain is not None:
            assert plan.route(domain) == plan.schedule(domain).steps
            assert plan.route(domain) is plan.route(domain)
    _assert_recorded_match(plan)


def test_unknown_domains_are_rejected_and_schedules_are_read_only() -> None:
    plan = _compile(FORK_JOIN, {"out": "$steps.join.value"})
    schedule = plan.schedule()

    with pytest.raises(ContractError, match="no source 'nope'"):
        plan.schedule("nope")
    with pytest.raises(FrozenInstanceError):
        schedule.priorities = ()
    with pytest.raises(TypeError):
        schedule.index[("fork",)] = 9
    assert plan.schedule() is schedule


def test_hand_built_plans_without_recorded_dependencies_derive_them() -> None:
    plan = _compile(FORK_JOIN, {"out": "$steps.join.value"})
    bare = replace(
        plan,
        steps=tuple(replace(item, dependencies=()) for item in plan.steps),
        demand=None,
    )

    assert _edges(bare.schedule()) == _edges(plan.schedule())
    assert bare.schedule().priorities == plan.schedule().priorities


def test_recorded_ordering_only_dependencies_are_kept() -> None:
    plan = _compile(FORK_JOIN, {"out": "$steps.join.value"})
    steps = tuple(
        (
            replace(item, dependencies=item.dependencies + (("left",),))
            if item.path == ("right",)
            else item
        )
        for item in plan.steps
    )
    ordered = replace(plan, steps=steps, demand=None)

    assert _edges(ordered.schedule())[("right",)] == [("fork",), ("left",)]
    assert _ranked(ordered.schedule()) == [
        "fork",
        "left",
        "right",
        "right_tail",
        "left_tail",
        "join",
    ]


@pytest.mark.parametrize(
    ("dependencies", "message"),
    [
        ((("missing",),), r"dependency \$steps.missing must be an earlier step"),
        ((("join",),), r"dependency \$steps.join must be an earlier step"),
        ((("right",),), r"dependency \$steps.right must be an earlier step"),
        ((("fork",), ("fork",)), "repeats a recorded dependency"),
    ],
)
def test_malformed_recorded_dependencies_are_rejected(dependencies, message) -> None:
    plan = _compile(FORK_JOIN, {"out": "$steps.join.value"})
    steps = tuple(
        replace(item, dependencies=dependencies) if item.path == ("right",) else item
        for item in plan.steps
    )

    with pytest.raises(ContractError, match=message):
        replace(plan, steps=steps, demand=None)


def test_a_static_step_cannot_record_a_dependency_on_a_domain_step() -> None:
    plan = compile_definition(
        [
            step("scale", "on_cam", value="$sources.cam.image"),
            step("scale", "static", value="$inputs.factor"),
        ],
        outputs=[
            active_group("cams", "$sources.cam.image", out="$steps.on_cam.scaled")
        ],
        inputs=[parameter("factor", default=1.0, kind=["float"])],
    )
    steps = tuple(
        replace(item, dependencies=(("on_cam",),)) if item.path == ("static",) else item
        for item in plan.steps
    )

    with pytest.raises(ContractError, match="in domain None depends on"):
        replace(plan, steps=steps, demand=None)


def test_demand_narrowing_rebuilds_the_schedule_of_the_kept_steps() -> None:
    # Prunable fixture blocks: the unrequested branch and join are dropped.
    plan = compile_workflow(
        definition(
            [
                demand_step(Double, "fork", value="$inputs.value"),
                demand_step(Double, "left", value="$steps.fork.doubled"),
                demand_step(Double, "right", value="$steps.fork.doubled"),
                demand_step(
                    Join,
                    "join",
                    left="$steps.left.doubled",
                    right="$steps.right.doubled",
                ),
            ],
            {"left": "$steps.left.doubled", "join": "$steps.join.value"},
        ),
        catalogue=DEMAND_CATALOGUE,
        options=CompileOptions(requested_outputs=("left",)),
    )

    assert [item.path for item in plan.schedule().steps] == [("fork",), ("left",)]
    assert plan.schedule().predecessors == ((), (0,))
    _assert_consistent(plan.schedule())


def test_an_active_route_keeps_nested_and_gated_edges_inside_the_domain() -> None:
    child = workflow(
        [step("scale", "inner", value="$inputs.x")],
        {"out": "$steps.inner.scaled"},
        inputs=[parameter("x", kind=["float"])],
    )
    plan = compile_definition(
        [
            gate("decide", "$sources.cam.image", ["child"]),
            nested(
                "child",
                workflow_definition=child,
                parameter_bindings={"x": "$sources.cam.image"},
            ),
            step("scale", "after", value="$steps.child.out"),
            step("scale", "on_probe", value="$sources.probe.value"),
        ],
        outputs=[active_group("cams", "$sources.cam.image", out="$steps.after.scaled")],
    )
    cam = plan.schedule("cam")

    assert [item.path for item in cam.steps] == [
        ("decide",),
        ("child", "inner"),
        ("after",),
    ]
    assert _edges(cam) == {
        ("decide",): [],
        ("child", "inner"): [("decide",)],
        ("after",): [("child", "inner")],
    }
    assert _edges(plan.schedule("probe")) == {("on_probe",): []}
    _assert_consistent(cam)
    _assert_recorded_match(plan)
