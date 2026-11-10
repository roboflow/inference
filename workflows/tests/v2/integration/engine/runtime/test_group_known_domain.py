"""Decision 016 (phase): a delivered ``Group`` exposes its known child domain.

Compositions adapted from ``ecosystem-e4-group-contract-plan/probe_group.py``.
The gate continues when ``child > 1``:

    children        raw rows             Group values   indices   known_indices
    [10, 0, 20, 0]  [10, None, 20, None] [10, 20]       0, 2      0, 1, 2, 3
    []              []                   []             -         -
    [0, 0]          [None, None]         []             -         0, 1
    parent gated    absent               no Group       -         -

Survivor iteration and the default skip policy stay as they were.
"""

from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.declaration import (
    BlockParams,
    Group,
    Output,
    Ref,
)

from tests.v2.support.execution.blocks import ContinueIf, Echo, Recording

SETS = [[10, 0, 20, 0], [], [0, 0], [30]]


class Emit(Recording):
    type = "test/group_domain_emit@v1"
    outputs = {"children": Output(expand="regions")}

    class Params(BlockParams):
        data: Ref()

    def run(self, *, data) -> dict:
        return {"children": Batch.of(data)}


class SparseEmit(Recording):
    """Omits non-positive candidates instead of emitting placeholders."""

    type = "test/group_domain_sparse_emit@v1"
    outputs = {"children": Output(expand="regions")}

    class Params(BlockParams):
        data: Ref()

    def run(self, *, data) -> dict:
        kept = [(position,) for position, value in enumerate(data) if value > 0]
        children = Batch.of([data[index[0]] for index in kept], indices=kept)
        return {"children": children}


class Inspect(Recording):
    """Empty-accepting group consumer; records what it received."""

    type = "test/group_domain_inspect@v1"
    accepts_empty = True
    outputs = {"view": Output()}

    class Params(BlockParams):
        data: Group()

    def run(self, *, data) -> dict:
        view = None if data is None else describe(data)
        self.remember(view=view)
        return {"view": view}


class DefaultInspect(Inspect):
    type = "test/group_domain_default_inspect@v1"
    accepts_empty = False


class AnchoredInspect(Inspect):
    type = "test/group_domain_anchored_inspect@v1"

    class Params(BlockParams):
        data: Group()
        anchor: Ref()

    def run(self, *, data, anchor) -> dict:
        return super().run(data=data)


CATALOGUE = Catalogue(
    [Emit, SparseEmit, Inspect, DefaultInspect, AnchoredInspect, ContinueIf, Echo]
)


def describe(group: Batch) -> dict:
    view = {
        "values": list(group),
        "length": len(group),
        "parent": group.parent_index,
        "indices": list(group.indices),
        "known": list(group.known_indices),
    }

    return view


def workflow(inputs, steps, outputs) -> dict:
    definition = {
        "version": "2.0",
        "inputs": [
            {"type": "WorkflowBatchInput", "name": name, "kind": ["*"]}
            for name in inputs
        ],
        "steps": steps,
        "outputs": [
            {"type": "JsonField", "name": name, "selector": selector}
            for name, selector in outputs
        ],
    }

    return definition


def emit(data, *, name="emit", block=Emit) -> dict:
    return {"type": block.type, "name": name, "data": data}


def gate(value, *, name="gate", target="echo") -> dict:
    step = {
        "type": ContinueIf.type,
        "name": name,
        "value": value,
        "threshold": 1,
        "next_steps": [f"$steps.{target}"],
    }

    return step


def echo(value) -> dict:
    return {"type": Echo.type, "name": "echo", "value": value}


def inspect(data, *, name="inspect", block=Inspect) -> dict:
    return {"type": block.type, "name": name, "data": data}


def run(definition, inputs):
    session = compile_workflow(definition, catalogue=CATALOGUE).create_session()
    result = session.run(inputs)

    return session, result.rows()


def views(session, name="inspect"):
    return [call["view"] for call in session.instances[(name,)].calls]


def gated_children_workflow() -> dict:
    definition = workflow(
        ["sets"],
        [
            emit("$inputs.sets"),
            gate("$steps.emit.children"),
            echo("$steps.emit.children"),
            inspect("$steps.echo.value"),
            inspect("$steps.echo.value", name="default", block=DefaultInspect),
        ],
        [("raw", "$steps.echo.value"), ("group", "$steps.inspect.view")],
    )

    return definition


def test_group_keeps_interior_trailing_and_all_filtered_positions() -> None:
    session, rows = run(gated_children_workflow(), {"sets": SETS})

    assert [row["raw"] for row in rows] == [
        [10, None, 20, None],
        [],
        [None, None],
        [30],
    ]
    assert views(session) == [
        {
            "values": [10, 20],
            "length": 2,
            "parent": (0,),
            "indices": [(0, 0), (0, 2)],
            "known": [(0, 0), (0, 1), (0, 2), (0, 3)],
        },
        {"values": [], "length": 0, "parent": (1,), "indices": [], "known": []},
        {
            "values": [],
            "length": 0,
            "parent": (2,),
            "indices": [],
            "known": [(2, 0), (2, 1)],
        },
        {
            "values": [30],
            "length": 1,
            "parent": (3,),
            "indices": [(3, 0)],
            "known": [(3, 0)],
        },
    ]


def test_default_group_consumer_keeps_its_skip_policy() -> None:
    session, _ = run(gated_children_workflow(), {"sets": SETS})

    # All-filtered parent 2 is still skipped; the genuinely empty parent 1 runs.
    assert [view["parent"] for view in views(session, "default")] == [
        (0,),
        (1,),
        (3,),
    ]
    assert views(session, "default")[0]["known"] == [(0, 0), (0, 1), (0, 2), (0, 3)]
    assert views(session, "default")[1]["known"] == []


def test_sparse_producer_domain_holds_only_emitted_candidates() -> None:
    definition = workflow(
        ["sets"],
        [
            emit("$inputs.sets", block=SparseEmit),
            inspect("$steps.emit.children"),
        ],
        [("raw", "$steps.emit.children")],
    )

    session, rows = run(definition, {"sets": SETS})

    # Disclosed limit: candidates never emitted are not part of any domain.
    assert [row["raw"] for row in rows] == [[10, None, 20], [], [], [30]]
    assert [view["known"] for view in views(session)] == [
        [(0, 0), (0, 2)],
        [],
        [],
        [(3, 0)],
    ]


def test_none_placeholders_keep_their_positions_after_a_default_block() -> None:
    # The G2 DynamicCrop path: a producer emits positional None slots, a
    # default item block skips them, the group still knows every slot.
    definition = workflow(
        ["sets"],
        [
            emit("$inputs.sets"),
            echo("$steps.emit.children"),
            inspect("$steps.emit.children", name="direct"),
            inspect("$steps.echo.value"),
        ],
        [("raw", "$steps.echo.value")],
    )

    session, rows = run(definition, {"sets": [[10, None, 20, None], [None, None], []]})

    assert [row["raw"] for row in rows] == [[10, None, 20, None], [None, None], []]
    assert [view["values"] for view in views(session, "direct")] == [
        [10, None, 20, None],
        [None, None],
        [],
    ]
    assert [(view["values"], view["known"]) for view in views(session)] == [
        ([10, 20], [(0, 0), (0, 1), (0, 2), (0, 3)]),
        ([], [(1, 0), (1, 1)]),
        ([], []),
    ]


def test_nested_groups_report_the_domain_of_their_exact_parent() -> None:
    definition = workflow(
        ["sets"],
        [
            emit("$inputs.sets", name="outer"),
            emit("$steps.outer.children", name="inner"),
            gate("$steps.inner.children"),
            echo("$steps.inner.children"),
            inspect("$steps.echo.value"),
        ],
        [("raw", "$steps.echo.value")],
    )

    session, rows = run(definition, {"sets": [[SETS[0], SETS[1]], [SETS[2], SETS[3]]]})

    assert [row["raw"] for row in rows] == [
        [[10, None, 20, None], []],
        [[None, None], [30]],
    ]
    assert [
        (view["parent"], view["indices"], view["known"]) for view in views(session)
    ] == [
        ((0, 0), [(0, 0, 0), (0, 0, 2)], [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3)]),
        ((0, 1), [], []),
        ((1, 0), [], [(1, 0, 0), (1, 0, 1)]),
        ((1, 1), [(1, 1, 0)], [(1, 1, 0)]),
    ]


def test_gate_inside_child_workflow_keeps_the_domain() -> None:
    child = workflow(
        ["items"],
        [gate("$inputs.items"), echo("$inputs.items")],
        [("value", "$steps.echo.value")],
    )
    definition = workflow(
        ["sets"],
        [
            emit("$inputs.sets"),
            {
                "type": "inner_workflow",
                "name": "child",
                "workflow_definition": child,
                "parameter_bindings": {"items": "$steps.emit.children"},
            },
            inspect("$steps.child.value"),
            inspect("$steps.child.value", name="default", block=DefaultInspect),
        ],
        [("raw", "$steps.child.value")],
    )

    session, rows = run(definition, {"sets": SETS})

    assert [row["raw"] for row in rows] == [
        [10, None, 20, None],
        [],
        [None, None],
        [30],
    ]
    assert [view["known"] for view in views(session)] == [
        [(0, 0), (0, 1), (0, 2), (0, 3)],
        [],
        [(2, 0), (2, 1)],
        [(3, 0)],
    ]
    assert [view["parent"] for view in views(session, "default")] == [
        (0,),
        (1,),
        (3,),
    ]


def test_filtered_input_crossing_child_workflow_keeps_the_domain() -> None:
    child = workflow(
        ["items"], [echo("$inputs.items")], [("value", "$steps.echo.value")]
    )
    definition = workflow(
        ["sets"],
        [
            emit("$inputs.sets"),
            gate("$steps.emit.children"),
            echo("$steps.emit.children"),
            {
                "type": "inner_workflow",
                "name": "child",
                "workflow_definition": child,
                "parameter_bindings": {"items": "$steps.echo.value"},
            },
            inspect("$steps.child.value"),
        ],
        [("raw", "$steps.child.value")],
    )

    session, rows = run(definition, {"sets": SETS})

    assert [row["raw"] for row in rows] == [
        [10, None, 20, None],
        [],
        [None, None],
        [30],
    ]
    assert [(view["values"], view["known"]) for view in views(session)] == [
        ([10, 20], [(0, 0), (0, 1), (0, 2), (0, 3)]),
        ([], []),
        ([], [(2, 0), (2, 1)]),
        ([30], [(3, 0)]),
    ]


def test_parent_filtered_before_expansion_stays_absent() -> None:
    definition = workflow(
        ["sets", "enabled"],
        [
            gate("$inputs.enabled", name="parent_gate", target="emit"),
            emit("$inputs.sets"),
            inspect("$steps.emit.children"),
            {
                "type": AnchoredInspect.type,
                "name": "anchored",
                "data": "$steps.emit.children",
                "anchor": "$inputs.enabled",
            },
        ],
        [("group", "$steps.inspect.view"), ("anchored", "$steps.anchored.view")],
    )

    session, rows = run(definition, {"sets": [[], [10]], "enabled": [0, 2]})

    # Parent 0 has no group at all: no domain, not an empty one.
    assert views(session) == [
        {
            "values": [10],
            "length": 1,
            "parent": (1,),
            "indices": [(1, 0)],
            "known": [(1, 0)],
        }
    ]
    assert views(session, "anchored")[0] is None
    assert rows[0]["group"] is None
