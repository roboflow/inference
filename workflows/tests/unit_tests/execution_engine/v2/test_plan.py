"""Tests of the V2 compiled-plan contract, sessions, observer and futures."""

import sys
import types
from concurrent.futures import Future
from typing import Dict, List

import pytest
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.context import NoExecutionContextError
from roboflow_workflows.execution_engine.v2.data import (
    Axis,
    Batch,
    EntryLayout,
    EntryMetadata,
    WorkflowsBuffer,
)
from roboflow_workflows.execution_engine.v2.declaration import (
    Block,
    BlockParams,
    Group,
    Output,
    Ref,
    Select,
    StepRef,
    spec_of,
)
from roboflow_workflows.execution_engine.v2.errors import (
    ContractError,
    StepExecutionError,
)
from roboflow_workflows.execution_engine.v2.kinds import FLOAT_KIND, INTEGER_KIND
from roboflow_workflows.execution_engine.v2.plan import (
    EXECUTION_MODULE,
    Binding,
    ChildInputPort,
    CompiledWorkflow,
    CompileOptions,
    Constant,
    ExecutionObserver,
    Gate,
    InputPort,
    PlannedChildInput,
    PlannedInput,
    PlannedOutput,
    PlannedStep,
    PlannedWorkflowOutput,
    RunResult,
    StepPort,
    resolve_futures,
)

SAMPLES = Axis(id="N", kind="sample")
CROPS = Axis(id="crop/children", kind="dynamic_nesting")
BATCH = EntryLayout(axes=(SAMPLES,))
NESTED = EntryLayout(axes=(SAMPLES, CROPS))
SCALAR = EntryLayout()


class Expand(Block):
    """Adds a child axis."""

    type = "test/expand@v1"
    outputs = {
        "children": Output(FLOAT_KIND, expand="children"),
        "count": Output(INTEGER_KIND),
    }

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)

    def run(self, *, value) -> dict:
        return {}


class Scale(Block):
    """Element-wise block with a literal-or-selector factor."""

    type = "test/scale@v1"
    outputs = {"scaled": Output(FLOAT_KIND)}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)
        factor: float | Ref(FLOAT_KIND) = 2.0

    def run(self, *, value, factor) -> dict:
        return {}


class SumWithParent(Block):
    """Parent element plus its group of children."""

    type = "test/sum_with_parent@v1"
    outputs = {
        "total": Output(FLOAT_KIND),
        "shares": Output(FLOAT_KIND, preserve="children"),
    }

    class Params(BlockParams):
        parent: Ref(FLOAT_KIND)
        children: Group(FLOAT_KIND)

    def run(self, *, parent, children) -> dict:
        return {}


class InvertMany(Block):
    """Vectorized block."""

    type = "test/invert_many@v1"
    outputs = {"inverted": Output(FLOAT_KIND)}

    class Params(BlockParams):
        values: Ref(FLOAT_KIND, batch="always")
        offset: float | Ref(FLOAT_KIND) = 0.0

    def run(self, *, values, offset) -> list:
        return []


class Gatekeeper(Block):
    """Control block."""

    type = "test/gatekeeper@v1"

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)
        next_steps: List[StepRef]

    def run(self, *, value, next_steps):
        return Select(next_steps)


class Sink(Block):
    """Output-free block with only a literal payload."""

    type = "test/sink@v1"

    class Params(BlockParams):
        payload: str = "alert"

    def run(self, *, payload) -> dict:
        return {}


CATALOGUE = Catalogue([Expand, Scale, SumWithParent, InvertMany, Gatekeeper, Sink])


def _step(
    block_class, path, *, raw=None, bindings=(), layout=BATCH, outputs=None, **extra
):
    spec = spec_of(block_class)
    params = spec.validate_params(raw or {})
    step = PlannedStep(
        path=path,
        spec=spec,
        namespace="",
        params=params,
        bindings=bindings,
        invocation_layout=layout,
        outputs=outputs if outputs is not None else {},
        **extra,
    )

    return step


def _value_binding(field, source, layout, mode, **extra):
    binding = Binding(
        field=field,
        position=(),
        selector=f"${field}",
        source=source,
        source_layout=layout,
        mode=mode,
        **extra,
    )

    return binding


VALUES = PlannedInput(name="values", kinds=("float",), layout=BATCH)
FACTOR = PlannedInput(
    name="factor", kinds=("float",), layout=SCALAR, required=False, default=2.0
)


def _expand_step():
    step = _step(
        Expand,
        ("expand",),
        raw={"value": "$inputs.values"},
        bindings=(_value_binding("value", InputPort("values"), BATCH, "element"),),
        outputs={
            "children": PlannedOutput(
                "children", ("float",), NESTED, transform="expand"
            ),
            "count": PlannedOutput("count", ("integer",), BATCH),
        },
    )

    return step


def _reference_plan() -> CompiledWorkflow:
    expand = _expand_step()
    scale_child = _step(
        Scale,
        ("child", "scale"),
        raw={"value": "$steps.expand.children", "factor": "$inputs.factor"},
        bindings=(
            _value_binding(
                "value", StepPort(("expand",), "children"), NESTED, "element"
            ),
            _value_binding("factor", InputPort("factor"), SCALAR, "constant"),
        ),
        layout=NESTED,
        outputs={"scaled": PlannedOutput("scaled", ("float",), NESTED)},
    )
    reduce = _step(
        SumWithParent,
        ("reduce",),
        raw={"parent": "$inputs.values", "children": "$steps.child.scaled"},
        bindings=(
            _value_binding("parent", InputPort("values"), BATCH, "element"),
            _value_binding(
                "children", StepPort(("child", "scale"), "scaled"), NESTED, "group"
            ),
        ),
        outputs={
            "total": PlannedOutput("total", ("float",), BATCH),
            "shares": PlannedOutput(
                "shares",
                ("float",),
                NESTED,
                transform="preserve",
                group_field="children",
            ),
        },
    )
    plan = CompiledWorkflow(
        inputs={"values": VALUES, "factor": FACTOR},
        steps=(expand, scale_child, reduce),
        outputs=(
            PlannedWorkflowOutput(
                "total", "$steps.reduce.total", StepPort(("reduce",), "total")
            ),
            PlannedWorkflowOutput("all", "$steps.expand.*", StepPort(("expand",), "*")),
        ),
        catalogue=CATALOGUE,
    )

    return plan


def test_hand_built_reference_plan_is_valid_and_describable() -> None:
    plan = _reference_plan()

    description = plan.describe()

    assert [step["path"] for step in description["steps"]] == [
        "$steps.expand",
        "$steps.child/scale",
        "$steps.reduce",
    ]
    reduce = description["steps"][2]
    assert reduce["invocation_axes"] == ["N"]
    assert reduce["bindings"][1] == {
        "field": ["children"],
        "selector": "$children",
        "source": "$steps.child/scale.scaled",
        "source_axes": ["N", "crop/children"],
        "mode": "group",
        "batch": "never",
        "cast_axes": None,
    }
    assert reduce["outputs"]["shares"]["axes"] == ["N", "crop/children"]
    assert description["outputs"]["all"]["source"] == "$steps.expand.*"
    assert plan.step(("child", "scale")).binding_for("factor").mode == "constant"


@pytest.mark.parametrize(
    "mode, source_layout, invocation_layout",
    [
        ("element", BATCH, BATCH),
        ("ancestor", BATCH, NESTED),
        ("constant", SCALAR, NESTED),
        ("constant", SCALAR, SCALAR),
    ],
)
def test_item_binding_modes_match_layouts(
    mode, source_layout, invocation_layout
) -> None:
    step = _step(
        Scale,
        ("scale",),
        raw={"value": "$inputs.values"},
        bindings=(_value_binding("value", InputPort("values"), source_layout, mode),),
        layout=invocation_layout,
    )

    assert step.bindings[0].mode == mode


@pytest.mark.parametrize(
    "mode, source_layout, invocation_layout",
    [
        ("element", NESTED, BATCH),
        ("ancestor", BATCH, BATCH),
        ("constant", BATCH, BATCH),
        ("group", NESTED, BATCH),
    ],
)
def test_item_binding_modes_reject_inconsistent_layouts(
    mode, source_layout, invocation_layout
) -> None:
    with pytest.raises(ContractError):
        _step(
            Scale,
            ("scale",),
            raw={"value": "$inputs.values"},
            bindings=(
                _value_binding("value", InputPort("values"), source_layout, mode),
            ),
            layout=invocation_layout,
        )


def test_constant_group_casts_at_scalar_and_batched_invocation_levels() -> None:
    raw = {"parent": "$inputs.values", "children": "$inputs.values"}
    cast_axis = Axis(id="reduce/children/cast", kind="dynamic_nesting")
    at_scalar = _step(
        SumWithParent,
        ("reduce",),
        raw=raw,
        bindings=(
            _value_binding("parent", Constant(1.0), SCALAR, "constant"),
            _value_binding(
                "children",
                Constant(5.0),
                SCALAR,
                "constant_group",
                cast_layout=EntryLayout(axes=(cast_axis,)),
            ),
        ),
        layout=SCALAR,
    )
    per_parent_cast = EntryLayout(axes=(SAMPLES, cast_axis))
    beside_parent = _step(
        SumWithParent,
        ("reduce",),
        raw=raw,
        bindings=(
            _value_binding("parent", InputPort("values"), BATCH, "element"),
            _value_binding(
                "children",
                Constant(5.0),
                SCALAR,
                "constant_group",
                cast_layout=per_parent_cast,
            ),
        ),
        outputs={
            "shares": PlannedOutput(
                "shares",
                ("float",),
                per_parent_cast,
                transform="preserve",
                group_field="children",
            )
        },
    )

    assert at_scalar.bindings[1].group_layout.axis_ids == ("reduce/children/cast",)
    assert beside_parent.bindings[1].source_layout == SCALAR
    assert beside_parent.outputs["shares"].layout == per_parent_cast
    with pytest.raises(ContractError, match="needs cast_layout"):
        _value_binding("children", Constant(5.0), SCALAR, "constant_group")
    with pytest.raises(ContractError, match="only allowed in mode"):
        _value_binding("value", Constant(5.0), SCALAR, "constant", cast_layout=BATCH)
    with pytest.raises(ContractError, match="mode 'constant_group'"):
        _step(
            SumWithParent,
            ("reduce",),
            raw=raw,
            bindings=(
                _value_binding(
                    "children",
                    Constant(5.0),
                    SCALAR,
                    "constant_group",
                    cast_layout=BATCH,
                ),
            ),
        )


def test_group_modes_reject_item_modes() -> None:
    raw = {"parent": "$inputs.values", "children": "$inputs.values"}

    with pytest.raises(ContractError, match="cannot use mode 'element'"):
        _step(
            SumWithParent,
            ("reduce",),
            raw=raw,
            bindings=(
                _value_binding("children", InputPort("values"), BATCH, "element"),
            ),
        )
    with pytest.raises(ContractError, match="Constant source"):
        _value_binding("value", Constant(1.0), SCALAR, "element")
    with pytest.raises(ContractError, match="unknown mode"):
        _value_binding("value", InputPort("values"), BATCH, "zip")


def test_bindings_must_target_declared_selector_positions() -> None:
    with pytest.raises(ContractError, match="not a declared selector position"):
        _step(
            Sink,
            ("sink",),
            bindings=(
                _value_binding("payload", InputPort("values"), BATCH, "element"),
            ),
        )
    with pytest.raises(ContractError, match="control target"):
        _step(
            Gatekeeper,
            ("gate",),
            raw={"value": "$inputs.values", "next_steps": ["$steps.sink"]},
            bindings=(
                Binding(
                    "next_steps",
                    (0,),
                    "$steps.sink",
                    InputPort("values"),
                    BATCH,
                    "element",
                ),
            ),
        )


def test_vectorized_blocks_require_constant_non_batched_bindings() -> None:
    raw = {"values": "$inputs.values", "offset": "$inputs.values"}
    values = _value_binding(
        "values", InputPort("values"), BATCH, "element", batch="always"
    )

    constant_offset = _step(
        InvertMany,
        ("invert",),
        raw=raw,
        bindings=(
            values,
            _value_binding("offset", InputPort("factor"), SCALAR, "constant"),
        ),
    )

    assert constant_offset.spec.accepts_batches is True
    assert constant_offset.delivers_batches is True
    with pytest.raises(ContractError, match="must be constant"):
        _step(
            InvertMany,
            ("invert",),
            raw=raw,
            bindings=(
                values,
                _value_binding("offset", InputPort("values"), BATCH, "element"),
            ),
        )


def test_output_layouts_are_checked_against_the_invocation_layout() -> None:
    with pytest.raises(ContractError, match="output 'count'"):
        _step(
            Expand,
            ("expand",),
            raw={"value": "$inputs.values"},
            bindings=(_value_binding("value", InputPort("values"), BATCH, "element"),),
            outputs={"count": PlannedOutput("count", ("integer",), NESTED)},
        )


def _gated_plan(*, target_paths, gate_target="$steps.child"):
    gate = _step(
        Gatekeeper,
        ("gate",),
        raw={"value": "$inputs.values", "next_steps": ["$steps.child"]},
        bindings=(_value_binding("value", InputPort("values"), BATCH, "element"),),
        control_targets={"$steps.child": target_paths},
    )
    child_steps = [
        _step(
            Sink,
            path,
            gates=(
                Gate(controller=("gate",), target=gate_target, controller_layout=BATCH),
            ),
        )
        for path in (("child", "first"), ("child", "second"))
    ]
    plan = CompiledWorkflow(
        inputs={"values": VALUES},
        steps=(gate, *child_steps),
        outputs=(),
        catalogue=CATALOGUE,
    )

    return plan


def test_control_target_can_govern_every_step_of_a_nested_workflow() -> None:
    plan = _gated_plan(target_paths=(("child", "first"), ("child", "second")))

    description = plan.describe()

    assert description["steps"][0]["control_targets"] == {
        "$steps.child": ["$steps.child/first", "$steps.child/second"]
    }
    assert description["steps"][2]["gates"] == [
        {
            "controller": "$steps.gate",
            "target": "$steps.child",
            "controller_axes": ["N"],
        }
    ]


def test_gates_must_match_their_controller() -> None:
    with pytest.raises(ContractError, match="does not govern"):
        _gated_plan(target_paths=(("child", "first"),))
    with pytest.raises(ContractError, match="later step"):
        _gated_plan(
            target_paths=(("child", "first"), ("child", "second"), ("missing",))
        )
    with pytest.raises(ContractError, match="non-control block"):
        _step(Sink, ("sink",), control_targets={"$steps.x": (("x",),)})


@pytest.mark.parametrize(
    "outputs, steps_order, fragment",
    [
        (
            (PlannedWorkflowOutput("x", "$inputs.missing", InputPort("missing")),),
            None,
            "unknown workflow input",
        ),
        (
            (
                PlannedWorkflowOutput(
                    "x", "$steps.expand.nope", StepPort(("expand",), "nope")
                ),
            ),
            None,
            "no output",
        ),
        ((), "reversed", "not an earlier step"),
        ((), "duplicate", "twice"),
    ],
)
def test_plan_references_are_validated(outputs, steps_order, fragment) -> None:
    expand = _expand_step()
    scale = _step(
        Scale,
        ("scale",),
        raw={"value": "$steps.expand.count"},
        bindings=(
            _value_binding("value", StepPort(("expand",), "count"), BATCH, "element"),
        ),
    )
    steps = {"reversed": (scale, expand), "duplicate": (expand, expand)}.get(
        steps_order, (expand, scale)
    )

    with pytest.raises(ContractError, match=fragment):
        CompiledWorkflow(
            inputs={"values": VALUES}, steps=steps, outputs=outputs, catalogue=CATALOGUE
        )


def test_session_run_and_rows_delegate_to_the_execution_module(monkeypatch) -> None:
    calls = []
    fake = types.ModuleType(EXECUTION_MODULE)
    fake.run_session = (
        lambda session, *, inputs: calls.append(("run", session, inputs)) or "result"
    )
    fake.build_rows = (
        lambda result, *, serialize: calls.append(("rows", serialize)) or []
    )
    monkeypatch.setitem(sys.modules, EXECUTION_MODULE, fake)
    session = _reference_plan().create_session()

    result = session.run({"values": [1.0]})

    assert result == "result"
    assert calls[0][1] is session and calls[0][2] == {"values": [1.0]}
    assert set(session.instances) == {("expand",), ("child", "scale"), ("reduce",)}


def test_observer_defaults_are_no_ops() -> None:
    observer = ExecutionObserver()

    observer.on_run_started(session_id="s", run_id="r")
    observer.on_step_started(step=("a",), block_type="t")
    observer.on_invocation(step=("a",), index=(0,), arguments={"step": 1}, result={})
    observer.on_invocation_skipped(step=("a",), index=(1,), reason="denied_by_gate")
    observer.on_step_finished(step=("a",), invocations=1, skipped=1)
    observer.on_run_finished(run_id="r", result=None, error=None)


def test_resolve_futures_waits_inside_results_without_copying_payloads() -> None:
    payload = {"large": [1, 2, 3]}
    future = Future()
    future.set_result(payload)
    batch = Batch([1, future], indices=[(0,), (2,)])
    plain = {"a": [1, 2], "b": payload}

    resolved = resolve_futures({"value": future, "items": [future, 3], "batch": batch})

    assert resolved["value"] is payload
    assert resolved["items"] == [payload, 3]
    assert resolved["batch"].indices == ((0,), (2,))
    assert resolved["batch"][1] is payload
    assert resolve_futures(plain) is plain


def test_resolve_futures_propagates_failures() -> None:
    future = Future()
    future.set_exception(RuntimeError("remote failure"))

    with pytest.raises(RuntimeError, match="remote failure"):
        resolve_futures({"value": future})


def test_compile_options_and_step_errors() -> None:
    options = CompileOptions(mutation_conflicts="error")
    error = StepExecutionError(
        "boom", step_path=("child", "scale"), block_type="test/scale@v1", index=(0, 1)
    )

    assert options.max_nested_depth == 4 and options.max_nested_count == 32
    assert str(error) == "$steps.child/scale at index [0, 1] (test/scale@v1): boom"
    with pytest.raises(ContractError, match="mutation_conflicts"):
        CompileOptions(mutation_conflicts="ignore")
    with pytest.raises(ContractError, match="non-negative"):
        CompileOptions(max_nested_depth=-1)


OTHER_CROPS = Axis(id="other/children", kind="dynamic_nesting")
OTHER_NESTED = EntryLayout(axes=(SAMPLES, OTHER_CROPS))


class Consensus(Block):
    """list[Ref(batch="always")]: the block receives list[Batch]."""

    type = "test/consensus@v1"
    outputs = {"merged": Output(FLOAT_KIND)}

    class Params(BlockParams):
        predictions: List[Ref(FLOAT_KIND, batch="always")]

    def run(self, *, predictions) -> list:
        return [{"merged": sum(values)} for values in zip(*predictions)]


class Csv(Block):
    """dict of scalar-or-batch leaves: batches only for varying leaves."""

    type = "test/csv@v1"
    outputs = {"csv": Output()}

    class Params(BlockParams):
        columns: Dict[str, str | float | Ref(batch="if_varying")]

    def run(self, *, columns):
        return {"csv": columns}


class NamedGroups(Block):
    """dict of Group leaves beside a parent reference."""

    type = "test/named_groups@v1"
    outputs = {
        "summary": Output(source="parent"),
        "labelled": Output(preserve="groups"),
    }

    class Params(BlockParams):
        parent: Ref()
        groups: Dict[str, Group()]

    def run(self, *, parent, groups) -> dict:
        return {}


class StitchAndTranslate(Block):
    """Parent-level and child-level outputs from one invocation."""

    type = "test/stitch_and_translate@v1"
    outputs = {
        "stitched": Output(source="image"),
        "translated": Output(preserve="predictions"),
    }

    class Params(BlockParams):
        image: Ref()
        predictions: Group()

    def run(self, *, image, predictions) -> dict:
        return {}


class Generate(Block):
    """Input-free source creating its own root axis."""

    type = "test/generate@v1"
    outputs = {"items": Output(expand="generated")}

    def run(self) -> dict:
        return {}


PATTERN_CATALOGUE = Catalogue(
    [Consensus, Csv, NamedGroups, StitchAndTranslate, Generate]
)


def _leaf(field, position, source, layout, mode, **extra):
    binding = Binding(
        field=field,
        position=position,
        selector="$steps.x.y",
        source=source,
        source_layout=layout,
        mode=mode,
        **extra,
    )

    return binding


def test_consensus_leaves_deliver_batches_even_when_all_are_constant() -> None:
    varying = _step(
        Consensus,
        ("consensus",),
        raw={"predictions": ["$inputs.values", "$inputs.values"]},
        bindings=(
            _leaf(
                "predictions",
                (0,),
                InputPort("values"),
                BATCH,
                "element",
                batch="always",
            ),
            _leaf(
                "predictions",
                (1,),
                InputPort("values"),
                BATCH,
                "element",
                batch="always",
            ),
        ),
    )
    constant = _step(
        Consensus,
        ("consensus",),
        raw={"predictions": ["$inputs.values"]},
        bindings=(
            _leaf(
                "predictions",
                (0,),
                InputPort("factor"),
                SCALAR,
                "constant",
                batch="always",
            ),
        ),
    )

    assert [binding.position for binding in varying.bindings_for("predictions")] == [
        (0,),
        (1,),
    ]
    assert varying.delivers_batches is True
    assert constant.delivers_batches is True


def test_csv_scalar_only_step_is_called_per_invocation() -> None:
    raw = {"columns": {"camera": "north", "count": "$inputs.values"}}
    scalar_only = _step(
        Csv,
        ("csv",),
        raw=raw,
        bindings=(
            _leaf(
                "columns",
                ("count",),
                InputPort("factor"),
                SCALAR,
                "constant",
                batch="if_varying",
            ),
        ),
        layout=SCALAR,
    )
    mixed = _step(
        Csv,
        ("csv",),
        raw=raw,
        bindings=(
            _leaf(
                "columns",
                ("count",),
                InputPort("values"),
                BATCH,
                "element",
                batch="if_varying",
            ),
        ),
    )

    assert scalar_only.delivers_batches is False
    assert mixed.delivers_batches is True
    with pytest.raises(ContractError, match="copies batch mode"):
        _step(
            Csv,
            ("csv",),
            raw=raw,
            bindings=(
                _leaf("columns", ("count",), InputPort("values"), BATCH, "element"),
            ),
        )


def _named_groups_step(people_binding, *, outputs=None):
    step = _step(
        NamedGroups,
        ("named",),
        raw={
            "parent": "$inputs.values",
            "groups": {"cars": "$steps.a.b", "people": "$steps.c.d"},
        },
        bindings=(
            _leaf("parent", (), InputPort("values"), BATCH, "element"),
            _leaf(
                "groups", ("cars",), StepPort(("expand",), "children"), NESTED, "group"
            ),
            people_binding,
        ),
        outputs=outputs,
    )

    return step


def test_compound_groups_preserve_only_a_shared_child_layout() -> None:
    labelled = {
        "summary": PlannedOutput("summary", ("*",), BATCH, source_field="parent"),
        "labelled": PlannedOutput(
            "labelled", ("*",), NESTED, transform="preserve", group_field="groups"
        ),
    }

    shared = _named_groups_step(
        _leaf(
            "groups", ("people",), StepPort(("expand",), "children"), NESTED, "group"
        ),
        outputs=labelled,
    )

    assert shared.outputs["labelled"].layout == NESTED
    assert shared.outputs["summary"].source_field == "parent"
    with pytest.raises(ContractError, match="different layouts"):
        _named_groups_step(
            _leaf("groups", ("people",), InputPort("values"), OTHER_NESTED, "group"),
            outputs=labelled,
        )


def test_compound_group_scalar_leaf_is_cast_per_parent() -> None:
    cast_layout = EntryLayout(
        axes=(SAMPLES, Axis(id="named/groups/cast", kind="dynamic_nesting"))
    )

    step = _named_groups_step(
        _leaf(
            "groups",
            ("people",),
            Constant("constant"),
            SCALAR,
            "constant_group",
            cast_layout=cast_layout,
        )
    )

    people = step.binding_for("groups", ("people",))
    assert people.source_layout == SCALAR
    assert people.group_layout == cast_layout
    assert step.delivers_batches is False


def test_one_block_emits_parent_and_child_layouts() -> None:
    step = _step(
        StitchAndTranslate,
        ("stitch",),
        raw={"image": "$inputs.values", "predictions": "$steps.expand.children"},
        bindings=(
            _leaf("image", (), InputPort("values"), BATCH, "element"),
            _leaf(
                "predictions", (), StepPort(("expand",), "children"), NESTED, "group"
            ),
        ),
        outputs={
            "stitched": PlannedOutput("stitched", ("*",), BATCH, source_field="image"),
            "translated": PlannedOutput(
                "translated",
                ("*",),
                NESTED,
                transform="preserve",
                group_field="predictions",
            ),
        },
    )

    assert step.outputs["stitched"].layout.axis_ids == ("N",)
    assert step.outputs["translated"].layout.axis_ids == ("N", "crop/children")
    unbound = _step(
        StitchAndTranslate,
        ("stitch",),
        raw={"image": "$inputs.values", "predictions": "$steps.expand.children"},
        bindings=(
            _leaf(
                "predictions", (), StepPort(("expand",), "children"), NESTED, "group"
            ),
        ),
        outputs={
            "stitched": PlannedOutput("stitched", ("*",), BATCH, source_field="image")
        },
    )

    assert unbound.outputs["stitched"].source_field == "image"  # no context (R4/B1)
    with pytest.raises(ContractError, match="not a data field"):
        _step(
            StitchAndTranslate,
            ("stitch",),
            raw={"image": "$inputs.values", "predictions": "$steps.expand.children"},
            bindings=(_leaf("image", (), InputPort("values"), BATCH, "element"),),
            outputs={
                "stitched": PlannedOutput(
                    "stitched", ("*",), BATCH, source_field="missing"
                )
            },
        )


def _generated_plan(second_axis_id="second/generated"):
    generated = EntryLayout(axes=(Axis(id="source/generated", kind="dynamic_nesting"),))
    second = EntryLayout(axes=(Axis(id=second_axis_id, kind="dynamic_nesting"),))
    steps = (
        _expand_step(),
        _step(
            Generate,
            ("source",),
            layout=SCALAR,
            outputs={
                "items": PlannedOutput("items", ("*",), generated, transform="expand")
            },
        ),
        _step(
            Generate,
            ("second",),
            layout=SCALAR,
            outputs={
                "items": PlannedOutput("items", ("*",), second, transform="expand")
            },
        ),
    )
    plan = CompiledWorkflow(
        inputs={"values": VALUES}, steps=steps, outputs=(), catalogue=PATTERN_CATALOGUE
    )

    return plan


def test_axis_origins_distinguish_inputs_expansions_and_generated_roots() -> None:
    plan = _generated_plan()

    assert plan.axis_origin("N").kind == "input"
    assert plan.axis_origin("crop/children").describe() == "$steps.expand.children"
    assert plan.axis_origin("source/generated").step == ("source",)
    assert plan.describe()["axis_origins"]["second/generated"] == "$steps.second.items"
    with pytest.raises(ContractError, match="no axis"):
        plan.axis_origin("missing")
    with pytest.raises(ContractError, match="own axis identity"):
        _generated_plan(second_axis_id="source/generated")


def _run_result(plan, *, entries, selections, statuses):
    buffer = WorkflowsBuffer(
        lineage_id="plan",
        pulse_id=0,
        data={key: value for key, (value, _) in entries.items()},
        layout={key: layout for key, (_, layout) in entries.items()},
        metadata={key: EntryMetadata() for key in entries},
    )
    result = RunResult(
        outputs=buffer,
        selections=selections,
        statuses=statuses,
        filtered_paths={
            key: () for key, status in statuses.items() if status == "filtered"
        },
        plan=plan,
        session_id="session",
        run_id="run",
        trace=[{"event": "run_started"}],
    )

    return result


def test_wildcard_result_keeps_one_entry_per_port_layout() -> None:
    plan = _reference_plan()
    entries = {
        "total": (Batch([1.0, 2.0]), BATCH),
        "all/children": (
            Batch([Batch([10.0], parent_index=(0,)), Batch([], parent_index=(1,))]),
            NESTED,
        ),
        "all/count": (Batch([1, 0]), BATCH),
    }
    selections = {
        "total": {"$steps.reduce.total": "total"},
        "all": {
            "$steps.expand.children": "all/children",
            "$steps.expand.count": "all/count",
        },
    }

    result = _run_result(
        plan,
        entries=entries,
        selections=selections,
        statuses={key: "complete" for key in entries},
    )

    assert result.outputs.layout["all/children"] == NESTED
    assert result.outputs.layout["all/count"] == BATCH
    assert result.trace == ({"event": "run_started"},)


def test_run_result_statuses_must_match_buffer_entries() -> None:
    plan = _reference_plan()
    selections = {"total": {"$steps.reduce.total": "total"}}

    filtered = _run_result(
        plan, entries={}, selections=selections, statuses={"total": "filtered"}
    )

    assert filtered.filtered_paths == {"total": ()}
    with pytest.raises(ContractError, match="complete entries"):
        _run_result(
            plan, entries={}, selections=selections, statuses={"total": "complete"}
        )
    with pytest.raises(ContractError, match="cover exactly"):
        _run_result(plan, entries={}, selections=selections, statuses={})


@pytest.mark.parametrize("name", ["2026", "camera-1", "parse_json"])
def test_step_and_input_names_are_selector_segments(name) -> None:
    step = _step(Sink, (name,), layout=SCALAR)
    item = PlannedInput(name=name, kinds=("*",), layout=SCALAR)

    assert step.path == (name,) and item.name == name
    with pytest.raises(ContractError, match="selector segments"):
        _step(Sink, ("bad name",), layout=SCALAR)
    with pytest.raises(ContractError, match="letters, digits"):
        PlannedInput(name="bad\n", kinds=("*",), layout=SCALAR)


def _child_plan(child_inputs, *, binding_source, binding_layout=BATCH, outputs=()):
    step = _step(
        Scale,
        ("child", "scale"),
        raw={"value": "$inputs.x"},
        bindings=(_value_binding("value", binding_source, binding_layout, "element"),),
        outputs={"scaled": PlannedOutput("scaled", ("float",), BATCH)},
    )
    plan = CompiledWorkflow(
        inputs={"values": VALUES},
        steps=(step,),
        outputs=outputs,
        catalogue=CATALOGUE,
        child_inputs=child_inputs,
    )

    return plan


def test_child_inputs_are_shared_ports_with_their_source_layout() -> None:
    outer = PlannedChildInput(
        scope=("child",),
        name="x",
        kinds=("float",),
        layout=BATCH,
        source=InputPort("values"),
    )
    inner = PlannedChildInput(
        scope=("child", "grandchild"),
        name="y",
        kinds=("*",),
        layout=BATCH,
        source=ChildInputPort(("child",), "x"),
    )
    default = PlannedChildInput(
        scope=("child",), name="k", kinds=("float",), layout=SCALAR, source=Constant(3)
    )
    plan = _child_plan(
        (outer, inner, default),
        binding_source=ChildInputPort(("child", "grandchild"), "y"),
        outputs=(
            PlannedWorkflowOutput(
                "k", "$steps.child.k", ChildInputPort(("child",), "k")
            ),
        ),
    )

    assert plan.child_input(ChildInputPort(("child",), "x")) is outer
    assert plan.describe()["child_inputs"][1] == {
        "port": "$steps.child/grandchild: $inputs.y",
        "kinds": ["*"],
        "axes": ["N"],
        "source": "$steps.child: $inputs.x",
    }
    assert set(plan.describe()["axis_origins"]) == {"N"}
    with pytest.raises(ContractError, match="no child input"):
        plan.child_input(ChildInputPort(("child",), "missing"))


@pytest.mark.parametrize(
    "child_inputs, binding_source, fragment",
    [
        ((), ChildInputPort(("child",), "x"), "unknown"),
        (
            (
                PlannedChildInput(
                    ("child", "g"), "y", ("*",), BATCH, ChildInputPort(("child",), "x")
                ),
                PlannedChildInput(("child",), "x", ("*",), BATCH, InputPort("values")),
            ),
            ChildInputPort(("child", "g"), "y"),
            "later-listed",
        ),
        (
            (
                PlannedChildInput(("child",), "x", ("*",), BATCH, InputPort("values")),
                PlannedChildInput(("child",), "x", ("*",), BATCH, InputPort("values")),
            ),
            ChildInputPort(("child",), "x"),
            "twice",
        ),
        (
            (PlannedChildInput(("child",), "x", ("*",), SCALAR, InputPort("values")),),
            ChildInputPort(("child",), "x"),
            "differ",
        ),
    ],
)
def test_child_input_references_are_validated(
    child_inputs, binding_source, fragment
) -> None:
    with pytest.raises(ContractError, match=fragment):
        _child_plan(child_inputs, binding_source=binding_source)


def test_binding_source_layout_must_match_the_source() -> None:
    with pytest.raises(ContractError, match="differ from the axes"):
        CompiledWorkflow(
            inputs={"values": VALUES, "factor": FACTOR},
            steps=(
                _step(
                    Scale,
                    ("scale",),
                    raw={"value": "$inputs.factor"},
                    bindings=(
                        _value_binding("value", InputPort("factor"), BATCH, "element"),
                    ),
                ),
            ),
            outputs=(),
            catalogue=CATALOGUE,
        )


class ContextAware(Block):
    """Records the execution context seen by its constructor."""

    type = "test/context_aware@v1"

    def __init__(self):
        self.constructed_in = self.execution_context

    def run(self) -> dict:
        return {}


def test_constructors_run_in_the_session_execution_context() -> None:
    step = _step(ContextAware, ("child", "aware"), layout=SCALAR)
    plan = CompiledWorkflow(
        inputs={}, steps=(step,), outputs=(), catalogue=Catalogue([ContextAware])
    )

    session = plan.create_session()

    context = session.instances[("child", "aware")].constructed_in
    assert context.session_id == session.session_id
    assert context.step_path == ("child", "aware")
    assert context.block_type == "test/context_aware@v1"
    assert context.run_id is None
    with pytest.raises(NoExecutionContextError):
        session.instances[("child", "aware")].execution_context
