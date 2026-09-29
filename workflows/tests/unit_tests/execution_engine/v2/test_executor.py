"""Behavioural tests of the V2 serial reference executor.

All blocks here use tiny numeric payloads; no image type is involved, which
proves the generic core does not inspect payload types.
"""

import re
from fractions import Fraction
from typing import Any, Callable, Dict, List, Mapping

import pytest
from roboflow_workflows.execution_engine.v2 import (
    Batch,
    BlockContract,
    ContractError,
    EntryMetadata,
    Index,
    InputSpec,
    InputValue,
    OutputSpec,
    Registry,
    SampleContext,
    TemporalContext,
    Timestamp,
    WorkflowExecutionError,
    compile_workflow,
)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class _Expand:
    """Item block: value -> children ``value * 10 + i`` and preserved count."""

    def __init__(self, config: Mapping[str, Any]):
        self._counts = {str(k): v for k, v in config.get("counts", {}).items()}
        self._default = config.get("default", 1)
        self._indices = config.get("indices")

    def run(self, value: Any) -> Dict[str, Any]:
        count = self._counts.get(str(value), self._default)
        children = [value * 10 + i for i in range(count)]
        indices = None
        if self._indices is not None:
            indices = [tuple(item) for item in self._indices[:count]]
        return {"children": Batch.of(children, indices=indices), "count": count}


class _Scale:
    def __init__(self, config: Mapping[str, Any], calls: List[Dict[str, Any]]):
        self._factor = config.get("factor", 2)
        self._calls = calls

    def run(self, value: Any) -> Dict[str, Any]:
        self._calls.append({"block": "scale", "value": value})
        return {"scaled": value * self._factor}


class _Sum:
    """Reducer: parent item plus child group -> total and size."""

    def __init__(self, config: Mapping[str, Any], calls: List[Dict[str, Any]]):
        self._calls = calls

    def run(self, parent: Any, children: Batch) -> Dict[str, Any]:
        self._calls.append(
            {
                "block": "sum",
                "parent": parent,
                "children": list(children),
                "indices": children.indices,
                "parent_index": children.parent_index,
                "layout": children.layout,
                "metadata": children.metadata,
            }
        )
        return {"total": parent + sum(children), "size": len(children)}


class _Threshold:
    def __init__(self, config: Mapping[str, Any]):
        self._minimum = config["minimum"]

    def run(self, value: Any) -> Dict[str, Any]:
        return {"keep": value >= self._minimum}


class _EchoGroup:
    def __init__(self, config: Mapping[str, Any]):
        pass

    def run(self, group: Batch) -> Dict[str, Any]:
        return {"same": group, "size": len(group)}


class _Failing:
    def __init__(self, config: Mapping[str, Any]):
        self._poison = config["poison"]

    def run(self, value: Any) -> Dict[str, Any]:
        if value == self._poison:
            raise ZeroDivisionError(f"poisoned by {value}")
        return {"out": value}


class _Counter:
    """Instance state: reports how many times this instance ran."""

    def __init__(self, config: Mapping[str, Any]):
        self._calls = 0

    def run(self, value: Any) -> Dict[str, Any]:
        self._calls += 1
        return {"count": self._calls}


class _Custom:
    def __init__(self, run: Callable[..., Any]):
        self._run = run

    def run(self, **kwargs: Any) -> Any:
        return self._run(**kwargs)


ITEM_NUMBER = InputSpec("number")
BATCH_NUMBER = InputSpec("number", view="batch")


def _registry(calls: List[Dict[str, Any]]) -> Registry:
    registry = Registry()
    registry.register_kind("number", _is_number)
    registry.register_kind("boolean", lambda v: isinstance(v, bool))
    registry.register_kind("record", lambda v: isinstance(v, dict))
    registry.register_kind("anything")
    registry.register_block(
        "expand",
        contract=BlockContract(
            reference="value",
            inputs={"value": ITEM_NUMBER},
            outputs={
                "children": OutputSpec("number", transform="append", axis="kids"),
                "count": OutputSpec("number"),
            },
        ),
        factory=_Expand,
    )
    registry.register_block(
        "scale",
        contract=BlockContract(
            reference="value",
            inputs={"value": ITEM_NUMBER},
            outputs={"scaled": OutputSpec("number")},
        ),
        factory=lambda config: _Scale(config, calls),
    )
    registry.register_block(
        "sum",
        contract=BlockContract(
            reference="children",
            inputs={"parent": ITEM_NUMBER, "children": BATCH_NUMBER},
            outputs={
                "total": OutputSpec("number", transform="collapse"),
                "size": OutputSpec("number", transform="collapse"),
            },
        ),
        factory=lambda config: _Sum(config, calls),
    )
    registry.register_block(
        "threshold",
        contract=BlockContract(
            reference="value",
            inputs={"value": ITEM_NUMBER},
            outputs={"keep": OutputSpec("boolean")},
        ),
        factory=_Threshold,
    )
    registry.register_block(
        "echo_group",
        contract=BlockContract(
            reference="group",
            inputs={"group": BATCH_NUMBER},
            outputs={
                "same": OutputSpec("number", transform="preserve"),
                "size": OutputSpec("number", transform="collapse"),
            },
        ),
        factory=_EchoGroup,
    )
    registry.register_block(
        "failing",
        contract=BlockContract(
            reference="value",
            inputs={"value": ITEM_NUMBER},
            outputs={"out": OutputSpec("number")},
        ),
        factory=_Failing,
    )
    registry.register_block(
        "counter",
        contract=BlockContract(
            reference="value",
            inputs={"value": ITEM_NUMBER},
            outputs={"count": OutputSpec("number")},
        ),
        factory=_Counter,
    )
    return registry


def _register_custom(
    registry: Registry, name: str, *, contract: BlockContract, run: Callable[..., Any]
) -> None:
    registry.register_block(
        name, contract=contract, factory=lambda config: _Custom(run)
    )


def _samples_input(name: str = "values", kind: str = "number") -> Dict[str, Any]:
    return {
        "name": name,
        "kind": kind,
        "axes": [{"id": "samples", "kind": "sample", "stationary": True}],
    }


def _definition(steps: List[Dict[str, Any]], outputs: Dict[str, str], inputs=None):
    return {
        "version": "2.0",
        "inputs": inputs if inputs is not None else [_samples_input()],
        "steps": steps,
        "outputs": [
            {"name": name, "selector": selector} for name, selector in outputs.items()
        ],
    }


def _nested(batch: Batch) -> Any:
    if isinstance(batch, Batch):
        return {index: _nested(item) for index, item in batch.iter_with_indices()}
    return batch


def _sample_metadata() -> EntryMetadata:
    return EntryMetadata(
        sample={
            (0,): SampleContext("a"),
            (1,): SampleContext("b"),
            (2,): SampleContext("c"),
        },
        temporal={
            (): TemporalContext(
                observed_coverage=Timestamp(
                    ticks=7, time_base=Fraction(1, 30), clock_id="clk"
                )
            )
        },
    )


# --- ragged expansion, parent references, reduction -------------------------


def test_ragged_children_reach_reducer_with_parent_and_empty_group() -> None:
    calls: List[Dict[str, Any]] = []
    registry = _registry(calls)
    definition = _definition(
        steps=[
            {
                "name": "expand",
                "type": "expand",
                "inputs": {"value": "$inputs.values"},
                "config": {"counts": {"1": 2, "2": 0, "3": 1}},
            },
            {
                "name": "sum",
                "type": "sum",
                "inputs": {
                    "parent": "$inputs.values",
                    "children": "$steps.expand.children",
                },
            },
        ],
        outputs={
            "children": "$steps.expand.children",
            "count": "$steps.expand.count",
            "total": "$steps.sum.total",
            "size": "$steps.sum.size",
        },
    )
    plan = compile_workflow(definition, registry=registry)
    metadata = _sample_metadata()

    result = plan.run(
        inputs={"values": InputValue(Batch.of([1, 2, 3]), metadata=metadata)}
    )

    assert _nested(result.outputs.data["children"]) == {
        (0,): {(0, 0): 10, (0, 1): 11},
        (1,): {},
        (2,): {(2, 0): 30},
    }
    assert list(result.outputs.data["count"]) == [2, 0, 1]
    assert list(result.outputs.data["size"]) == [2, 0, 1]
    assert list(result.outputs.data["total"]) == [1 + 10 + 11, 2, 3 + 30]
    assert result.outputs.layout["children"].axis_ids == ("samples", "expand/kids")
    assert result.outputs.layout["count"].axis_ids == ("samples",)
    assert result.outputs.layout["total"].axis_ids == ("samples",)
    assert result.outputs.layout["children"].axes[1].kind == "dynamic_nesting"
    assert result.statuses == {
        "children": "complete",
        "count": "complete",
        "total": "complete",
        "size": "complete",
    }
    sums = [call for call in calls if call["block"] == "sum"]
    assert [call["parent"] for call in sums] == [1, 2, 3]
    assert [call["children"] for call in sums] == [[10, 11], [], [30]]
    assert [call["indices"] for call in sums] == [((0, 0), (0, 1)), (), ((2, 0),)]
    assert [call["parent_index"] for call in sums] == [(0,), (1,), (2,)]
    assert all(call["layout"].axis_ids == ("samples", "expand/kids") for call in sums)
    assert sums[0]["metadata"].sample_at((0, 1)).source_id == "a"
    assert sums[1]["metadata"].sample_at((1,)).source_id == "b"
    assert result.outputs.metadata["children"].sample_at((2, 0)).source_id == "c"
    assert result.outputs.metadata["total"].sample_at((1,)).source_id == "b"
    assert (
        result.outputs.metadata["total"].temporal_at((1,)).observed_coverage.ticks == 7
    )
    assert not result.outputs.layout["total"].has_time


def test_same_leaf_maps_at_multiple_depths_and_reduces_twice() -> None:
    calls: List[Dict[str, Any]] = []
    registry = _registry(calls)
    definition = _definition(
        steps=[
            {"name": "s0", "type": "scale", "inputs": {"value": "$inputs.values"}},
            {
                "name": "e1",
                "type": "expand",
                "inputs": {"value": "$inputs.values"},
                "config": {"counts": {"1": 2, "2": 0, "3": 1}},
            },
            {"name": "s1", "type": "scale", "inputs": {"value": "$steps.e1.children"}},
            {
                "name": "e2",
                "type": "expand",
                "inputs": {"value": "$steps.s1.scaled"},
                "config": {"counts": {"22": 0}, "default": 2},
            },
            {"name": "s2", "type": "scale", "inputs": {"value": "$steps.e2.children"}},
            {
                "name": "r2",
                "type": "sum",
                "inputs": {
                    "parent": "$steps.s1.scaled",
                    "children": "$steps.s2.scaled",
                },
            },
            {
                "name": "r1",
                "type": "sum",
                "inputs": {"parent": "$steps.s0.scaled", "children": "$steps.r2.total"},
            },
        ],
        outputs={
            "s2": "$steps.s2.scaled",
            "r2": "$steps.r2.total",
            "r2_size": "$steps.r2.size",
            "r1": "$steps.r1.total",
            "r1_size": "$steps.r1.size",
        },
    )
    plan = compile_workflow(definition, registry=registry)

    result = plan.run(inputs={"values": Batch.of([1, 2, 3])})

    # e1 children: 1 -> [10, 11], 2 -> [], 3 -> [30]; s1 doubles: [20, 22], [], [60]
    # e2 children: 20 -> [200, 201], 22 -> [], 60 -> [600, 601]; s2 doubles them.
    assert _nested(result.outputs.data["s2"]) == {
        (0,): {
            (0, 0): {(0, 0, 0): 400, (0, 0, 1): 402},
            (0, 1): {},
        },
        (1,): {},
        (2,): {(2, 0): {(2, 0, 0): 1200, (2, 0, 1): 1202}},
    }
    assert result.outputs.layout["s2"].axis_ids == ("samples", "e1/kids", "e2/kids")
    assert _nested(result.outputs.data["r2"]) == {
        (0,): {(0, 0): 20 + 400 + 402, (0, 1): 22},
        (1,): {},
        (2,): {(2, 0): 60 + 1200 + 1202},
    }
    assert _nested(result.outputs.data["r2_size"]) == {
        (0,): {(0, 0): 2, (0, 1): 0},
        (1,): {},
        (2,): {(2, 0): 2},
    }
    assert result.outputs.layout["r2"].axis_ids == ("samples", "e1/kids")
    assert list(result.outputs.data["r1"]) == [2 + 822 + 22, 4, 6 + 2462]
    assert list(result.outputs.data["r1_size"]) == [2, 0, 1]
    assert result.outputs.layout["r1"].axis_ids == ("samples",)
    scale_values = [call["value"] for call in calls if call["block"] == "scale"]
    assert scale_values == [1, 2, 3, 10, 11, 30, 200, 201, 600, 601]


def test_singleton_sparse_and_wholly_empty_inputs_keep_axes() -> None:
    calls: List[Dict[str, Any]] = []
    registry = _registry(calls)
    definition = _definition(
        steps=[
            {
                "name": "expand",
                "type": "expand",
                "inputs": {"value": "$inputs.values"},
                "config": {"counts": {"5": 2}, "indices": [[0], [3]]},
            },
            {
                "name": "sum",
                "type": "sum",
                "inputs": {
                    "parent": "$inputs.values",
                    "children": "$steps.expand.children",
                },
            },
        ],
        outputs={"children": "$steps.expand.children", "total": "$steps.sum.total"},
    )
    plan = compile_workflow(definition, registry=registry)

    singleton = plan.run(inputs={"values": Batch.of([5], indices=[(4,)])})
    empty = plan.run(inputs={"values": Batch.empty()})

    assert _nested(singleton.outputs.data["children"]) == {
        (4,): {(4, 0): 50, (4, 3): 51}
    }
    assert _nested(singleton.outputs.data["total"]) == {(4,): 5 + 50 + 51}
    assert singleton.outputs.layout["total"].axis_ids == ("samples",)
    assert _nested(empty.outputs.data["children"]) == {}
    assert _nested(empty.outputs.data["total"]) == {}
    assert empty.outputs.layout["children"].axis_ids == ("samples", "expand/kids")
    assert empty.statuses == {"children": "complete", "total": "complete"}
    assert [call for call in calls if call["block"] == "sum"][-1]["parent"] == 5
    assert len([call for call in calls if call["block"] == "sum"]) == 1


def test_batch_preserve_echoes_group_and_carries_metadata() -> None:
    registry = _registry([])
    definition = _definition(
        steps=[
            {
                "name": "expand",
                "type": "expand",
                "inputs": {"value": "$inputs.values"},
                "config": {"counts": {"1": 2, "2": 0}},
            },
            {
                "name": "echo",
                "type": "echo_group",
                "inputs": {"group": "$steps.expand.children"},
            },
        ],
        outputs={"same": "$steps.echo.same", "size": "$steps.echo.size"},
    )
    plan = compile_workflow(definition, registry=registry)

    result = plan.run(
        inputs={"values": InputValue(Batch.of([1, 2]), metadata=_sample_metadata_two())}
    )

    assert _nested(result.outputs.data["same"]) == {
        (0,): {(0, 0): 10, (0, 1): 11},
        (1,): {},
    }
    assert list(result.outputs.data["size"]) == [2, 0]
    assert result.outputs.layout["same"].axis_ids == ("samples", "expand/kids")
    assert result.outputs.metadata["same"].sample_at((0, 1)).source_id == "a"


def test_container_payloads_shaped_like_index_pairs_stay_payloads() -> None:
    registry = _registry([])
    _register_custom(
        registry,
        "wrap",
        contract=BlockContract(
            reference="value",
            inputs={"value": ITEM_NUMBER},
            outputs={
                "pairs": OutputSpec("anything"),
                "parts": OutputSpec("anything", transform="append", axis="parts"),
            },
        ),
        run=lambda value: {
            "pairs": (((0,), value), ((1,), value)),
            "parts": Batch.of([((0,), value), [value, value]]),
        },
    )
    _register_custom(
        registry,
        "gather",
        contract=BlockContract(
            reference="parts",
            inputs={"parts": InputSpec("anything", view="batch")},
            outputs={
                "same": OutputSpec("anything"),
                "listed": OutputSpec("anything", transform="collapse"),
            },
        ),
        run=lambda parts: {"same": parts, "listed": list(parts.iter_with_indices())},
    )
    definition = _definition(
        steps=[
            {"name": "wrap", "type": "wrap", "inputs": {"value": "$inputs.values"}},
            {
                "name": "gather",
                "type": "gather",
                "inputs": {"parts": "$steps.wrap.parts"},
            },
        ],
        outputs={
            "pairs": "$steps.wrap.pairs",
            "parts": "$steps.wrap.parts",
            "same": "$steps.gather.same",
            "listed": "$steps.gather.listed",
        },
    )
    plan = compile_workflow(definition, registry=registry)

    result = plan.run(inputs={"values": Batch.of([1, 2])})

    parts = {
        (0,): {(0, 0): ((0,), 1), (0, 1): [1, 1]},
        (1,): {(1, 0): ((0,), 2), (1, 1): [2, 2]},
    }
    assert _nested(result.outputs.data["pairs"]) == {
        (0,): (((0,), 1), ((1,), 1)),
        (1,): (((0,), 2), ((1,), 2)),
    }
    assert _nested(result.outputs.data["parts"]) == parts
    assert _nested(result.outputs.data["same"]) == parts
    assert _nested(result.outputs.data["listed"]) == {
        (0,): [((0, 0), ((0,), 1)), ((0, 1), [1, 1])],
        (1,): [((1, 0), ((0,), 2)), ((1, 1), [2, 2])],
    }
    assert result.outputs.layout["pairs"].axis_ids == ("samples",)
    assert result.outputs.layout["same"].axis_ids == ("samples", "wrap/parts")
    assert result.outputs.layout["listed"].axis_ids == ("samples",)


def _sample_metadata_two() -> EntryMetadata:
    return EntryMetadata(sample={(0,): SampleContext("a"), (1,): SampleContext("b")})


# --- gates, filtering, empty versus filtered --------------------------------


def _gated_definition() -> Dict[str, Any]:
    return _definition(
        steps=[
            {
                "name": "expand",
                "type": "expand",
                "inputs": {"value": "$inputs.values"},
                "config": {"counts": {"1": 2, "2": 0, "3": 1, "4": 2}},
            },
            {
                "name": "gate",
                "type": "threshold",
                "inputs": {"value": "$steps.expand.children"},
                "config": {"minimum": 11},
            },
            {
                "name": "kept",
                "type": "scale",
                "inputs": {"value": "$steps.expand.children"},
                "when": "$steps.gate.keep",
            },
            {
                "name": "kept_sum",
                "type": "sum",
                "inputs": {
                    "parent": "$inputs.values",
                    "children": "$steps.kept.scaled",
                },
            },
            {
                "name": "all_sum",
                "type": "sum",
                "inputs": {
                    "parent": "$inputs.values",
                    "children": "$steps.expand.children",
                },
            },
        ],
        outputs={
            "kept": "$steps.kept.scaled",
            "kept_total": "$steps.kept_sum.total",
            "kept_size": "$steps.kept_sum.size",
            "all_total": "$steps.all_sum.total",
        },
    )


def test_gate_keeps_indices_and_distinguishes_all_filtered_from_empty() -> None:
    calls: List[Dict[str, Any]] = []
    registry = _registry(calls)
    plan = compile_workflow(_gated_definition(), registry=registry)

    # children: 1 -> [10, 11]; 2 -> []; 3 -> [30]; 4 -> [40, 41]; gate keeps >= 11
    result = plan.run(inputs={"values": Batch.of([1, 2, 3, 4])})

    assert _nested(result.outputs.data["kept"]) == {
        (0,): {(0, 1): 22},
        (1,): {},
        (2,): {(2, 0): 60},
        (3,): {(3, 0): 80, (3, 1): 82},
    }
    assert result.filtered_paths["kept"] == ((0, 0),)
    assert _nested(result.outputs.data["kept_total"]) == {
        (0,): 1 + 22,
        (1,): 2,
        (2,): 3 + 60,
        (3,): 4 + 80 + 82,
    }
    assert _nested(result.outputs.data["kept_size"]) == {
        (0,): 1,
        (1,): 0,
        (2,): 1,
        (3,): 2,
    }
    assert list(result.outputs.data["all_total"]) == [22, 2, 33, 85]
    assert result.filtered_paths["kept_total"] == ()
    kept_sums = [call for call in calls if call["block"] == "sum"][:4]
    assert kept_sums[0]["indices"] == ((0, 1),)
    assert kept_sums[1]["indices"] == ()
    assert kept_sums[1]["parent_index"] == (1,)


def test_parent_with_all_children_gated_is_filtered_not_empty() -> None:
    calls: List[Dict[str, Any]] = []
    registry = _registry(calls)
    definition = _gated_definition()
    definition["steps"][1]["config"] = {"minimum": 30}
    plan = compile_workflow(definition, registry=registry)

    # gate keeps >= 30: parent 0 has candidates [10, 11] all gated; parent 1 is empty.
    result = plan.run(inputs={"values": Batch.of([1, 2, 3, 4])})

    assert _nested(result.outputs.data["kept"]) == {
        (1,): {},
        (2,): {(2, 0): 60},
        (3,): {(3, 0): 80, (3, 1): 82},
    }
    assert result.filtered_paths["kept"] == ((0,),)
    assert _nested(result.outputs.data["kept_total"]) == {(1,): 2, (2,): 63, (3,): 166}
    assert _nested(result.outputs.data["kept_size"]) == {(1,): 0, (2,): 1, (3,): 2}
    assert result.filtered_paths["kept_total"] == ((0,),)
    assert result.statuses["kept_total"] == "complete"
    assert list(result.outputs.data["all_total"]) == [22, 2, 33, 85]
    assert result.filtered_paths["all_total"] == ()
    kept_sums = [call for call in calls if call["block"] == "sum"][:3]
    assert [call["parent"] for call in kept_sums] == [2, 3, 4]
    filtered_events = [
        event
        for event in result.trace
        if event["event"] == "invocation_filtered"
        and event["node"] == "$steps.kept_sum"
    ]
    assert filtered_events == [
        {
            "event": "invocation_filtered",
            "node": "$steps.kept_sum",
            "index": [0],
            "reason": "upstream_filtered",
            "ports": ["children"],
        }
    ]


def test_filtered_paths_use_sparse_logical_indices_and_separate_empty_from_filtered() -> (
    None
):
    calls: List[Dict[str, Any]] = []
    registry = _registry(calls)
    definition = _gated_definition()
    definition["steps"][1]["config"] = {"minimum": 30}

    # sparse parents (2,), (5,), (9,): children [10, 11] all gated, [] empty,
    # [40, 41] kept.
    result = compile_workflow(definition, registry=registry).run(
        inputs={"values": Batch.of([1, 2, 4], indices=[(2,), (5,), (9,)])}
    )

    assert _nested(result.outputs.data["kept"]) == {
        (5,): {},
        (9,): {(9, 0): 80, (9, 1): 82},
    }
    assert result.filtered_paths["kept"] == ((2,),)
    assert result.filtered_paths["kept_total"] == ((2,),)
    assert _nested(result.outputs.data["kept_total"]) == {(5,): 2, (9,): 166}
    assert result.filtered_paths["all_total"] == ()
    assert _nested(result.outputs.data["all_total"]) == {(2,): 22, (5,): 2, (9,): 85}
    assert result.statuses["kept"] == "complete"
    resolved = {
        event["name"]: event["filtered_paths"]
        for event in result.trace
        if event["event"] == "output_resolved"
    }
    assert resolved["kept"] == [[2]] and resolved["all_total"] == []


def test_ancestor_gate_masks_descendants_only_on_dependent_path() -> None:
    calls: List[Dict[str, Any]] = []
    registry = _registry(calls)
    definition = _definition(
        steps=[
            {
                "name": "parent_gate",
                "type": "threshold",
                "inputs": {"value": "$inputs.values"},
                "config": {"minimum": 2},
            },
            {
                "name": "expand",
                "type": "expand",
                "inputs": {"value": "$inputs.values"},
                "config": {"counts": {"1": 2, "2": 0, "3": 1}},
            },
            {
                "name": "kept",
                "type": "scale",
                "inputs": {"value": "$steps.expand.children"},
                "when": "$steps.parent_gate.keep",
            },
            {
                "name": "kept_sum",
                "type": "sum",
                "inputs": {
                    "parent": "$inputs.values",
                    "children": "$steps.kept.scaled",
                },
            },
            {
                "name": "free",
                "type": "scale",
                "inputs": {"value": "$steps.expand.children"},
            },
        ],
        outputs={
            "kept": "$steps.kept.scaled",
            "kept_total": "$steps.kept_sum.total",
            "free": "$steps.free.scaled",
        },
    )
    plan = compile_workflow(definition, registry=registry)

    result = plan.run(inputs={"values": Batch.of([1, 2, 3])})

    assert _nested(result.outputs.data["kept"]) == {(1,): {}, (2,): {(2, 0): 60}}
    assert result.filtered_paths["kept"] == ((0,),)
    assert _nested(result.outputs.data["kept_total"]) == {(1,): 2, (2,): 63}
    assert _nested(result.outputs.data["free"]) == {
        (0,): {(0, 0): 20, (0, 1): 22},
        (1,): {},
        (2,): {(2, 0): 60},
    }
    assert result.filtered_paths["free"] == ()
    filtered_events = [
        event
        for event in result.trace
        if event["event"] == "invocation_filtered" and event["node"] == "$steps.kept"
    ]
    # The ancestor decision is recorded once, at the path it addresses.
    assert [event["index"] for event in filtered_events] == [[0]]
    assert {event["reason"] for event in filtered_events} == {"gate_false"}


def test_root_level_gate_filters_whole_output_and_sibling_stays_complete() -> None:
    registry = _registry([])
    definition = _definition(
        inputs=[{"name": "value", "kind": "number"}],
        steps=[
            {
                "name": "gate",
                "type": "threshold",
                "inputs": {"value": "$inputs.value"},
                "config": {"minimum": 100},
            },
            {
                "name": "gated",
                "type": "scale",
                "inputs": {"value": "$inputs.value"},
                "when": "$steps.gate.keep",
            },
            {
                "name": "downstream",
                "type": "scale",
                "inputs": {"value": "$steps.gated.scaled"},
            },
            {"name": "sibling", "type": "scale", "inputs": {"value": "$inputs.value"}},
        ],
        outputs={
            "gated": "$steps.gated.scaled",
            "downstream": "$steps.downstream.scaled",
            "sibling": "$steps.sibling.scaled",
        },
    )
    plan = compile_workflow(definition, registry=registry)

    result = plan.run(inputs={"value": 5})

    assert result.statuses == {
        "gated": "filtered",
        "downstream": "filtered",
        "sibling": "complete",
    }
    assert set(result.outputs.data) == {"sibling"}
    assert result.outputs.data["sibling"] == 10
    assert result.outputs.layout["sibling"].axes == ()
    assert result.filtered_paths["gated"] == ((),)
    assert result.filtered_paths["downstream"] == ((),)
    step_events = {
        event["node"]: event["status"]
        for event in result.trace
        if event["event"] == "step_completed"
    }
    assert step_events == {
        "$steps.gate": "complete",
        "$steps.gated": "filtered",
        "$steps.downstream": "filtered",
        "$steps.sibling": "complete",
    }
    assert [event["event"] for event in result.trace][-1] == "run_completed"


def test_all_top_level_items_gated_makes_entry_filtered_but_empty_input_stays_complete() -> (
    None
):
    registry = _registry([])
    definition = _definition(
        steps=[
            {
                "name": "gate",
                "type": "threshold",
                "inputs": {"value": "$inputs.values"},
                "config": {"minimum": 100},
            },
            {
                "name": "gated",
                "type": "scale",
                "inputs": {"value": "$inputs.values"},
                "when": "$steps.gate.keep",
            },
        ],
        outputs={"gated": "$steps.gated.scaled"},
    )
    plan = compile_workflow(definition, registry=registry)

    gated = plan.run(inputs={"values": Batch.of([1, 2])})
    empty = plan.run(inputs={"values": Batch.empty()})

    assert gated.statuses == {"gated": "filtered"}
    assert gated.filtered_paths["gated"] == ((),)
    assert empty.statuses == {"gated": "complete"}
    assert _nested(empty.outputs.data["gated"]) == {}


def test_two_batched_inputs_intersect_filtered_children_and_reject_domain_mismatch() -> (
    None
):
    calls: List[Dict[str, Any]] = []
    registry = _registry(calls)
    _register_custom(
        registry,
        "pair",
        contract=BlockContract(
            reference="left",
            inputs={"left": BATCH_NUMBER, "right": BATCH_NUMBER},
            outputs={"pairs": OutputSpec("record", transform="collapse")},
        ),
        run=lambda left, right: {
            "pairs": {
                "left": list(left),
                "right": list(right),
                "indices": list(left.indices),
            }
        },
    )
    definition = _definition(
        steps=[
            {
                "name": "expand",
                "type": "expand",
                "inputs": {"value": "$inputs.values"},
                "config": {"counts": {"1": 2, "2": 0, "3": 1}},
            },
            {
                "name": "gate",
                "type": "threshold",
                "inputs": {"value": "$steps.expand.children"},
                "config": {"minimum": 11},
            },
            {
                "name": "kept",
                "type": "scale",
                "inputs": {"value": "$steps.expand.children"},
                "when": "$steps.gate.keep",
            },
            {
                "name": "pair",
                "type": "pair",
                "inputs": {
                    "left": "$steps.expand.children",
                    "right": "$steps.kept.scaled",
                },
            },
        ],
        outputs={"pairs": "$steps.pair.pairs"},
    )
    plan = compile_workflow(definition, registry=registry)

    result = plan.run(inputs={"values": Batch.of([1, 2, 3])})

    assert _nested(result.outputs.data["pairs"]) == {
        (0,): {"left": [11], "right": [22], "indices": [(0, 1)]},
        (1,): {"left": [], "right": [], "indices": []},
        (2,): {"left": [30], "right": [60], "indices": [(2, 0)]},
    }
    assert result.filtered_paths["pairs"] == ()


def test_domain_mismatch_between_sibling_inputs_is_reported_with_index() -> None:
    registry = _registry([])
    definition = _definition(
        inputs=[_samples_input("a"), _samples_input("b")],
        steps=[
            {
                "name": "combine",
                "type": "combine",
                "inputs": {"left": "$inputs.a", "right": "$inputs.b"},
            }
        ],
        outputs={"sum": "$steps.combine.sum"},
    )
    _register_custom(
        registry,
        "combine",
        contract=BlockContract(
            reference="left",
            inputs={"left": ITEM_NUMBER, "right": ITEM_NUMBER},
            outputs={"sum": OutputSpec("number")},
        ),
        run=lambda left, right: {"sum": left + right},
    )
    plan = compile_workflow(definition, registry=registry)

    aligned = plan.run(inputs={"a": Batch.of([1, 2]), "b": Batch.of([10, 20])})
    assert list(aligned.outputs.data["sum"]) == [11, 22]

    with pytest.raises(WorkflowExecutionError) as info:
        plan.run(inputs={"a": Batch.of([1, 2]), "b": Batch.of([10])})
    assert "$steps.combine input domains disagree at index [1]" in str(info.value)
    assert "absent (and not filtered) on ['right']" in str(info.value)


# --- metadata --------------------------------------------------------------


def test_collapse_resolves_child_contexts_common_or_none_without_touching_upstream() -> (
    None
):
    calls: List[Dict[str, Any]] = []
    registry = _registry(calls)
    definition = _definition(
        inputs=[
            {
                "name": "parents",
                "kind": "number",
                "axes": [{"id": "samples", "kind": "sample"}],
            },
            {
                "name": "children",
                "kind": "number",
                "axes": [
                    {"id": "samples", "kind": "sample"},
                    {"id": "regions", "kind": "dynamic_nesting"},
                ],
            },
        ],
        steps=[
            {
                "name": "sum",
                "type": "sum",
                "inputs": {"parent": "$inputs.parents", "children": "$inputs.children"},
            },
            {
                "name": "scaled",
                "type": "scale",
                "inputs": {"value": "$inputs.children"},
            },
        ],
        outputs={"total": "$steps.sum.total", "scaled": "$steps.scaled.scaled"},
    )
    plan = compile_workflow(definition, registry=registry)
    children = Batch(
        [
            Batch([1, 2], indices=[(0, 0), (0, 1)], parent_index=(0,)),
            Batch([3, 4], indices=[(1, 0), (1, 1)], parent_index=(1,)),
            Batch([5, 6], indices=[(2, 0), (2, 1)], parent_index=(2,)),
            Batch.empty(parent_index=(3,)),
        ],
        indices=[(0,), (1,), (2,), (3,)],
    )
    child_metadata = EntryMetadata(
        sample={
            (): SampleContext("root"),
            (0,): SampleContext("a"),
            (0, 1): None,
            (1,): SampleContext("b"),
            (2,): SampleContext("c"),
            (2, 0): SampleContext("z"),
            (2, 1): SampleContext("z"),
            (3,): SampleContext("d"),
        }
    )

    result = plan.run(
        inputs={
            "parents": Batch.of([0, 0, 0, 0]),
            "children": InputValue(children, metadata=child_metadata),
        }
    )

    total_metadata = result.outputs.metadata["total"]
    assert (
        total_metadata.sample_at((0,)) is None
    ), "conflicting children -> explicit None"
    assert (0,) in total_metadata.sample and total_metadata.sample[(0,)] is None
    assert total_metadata.sample_at((1,)).source_id == "b"
    assert (
        total_metadata.sample_at((2,)).source_id == "z"
    ), "common child override replaces parent"
    assert (
        total_metadata.sample_at((3,)).source_id == "d"
    ), "empty group keeps parent context"
    assert all(len(index) <= 1 for index in total_metadata.sample)
    assert (
        child_metadata.sample_at((2, 0)).source_id == "z"
    ), "upstream metadata unchanged"
    assert child_metadata.sample_at((0, 1)) is None
    assert result.outputs.metadata["scaled"].sample_at((0, 1)) is None
    assert result.outputs.metadata["scaled"].sample_at((2, 1)).source_id == "z"
    assert (
        result.outputs.metadata["scaled"] is child_metadata
    ), "preserve shares metadata"
    with pytest.raises(TypeError):
        result.outputs.metadata["total"].sample[(0,)] = SampleContext("x")


def test_preserve_and_append_inherit_metadata_and_drop_filtered_paths() -> None:
    registry = _registry([])
    definition = _definition(
        steps=[
            {
                "name": "gate",
                "type": "threshold",
                "inputs": {"value": "$inputs.values"},
                "config": {"minimum": 2},
            },
            {
                "name": "expand",
                "type": "expand",
                "inputs": {"value": "$inputs.values"},
                "when": "$steps.gate.keep",
                "config": {"default": 1},
            },
        ],
        outputs={"children": "$steps.expand.children"},
    )
    plan = compile_workflow(definition, registry=registry)

    result = plan.run(
        inputs={"values": InputValue(Batch.of([1, 2, 3]), metadata=_sample_metadata())}
    )

    metadata = result.outputs.metadata["children"]
    assert set(metadata.sample) == {(1,), (2,)}, "filtered path (0,) dropped"
    assert metadata.sample_at((1, 0)).source_id == "b"
    assert metadata.temporal_at((2, 0)).observed_coverage.ticks == 7
    assert _nested(result.outputs.data["children"]) == {
        (1,): {(1, 0): 20},
        (2,): {(2, 0): 30},
    }
    assert result.filtered_paths["children"] == ((0,),)


# --- errors with context and causes ----------------------------------------


def test_block_exception_is_wrapped_with_step_index_and_cause() -> None:
    registry = _registry([])
    definition = _definition(
        steps=[
            {
                "name": "boom",
                "type": "failing",
                "inputs": {"value": "$inputs.values"},
                "config": {"poison": 2},
            }
        ],
        outputs={"out": "$steps.boom.out"},
    )
    plan = compile_workflow(definition, registry=registry)

    with pytest.raises(WorkflowExecutionError) as info:
        plan.run(inputs={"values": Batch.of([1, 2, 3])})

    assert "$steps.boom block 'failing' failed at index [1]" in str(info.value)
    assert "ZeroDivisionError: poisoned by 2" in str(info.value)
    assert isinstance(info.value.__cause__, ZeroDivisionError)
    trace = info.value.trace
    assert trace[-1]["event"] == "run_failed"
    started = [event for event in trace if event["event"] == "step_started"]
    assert started[-1]["node"] == "$steps.boom" and started[-1]["status"] == "pending"
    completed = [event for event in trace if event["event"] == "invocation_completed"]
    assert [event["index"] for event in completed] == [[0]]


def test_output_kind_failure_names_step_output_index_and_keeps_cause() -> None:
    registry = _registry([])
    _register_custom(
        registry,
        "bad_kind",
        contract=BlockContract(
            reference="value",
            inputs={"value": ITEM_NUMBER},
            outputs={"out": OutputSpec("number")},
        ),
        run=lambda value: {"out": "text" if value == 2 else value},
    )
    plan = compile_workflow(
        _definition(
            steps=[
                {"name": "s", "type": "bad_kind", "inputs": {"value": "$inputs.values"}}
            ],
            outputs={"out": "$steps.s.out"},
        ),
        registry=registry,
    )

    with pytest.raises(WorkflowExecutionError) as info:
        plan.run(inputs={"values": Batch.of([1, 2])})

    assert "$steps.s.out payload at index [1] does not satisfy kind 'number'" in str(
        info.value
    )
    assert isinstance(info.value.__cause__, ContractError)


@pytest.mark.parametrize(
    "returned, expected",
    [
        ({}, "returned an empty mapping; this is a missing result"),
        ({"other": 1}, "Missing: ['out']; unknown: ['other']"),
        ([1], "returned list; blocks must return a mapping"),
        (
            {"out": Batch.of([1])},
            "returned a Batch but an item preserve output must return one payload",
        ),
    ],
)
def test_malformed_results_are_reported_with_step_and_index(returned, expected) -> None:
    registry = _registry([])
    _register_custom(
        registry,
        "custom",
        contract=BlockContract(
            reference="value",
            inputs={"value": ITEM_NUMBER},
            outputs={"out": OutputSpec("number")},
        ),
        run=lambda value: returned,
    )
    plan = compile_workflow(
        _definition(
            steps=[
                {"name": "s", "type": "custom", "inputs": {"value": "$inputs.values"}}
            ],
            outputs={"out": "$steps.s.out"},
        ),
        registry=registry,
    )

    with pytest.raises(WorkflowExecutionError) as info:
        plan.run(inputs={"values": Batch.of([7])})

    assert "$steps.s" in str(info.value) and "index [0]" in str(info.value)
    assert expected in str(info.value)


@pytest.mark.parametrize(
    "returned, expected",
    [
        ({"kids": [1, 2]}, "an append output must return a Batch of children"),
        ({"kids": Batch.of([Batch.of([1])])}, "returned a nested Batch child"),
        (
            {"kids": Batch([1], indices=[(0, 0)], parent_index=(0,))},
            "local one-component indices",
        ),
    ],
)
def test_malformed_append_results_are_rejected(returned, expected) -> None:
    registry = _registry([])
    _register_custom(
        registry,
        "custom",
        contract=BlockContract(
            reference="value",
            inputs={"value": ITEM_NUMBER},
            outputs={"kids": OutputSpec("number", transform="append", axis="k")},
        ),
        run=lambda value: returned,
    )
    plan = compile_workflow(
        _definition(
            steps=[
                {"name": "s", "type": "custom", "inputs": {"value": "$inputs.values"}}
            ],
            outputs={"kids": "$steps.s.kids"},
        ),
        registry=registry,
    )

    with pytest.raises(WorkflowExecutionError, match=re.escape(expected)):
        plan.run(inputs={"values": Batch.of([7])})


def test_batch_preserve_must_echo_supplied_indices() -> None:
    registry = _registry([])
    _register_custom(
        registry,
        "renumber",
        contract=BlockContract(
            reference="group",
            inputs={"group": BATCH_NUMBER},
            outputs={"same": OutputSpec("number", transform="preserve")},
        ),
        run=lambda group: {"same": Batch.of(list(group))},
    )
    plan = compile_workflow(
        _definition(
            steps=[
                {"name": "s", "type": "renumber", "inputs": {"group": "$inputs.values"}}
            ],
            outputs={"same": "$steps.s.same"},
        ),
        registry=registry,
    )

    with pytest.raises(WorkflowExecutionError) as info:
        plan.run(inputs={"values": Batch.of([7, 8], indices=[(3,), (5,)])})

    assert (
        "returned child indices [[0], [1]] but the supplied group domain is [[3], [5]]"
        in str(info.value)
    )


def test_shared_axis_key_requires_corresponding_child_domains() -> None:
    registry = _registry([])
    _register_custom(
        registry,
        "twin",
        contract=BlockContract(
            reference="value",
            inputs={"value": ITEM_NUMBER},
            outputs={
                "left": OutputSpec("number", transform="append", axis="k"),
                "right": OutputSpec("number", transform="append", axis="k"),
            },
        ),
        run=lambda value: {"left": Batch.of([1, 2]), "right": Batch.of([1])},
    )
    plan = compile_workflow(
        _definition(
            steps=[
                {"name": "s", "type": "twin", "inputs": {"value": "$inputs.values"}}
            ],
            outputs={"left": "$steps.s.left"},
        ),
        registry=registry,
    )

    with pytest.raises(WorkflowExecutionError, match=re.escape("share axis key 'k'")):
        plan.run(inputs={"values": Batch.of([7])})


def test_gate_values_must_be_actual_bools() -> None:
    plan = compile_workflow(
        _definition(
            steps=[
                {"name": "g", "type": "truthy", "inputs": {"value": "$inputs.values"}},
                {
                    "name": "s",
                    "type": "scale",
                    "inputs": {"value": "$inputs.values"},
                    "when": "$steps.g.keep",
                },
            ],
            outputs={"s": "$steps.s.scaled"},
        ),
        registry=_registry_with_unchecked_boolean(),
    )

    with pytest.raises(
        WorkflowExecutionError, match=re.escape("gate values must be actual bools")
    ):
        plan.run(inputs={"values": Batch.of([7])})


def _registry_with_unchecked_boolean() -> Registry:
    """Registry whose `boolean` kind has no validator, so 1 reaches the gate."""
    registry = Registry()
    registry.register_kind("number", _is_number)
    registry.register_kind("boolean")
    _register_custom(
        registry,
        "truthy",
        contract=BlockContract(
            reference="value",
            inputs={"value": ITEM_NUMBER},
            outputs={"keep": OutputSpec("boolean")},
        ),
        run=lambda value: {"keep": 1},
    )
    registry.register_block(
        "scale",
        contract=BlockContract(
            reference="value",
            inputs={"value": ITEM_NUMBER},
            outputs={"scaled": OutputSpec("number")},
        ),
        factory=lambda config: _Scale(config, []),
    )
    return registry


@pytest.mark.parametrize(
    "value, expected",
    [
        ("x", "$inputs.values payload at index [0] does not satisfy kind 'number'"),
        (
            Batch.of([Batch.of([1])]),
            "received a Batch at index [0]; there is no automatic payload-to-Batch casting",
        ),
        ([1, 2], "received list at index [] where a Batch was expected"),
        (
            Batch([1], indices=[(0, 0)], parent_index=(0,)),
            "expected a full path of 1 non-negative integers",
        ),
    ],
)
def test_input_binding_errors_name_input_and_index(value, expected) -> None:
    registry = _registry([])
    plan = compile_workflow(
        _definition(
            steps=[
                {"name": "s", "type": "scale", "inputs": {"value": "$inputs.values"}}
            ],
            outputs={"s": "$steps.s.scaled"},
        ),
        registry=registry,
    )

    with pytest.raises(
        (WorkflowExecutionError, ContractError), match=re.escape(expected)
    ):
        plan.run(
            inputs={"values": Batch.of([value]) if isinstance(value, str) else value}
        )


def test_missing_and_unknown_workflow_inputs_are_rejected() -> None:
    registry = _registry([])
    plan = compile_workflow(
        _definition(
            steps=[
                {"name": "s", "type": "scale", "inputs": {"value": "$inputs.values"}}
            ],
            outputs={"s": "$steps.s.scaled"},
        ),
        registry=registry,
    )

    with pytest.raises(
        WorkflowExecutionError,
        match=re.escape("Missing: ['values']; unknown: ['other']"),
    ):
        plan.run(inputs={"other": Batch.of([1])})


# --- invocation freshness and reuse -----------------------------------------


def test_plan_reuse_after_failure_has_fresh_state_and_distinct_identities() -> None:
    calls: List[Dict[str, Any]] = []
    registry = _registry(calls)
    definition = _definition(
        steps=[
            {"name": "count", "type": "counter", "inputs": {"value": "$inputs.values"}},
            {
                "name": "boom",
                "type": "failing",
                "inputs": {"value": "$inputs.values"},
                "config": {"poison": 99},
            },
            {
                "name": "expand",
                "type": "expand",
                "inputs": {"value": "$inputs.values"},
                "config": {"counts": {"1": 2, "2": 0}},
            },
            {
                "name": "sum",
                "type": "sum",
                "inputs": {
                    "parent": "$inputs.values",
                    "children": "$steps.expand.children",
                },
            },
        ],
        outputs={"count": "$steps.count.count", "total": "$steps.sum.total"},
    )
    plan = compile_workflow(definition, registry=registry)

    first = plan.run(inputs={"values": Batch.of([1, 2])})
    second = plan.run(
        inputs={"values": Batch.of([2, 2, 1], indices=[(0,), (3,), (7,)])}
    )
    with pytest.raises(WorkflowExecutionError):
        plan.run(inputs={"values": Batch.of([1, 99])})
    third = plan.run(inputs={"values": Batch.of([1, 2])})
    fresh = compile_workflow(definition, registry=registry).run(
        inputs={"values": Batch.of([1, 2])}
    )

    assert list(first.outputs.data["count"]) == [
        1,
        2,
    ], "one instance per run, counter restarts"
    assert list(second.outputs.data["count"]) == [1, 2, 3]
    assert list(third.outputs.data["count"]) == [1, 2]
    assert _nested(second.outputs.data["total"]) == {
        (0,): 2,
        (3,): 2,
        (7,): 1 + 10 + 11,
    }
    assert _nested(third.outputs.data["total"]) == _nested(fresh.outputs.data["total"])
    assert third.outputs.data["total"] == fresh.outputs.data["total"]
    assert third.statuses == fresh.statuses
    assert len({first.invocation_id, second.invocation_id, third.invocation_id}) == 3
    assert (
        first.outputs.pulse_id,
        second.outputs.pulse_id,
        third.outputs.pulse_id,
    ) == (0, 1, 3)
    assert first.outputs.lineage_id == third.outputs.lineage_id
    assert fresh.outputs.lineage_id != first.outputs.lineage_id
    assert fresh.outputs.pulse_id == 0
    assert all(event["event"] != "run_failed" for event in third.trace)


def test_no_block_runs_before_run_and_factories_are_fresh_per_invocation() -> None:
    instances: List[int] = []

    class _Tracking:
        def __init__(self, config: Mapping[str, Any]):
            instances.append(id(self))

        def run(self, value: Any) -> Dict[str, Any]:
            return {"out": value}

    registry = Registry()
    registry.register_kind("number", _is_number)
    registry.register_block(
        "tracking",
        contract=BlockContract(
            reference="value",
            inputs={"value": ITEM_NUMBER},
            outputs={"out": OutputSpec("number")},
        ),
        factory=_Tracking,
    )
    plan = compile_workflow(
        _definition(
            steps=[
                {"name": "t", "type": "tracking", "inputs": {"value": "$inputs.values"}}
            ],
            outputs={"out": "$steps.t.out"},
        ),
        registry=registry,
    )
    compiled_instances = len(instances)

    plan.run(inputs={"values": Batch.of([1])})
    plan.run(inputs={"values": Batch.of([1])})

    assert compiled_instances == 1, "factory validates config at compile time"
    assert len(instances) == 3, "one fresh instance per invocation"


class _History:
    """Keeps nested config containers and appends to them while running."""

    def __init__(self, config: Mapping[str, Any]):
        self._history = config["history"]
        self._seen = config["nested"]["seen"]

    def run(self, value: Any) -> Dict[str, Any]:
        if value == 99:
            self._history.append("before failure")
            raise RuntimeError("fails after mutating its config")
        self._history.append(value)
        self._seen[str(value)] = True
        return {"length": len(self._history), "seen": len(self._seen)}


def _history_plan():
    registry = Registry()
    registry.register_kind("number", _is_number)
    registry.register_block(
        "history",
        contract=BlockContract(
            reference="value",
            inputs={"value": ITEM_NUMBER},
            outputs={"length": OutputSpec("number"), "seen": OutputSpec("number")},
        ),
        factory=_History,
    )
    definition = _definition(
        inputs=[{"name": "value", "kind": "number"}],
        steps=[
            {
                "name": "h",
                "type": "history",
                "inputs": {"value": "$inputs.value"},
                "config": {"history": [], "nested": {"seen": {}}},
            }
        ],
        outputs={"length": "$steps.h.length", "seen": "$steps.h.seen"},
    )
    plan = compile_workflow(definition, registry=registry)
    return plan, definition


def test_nested_config_state_does_not_leak_between_runs_or_into_the_plan() -> None:
    plan, definition = _history_plan()

    first = plan.run(inputs={"value": 1})
    second = plan.run(inputs={"value": 2})
    with pytest.raises(WorkflowExecutionError) as info:
        plan.run(inputs={"value": 99})
    after_failure = plan.run(inputs={"value": 3})

    lengths = [first.outputs.data["length"], second.outputs.data["length"]]
    assert lengths == [1, 1], "each run gets a fresh instance with its own config copy"
    assert [first.outputs.data["seen"], second.outputs.data["seen"]] == [1, 1]
    assert isinstance(info.value.__cause__, RuntimeError)
    assert after_failure.outputs.data["length"] == 1
    assert after_failure.outputs.data["seen"] == 1
    step = plan.steps[0]
    assert dict(step.config["nested"]["seen"]) == {}
    assert step.config["history"] == ()
    assert step.materialize_config() == {"history": [], "nested": {"seen": {}}}
    assert definition["steps"][0]["config"] == {"history": [], "nested": {"seen": {}}}


def test_external_edits_of_definition_and_exposed_config_do_not_reach_blocks() -> None:
    plan, definition = _history_plan()
    definition["steps"][0]["config"]["history"].extend([7, 8, 9])
    definition["steps"][0]["config"]["nested"]["seen"]["x"] = True

    with pytest.raises(TypeError):
        plan.steps[0].config["nested"]["seen"]["x"] = True
    with pytest.raises(AttributeError):
        plan.steps[0].config["history"].append(1)
    result = plan.run(inputs={"value": 1})

    assert result.outputs.data["length"] == 1
    assert result.outputs.data["seen"] == 1


def test_plain_values_are_accepted_and_trace_is_json_friendly() -> None:
    import json

    registry = _registry([])
    plan = compile_workflow(
        _definition(
            inputs=[{"name": "value", "kind": "number"}],
            steps=[
                {"name": "s", "type": "scale", "inputs": {"value": "$inputs.value"}}
            ],
            outputs={"s": "$steps.s.scaled"},
        ),
        registry=registry,
    )

    result = plan.run(inputs={"value": 21})

    assert result.outputs.data["s"] == 42
    assert result.outputs.metadata["s"].is_empty
    assert json.loads(json.dumps(list(result.trace)))[0]["event"] == "run_started"
    assert [event["event"] for event in result.trace] == [
        "run_started",
        "input_bound",
        "step_started",
        "invocation_completed",
        "step_completed",
        "output_resolved",
        "run_completed",
    ]


# --- review corrections: lineage masks, batch alignment, group domains -------


def _depth_input(name: str, depth: int) -> Dict[str, Any]:
    axes = [
        {"id": "samples", "kind": "sample"},
        {"id": "children", "kind": "dynamic_nesting"},
        {"id": "grandchildren", "kind": "dynamic_nesting"},
    ][:depth]
    return {"name": name, "kind": "number", "axes": axes}


def _register_reexpand(registry: Registry, calls: List[Dict[str, Any]]) -> None:
    """Parent item + old child group -> fresh children, preserved old, count."""

    def run(parent: Any, children: Batch) -> Dict[str, Any]:
        calls.append({"block": "reexpand", "parent": parent, "old": children.indices})
        kept = Batch(
            [value + 1 for value in children],
            indices=children.indices,
            parent_index=children.parent_index,
        )
        return {
            "fresh": Batch.of(
                [parent * 100 + local for local in (2, 5, 9)],
                indices=[(2,), (5,), (9,)],
            ),
            "kept": kept,
            "count": len(children),
        }

    _register_custom(
        registry,
        "reexpand",
        contract=BlockContract(
            reference="children",
            inputs={"parent": ITEM_NUMBER, "children": BATCH_NUMBER},
            outputs={
                "fresh": OutputSpec(
                    "number", transform="append", source="parent", axis="fresh"
                ),
                "kept": OutputSpec("number", transform="preserve"),
                "count": OutputSpec("number", transform="collapse"),
            },
        ),
        run=run,
    )
    _register_custom(
        registry,
        "old_gate",
        contract=BlockContract(
            reference="value",
            inputs={"value": ITEM_NUMBER},
            outputs={"keep": OutputSpec("boolean")},
        ),
        run=lambda value: {"keep": value not in (11, 30)},
    )


def _reexpand_definition() -> Dict[str, Any]:
    return _definition(
        steps=[
            {
                "name": "expand",
                "type": "expand",
                "inputs": {"value": "$inputs.values"},
                "config": {
                    "counts": {"1": 3, "2": 0, "3": 1, "4": 2},
                    "indices": [[2], [5], [9]],
                },
            },
            {
                "name": "old_gate",
                "type": "old_gate",
                "inputs": {"value": "$steps.expand.children"},
            },
            {
                "name": "kept_old",
                "type": "scale",
                "inputs": {"value": "$steps.expand.children"},
                "when": "$steps.old_gate.keep",
            },
            {
                "name": "parent_gate",
                "type": "threshold",
                "inputs": {"value": "$inputs.values"},
                "config": {"minimum": -1},
            },
            {
                "name": "reexpand",
                "type": "reexpand",
                "inputs": {
                    "parent": "$inputs.values",
                    "children": "$steps.kept_old.scaled",
                },
                "when": "$steps.parent_gate.keep",
            },
            {
                "name": "after",
                "type": "scale",
                "inputs": {"value": "$steps.reexpand.fresh"},
            },
            {
                "name": "fresh_sum",
                "type": "sum",
                "inputs": {
                    "parent": "$inputs.values",
                    "children": "$steps.after.scaled",
                },
            },
            {
                "name": "old_after",
                "type": "scale",
                "inputs": {"value": "$steps.reexpand.kept"},
            },
        ],
        outputs={
            "fresh": "$steps.reexpand.fresh",
            "kept": "$steps.reexpand.kept",
            "count": "$steps.reexpand.count",
            "after": "$steps.after.scaled",
            "fresh_sum": "$steps.fresh_sum.total",
            "old_after": "$steps.old_after.scaled",
        },
    )


def test_old_child_masks_do_not_leak_into_fresh_append_lineage() -> None:
    calls: List[Dict[str, Any]] = []
    registry = _registry(calls)
    _register_reexpand(registry, calls)
    # parent_gate drops value 4 only
    definition = _reexpand_definition()
    definition["steps"][3]["type"] = "not_four"
    _register_custom(
        registry,
        "not_four",
        contract=BlockContract(
            reference="value",
            inputs={"value": ITEM_NUMBER},
            outputs={"keep": OutputSpec("boolean")},
        ),
        run=lambda value: {"keep": value != 4},
    )
    definition["steps"][3].pop("config")
    plan = compile_workflow(definition, registry=registry)
    metadata = EntryMetadata(
        sample={
            (3,): SampleContext("p3"),
            (7,): SampleContext("p7"),
            (8,): SampleContext("p8"),
            (9,): SampleContext("p9"),
        }
    )

    # Parents (3,)=1 (7,)=2 (8,)=3 (9,)=4. Old children at local 2, 5, 9:
    # (3,): 10, 11, 12 -> gate drops 11 at (3, 5); (7,): genuine empty;
    # (8,): 30 -> all gated; (9,): 40, 41 but the parent gate drops (9,).
    result = plan.run(
        inputs={
            "values": InputValue(
                Batch.of([1, 2, 3, 4], indices=[(3,), (7,), (8,), (9,)]),
                metadata=metadata,
            )
        }
    )

    fresh = {
        (3,): {(3, 2): 102, (3, 5): 105, (3, 9): 109},
        (7,): {(7, 2): 202, (7, 5): 205, (7, 9): 209},
    }
    assert _nested(result.outputs.data["fresh"]) == fresh
    assert result.outputs.layout["fresh"].axis_ids == ("samples", "reexpand/fresh")
    assert result.filtered_paths["fresh"] == ((8,), (9,))
    assert _nested(result.outputs.data["after"]) == {
        parent: {index: value * 2 for index, value in children.items()}
        for parent, children in fresh.items()
    }
    assert result.filtered_paths["after"] == ((8,), (9,))
    assert _nested(result.outputs.data["fresh_sum"]) == {
        (3,): 1 + 204 + 210 + 218,
        (7,): 2 + 404 + 410 + 418,
    }
    fresh_sums = [call for call in calls if call["block"] == "sum"]
    assert [call["indices"] for call in fresh_sums] == [
        ((3, 2), (3, 5), (3, 9)),
        ((7, 2), (7, 5), (7, 9)),
    ]
    # The preserved old child keeps its own old-axis mask.
    assert _nested(result.outputs.data["kept"]) == {
        (3,): {(3, 2): 21, (3, 9): 25},
        (7,): {},
    }
    assert result.outputs.layout["kept"].axis_ids == ("samples", "expand/kids")
    assert result.filtered_paths["kept"] == ((3, 5), (8,), (9,))
    assert _nested(result.outputs.data["old_after"]) == {
        (3,): {(3, 2): 42, (3, 9): 50},
        (7,): {},
    }
    assert result.filtered_paths["old_after"] == ((3, 5), (8,), (9,))
    assert _nested(result.outputs.data["count"]) == {(3,): 2, (7,): 0}
    assert result.filtered_paths["count"] == ((8,), (9,))
    reexpand_calls = [call for call in calls if call["block"] == "reexpand"]
    assert [(call["parent"], call["old"]) for call in reexpand_calls] == [
        (1, ((3, 2), (3, 9))),
        (2, ()),
    ]
    assert result.outputs.metadata["fresh"].sample_at((3, 5)).source_id == "p3"
    assert result.outputs.metadata["kept"].sample_at((3, 9)).source_id == "p3"
    assert result.outputs.metadata["count"].sample_at((7,)).source_id == "p7"
    assert set(result.outputs.metadata["fresh"].sample) == {(3,), (7,)}


def _register_weighted(registry: Registry, calls: List[Dict[str, Any]]) -> None:
    def run(left: Batch, right: Batch) -> Dict[str, Any]:
        calls.append({"left": left.indices, "right": right.indices})
        return {"total": sum(a * b for a, b in zip(left, right))}

    _register_custom(
        registry,
        "weighted",
        contract=BlockContract(
            reference="left",
            inputs={"left": BATCH_NUMBER, "right": BATCH_NUMBER},
            outputs={"total": OutputSpec("number", transform="collapse")},
        ),
        run=run,
    )


def test_batch_views_are_aligned_to_one_sorted_logical_sequence() -> None:
    calls: List[Dict[str, Any]] = []
    registry = _registry([])
    _register_weighted(registry, calls)
    plan = compile_workflow(
        _definition(
            inputs=[_samples_input("left"), _samples_input("right")],
            steps=[
                {
                    "name": "w",
                    "type": "weighted",
                    "inputs": {"left": "$inputs.left", "right": "$inputs.right"},
                }
            ],
            outputs={"total": "$steps.w.total"},
        ),
        registry=registry,
    )

    result = plan.run(
        inputs={
            "left": Batch.of([9, 2, 5], indices=[(9,), (2,), (5,)]),
            "right": Batch.of([50, 90, 20], indices=[(5,), (9,), (2,)]),
        }
    )

    assert result.outputs.data["total"] == 2 * 20 + 5 * 50 + 9 * 90
    assert calls == [{"left": ((2,), (5,), (9,)), "right": ((2,), (5,), (9,))}]


def test_batch_alignment_holds_when_only_one_side_was_filtered() -> None:
    calls: List[Dict[str, Any]] = []
    registry = _registry([])
    _register_weighted(registry, calls)
    plan = compile_workflow(
        _definition(
            inputs=[_depth_input("left", 2), _depth_input("right", 2)],
            steps=[
                {
                    "name": "gate",
                    "type": "threshold",
                    "inputs": {"value": "$inputs.left"},
                    "config": {"minimum": 3},
                },
                {
                    "name": "kept",
                    "type": "scale",
                    "inputs": {"value": "$inputs.left"},
                    "when": "$steps.gate.keep",
                    "config": {"factor": 1},
                },
                {
                    "name": "w",
                    "type": "weighted",
                    "inputs": {"left": "$steps.kept.scaled", "right": "$inputs.right"},
                },
            ],
            outputs={"total": "$steps.w.total"},
        ),
        registry=registry,
    )
    # Same physical order on both sides; filtering (4, 9) rebuilds only the
    # left group, the right group stays in its original descending order.
    left = Batch(
        [
            Batch(
                [9, 1, 5, 2],
                indices=[(4, 9), (4, 7), (4, 5), (4, 2)],
                parent_index=(4,),
            )
        ],
        indices=[(4,)],
    )
    right = Batch(
        [
            Batch(
                [90, 70, 50, 20],
                indices=[(4, 9), (4, 7), (4, 5), (4, 2)],
                parent_index=(4,),
            )
        ],
        indices=[(4,)],
    )

    result = plan.run(inputs={"left": left, "right": right})

    # gate keeps >= 3: (4, 9)=9 and (4, 5)=5 survive; (4, 7) and (4, 2) dropped
    assert calls == [{"left": ((4, 5), (4, 9)), "right": ((4, 5), (4, 9))}]
    assert _nested(result.outputs.data["total"]) == {(4,): 5 * 50 + 9 * 90}


def test_batch_preserve_output_follows_canonical_order_for_permuted_input() -> None:
    registry = _registry([])
    plan = compile_workflow(
        _definition(
            steps=[
                {
                    "name": "echo",
                    "type": "echo_group",
                    "inputs": {"group": "$inputs.values"},
                }
            ],
            outputs={"same": "$steps.echo.same"},
        ),
        registry=registry,
    )

    result = plan.run(inputs={"values": Batch.of([9, 2], indices=[(9,), (2,)])})

    assert _nested(result.outputs.data["same"]) == {(2,): 2, (9,): 9}


def _register_pair(registry: Registry) -> None:
    _register_custom(
        registry,
        "pair",
        contract=BlockContract(
            reference="left",
            inputs={"left": ITEM_NUMBER, "right": ITEM_NUMBER},
            outputs={"sum": OutputSpec("number")},
        ),
        run=lambda left, right: {"sum": left + right},
    )


def _pair_plan(registry: Registry, *, left: str = "$inputs.left"):
    definition = _definition(
        inputs=[_depth_input("left", 2), _depth_input("right", 2)],
        steps=[
            {
                "name": "pair",
                "type": "pair",
                "inputs": {"left": left, "right": "$inputs.right"},
            }
        ],
        outputs={"sum": "$steps.pair.sum"},
    )
    return definition


def test_unmatched_genuine_empty_parents_are_rejected_for_item_inputs() -> None:
    registry = _registry([])
    _register_pair(registry)
    plan = compile_workflow(_pair_plan(registry), registry=registry)
    left = Batch(
        [Batch([1], parent_index=(2,)), Batch.empty(parent_index=(5,))],
        indices=[(2,), (5,)],
    )
    right = Batch([Batch([10], parent_index=(2,))], indices=[(2,)])

    with pytest.raises(WorkflowExecutionError) as info:
        plan.run(inputs={"left": left, "right": right})

    assert "$steps.pair input domains disagree at index [5]" in str(info.value)
    assert "absent (and not filtered) on ['right']" in str(info.value)
    assert "including empty groups" in str(info.value)


def test_matching_empty_parents_and_sparse_domains_are_kept() -> None:
    registry = _registry([])
    _register_pair(registry)
    plan = compile_workflow(_pair_plan(registry), registry=registry)
    left = Batch(
        [
            Batch([1], indices=[(2, 4)], parent_index=(2,)),
            Batch.empty(parent_index=(5,)),
        ],
        indices=[(2,), (5,)],
    )
    right = Batch(
        [
            Batch.empty(parent_index=(5,)),
            Batch([10], indices=[(2, 4)], parent_index=(2,)),
        ],
        indices=[(5,), (2,)],
    )

    result = plan.run(inputs={"left": left, "right": right})

    assert _nested(result.outputs.data["sum"]) == {(2,): {(2, 4): 11}, (5,): {}}
    assert result.statuses == {"sum": "complete"}
    assert result.filtered_paths["sum"] == ()


def test_explicitly_filtered_parent_may_be_absent_on_one_input() -> None:
    registry = _registry([])
    _register_pair(registry)
    definition = _pair_plan(registry, left="$steps.kept.scaled")
    definition["steps"][:0] = [
        {
            "name": "gate",
            "type": "threshold",
            "inputs": {"value": "$inputs.left"},
            "config": {"minimum": 3},
        },
        {
            "name": "kept",
            "type": "scale",
            "inputs": {"value": "$inputs.left"},
            "when": "$steps.gate.keep",
            "config": {"factor": 1},
        },
    ]
    plan = compile_workflow(definition, registry=registry)
    # Left group (5,) has candidates that are all gated, so it is filtered on
    # the left; right holds (5,) as a genuine empty group and (6,) populated.
    left = Batch(
        [
            Batch([4], parent_index=(2,)),
            Batch([1], parent_index=(5,)),
            Batch([2, 1], parent_index=(6,)),
        ],
        indices=[(2,), (5,), (6,)],
    )
    right = Batch(
        [
            Batch([40], parent_index=(2,)),
            Batch.empty(parent_index=(5,)),
            Batch([60, 61], parent_index=(6,)),
        ],
        indices=[(2,), (5,), (6,)],
    )

    result = plan.run(inputs={"left": left, "right": right})

    assert _nested(result.outputs.data["sum"]) == {(2,): {(2, 0): 44}}
    assert result.filtered_paths["sum"] == ((5,), (6,))


def test_deeper_mixed_inputs_compare_empty_parent_domains() -> None:
    calls: List[Dict[str, Any]] = []
    registry = _registry(calls)
    plan = compile_workflow(
        _definition(
            inputs=[_depth_input("parent", 2), _depth_input("children", 3)],
            steps=[
                {
                    "name": "sum",
                    "type": "sum",
                    "inputs": {
                        "parent": "$inputs.parent",
                        "children": "$inputs.children",
                    },
                }
            ],
            outputs={"total": "$steps.sum.total"},
        ),
        registry=registry,
    )
    parent = Batch(
        [
            Batch([1, 2], parent_index=(2,)),
            Batch.empty(parent_index=(5,)),
        ],
        indices=[(2,), (5,)],
    )
    matching_children = Batch(
        [
            Batch(
                [
                    Batch([10, 20], parent_index=(2, 0)),
                    Batch.empty(parent_index=(2, 1)),
                ],
                parent_index=(2,),
            ),
            Batch.empty(parent_index=(5,)),
        ],
        indices=[(2,), (5,)],
    )
    mismatching_children = Batch(
        [matching_children[0], Batch.empty(parent_index=(6,))],
        indices=[(2,), (6,)],
    )

    result = plan.run(inputs={"parent": parent, "children": matching_children})
    with pytest.raises(WorkflowExecutionError) as info:
        plan.run(inputs={"parent": parent, "children": mismatching_children})

    assert _nested(result.outputs.data["total"]) == {
        (2,): {(2, 0): 31, (2, 1): 2},
        (5,): {},
    }
    assert [call["indices"] for call in calls] == [((2, 0, 0), (2, 0, 1)), ()]
    assert "$steps.sum input domains disagree at index [5]" in str(info.value)
    assert "absent (and not filtered) on ['children']" in str(info.value)
    assert len(calls) == 2, "no block runs before the domain mismatch is reported"


# --- decision 003: ancestor gates also filter empty descendant groups --------


def _tree_of(content: Dict[Index, Any], parent: Index = ()) -> Batch:
    """Build a nested Batch from ``{index: child}``; dicts become groups."""
    indices = sorted(content)
    return Batch(
        [
            (
                _tree_of(content[index], index)
                if isinstance(content[index], dict)
                else content[index]
            )
            for index in indices
        ],
        indices=indices,
        parent_index=parent,
    )


def _events(result: Any, node: str) -> List[Any]:
    return [
        (event["index"], event["reason"])
        for event in result.trace
        if event["event"] == "invocation_filtered" and event["node"] == node
    ]


def test_false_ancestor_gate_filters_sparse_nested_empty_groups() -> None:
    registry = _registry([])
    plan = compile_workflow(
        _definition(
            inputs=[_depth_input("flags", 1), _depth_input("values", 3)],
            steps=[
                {
                    "name": "gate",
                    "type": "threshold",
                    "inputs": {"value": "$inputs.flags"},
                    "config": {"minimum": 1},
                },
                {
                    "name": "gated",
                    "type": "scale",
                    "inputs": {"value": "$inputs.values"},
                    "when": "$steps.gate.keep",
                },
                {
                    "name": "inner",
                    "type": "echo_group",
                    "inputs": {"group": "$steps.gated.scaled"},
                },
                {
                    "name": "outer",
                    "type": "echo_group",
                    "inputs": {"group": "$steps.inner.size"},
                },
                {
                    "name": "sibling",
                    "type": "scale",
                    "inputs": {"value": "$inputs.values"},
                },
                {
                    "name": "sibling_inner",
                    "type": "echo_group",
                    "inputs": {"group": "$steps.sibling.scaled"},
                },
            ],
            outputs={
                "gated": "$steps.gated.scaled",
                "inner": "$steps.inner.size",
                "outer": "$steps.outer.size",
                "sibling_inner": "$steps.sibling_inner.size",
            },
        ),
        registry=registry,
    )
    # (2,) true: a leaf plus a nested empty group; (4,) false with a leaf;
    # (6,) true and empty; (8,) false with only a nested empty group.
    flags = Batch.of([1, 0, 1, 0], indices=[(2,), (4,), (6,), (8,)])
    values = _tree_of(
        {
            (2,): {(2, 1): {(2, 1, 0): 5}, (2, 3): {}},
            (4,): {(4, 0): {(4, 0, 0): 7}},
            (6,): {},
            (8,): {(8, 0): {}},
        }
    )

    result = plan.run(inputs={"flags": flags, "values": values})

    assert _nested(result.outputs.data["gated"]) == {
        (2,): {(2, 1): {(2, 1, 0): 10}, (2, 3): {}},
        (6,): {},
    }
    assert result.filtered_paths["gated"] == ((4,), (8,))
    assert _nested(result.outputs.data["inner"]) == {
        (2,): {(2, 1): 1, (2, 3): 0},
        (6,): {},
    }
    assert result.filtered_paths["inner"] == ((4,), (8,))
    assert _nested(result.outputs.data["outer"]) == {(2,): 2, (6,): 0}
    assert result.filtered_paths["outer"] == ((4,), (8,))
    assert _nested(result.outputs.data["sibling_inner"]) == {
        (2,): {(2, 1): 1, (2, 3): 0},
        (4,): {(4, 0): 1},
        (6,): {},
        (8,): {(8, 0): 0},
    }
    assert result.filtered_paths["sibling_inner"] == ()
    assert set(result.statuses.values()) == {"complete"}
    assert _events(result, "$steps.gated") == [([4], "gate_false"), ([8], "gate_false")]


def test_filtered_ancestor_gate_filters_empty_group() -> None:
    registry = _registry([])
    plan = compile_workflow(
        _definition(
            inputs=[_depth_input("flags", 1), _depth_input("values", 2)],
            steps=[
                {
                    "name": "pre",
                    "type": "threshold",
                    "inputs": {"value": "$inputs.flags"},
                    "config": {"minimum": 1},
                },
                {
                    "name": "gate",
                    "type": "threshold",
                    "inputs": {"value": "$inputs.flags"},
                    "when": "$steps.pre.keep",
                    "config": {"minimum": 0},
                },
                {
                    "name": "gated",
                    "type": "scale",
                    "inputs": {"value": "$inputs.values"},
                    "when": "$steps.gate.keep",
                },
                {
                    "name": "size",
                    "type": "echo_group",
                    "inputs": {"group": "$steps.gated.scaled"},
                },
            ],
            outputs={"gated": "$steps.gated.scaled", "size": "$steps.size.size"},
        ),
        registry=registry,
    )

    result = plan.run(
        inputs={
            "flags": Batch.of([1, 0], indices=[(0,), (3,)]),
            "values": _tree_of({(0,): {}, (3,): {}}),
        }
    )

    assert _nested(result.outputs.data["gated"]) == {(0,): {}}
    assert result.filtered_paths["gated"] == ((3,),)
    assert _nested(result.outputs.data["size"]) == {(0,): 0}
    assert result.filtered_paths["size"] == ((3,),)
    assert _events(result, "$steps.gated") == [([3], "gate_filtered")]


@pytest.mark.parametrize("flag, expected_status", [(0, "filtered"), (1, "complete")])
def test_scalar_gate_over_entirely_empty_grouped_input(flag, expected_status) -> None:
    registry = _registry([])
    plan = compile_workflow(
        _definition(
            inputs=[{"name": "flag", "kind": "number"}, _samples_input()],
            steps=[
                {
                    "name": "gate",
                    "type": "threshold",
                    "inputs": {"value": "$inputs.flag"},
                    "config": {"minimum": 1},
                },
                {
                    "name": "gated",
                    "type": "scale",
                    "inputs": {"value": "$inputs.values"},
                    "when": "$steps.gate.keep",
                },
                {
                    "name": "size",
                    "type": "echo_group",
                    "inputs": {"group": "$steps.gated.scaled"},
                },
                {
                    "name": "sibling",
                    "type": "echo_group",
                    "inputs": {"group": "$inputs.values"},
                },
            ],
            outputs={
                "gated": "$steps.gated.scaled",
                "size": "$steps.size.size",
                "sibling": "$steps.sibling.size",
            },
        ),
        registry=registry,
    )

    result = plan.run(inputs={"flag": flag, "values": Batch.empty()})

    assert result.statuses == {
        "gated": expected_status,
        "size": expected_status,
        "sibling": "complete",
    }
    assert result.outputs.data["sibling"] == 0
    if expected_status == "filtered":
        assert set(result.outputs.data) == {"sibling"}
        assert result.filtered_paths["gated"] == ((),)
        assert result.filtered_paths["size"] == ((),)
        assert _events(result, "$steps.gated") == [([], "gate_false")]
    else:
        assert _nested(result.outputs.data["gated"]) == {}
        assert result.outputs.data["size"] == 0
        assert _events(result, "$steps.gated") == []


def test_child_level_gate_without_children_fabricates_no_parent_decision() -> None:
    registry = _registry([])
    plan = compile_workflow(
        _definition(
            inputs=[_depth_input("mids", 2), _depth_input("values", 3)],
            steps=[
                {
                    "name": "gate",
                    "type": "threshold",
                    "inputs": {"value": "$inputs.mids"},
                    "config": {"minimum": 1},
                },
                {
                    "name": "gated",
                    "type": "scale",
                    "inputs": {"value": "$inputs.values"},
                    "when": "$steps.gate.keep",
                },
            ],
            outputs={"gated": "$steps.gated.scaled"},
        ),
        registry=registry,
    )
    # (0, 0) has a false decision over an empty group; (1,) has no children,
    # so the child-level gate has nothing to decide there.
    mids = _tree_of({(0,): {(0, 0): 0, (0, 1): 5}, (1,): {}})
    values = _tree_of({(0,): {(0, 0): {}, (0, 1): {(0, 1, 0): 3}}, (1,): {}})

    result = plan.run(inputs={"mids": mids, "values": values})

    assert _nested(result.outputs.data["gated"]) == {
        (0,): {(0, 1): {(0, 1, 0): 6}},
        (1,): {},
    }
    assert result.filtered_paths["gated"] == ((0, 0),)
    assert result.statuses["gated"] == "complete"
    assert _events(result, "$steps.gated") == [([0, 0], "gate_false")]


def test_missing_ancestor_decision_for_an_empty_path_is_an_error() -> None:
    registry = _registry([])
    plan = compile_workflow(
        _definition(
            inputs=[_depth_input("flags", 1), _depth_input("values", 2)],
            steps=[
                {
                    "name": "gate",
                    "type": "threshold",
                    "inputs": {"value": "$inputs.flags"},
                    "config": {"minimum": 1},
                },
                {
                    "name": "gated",
                    "type": "scale",
                    "inputs": {"value": "$inputs.values"},
                    "when": "$steps.gate.keep",
                },
            ],
            outputs={"gated": "$steps.gated.scaled"},
        ),
        registry=registry,
    )

    with pytest.raises(WorkflowExecutionError) as info:
        plan.run(
            inputs={
                "flags": Batch.of([1], indices=[(2,)]),
                "values": _tree_of({(2,): {(2, 0): 4}, (5,): {}}),
            }
        )

    assert "$steps.gated gate '$steps.gate.keep' has no value at index [5]" in str(
        info.value
    )
    assert "possibly as an empty group" in str(info.value)
    assert info.value.trace[-1]["event"] == "run_failed"
