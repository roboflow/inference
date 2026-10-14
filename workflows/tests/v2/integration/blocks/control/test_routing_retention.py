"""delta_filter keeps each lane's last value by reference (``retains_inputs``).

    fresh ─▶ delta(value) ──next_steps──▶ bump(value, mutates)

The engine copies nothing. The compiler reports ``bump`` because it changes
the object ``delta`` keeps for the next comparison. Objects the caller passes
in are not reported: reusing one across runs is the caller's responsibility.
"""

import re
from typing import Any, Dict, List

import pytest
import torch
from roboflow_workflows.v2.blocks.control.delta_filter.v1 import DeltaFilterBlock
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref
from roboflow_workflows.v2.engine.errors import MutationConflictError
from roboflow_workflows.v2.engine.plan import CompileOptions


class _Fresh(Block):
    """Test block: a new ``{"count": 0}`` per call."""

    type = "test/fresh@v1"
    outputs = {"value": Output(source="image")}

    class Params(BlockParams):
        image: Ref(IMAGE_KIND)

    def run(self, *, image: ImageData) -> Dict[str, Any]:
        return {"value": {"count": 0}}


class _Bump(Block):
    """Test block: increments ``value["count"]`` in place."""

    type = "test/bump@v1"
    mutates = ("value",)
    outputs = {"count": Output()}

    class Params(BlockParams):
        value: Ref()

    def run(self, *, value: Any) -> Dict[str, Any]:
        value["count"] += 1

        return {"count": value["count"]}


class _Mark(Block):
    """Test block: outputs ``True`` when it runs."""

    type = "test/mark@v1"
    outputs = {"ran": Output(source="image")}

    class Params(BlockParams):
        image: Ref(IMAGE_KIND)

    def run(self, *, image: ImageData) -> Dict[str, Any]:
        return {"ran": True}


CATALOGUE = Catalogue([DeltaFilterBlock, _Fresh, _Bump, _Mark], kinds=[IMAGE_KIND])
IMAGE_INPUT = {"type": "WorkflowImage", "name": "image"}
VALUE_INPUT = {"type": "WorkflowParameter", "name": "value"}
STRICT = CompileOptions(mutation_conflicts="error")


def _image() -> ImageData:
    image = ImageData.from_tensor(torch.zeros((3, 4, 4), dtype=torch.uint8))

    return image


def _definition(
    steps: List[dict], outputs: Dict[str, str], *, inputs: List[dict]
) -> dict:
    definition = {
        "version": "2.0",
        "inputs": inputs,
        "steps": steps,
        "outputs": [
            {"type": "JsonField", "name": name, "selector": selector}
            for name, selector in outputs.items()
        ],
    }

    return definition


def _delta(value: str, *, next_step: str) -> dict:
    step = {
        "type": DeltaFilterBlock.type,
        "name": "delta",
        "image": "$inputs.image",
        "value": value,
        "next_steps": [next_step],
    }

    return step


# delta(value from the caller) -> mark: mark runs when the value changed.
CALLER_VALUE = _definition(
    [
        _delta("$inputs.value", next_step="$steps.mark"),
        {"type": _Mark.type, "name": "mark", "image": "$inputs.image"},
    ],
    {"ran": "$steps.mark.ran"},
    inputs=[IMAGE_INPUT, VALUE_INPUT],
)
# fresh -> delta -> bump: bump changes the object delta keeps.
BUMP_AFTER_DELTA = _definition(
    [
        {"type": _Fresh.type, "name": "fresh", "image": "$inputs.image"},
        _delta("$steps.fresh.value", next_step="$steps.bump"),
        {"type": _Bump.type, "name": "bump", "value": "$steps.fresh.value"},
    ],
    {"count": "$steps.bump.count"},
    inputs=[IMAGE_INPUT],
)


def _ran(session: Any, value: Any) -> bool:
    [row] = session.run({"image": _image(), "value": value}).rows()

    return row["ran"] is not None


def test_the_declaration_keeps_equal_values_filtered_without_warnings():
    plan = compile_workflow(CALLER_VALUE, catalogue=CATALOGUE, options=STRICT)
    session = plan.create_session()

    ran = [_ran(session, value) for value in (3, 3, 4, 4.0, True, 1, [1], [1])]

    assert DeltaFilterBlock.retains_inputs == ("value",)
    assert ran == [True, False, True, False, True, False, True, False]
    assert plan.warnings == ()
    session.close()


def test_a_step_mutating_the_kept_value_warns_and_strict_compilation_rejects_it():
    keeper = (
        "$steps.bump value ($steps.fresh.value) is mutated in place after "
        "$steps.delta keeps ['input value ($steps.fresh.value)']"
    )

    plan = compile_workflow(BUMP_AFTER_DELTA, catalogue=CATALOGUE)

    with pytest.raises(MutationConflictError, match=re.escape(keeper)):
        compile_workflow(BUMP_AFTER_DELTA, catalogue=CATALOGUE, options=STRICT)
    assert len(plan.warnings) == 1 and keeper in plan.warnings[0]


def test_a_warned_mutation_changes_the_kept_value_because_nothing_is_copied():
    # Every run's fresh {"count": 0} equals the last one, but bump turned the
    # kept object into {"count": 1}, so each run counts as a change.
    session = compile_workflow(BUMP_AFTER_DELTA, catalogue=CATALOGUE).create_session()

    counts = [session.run({"image": _image()}).rows() for _ in range(3)]

    assert counts == [[{"count": 1}]] * 3
    session.close()


def test_a_caller_object_changed_in_place_and_passed_again_is_no_change():
    session = compile_workflow(CALLER_VALUE, catalogue=CATALOGUE).create_session()
    value = [1]

    first = _ran(session, value)
    value.append(2)
    reused = _ran(session, value)
    new_object = _ran(session, [1, 2, 3])

    assert [first, reused, new_object] == [True, False, True]
    session.close()
