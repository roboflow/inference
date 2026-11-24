"""Capture and replay keep the known domain of a Group held in a payload.

A block receives a ``Group`` and returns it inside a dictionary payload. The
gate continues when ``child > 2.5``; each pulse expands to 4 children
``value + k``:

    pulse value   children          Group indices   known_indices
    1.0           1, 2, 3, 4        2, 3            0, 1, 2, 3
    -10.0         -10 .. -7         -               0, 1, 2, 3   (all filtered)

The recording and the replayed result must report the same domain as the
live result, so an all-filtered group never turns into a true-empty group.
"""

from pathlib import Path
from typing import Any, Dict, List

from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.declaration import BlockParams, Group, Output

from tests.v2.integration.engine.recording.test_capture_run import (
    WAIT,
    capture,
    read_chunks,
)
from tests.v2.support.execution.blocks import ContinueIf, Echo, Expand, Recording
from tests.v2.support.recording.capture import Feed, feed_values, field, group


class Hold(Recording):
    """Returns the received group inside a dictionary payload."""

    type = "test/hold_group@v1"
    accepts_empty = True
    outputs = {"held": Output()}

    class Params(BlockParams):
        data: Group()

    def run(self, *, data) -> dict:
        return {"held": {"group": data}}


CATALOGUE = Catalogue([ContinueIf, Echo, Expand, Hold], sources=[Feed])
EXPECTED = [
    ([(2,), (3,)], [(0,), (1,), (2,), (3,)]),
    ([], [(0,), (1,), (2,), (3,)]),
]


def definition() -> Dict[str, Any]:
    return {
        "version": "2.0",
        "inputs": [{"type": "WorkflowParameter", "name": "capture_dir"}],
        "sources": [{"type": "test/recorded_feed@v1", "name": "cam", "feed": "cam"}],
        "steps": [
            {
                "type": "test/expand@v1",
                "name": "expand",
                "value": "$sources.cam.value",
                "count": 4,
            },
            {
                "type": "test/continue_if@v1",
                "name": "keep",
                "value": "$steps.expand.children",
                "threshold": 2.5,
                "next_steps": ["$steps.kept"],
            },
            {"type": "test/echo@v1", "name": "kept", "value": "$steps.expand.children"},
            {"type": "test/hold_group@v1", "name": "hold", "data": "$steps.kept.value"},
        ],
        "outputs": [
            group("frames", "$sources.cam.value", field("held", "$steps.hold.held"))
        ],
        "recording": {
            "type": "file",
            "directory": "$inputs.capture_dir",
            "groups": ["frames"],
        },
        "retrospective": {
            "type": "workflow",
            "input_group": "frames",
            "workflow": {
                "steps": [],
                "outputs": [
                    group(
                        "reviewed",
                        "$sources.frames.held",
                        field("held", "$sources.frames.held"),
                    )
                ],
            },
        },
    }


def domain(result_value: Dict[str, Any]) -> tuple:
    held = result_value["group"]

    return list(held.indices), list(held.known_indices)


def test_capture_and_replay_keep_the_known_domain_of_a_held_group(
    tmp_path: Path,
) -> None:
    plan = compile_workflow(definition(), catalogue=CATALOGUE)
    delivered: List[Any] = []

    capture(
        plan,
        tmp_path / "rec",
        {"cam": feed_values(1.0, -10.0)},
        handlers={"frames": delivered.append},
    )
    chunks = read_chunks(tmp_path / "rec", "frames", catalogue=CATALOGUE)
    replayed: List[Any] = []
    run = plan.retrospective.start(
        recording=tmp_path / "rec", handlers={"reviewed": replayed.append}
    )
    assert run.wait(WAIT)

    assert [domain(result.outputs.data["held"]) for result in delivered] == EXPECTED
    assert [domain(chunk.values["held"]) for chunk in chunks] == EXPECTED
    assert [domain(result.outputs.data["held"]) for result in replayed] == EXPECTED
