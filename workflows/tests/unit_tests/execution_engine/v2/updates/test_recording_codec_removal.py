"""A graph update must not remove the decoder of an existing recording."""

from pathlib import Path

import pytest
from roboflow_workflows.execution_engine.v2 import compile_workflow
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.declaration import (
    Block,
    BlockParams,
    Output,
    Ref,
)
from roboflow_workflows.execution_engine.v2.errors import IncompatibleUpdateError
from roboflow_workflows.execution_engine.v2.kinds import FLOAT_KIND, Kind

from tests.unit_tests.execution_engine.v2.recording.test_capture_run import (
    Feed,
    feed_values,
    field,
    group,
)
from tests.unit_tests.execution_engine.v2.recording.test_codecs import (
    POINT_CODEC,
    Point,
)

POINT_KIND = Kind(
    "test/recorded_point", validate=lambda value: isinstance(value, Point)
)


class Pointify(Block):
    """Produce a custom payload that requires a registered codec."""

    type = "test/recorded_point@v1"
    outputs = {"point": Output(POINT_KIND)}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)

    def run(self, *, value):
        result = {"point": Point(value, value + 1)}

        return result


def _plan(*, codecs):
    definition = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowParameter", "name": "capture_dir"}],
        "sources": [{"type": Feed.type, "name": "cam", "feed": "cam"}],
        "steps": [
            {"type": Pointify.type, "name": "pointify", "value": "$sources.cam.value"}
        ],
        "outputs": [
            group(
                "frames", "$sources.cam.value", field("point", "$steps.pointify.point")
            )
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
                "version": "2.0",
                "inputs": [],
                "steps": [],
                "outputs": [
                    group(
                        "reviewed",
                        "$sources.frames.point",
                        field("point", "$sources.frames.point"),
                    )
                ],
            },
        },
    }
    plan = compile_workflow(
        definition, catalogue=Catalogue([Pointify], sources=[Feed], codecs=codecs)
    )

    return plan


def test_removing_a_codec_used_by_a_recording_is_refused(tmp_path: Path) -> None:
    old = _plan(codecs=[POINT_CODEC])
    new = _plan(codecs=[])
    directory = tmp_path / "capture"
    with old.create_session({"feeds": {"cam": feed_values(1.0)}}) as session:
        capture = session.start({"capture_dir": str(directory)}, handlers={})
        assert capture.wait(10)
        assert capture.state == "finished"
        replay = old.retrospective.start(recording=directory, handlers={})
        assert replay.wait(10)
        assert replay.state == "finished"

        assert session.assess_update(new).kind == "unsupported"
        for reset in (False, True):
            with pytest.raises(IncompatibleUpdateError):
                session.prepare_update(new, reset=reset)
