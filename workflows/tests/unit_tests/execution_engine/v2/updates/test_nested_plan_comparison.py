"""Updates of plans whose handlers and retrospective are nested plans.

A host may compile every plan against a new but equivalent catalogue. A
nested plan then compares like the root plan: by content, its catalogue by
providers. An identical handler is preserved and an identical retrospective
is supported, even though each compile generates a new replay source. A
changed handler block, provider, kind or parameter still needs a reset; a
changed retrospective, recorded group or recording codec stays unsupported.
A new unrelated block with its own kind and codec needs neither.
"""

import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.declaration import (
    Block,
    BlockParams,
    Output,
    Ref,
)
from roboflow_workflows.execution_engine.v2.errors import (
    GraphUpdateError,
    IncompatibleUpdateError,
)
from roboflow_workflows.execution_engine.v2.events import Event
from roboflow_workflows.execution_engine.v2.kinds import FLOAT_KIND, STRING_KIND, Kind
from roboflow_workflows.execution_engine.v2.recording.codecs import (
    PayloadCodec,
    type_name_of,
)
from roboflow_workflows.execution_engine.v2.recording.store import inspect_recording

from tests.unit_tests.execution_engine.v2.recording.test_capture_run import (
    Feed,
    feed_values,
    field,
    group,
)
from tests.unit_tests.execution_engine.v2.recording.test_codecs import (
    POINT_CODEC,
    encode_point,
)
from tests.unit_tests.execution_engine.v2.updates.blocks import WAIT

LABELS: List[str] = []
"""Every label a handler produced, in call order."""
SCALED: List[float] = []
"""Every value a retrospective produced, in call order."""


class Seen(Block):
    """Passes its value on and emits ``seen`` with it."""

    type = "test/nested_seen@v1"
    outputs = {"value": Output(FLOAT_KIND)}
    events = {"seen": Event({"value": FLOAT_KIND})}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)

    def run(self, *, value) -> dict:
        self.emit("seen", value=value)
        return {"value": value}


class Label(Block):
    """Handler step: labels a value with the catalogue-provided ``label``."""

    type = "test/nested_label@v1"
    outputs = {"text": Output(STRING_KIND)}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)
        offset: float = 0.0

    def __init__(self, *, label: str) -> None:
        self.label = label

    def run(self, *, value, offset) -> dict:
        text = f"{self.label}:{value + offset}"
        LABELS.append(text)
        return {"text": text}


class OtherLabel(Label):
    """Another class registered under the handler step's type."""

    type = Label.type


class Scale(Block):
    """Retrospective step: multiplies a recorded value."""

    type = "test/nested_scale@v1"
    outputs = {"value": Output(FLOAT_KIND)}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)
        factor: float = 1.0

    def run(self, *, value, factor) -> dict:
        SCALED.append(value * factor)
        return {"value": value * factor}


class OtherScale(Scale):
    """Another class registered under the retrospective step's type."""

    type = Scale.type


NOTE_KIND = Kind("test/note")
STRICT_NOTE_KIND = Kind("test/note", validate=lambda value: isinstance(value, str))
"""The same kind name with another validator."""
OTHER_POINT_CODEC = PayloadCodec(
    POINT_CODEC.name,
    POINT_CODEC.type_name,
    encode_point,
    lambda node, source, decoder: None,
)
"""The same codec name and type with another decoder."""


class Extra:
    """A payload type only the unrelated block produces."""


EXTRA_KIND = Kind("test/extra")
EXTRA_CODEC = PayloadCodec(
    "test/extra@1",
    type_name_of(Extra),
    lambda value, sink, encoder: {},
    lambda node, source, decoder: Extra(),
)


class Unrelated(Block):
    """A block no definition here uses, with a kind and a codec of its own."""

    type = "test/nested_unrelated@v1"
    outputs = {"extra": Output(EXTRA_KIND)}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)

    def run(self, *, value) -> dict:
        return {"extra": Extra()}


def catalogue(
    *,
    label_block=Label,
    scale_block=Scale,
    label="a",
    kind=NOTE_KIND,
    codec=POINT_CODEC,
    unrelated: bool = False,
) -> Catalogue:
    """A new catalogue on every call, as a host loading it per compile."""
    blocks = [Seen, label_block, scale_block, *([Unrelated] if unrelated else [])]
    codecs = [codec, *([EXTRA_CODEC] if unrelated else [])]
    built = Catalogue(
        blocks,
        sources=[Feed],
        kinds=[kind],
        providers={"label": label},
        codecs=codecs,
    )

    return built


def definition(
    *,
    offset: float = 0.0,
    factor: float = 10.0,
    recorded_fields: int = 1,
    retrospective: bool = True,
) -> Dict[str, Any]:
    """Feed ``cam`` -> ``seen`` (event) -> handler; record ``frames``; replay it."""
    fields = [field("value", "$steps.seen.value")]
    if recorded_fields > 1:
        fields.append(field("raw", "$sources.cam.value"))
    built = {
        "version": "2.0",
        "inputs": [
            {"type": "WorkflowParameter", "name": "capture_dir"},
            {
                "type": "WorkflowParameter",
                "name": "note",
                "kind": ["test/note"],
                "default_value": "",
            },
        ],
        "sources": [{"type": "test/recorded_feed@v1", "name": "cam", "feed": "cam"}],
        "steps": [
            {
                "type": "test/nested_seen@v1",
                "name": "seen",
                "value": "$sources.cam.value",
            }
        ],
        "handlers": [
            {
                "name": "on_seen",
                "on": "$steps.seen.events.seen",
                "execution": {"mode": "sync"},
                "bindings": {"value": "$event.value"},
                "workflow": {
                    "version": "2.0",
                    "inputs": [{"name": "value", "kind": ["float"], "axes": []}],
                    "steps": [
                        {
                            "type": "test/nested_label@v1",
                            "name": "label",
                            "value": "$inputs.value",
                            "offset": offset,
                        }
                    ],
                    "outputs": [
                        {
                            "type": "JsonField",
                            "name": "text",
                            "selector": "$steps.label.text",
                        }
                    ],
                },
            }
        ],
        "outputs": [group("frames", "$sources.cam.value", *fields)],
        "recording": {
            "type": "file",
            "directory": "$inputs.capture_dir",
            "groups": ["frames"],
        },
        "retrospective": {
            "type": "workflow",
            "input_group": "frames",
            "workflow": {
                "steps": [
                    {
                        "type": "test/nested_scale@v1",
                        "name": "scale",
                        "value": "$sources.frames.value",
                        "factor": factor,
                    }
                ],
                "outputs": [
                    group(
                        "reviewed",
                        "$sources.frames.value",
                        field("scaled", "$steps.scale.value"),
                    )
                ],
            },
        },
    }
    if not retrospective:
        del built["retrospective"]

    return built


def compile_plan(built: Optional[Dict[str, Any]] = None, **catalogue_options: Any):
    plan = compile_workflow(
        built or definition(), catalogue=catalogue(**catalogue_options)
    )

    return plan


def reasons(assessment) -> List[tuple]:
    found = sorted(
        (change.component, change.reason) for change in assessment.preserve_blocked_by
    )

    return found


def replay(plan, directory: Path) -> List[float]:
    SCALED.clear()
    run = plan.retrospective.start(recording=directory, handlers={})
    assert run.wait(WAIT)

    return list(SCALED)


@pytest.fixture(autouse=True)
def _clear_logs():
    LABELS.clear()
    SCALED.clear()
    yield


def test_an_identical_plan_against_a_new_catalogue_is_a_preserving_update() -> None:
    old, new = compile_plan(), compile_plan()
    session = old.create_session({"feeds": {"cam": []}})
    instance = session.instances[("seen",)]

    assessment = session.assess_update(new)
    receipt = session.apply_update(session.prepare_update(new))

    assert old.catalogue is not new.catalogue
    assert old.retrospective.plan.sources["frames"].spec.source_class is not (
        new.retrospective.plan.sources["frames"].spec.source_class
    )
    assert (assessment.kind, reasons(assessment)) == ("preserve", [])
    assert not receipt.reset
    assert session.instances[("seen",)] is instance
    session.close()


def test_an_unrelated_block_with_its_own_kind_and_codec_needs_no_reset() -> None:
    session = compile_plan().create_session({"feeds": {"cam": []}})

    assessment = session.assess_update(compile_plan(unrelated=True))
    session.close()

    assert (assessment.kind, reasons(assessment)) == ("preserve", [])


def test_stopped_recording_updates_idle_then_records_and_replays_again(
    tmp_path: Path,
) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    plan, update = compile_plan(), compile_plan()
    gate, two_delivered = threading.Event(), threading.Event()
    session = plan.create_session(
        {"feeds": {"cam": [*feed_values(1.0, 2.0), gate, *feed_values(3.0)]}}
    )
    delivered: List[int] = []

    def on_frames(result) -> None:
        delivered.append(result.pulse.sequence)
        if len(delivered) == 2:
            two_delivered.set()

    run = session.start({"capture_dir": str(first)}, handlers={"frames": on_frames})
    assert two_delivered.wait(WAIT)
    live = session.prepare_update(update)
    with pytest.raises(GraphUpdateError):
        run.apply_update(live, timeout=WAIT)
    live.discard()
    run.stop()
    gate.set()
    assert run.wait(WAIT)
    stopped = inspect_recording(first).status

    idle = session.assess_update(update)
    receipt = session.apply_update(session.prepare_update(update))
    restarted = session.start({"capture_dir": str(second)}, handlers={})
    assert restarted.wait(WAIT)
    session.close()

    assert (run.state, stopped) == ("finished", "stopped")
    assert (idle.kind, receipt.reset) == ("preserve", False)
    assert (restarted.state, inspect_recording(second).status) == (
        "finished",
        "complete",
    )
    assert LABELS == ["a:1.0", "a:2.0", "a:1.0", "a:2.0", "a:3.0"]
    assert replay(plan, first) == [10.0, 20.0]
    assert replay(update, second) == [10.0, 20.0, 30.0]


@pytest.mark.parametrize(
    "retrospective, changed, expected, label",
    [
        (
            True,
            {"label_block": OtherLabel},
            [("reactions", "changed")],
            "a:1.0",
        ),
        (True, {"offset": 0.5}, [("reactions", "changed")], "a:1.5"),
        (
            False,
            {"label": "b"},
            [("catalogue", "changed"), ("reactions", "changed")],
            "b:1.0",
        ),
        (
            False,
            {"kind": STRICT_NOTE_KIND},
            [("catalogue", "changed"), ("reactions", "changed")],
            "a:1.0",
        ),
    ],
    ids=["handler-block-class", "handler-params", "handler-provider", "kind"],
)
def test_a_changed_handler_still_needs_a_reset(
    retrospective, changed, expected, label, tmp_path: Path
) -> None:
    offset = changed.pop("offset", 0.0)
    session = compile_plan(definition(retrospective=retrospective)).create_session(
        {"feeds": {"cam": feed_values(1.0)}}
    )
    update = compile_plan(
        definition(offset=offset, retrospective=retrospective), **changed
    )

    assessment = session.assess_update(update)
    with pytest.raises(IncompatibleUpdateError):
        session.prepare_update(update)
    receipt = session.apply_update(session.prepare_update(update, reset=True))
    run = session.start({"capture_dir": str(tmp_path / "rec")}, handlers={})
    assert run.wait(WAIT)
    session.close()

    assert (assessment.kind, reasons(assessment)) == ("reset", expected)
    assert receipt.reset
    assert LABELS == [label]


@pytest.mark.parametrize(
    "changed", [{"label": "b"}, {"kind": STRICT_NOTE_KIND}], ids=["provider", "kind"]
)
def test_a_changed_catalogue_of_a_plan_with_a_retrospective_stays_unsupported(
    changed,
) -> None:
    # The replay resolves resources and kinds from the same catalogue.
    session = compile_plan().create_session({"feeds": {"cam": []}})

    assessment = session.assess_update(compile_plan(**changed))
    session.close()

    assert (assessment.kind, reasons(assessment)) == (
        "unsupported",
        [
            ("catalogue", "changed"),
            ("reactions", "changed"),
            ("retrospective", "changed"),
        ],
    )


@pytest.mark.parametrize(
    "changed, expected",
    [
        (
            lambda: compile_plan(definition(factor=2.0)),
            [("retrospective", "changed")],
        ),
        (
            lambda: compile_plan(scale_block=OtherScale),
            [("retrospective", "changed")],
        ),
        (
            lambda: compile_plan(definition(recorded_fields=2)),
            [("recording", "changed"), ("retrospective", "changed")],
        ),
        (
            lambda: compile_plan(codec=OTHER_POINT_CODEC),
            [("recording", "changed"), ("retrospective", "changed")],
        ),
    ],
    ids=[
        "retrospective-params",
        "retrospective-block-class",
        "recorded-group",
        "recording-codec",
    ],
)
def test_a_changed_retrospective_or_recorded_group_stays_unsupported(
    changed, expected
) -> None:
    session = compile_plan().create_session({"feeds": {"cam": []}})
    update = changed()

    assessment = session.assess_update(update)
    for reset in (False, True):
        with pytest.raises(IncompatibleUpdateError):
            session.prepare_update(update, reset=reset)
    session.close()

    assert (assessment.kind, reasons(assessment)) == ("unsupported", expected)
