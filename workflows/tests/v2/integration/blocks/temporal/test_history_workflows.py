"""``roboflow_core/frame_delay@v1`` and ``roboflow_core/image_copy@v1`` in compiled workflows.

Two cameras share one sparse source, shaped like ``video/stream_set@v1``: a
stationary ``camera`` axis whose absent cameras are omitted. Camera 0 has a
gap, a restart and a repeated frame number; camera 1 is absent from most
pulses and repeats its last frame::

    pulse         0  1  2  3  4  5  6  7  8  9
    cam 0 frame   1  2  4  -  5  1  2  3  3  4
    cam 1 frame   1  -  -  4  5  -  6  -  6  -

``data`` is the image itself, whose pixel value ``10 * camera + pulse``
names the pulse that stored it. The same frames run passively (one image per
``session.run`` or pipeline submission) and actively, serially and pipelined
with ``W = 2``, in both block execution modes. Every run must give the answer
of ``expected``, a direct transcription of latest V1.
"""

import gc
import re
import weakref
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pytest
import torch
from roboflow_workflows.execution_engine.entities.base import VideoMetadata
from roboflow_workflows.v2.blocks.control.continue_if_boolean.v1 import ContinueIfBlock
from roboflow_workflows.v2.blocks.image.image_copy.v1 import ImageCopyBlock
from roboflow_workflows.v2.blocks.temporal import FrameDelayBlock
from roboflow_workflows.v2.blocks.temporal.frame_delay.v1 import FrameHistoryCapacityError
from roboflow_workflows.v2.data.batch import (
    Axis,
    Batch,
    EntryLayout,
    EntryMetadata,
    InputValue,
    SampleContext,
    TemporalContext,
    Timestamp,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import BOOLEAN_KIND, INTEGER_KIND
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref
from roboflow_workflows.v2.engine.errors import (
    MutationConflictError,
    ParamsValidationError,
    StepExecutionError,
)
from roboflow_workflows.v2.engine.pipelining.options import PipelineOptions
from roboflow_workflows.v2.engine.plan import CompileOptions
from roboflow_workflows.v2.engine.sources import Emission, Source, SourceOutput

WAIT = 20.0
MODES = ["run", "phases"]
OFFSET = -1
T0 = datetime(2026, 10, 13)
CLOCK = "test:observed"
PULSES = [
    [(0, 1), (1, 1)],
    [(0, 2)],
    [(0, 4)],
    [(1, 4)],
    [(0, 5), (1, 5)],
    [(0, 1)],
    [(0, 2), (1, 6)],
    [(0, 3)],
    [(0, 3), (1, 6)],
    [(0, 4)],
]
# (pulse, camera) -> (reference frame, delayed frame number, delayed pixel)
Row = Tuple[int, Optional[int], Optional[int]]
Rows = Dict[Tuple[int, int], Row]


def expected() -> Rows:
    """V1: restart on a strictly lower number; an equal number replaces."""
    rows, kept = {}, {}
    for pulse, members in enumerate(PULSES):
        for camera, number in members:
            frames = kept.setdefault(camera, {})
            if frames and number < max(frames):
                frames.clear()
            frames[number] = 10 * camera + pulse
            target = number + OFFSET
            delayed = (target, frames[target]) if target in frames else (None, None)
            rows[(pulse, camera)] = (number, *delayed)

    return rows


def test_the_oracle_covers_gap_restart_and_repeat():
    rows = expected()

    assert rows[(2, 0)] == (4, None, None)
    assert rows[(5, 0)] == (1, None, None)
    assert rows[(8, 0)] == (3, 2, 6)
    assert rows[(9, 0)] == (4, 3, 8)
    assert rows[(8, 1)] == (6, 5, 14)


def frame(camera: int, number: int, pulse: int, tensors: list) -> ImageData:
    pixels = torch.full((3, 2, 2), 10 * camera + pulse, dtype=torch.uint8)
    tensors.append(pixels)
    metadata = VideoMetadata(
        video_identifier=f"set/{camera}",
        frame_number=number,
        frame_timestamp=T0 + timedelta(seconds=pulse),
    )
    image = ImageData.from_tensor(pixels, video_metadata=metadata)

    return image


class SparseCameras(Source):
    """Stationary ``camera`` axis; absent cameras are omitted, not compacted."""

    type = "test/sparse_cameras@v1"
    outputs = {
        "image": SourceOutput(
            IMAGE_KIND,
            layout=EntryLayout((Axis(id="camera", kind="sample", stationary=True),)),
        )
    }

    def __init__(self, tensors: list) -> None:
        self._tensors = tensors

    def open(self) -> None:
        self._pulse = 0

    def read(self) -> Optional[Emission]:
        if self._pulse == len(PULSES):
            return None

        pulse, self._pulse = self._pulse, self._pulse + 1
        images, indices, sample, temporal = [], [], {}, {}
        for camera, number in PULSES[pulse]:
            images.append(frame(camera, number, pulse, self._tensors))
            indices.append((camera,))
            sample[(camera,)] = SampleContext(source_id=f"set/{camera}")
            temporal[(camera,)] = TemporalContext(
                observed_coverage=Timestamp(pulse, 1, CLOCK)
            )
        value = Batch.of(images, indices=indices)
        emission = Emission(
            {
                "image": InputValue(
                    value, EntryMetadata(sample=sample, temporal=temporal)
                )
            }
        )

        return emission


class _Label(Block):
    """Test block: ``None`` for frame numbers in ``nulls``, else a label."""

    type = "test/label@v1"
    outputs = {"value": Output(source="image")}

    class Params(BlockParams):
        image: Ref(IMAGE_KIND)
        nulls: List[int] = []

    def run(self, *, image: ImageData, nulls: List[int]) -> Dict[str, Any]:
        number = image.video_metadata.frame_number
        value = None if number in nulls else f"label-{number}"

        return {"value": value}


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
        after: None | Ref(allow_none=True) = None

    def run(self, *, value: Any, after: Any) -> Dict[str, Any]:
        value["count"] += 1

        return {"count": value["count"]}


class _Pass(Block):
    """Test block: ``value`` once per image, with ``value``'s context.

    A workflow parameter has no context, so neither has the output.
    """

    type = "test/pass@v1"
    outputs = {"value": Output(source="value")}

    class Params(BlockParams):
        image: Ref(IMAGE_KIND)
        value: Ref(allow_none=True)

    def run(self, *, image: ImageData, value: Any) -> Dict[str, Any]:
        return {"value": value}


CATALOGUE = Catalogue(
    [FrameDelayBlock, ImageCopyBlock, ContinueIfBlock, _Label, _Fresh, _Bump, _Pass],
    sources=[SparseCameras],
    kinds=[IMAGE_KIND, BOOLEAN_KIND, INTEGER_KIND],
)
IMAGE_INPUT = {"type": "WorkflowImage", "name": "image"}
STEPS = [
    {
        "type": FrameDelayBlock.type,
        "name": "delay",
        "image": "$IMAGE",
        "data": "$IMAGE",
        "offset": OFFSET,
    },
    {"type": ImageCopyBlock.type, "name": "copy", "image": "$IMAGE", "device": "cpu"},
]
FIELDS = {
    "image": "$steps.delay.output",
    "ok": "$steps.delay.is_available",
    "ref": "$steps.delay.reference_frame_number",
    "copy": "$steps.copy.image",
}


def _steps(image: str) -> list:
    steps = [
        {key: image if value == "$IMAGE" else value for key, value in step.items()}
        for step in STEPS
    ]

    return steps


def _outputs(fields: Dict[str, str]) -> List[dict]:
    outputs = [
        {"type": "JsonField", "name": name, "selector": selector}
        for name, selector in fields.items()
    ]

    return outputs


def _row(values: dict) -> Row:
    delayed = values["image"]
    assert values["ok"] is (delayed is not None)
    if delayed is None:
        return values["ref"], None, None

    row = values["ref"], delayed.video_metadata.frame_number
    row += (int(delayed.tensor_image.max()),)

    return row


# ------------------------------------------------------------------ passive
def _passive_plan(mode: str, *, steps: list, fields: Dict[str, str], inputs=()):
    definition = {
        "version": "2.0",
        "inputs": [IMAGE_INPUT, *inputs],
        "steps": steps,
        "outputs": _outputs(fields),
    }
    plan = compile_workflow(
        definition, catalogue=CATALOGUE, options=CompileOptions(block_execution=mode)
    )

    return plan


def _passive_inputs(tensors: list):
    for pulse, members in enumerate(PULSES):
        for camera, number in members:
            yield (pulse, camera), {"image": frame(camera, number, pulse, tensors)}


def _single(value):
    if isinstance(value, Batch):
        members = [member for _, member in value.iter_with_indices()]
        assert len(members) <= 1
        return members[0] if members else None

    return value


def _passive_values(result) -> dict:
    values = {name: _single(value) for name, value in result.outputs.data.items()}

    return values


@pytest.mark.parametrize("mode", MODES)
def test_passive_run_keys_by_video_identifier(mode):
    plan = _passive_plan(mode, steps=_steps("$inputs.image"), fields=FIELDS)
    session = plan.create_session()
    rows = {
        key: _row(_passive_values(session.run(inputs)))
        for key, inputs in _passive_inputs([])
    }
    session.close()

    assert rows == expected()


@pytest.mark.parametrize("mode", MODES)
def test_passive_pipeline_matches_serial(mode):
    plan = _passive_plan(mode, steps=_steps("$inputs.image"), fields=FIELDS)
    session = plan.create_session()
    with session.pipeline(options=PipelineOptions(max_in_flight=2)) as pipeline:
        futures = [
            (key, pipeline.submit(inputs)) for key, inputs in _passive_inputs([])
        ]
    rows = {key: _row(_passive_values(future.result())) for key, future in futures}
    session.close()

    assert rows == expected()


def test_passive_kept_frames_and_copies_release_the_producer_tensors():
    tensors = []
    plan = _passive_plan("run", steps=_steps("$inputs.image"), fields=FIELDS)
    session = plan.create_session()
    results = [session.run(inputs) for _, inputs in _passive_inputs(tensors)]
    produced = [weakref.ref(tensor.untyped_storage()) for tensor in tensors]
    del tensors
    gc.collect()

    assert all(storage() is None for storage in produced)
    copies = [_passive_values(result)["copy"] for result in results]
    assert [c.video_metadata.frame_number for c in copies] == [
        number for members in PULSES for _, number in members
    ]
    session.close()


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("count, limit", [(17, None), (20, None), (20, 20)])
def test_compiled_delay_keeps_interleaved_camera_history_and_resets(mode, count, limit):
    step = {**STEPS[0], "image": "$inputs.image", "data": "$inputs.image"}
    if limit is not None:
        step["max_sources"] = limit
    fields = {name: FIELDS[name] for name in ("image", "ok", "ref")}
    plan = _passive_plan(mode, steps=[step], fields=fields)
    with plan.create_session() as session:
        for number in (1, 2):
            for camera in range(count):
                current = frame(camera, number, number, [])
                row = _passive_values(session.run({"image": current}))
                assert row["ref"] == number and row["ok"] is (number == 2)
                if number == 2:
                    delayed = row["image"]
                    assert delayed.video_metadata.video_identifier == f"set/{camera}"
                    assert delayed.video_metadata.frame_number == 1
                    assert int(delayed.tensor_image.max()) == 10 * camera + 1

        receipt = session.update(plan, reset=True)
        assert receipt.cleanup.wait(5) and receipt.cleanup.state == "done"
        # Previously admitted sources warm again; a whole new set also fits.
        for camera in range(count):
            row = _passive_values(session.run({"image": frame(camera, 3, 3, [])}))
            assert row["image"] is None and row["ok"] is False
        session.update(plan, reset=True)
        for camera in range(count):
            current = frame(camera, 1, 0, [])
            current.video_metadata.video_identifier = f"new/{camera}"
            row = _passive_values(session.run({"image": current}))
            assert row["ok"] is False


@pytest.mark.parametrize("mode", MODES)
def test_compiled_capacity_error_names_the_limit_and_processing_reset_releases_it(mode):
    step = {
        **STEPS[0],
        "image": "$inputs.image",
        "data": "$inputs.image",
        "max_sources": 2,
    }
    plan = _passive_plan(
        mode,
        steps=[step],
        fields={name: FIELDS[name] for name in ("image", "ok", "ref")},
    )
    with plan.create_session() as session:
        for camera in (0, 1):
            session.run({"image": frame(camera, 1, 0, [])})
        with pytest.raises(StepExecutionError) as failed:
            session.run({"image": frame(2, 1, 0, [])})
        cause = failed.value.__cause__
        assert isinstance(cause, FrameHistoryCapacityError)
        assert "max_sources=2" in str(cause) and "set/2" in str(cause)
        assert "reset processing" in str(cause)

        # A processing reset frees all source slots and starts with empty history.
        session.update(plan, reset=True)
        first = _passive_values(session.run({"image": frame(2, 1, 0, [])}))
        second = _passive_values(session.run({"image": frame(2, 2, 1, [])}))
        assert first["ok"] is False and second["ok"] is True
        assert second["image"].video_metadata.frame_number == 1


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "$inputs.limit"])
def test_compilation_rejects_invalid_or_selected_source_capacity(limit):
    step = {
        **STEPS[0],
        "image": "$inputs.image",
        "data": "$inputs.image",
        "max_sources": limit,
    }
    with pytest.raises(ParamsValidationError, match="max_sources"):
        _passive_plan("run", steps=[step], fields={"ok": FIELDS["ok"]})


def _one_camera(numbers: List[int]):
    for pulse, number in enumerate(numbers):
        yield frame(0, number, pulse, [])


@pytest.mark.parametrize("mode", MODES)
def test_a_selected_offset_keeps_the_deepest_history_seen(mode):
    steps = [{**STEPS[0], "image": "$inputs.image", "data": "$inputs.image"}]
    steps[0]["offset"] = "$inputs.offset"
    plan = _passive_plan(
        mode,
        steps=steps,
        fields={"image": "$steps.delay.output"},
        inputs=[{"type": "WorkflowParameter", "name": "offset"}],
    )
    session = plan.create_session()
    offsets = [-3, -1, -1, -1, -3, -1, -3]

    delayed = [
        _passive_values(session.run({"image": image, "offset": offset}))["image"]
        for image, offset in zip(_one_camera([1, 2, 3, 4, 5, 6, 7]), offsets)
    ]
    session.close()

    numbers = [None if d is None else d.video_metadata.frame_number for d in delayed]
    assert numbers == [None, 1, 2, 3, 2, 5, 4]


@pytest.mark.parametrize("mode", MODES)
def test_a_present_none_is_delayed_while_a_gated_value_is_never_kept(mode):
    # frame 1: present None; frame 3: the label step is gated off.
    steps = [
        {
            "type": ContinueIfBlock.type,
            "name": "gate",
            "condition": "$inputs.keep",
            "next_steps": ["$steps.label"],
        },
        {"type": _Label.type, "name": "label", "image": "$inputs.image", "nulls": [1]},
        {
            **STEPS[0],
            "image": "$inputs.image",
            "data": "$steps.label.value",
            "default_value": "none",
        },
    ]
    fields = {"value": "$steps.delay.output", "ok": "$steps.delay.is_available"}
    plan = _passive_plan(
        mode,
        steps=steps,
        fields=fields,
        inputs=[{"type": "WorkflowParameter", "name": "keep"}],
    )
    session = plan.create_session()
    keep = {1: True, 2: True, 3: False, 4: True, 5: True}

    rows = [
        _passive_values(session.run({"image": image, "keep": keep[number]}))
        for number, image in zip(keep, _one_camera(list(keep)))
    ]
    session.close()

    assert [(row.get("value", "-"), row.get("ok", "-")) for row in rows] == [
        ("none", False),  # nothing before frame 1
        (None, True),  # frame 1 kept its present None
        ("-", "-"),  # gated: no call, so no outputs at all
        ("none", False),  # frame 3 was never kept
        ("label-4", True),
    ]


def _sampled_frame(number: int, *, video: str, source: Optional[str]):
    metadata = VideoMetadata(
        video_identifier=video,
        frame_number=number,
        frame_timestamp=T0 + timedelta(seconds=number),
    )
    image = ImageData.from_tensor(
        torch.zeros((3, 2, 2), dtype=torch.uint8), video_metadata=metadata
    )
    if source is None:
        return image

    sample = {(): SampleContext(source_id=source)}
    sampled = InputValue(data=image, metadata=EntryMetadata(sample=sample))

    return sampled


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "video, sources, delayed",
    [
        # Without a video identifier the image's SampleContext keys history.
        ("", ["A", "B", "A", "B"], ["miss", "miss", "A1", "B1"]),
        # A video identifier wins: one history, B's frame 1 replaced A's.
        ("cam", ["A", "B", "A", "B"], ["miss", "miss", "B1", "B1"]),
        # Neither: every frame shares the default source.
        ("", [None] * 4, ["miss", "miss", "B1", "B1"]),
    ],
    ids=["image-source", "video-identifier", "no-source"],
)
def test_context_free_data_is_kept_under_the_image_source(
    mode, video, sources, delayed
):
    # "pass" returns a workflow parameter, so data has no context and the
    # delay call has no common one; only the image names the source.
    steps = [
        {
            "type": _Pass.type,
            "name": "pass",
            "image": "$inputs.image",
            "value": "$inputs.value",
        },
        {
            **STEPS[0],
            "image": "$inputs.image",
            "data": "$steps.pass.value",
            "default_value": "miss",
        },
    ]
    fields = {"value": "$steps.delay.output", "ok": "$steps.delay.is_available"}
    plan = _passive_plan(
        mode,
        steps=steps,
        fields=fields,
        inputs=[{"type": "WorkflowParameter", "name": "value"}],
    )
    session = plan.create_session()
    calls = zip(sources, [1, 1, 2, 2], ["A1", "B1", "A2", "B2"])

    rows = [
        _passive_values(
            session.run(
                {
                    "image": _sampled_frame(number, video=video, source=source),
                    "value": value,
                }
            )
        )
        for source, number, value in calls
    ]
    session.close()

    assert [row["value"] for row in rows] == delayed
    assert [row["ok"] for row in rows] == [value != "miss" for value in delayed]


# ------------------------------------------------------------------- active
def _run_active(
    mode: str,
    pipeline: Optional[PipelineOptions],
    tensors: list,
    *,
    before_close=None,
):
    definition = {
        "version": "2.0",
        "inputs": [],
        "sources": [{"type": SparseCameras.type, "name": "cams"}],
        "steps": _steps("$sources.cams.image"),
        "outputs": [
            {
                "type": "OutputGroup",
                "name": "delayed",
                "anchor": "$sources.cams.image",
                "outputs": _outputs(FIELDS),
            }
        ],
    }
    plan = compile_workflow(
        definition, catalogue=CATALOGUE, options=CompileOptions(block_execution=mode)
    )
    session = plan.create_session({"tensors": tensors})
    results = []
    run = session.start(handlers={"delayed": results.append}, pipeline=pipeline)
    try:
        assert run.wait(WAIT), "active run did not finish"
        if before_close is not None:
            before_close(results)
    finally:
        run.stop()
        session.close()

    return results


def _active_rows(results: list) -> Tuple[Rows, list]:
    rows, contexts = {}, []
    for pulse, result in enumerate(results):
        data, metadata = result.outputs.data, result.outputs.metadata
        for index, delayed in data["image"].iter_with_indices():
            values = {
                name: dict(data[name].iter_with_indices())[index] for name in data
            }
            rows[(pulse, index[0])] = _row(values)
            contexts.append((pulse, index, delayed, metadata["image"]))

    return rows, contexts


PIPELINES = [
    pytest.param(None, id="serial"),
    pytest.param(
        PipelineOptions(max_in_flight=3, max_workflow_parallelism=2), id="P3-W2"
    ),
]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("pipeline", PIPELINES)
def test_active_sparse_cameras_match_v1(mode, pipeline):
    results = _run_active(mode, pipeline, [])

    rows, _ = _active_rows(results)

    assert len(results) == len(PULSES)
    assert rows == expected()


@pytest.mark.parametrize("pipeline", PIPELINES)
def test_active_row_is_current_pulse_while_payload_keeps_its_own_frame(pipeline):
    results = _run_active("run", pipeline, [])

    _, contexts = _active_rows(results)

    delayed_rows = [row for row in contexts if row[2] is not None]
    assert delayed_rows
    for pulse, index, delayed, metadata in delayed_rows:
        assert metadata.temporal_at(index).observed_coverage.ticks == pulse
        assert metadata.sample_at(index).source_id == f"set/{index[0]}"
        own = delayed.video_metadata
        assert own.frame_timestamp < T0 + timedelta(seconds=pulse)
        assert own.frame_number == expected()[(pulse, index[0])][1]


@pytest.mark.parametrize("pipeline", PIPELINES)
def test_active_outputs_own_their_pixels_and_release_the_source_tensors(pipeline):
    # Checked while the session, and so the block's history, is still alive.
    tensors, checked = [], []

    def check(results: list) -> None:
        produced = [weakref.ref(tensor.untyped_storage()) for tensor in tensors]
        pointers = {tensor.data_ptr() for tensor in tensors}
        tensors.clear()
        gc.collect()
        outputs = [
            image
            for result in results
            for name in ("image", "copy")
            for _, image in result.outputs.data[name].iter_with_indices()
            if image is not None
        ]
        assert outputs
        assert all(image.tensor_image.data_ptr() not in pointers for image in outputs)
        assert [storage() is None for storage in produced] == [True] * len(produced)
        checked.append(True)

    _run_active("run", pipeline, tensors, before_close=check)

    assert checked == [True]


# ---------------------------------------------- mutation of retained values
def _mutation_plan(bumped: str, *, options: CompileOptions = CompileOptions()):
    steps = [
        {"type": _Fresh.type, "name": "fresh", "image": "$inputs.image"},
        {
            **STEPS[0],
            "image": "$inputs.image",
            "data": "$steps.fresh.value",
            "offset": "$inputs.offset",
        },
        {
            "type": _Bump.type,
            "name": "bump",
            "value": bumped,
            "after": "$steps.delay.is_available",
        },
    ]
    definition = {
        "version": "2.0",
        "inputs": [IMAGE_INPUT, {"type": "WorkflowParameter", "name": "offset"}],
        "steps": steps,
        "outputs": _outputs({"count": "$steps.bump.count"}),
    }
    plan = compile_workflow(definition, catalogue=CATALOGUE, options=options)

    return plan


@pytest.mark.parametrize(
    "bumped, keeper",
    [
        ("$steps.delay.output", "$steps.delay keeps ['output output'"),
        ("$steps.fresh.value", "$steps.delay keeps ['input data ($steps.fresh.value)'"),
    ],
    ids=["returned-output", "kept-input"],
)
def test_mutating_a_delayed_value_warns_and_strict_compilation_rejects_it(
    bumped, keeper
):
    plan = _mutation_plan(bumped)

    with pytest.raises(MutationConflictError, match=re.escape(keeper)):
        _mutation_plan(bumped, options=CompileOptions(mutation_conflicts="error"))

    assert any(keeper in warning for warning in plan.warnings)


def test_a_warned_mutation_of_the_output_changes_the_kept_value():
    # Frame 1's dict is returned at offset 0 and again at -1, bumped each time.
    plan = _mutation_plan("$steps.delay.output")
    session = plan.create_session()
    calls = zip(_one_camera([1, 2, 3]), [0, -1, -1])

    rows = [
        _passive_values(session.run({"image": image, "offset": offset}))
        for image, offset in calls
    ]
    session.close()

    assert [row["count"] for row in rows] == [1, 2, 1]
