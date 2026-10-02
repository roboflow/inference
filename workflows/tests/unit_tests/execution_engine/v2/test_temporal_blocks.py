"""Blocks that choose group members, and the first/last/selected context policies.

Workflows run through the real compiler, session and executor. Passive inputs
carry an explicit time axis and per-frame timestamps, so the tests read the
chosen frame's timestamp back from the output metadata::

    frames     [camera, window]   ImageData per (camera, frame), one timestamp each
    reference  [camera]           one ImageData per camera
    best       [camera]           Selected(frames.indices[best]) -> chosen frame + its PTS
    top        [camera, ranked]   Selection(...) -> chosen frames, each with its PTS
"""

from concurrent.futures import Future
from fractions import Fraction
from typing import Any, Dict, List, Optional, Sequence

import pytest
import torch
from roboflow_workflows.execution_engine.v2.blocks.control import ContinueIfBlock
from roboflow_workflows.execution_engine.v2.blocks.image import (
    HasBrightnessBlock,
    InvertBlock,
    MosaicBlock,
    StaticCropBlock,
)
from roboflow_workflows.execution_engine.v2.blocks.image_data import ImageData
from roboflow_workflows.execution_engine.v2.blocks.kinds import IMAGE_KIND
from roboflow_workflows.execution_engine.v2.blocks.temporal import (
    BestFrameBlock,
    TopKBrightestBlock,
)
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.data import (
    Batch,
    EntryMetadata,
    InputValue,
    SampleContext,
    TemporalContext,
    Timestamp,
)
from roboflow_workflows.execution_engine.v2.declaration import (
    Block,
    BlockParams,
    Group,
    Output,
    Ref,
    Selected,
    Selection,
)
from roboflow_workflows.execution_engine.v2.errors import (
    ContractError,
    DeclarationError,
    StepExecutionError,
    WorkflowCompileError,
)
from roboflow_workflows.execution_engine.v2.kinds import INTEGER_KIND
from roboflow_workflows.execution_engine.v2.plan import RunResult

CAMERA_AXIS = {"id": "camera", "kind": "sample", "stationary": True}
WINDOW_AXIS = {"id": "window", "kind": "time"}
FRAMES_INPUT = {"name": "frames", "kind": ["image"], "axes": [CAMERA_AXIS, WINDOW_AXIS]}
REFERENCE_INPUT = {"name": "reference", "kind": ["image"], "axes": [CAMERA_AXIS]}
IMAGES_INPUT = {"name": "images", "kind": ["image"], "axes": [CAMERA_AXIS]}


class _WindowSize(Block):
    """Test block: the number of frames in a window, timed three ways."""

    type = "test/window_size"
    outputs = {
        "first": Output(INTEGER_KIND, source="frames", context_policy="first"),
        "last": Output(INTEGER_KIND, source="frames", context_policy="last"),
        "common": Output(INTEGER_KIND, source="frames"),
    }

    class Params(BlockParams):
        frames: Group(IMAGE_KIND, temporal=True)

    def run(self, *, frames: Batch) -> Dict[str, Any]:
        value = len(frames)
        return {"first": value, "last": value, "common": value}


class _ReturnsAsIs(Block):
    """Test block: returns whatever ``answer`` says, to probe result checks."""

    type = "test/returns_as_is"
    answer: Any = None
    outputs = {
        "chosen": Output(IMAGE_KIND, source="frames", context_policy="selected"),
        "ranked": Output(
            IMAGE_KIND, expand="ranked", source="frames", context_policy="selected"
        ),
        "plain": Output(INTEGER_KIND, source="frames"),
    }

    class Params(BlockParams):
        frames: Group(IMAGE_KIND)

    def run(self, *, frames: Batch) -> Dict[str, Any]:
        return type(self).answer(frames)


def _catalogue() -> Catalogue:
    catalogue = Catalogue(
        [
            BestFrameBlock,
            TopKBrightestBlock,
            StaticCropBlock,
            MosaicBlock,
            InvertBlock,
            HasBrightnessBlock,
            ContinueIfBlock,
            _WindowSize,
            _ReturnsAsIs,
        ],
        kinds=[IMAGE_KIND],
    )

    return catalogue


def _image(value: int, *, name: str, size: int = 8) -> ImageData:
    pixels = torch.full((3, size, size), value, dtype=torch.uint8)
    image = ImageData.from_tensor(pixels, image_id=name)

    return image


def _at(milliseconds: int) -> TemporalContext:
    stamp = Timestamp(milliseconds, Fraction(1, 1000), "media")
    context = TemporalContext(observed_coverage=stamp, media_coverage=stamp)

    return context


def _camera(name: str) -> SampleContext:
    return SampleContext(source_id=name, source_type="camera")


def _timed_frames(
    values: Sequence[Sequence[int]],
    *,
    extra: Optional[Dict[Any, Any]] = None,
) -> InputValue:
    """Frames ``values[n][t]``, frame t of camera n timed at 1000*n + 100*t ms."""
    frames = [
        [_image(value, name=f"cam{n}-f{t}") for t, value in enumerate(row)]
        for n, row in enumerate(values)
    ]
    temporal = {
        (n, t): _at(1000 * n + 100 * t)
        for n, row in enumerate(values)
        for t in range(len(row))
    }
    temporal.update(extra or {})
    sample = {(n,): _camera(f"cam{n}") for n in range(len(values))}
    metadata = EntryMetadata(sample=sample, temporal=temporal)
    timed = InputValue(data=frames, metadata=metadata)

    return timed


def _references(values: Sequence[int]) -> InputValue:
    references = [_image(value, name=f"ref{n}") for n, value in enumerate(values)]
    sample = {(n,): _camera(f"cam{n}") for n in range(len(values))}
    timed = InputValue(data=references, metadata=EntryMetadata(sample=sample))

    return timed


def _output(name: str, selector: str) -> Dict[str, str]:
    return {"type": "JsonField", "name": name, "selector": selector}


def _run(definition: Dict[str, Any], inputs: Dict[str, Any]) -> RunResult:
    plan = compile_workflow({"version": "2.0", **definition}, catalogue=_catalogue())
    result = plan.create_session().run(inputs)

    return result


def _entry(result: RunResult, name: str) -> str:
    (entry,) = result.selections[name].values()

    return entry


def _data(result: RunResult, name: str) -> Any:
    return result.outputs.data[_entry(result, name)]


def _metadata(result: RunResult, name: str) -> EntryMetadata:
    return result.outputs.metadata[_entry(result, name)]


def _axis_kinds(result: RunResult, name: str) -> List[str]:
    layout = result.outputs.layout[_entry(result, name)]

    return [axis.kind for axis in layout.axes]


def _ms(context: Optional[TemporalContext]) -> Optional[int]:
    if context is None:
        return None

    return context.observed_coverage.ticks


BEST_FRAME_WORKFLOW = {
    "inputs": [FRAMES_INPUT, REFERENCE_INPUT],
    "steps": [
        {
            "type": "v2/best_frame",
            "name": "best",
            "frames": "$inputs.frames",
            "reference": "$inputs.reference",
        }
    ],
    "outputs": [
        _output("frame", "$steps.best.frame"),
        _output("difference", "$steps.best.difference"),
    ],
}


def test_best_frame_chooses_a_different_time_position_per_camera() -> None:
    # given
    frames = _timed_frames([[10, 50, 90], [200, 120, 30, 125]])
    references = _references([52, 118])

    # when
    result = _run(BEST_FRAME_WORKFLOW, {"frames": frames, "reference": references})

    # then
    chosen = _data(result, "frame")
    assert [image.image_id for image in chosen] == ["cam0-f1", "cam1-f1"]
    assert chosen[0] is frames.data[0][1]
    assert chosen[1] is frames.data[1][1]
    assert _axis_kinds(result, "frame") == ["sample"]
    metadata = _metadata(result, "frame")
    assert [_ms(metadata.temporal_at((n,))) for n in (0, 1)] == [100, 1100]
    assert [metadata.sample_at((n,)).source_id for n in (0, 1)] == ["cam0", "cam1"]


def test_transformed_selection_keeps_the_chosen_frame_context() -> None:
    # given
    frames = _timed_frames([[10, 50, 90], [200, 120, 30]])
    references = _references([88, 28])

    # when
    result = _run(BEST_FRAME_WORKFLOW, {"frames": frames, "reference": references})

    # then
    assert list(_data(result, "difference")) == [2.0, 2.0]
    metadata = _metadata(result, "difference")
    assert [_ms(metadata.temporal_at((n,))) for n in (0, 1)] == [200, 1200]
    assert metadata.sample_at((1,)).source_id == "cam1"


def test_explicit_none_on_the_chosen_frame_is_not_replaced_by_a_root_timestamp() -> (
    None
):
    # given: a root timestamp for the whole input, cleared on frame (0, 1)
    frames = _timed_frames([[10, 50, 90]], extra={(): _at(9999), (0, 1): None})
    references = _references([50])

    # when
    result = _run(BEST_FRAME_WORKFLOW, {"frames": frames, "reference": references})

    # then
    assert _data(result, "frame")[0] is frames.data[0][1]
    assert _metadata(result, "frame").temporal_at((0,)) is None


def test_best_frame_of_an_empty_window_chooses_nothing_and_has_no_timestamp() -> None:
    # given: camera 1 has an empty window
    frames = _timed_frames([[10, 50], []], extra={(): _at(9999)})
    references = _references([50, 50])

    # when
    result = _run(BEST_FRAME_WORKFLOW, {"frames": frames, "reference": references})

    # then
    assert list(_data(result, "frame"))[1] is None
    assert list(_data(result, "difference"))[1] is None
    assert _ms(_metadata(result, "frame").temporal_at((0,))) == 100
    assert _metadata(result, "frame").temporal_at((1,)) is None


def test_selection_uses_full_logical_indices_of_filtered_ragged_groups() -> None:
    # given: dark frames are filtered before choosing, so camera 0 delivers
    # only (0, 2) and (0, 3); a positional index would name the wrong frame
    definition = {
        "inputs": [FRAMES_INPUT, REFERENCE_INPUT],
        "steps": [
            {
                "type": "v2/has_brightness",
                "name": "bright",
                "image": "$inputs.frames",
                "minimum": 20.0,
            },
            {
                "type": "v2/continue_if",
                "name": "gate",
                "condition": "$steps.bright.keep",
                "next_steps": ["$steps.invert"],
            },
            {"type": "v2/invert", "name": "invert", "image": "$inputs.frames"},
            {
                "type": "v2/best_frame",
                "name": "best",
                "frames": "$steps.invert.image",
                "reference": "$inputs.reference",
            },
        ],
        "outputs": [_output("frame", "$steps.best.frame")],
    }
    frames = _timed_frames([[0, 5, 235, 205], [215]])
    references = _references([50, 40])

    # when
    result = _run(definition, {"frames": frames, "reference": references})

    # then: inverted 235 -> 20 and 205 -> 50; the closest is frame (0, 3)
    chosen = _data(result, "frame")
    assert [image.image_id for image in chosen] == ["cam0-f3", "cam1-f0"]
    metadata = _metadata(result, "frame")
    assert [_ms(metadata.temporal_at((n,))) for n in (0, 1)] == [300, 1000]


TOP_K_WORKFLOW = {
    "inputs": [FRAMES_INPUT, {"type": "WorkflowParameter", "name": "k"}],
    "steps": [
        {
            "type": "v2/top_k_brightest",
            "name": "top",
            "images": "$inputs.frames",
            "k": "$inputs.k",
        }
    ],
    "outputs": [
        _output("images", "$steps.top.images"),
        _output("brightness", "$steps.top.brightness"),
    ],
}


def test_top_k_returns_chosen_frames_brightest_first_with_their_own_timestamps() -> (
    None
):
    # given
    frames = _timed_frames([[10, 90, 50], [70]])

    # when
    result = _run(TOP_K_WORKFLOW, {"frames": frames, "k": 2})

    # then
    images = _data(result, "images")
    assert [[image.image_id for image in group] for group in images] == [
        ["cam0-f1", "cam0-f2"],
        ["cam1-f0"],
    ]
    assert images[0][0] is frames.data[0][1]
    assert [group.indices for group in images] == [((0, 0), (0, 1)), ((1, 0),)]
    assert _axis_kinds(result, "images") == ["sample", "dynamic_nesting"]
    assert [list(group) for group in _data(result, "brightness")] == [
        [90.0, 50.0],
        [70.0],
    ]
    metadata = _metadata(result, "images")
    assert [_ms(metadata.temporal_at(index)) for index in [(0, 0), (0, 1)]] == [
        100,
        200,
    ]
    assert metadata.temporal_at((0,)) is None
    assert metadata.sample_at((1, 0)).source_id == "cam1"


def test_a_single_selected_frame_is_still_a_collection() -> None:
    # given
    frames = _timed_frames([[10, 90, 50]])

    # when
    result = _run(TOP_K_WORKFLOW, {"frames": frames, "k": 1})

    # then
    (group,) = _data(result, "images")
    assert isinstance(group, Batch)
    assert group.indices == ((0, 0),)
    assert group[0] is frames.data[0][1]
    assert _axis_kinds(result, "images") == ["sample", "dynamic_nesting"]


def test_first_and_last_policies_time_a_collapsed_window_by_its_ends() -> None:
    # given
    definition = {
        "inputs": [FRAMES_INPUT],
        "steps": [
            {"type": "test/window_size", "name": "ends", "frames": "$inputs.frames"},
            {"type": "v2/mosaic", "name": "mosaic", "images": "$inputs.frames"},
        ],
        "outputs": [
            _output("first", "$steps.ends.first"),
            _output("last", "$steps.ends.last"),
            _output("common", "$steps.ends.common"),
            _output("mosaic", "$steps.mosaic.image"),
        ],
    }
    frames = _timed_frames([[10, 20, 30], [40, 50]], extra={(): _at(9999)})

    # when
    result = _run(definition, {"frames": frames})

    # then
    def times(name: str) -> List[Optional[int]]:
        metadata = _metadata(result, name)
        return [_ms(metadata.temporal_at((n,))) for n in (0, 1)]

    assert times("first") == [0, 1000]
    assert times("last") == [200, 1100]
    assert times("common") == [None, None]
    assert times("mosaic") == [200, 1100]
    assert _metadata(result, "last").sample_at((1,)).source_id == "cam1"
    assert _axis_kinds(result, "mosaic") == ["sample"]


def test_an_empty_window_does_not_inherit_an_unrelated_root_timestamp() -> None:
    # given: camera 0's window is empty; camera 1's frames share one time;
    # a root timestamp 9999 describes none of them
    definition = {
        "inputs": [FRAMES_INPUT],
        "steps": [
            {"type": "test/window_size", "name": "ends", "frames": "$inputs.frames"},
            {"type": "v2/mosaic", "name": "mosaic", "images": "$inputs.frames"},
        ],
        "outputs": [
            _output("common", "$steps.ends.common"),
            _output("first", "$steps.ends.first"),
            _output("mosaic", "$steps.mosaic.image"),
        ],
    }
    shared = _at(5000)
    frames = _timed_frames(
        [[], [40, 50]], extra={(): _at(9999), (1, 0): shared, (1, 1): shared}
    )

    # when
    result = _run(definition, {"frames": frames})

    # then: no timestamp for the empty window; its camera is still known
    for name in ("common", "first", "mosaic"):
        metadata = _metadata(result, name)
        assert metadata.temporal_at((0,)) is None, name
        assert metadata.sample_at((0,)).source_id == "cam0", name
    assert _ms(_metadata(result, "common").temporal_at((1,))) == 5000
    assert list(_data(result, "common")) == [0, 2]


def test_static_crop_keeps_every_region_position_and_its_geometry() -> None:
    # given
    definition = {
        "inputs": [IMAGES_INPUT],
        "steps": [
            {
                "type": "v2/static_crop",
                "name": "crop",
                "image": "$inputs.images",
                "regions": [[0, 0, 4, 4], [100, 100, 120, 120], [6, 2, 12, 12]],
                "outside": "none",
            },
            {"type": "v2/invert", "name": "invert", "image": "$steps.crop.crops"},
        ],
        "outputs": [
            _output("crops", "$steps.crop.crops"),
            _output("inverted", "$steps.invert.image"),
        ],
    }
    images = [_image(10, name="a", size=8), _image(20, name="b", size=8)]

    # when
    result = _run(definition, {"images": images})

    # then
    crops = _data(result, "crops")
    assert [group.indices for group in crops] == [
        ((0, 0), (0, 1), (0, 2)),
        ((1, 0), (1, 1), (1, 2)),
    ]
    assert crops[0][1] is None
    assert _axis_kinds(result, "crops") == ["sample", "static_nesting"]
    clipped = crops[1][2]
    assert clipped.size_hw == (6, 2)
    assert clipped.prediction_metadata()["root_parent_coordinates"] == [6, 2]
    assert clipped.root.frame_id == "b"
    inverted = _data(result, "inverted")
    assert [group.indices for group in inverted] == [((0, 0), (0, 2)), ((1, 0), (1, 2))]


def test_static_crop_rejects_a_region_outside_the_image_by_default() -> None:
    # given
    definition = {
        "inputs": [IMAGES_INPUT],
        "steps": [
            {
                "type": "v2/static_crop",
                "name": "crop",
                "image": "$inputs.images",
                "regions": [[0, 0, 4, 4], [100, 100, 120, 120]],
            }
        ],
        "outputs": [_output("crops", "$steps.crop.crops")],
    }

    # when / then
    with pytest.raises(StepExecutionError, match=r"rectangle 1 \[100, 100, 120, 120\]"):
        _run(definition, {"images": [_image(10, name="a")]})


def test_static_crop_regions_cannot_be_selected_at_run_time() -> None:
    # given
    definition = {
        "inputs": [IMAGES_INPUT, {"type": "WorkflowParameter", "name": "regions"}],
        "steps": [
            {
                "type": "v2/static_crop",
                "name": "crop",
                "image": "$inputs.images",
                "regions": "$inputs.regions",
            }
        ],
        "outputs": [_output("crops", "$steps.crop.crops")],
    }

    # when / then
    with pytest.raises(Exception, match="regions"):
        compile_workflow({"version": "2.0", **definition}, catalogue=_catalogue())


def test_crops_after_a_window_collapse_trailing_first_then_time() -> None:
    # given: [camera, window] -> crops [camera, window, regions]
    # -> mosaic per frame [camera, window] -> best frame per camera [camera]
    definition = {
        "inputs": [FRAMES_INPUT, REFERENCE_INPUT],
        "steps": [
            {
                "type": "v2/static_crop",
                "name": "crop",
                "image": "$inputs.frames",
                "regions": [[0, 0, 4, 4], [4, 4, 8, 8]],
            },
            {"type": "v2/mosaic", "name": "tiles", "images": "$steps.crop.crops"},
            {
                "type": "v2/best_frame",
                "name": "best",
                "frames": "$steps.tiles.image",
                "reference": "$inputs.reference",
            },
        ],
        "outputs": [
            _output("crops", "$steps.crop.crops"),
            _output("tiles", "$steps.tiles.image"),
            _output("best", "$steps.best.frame"),
        ],
    }
    frames = _timed_frames([[10, 60, 90], [30, 200]])
    references = _references([64, 190])

    # when
    result = _run(definition, {"frames": frames, "reference": references})

    # then
    assert _axis_kinds(result, "crops") == ["sample", "time", "static_nesting"]
    assert _axis_kinds(result, "tiles") == ["sample", "time"]
    assert _axis_kinds(result, "best") == ["sample"]
    tiles_metadata = _metadata(result, "tiles")
    assert _ms(tiles_metadata.temporal_at((1, 1))) == 1100
    best = _metadata(result, "best")
    assert [_ms(best.temporal_at((n,))) for n in (0, 1)] == [100, 1100]


def test_temporal_group_rejects_time_that_is_not_the_last_axis() -> None:
    # given: crops follow the window, so time is not the trailing axis
    definition = {
        "inputs": [FRAMES_INPUT, REFERENCE_INPUT],
        "steps": [
            {
                "type": "v2/static_crop",
                "name": "crop",
                "image": "$inputs.frames",
                "regions": [[0, 0, 4, 4]],
            },
            {
                "type": "v2/best_frame",
                "name": "best",
                "frames": "$steps.crop.crops",
                "reference": "$inputs.reference",
            },
        ],
        "outputs": [_output("best", "$steps.best.frame")],
    }

    # when / then
    with pytest.raises(
        WorkflowCompileError, match=r"last axis .*regions.* is static_nesting"
    ):
        compile_workflow({"version": "2.0", **definition}, catalogue=_catalogue())


def test_temporal_group_rejects_a_group_without_time() -> None:
    # given: cameras [camera] with no window; the group would collapse camera
    definition = {
        "inputs": [IMAGES_INPUT, {"name": "reference", "kind": ["image"]}],
        "steps": [
            {
                "type": "v2/best_frame",
                "name": "best",
                "frames": "$inputs.images",
                "reference": "$inputs.reference",
            }
        ],
        "outputs": [_output("best", "$steps.best.frame")],
    }

    # when / then
    with pytest.raises(WorkflowCompileError, match=r"last axis .*camera.* is sample"):
        compile_workflow({"version": "2.0", **definition}, catalogue=_catalogue())


PROBE_FRAMES = [[10, 20, 30]]


def _probe(answer: Any) -> RunResult:
    """Run ``test/returns_as_is``; outputs ``answer`` omits choose nothing."""
    _ReturnsAsIs.answer = staticmethod(
        lambda frames: {
            "chosen": None,
            "ranked": Selection([]),
            "plain": 0,
            **answer(frames),
        }
    )
    definition = {
        "inputs": [FRAMES_INPUT],
        "steps": [
            {"type": "test/returns_as_is", "name": "probe", "frames": "$inputs.frames"}
        ],
        "outputs": [
            _output("chosen", "$steps.probe.chosen"),
            _output("ranked", "$steps.probe.ranked"),
        ],
    }
    result = _run(definition, {"frames": _timed_frames(PROBE_FRAMES)})

    return result


def _done(value: Any) -> Future:
    future: Future = Future()
    future.set_result(value)

    return future


def _broken() -> Future:
    future: Future = Future()
    future.set_exception(ValueError("model server went away"))

    return future


def test_future_values_of_selected_and_selection_are_resolved() -> None:
    # given: transformed values arrive as futures, like other block results
    replacement = _image(99, name="replacement")
    delivered = {}

    def answer(frames: Batch) -> Dict[str, Any]:
        delivered["frames"] = frames
        return {
            "chosen": Selected(frames.indices[2], value=_done(replacement)),
            "ranked": Selection(
                [frames.indices[1], frames.indices[0]],
                values=[_done(replacement), frames[0]],
            ),
        }

    # when
    result = _probe(answer)

    # then: resolved payloads, each with its chosen frame's context
    frames = delivered["frames"]
    assert _data(result, "chosen")[0] is replacement
    assert _ms(_metadata(result, "chosen").temporal_at((0,))) == 200
    (ranked,) = _data(result, "ranked")
    assert ranked[0] is replacement
    assert ranked[1] is frames[0]
    ranked_metadata = _metadata(result, "ranked")
    assert [_ms(ranked_metadata.temporal_at(index)) for index in ranked.indices] == [
        100,
        0,
    ]


def test_unchanged_chosen_payloads_keep_their_identity_next_to_futures() -> None:
    # given
    def answer(frames: Batch) -> Dict[str, Any]:
        return {
            "chosen": Selected(frames.indices[1]),
            "ranked": Selection([frames.indices[0]]),
            "plain": _done(3),
        }

    # when
    result = _probe(answer)

    # then
    assert [image.image_id for image in _data(result, "chosen")] == ["cam0-f1"]
    assert _data(result, "ranked")[0][0].image_id == "cam0-f0"


@pytest.mark.parametrize(
    "answer",
    [
        lambda frames: {"chosen": Selected(frames.indices[0], value=_broken())},
        lambda frames: {"ranked": Selection([frames.indices[0]], values=[_broken()])},
    ],
    ids=["Selected", "Selection"],
)
def test_a_failed_future_inside_a_choice_names_the_step_and_its_error(
    answer: Any,
) -> None:
    # when / then
    with pytest.raises(StepExecutionError) as caught:
        _probe(answer)

    assert "a future returned by the block failed with ValueError" in str(caught.value)
    assert caught.value.step_path == ("probe",)
    assert isinstance(caught.value.__cause__, ValueError)


@pytest.mark.parametrize(
    ("answer", "message"),
    [
        (
            lambda frames: {"chosen": frames[0], "plain": 0},
            r"output 'chosen' declares context_policy='selected'; return Selected",
        ),
        (
            lambda frames: {"chosen": Selected((0, 9)), "plain": 0},
            r"selected \[0, 9\], which 'frames' did not deliver here",
        ),
        (
            lambda frames: {
                "chosen": Selected(frames.indices[0]),
                "plain": Selected((0, 0)),
            },
            r"output 'plain' returned Selected, but declares "
            r"context_policy='common_or_none'",
        ),
        (
            lambda frames: {"chosen": Selected(1), "plain": 0},
            r"full logical index from Batch.indices",
        ),
    ],
)
def test_wrong_selection_results_fail_with_the_output_and_expected_form(
    answer: Any, message: str
) -> None:
    # when / then
    with pytest.raises(StepExecutionError, match=message):
        _probe(answer)


def test_selection_rejects_repeated_members() -> None:
    # when / then
    with pytest.raises(ContractError, match=r"repeats member\(s\) \[\[0, 1\]\]"):
        Selection([(0, 1), (0, 1)])


@pytest.mark.parametrize(
    ("output", "message"),
    [
        (lambda: Output(context_policy="last"), "name that Group field with source"),
        (
            lambda: Output(preserve="frames", context_policy="first"),
            "keeps every member's context",
        ),
        (
            lambda: Output(
                expand="k", stationary=True, source="frames", context_policy="selected"
            ),
            "cannot be stationary",
        ),
    ],
)
def test_member_policy_declarations_are_checked(output: Any, message: str) -> None:
    # when / then
    with pytest.raises(DeclarationError, match=message):
        output()


def test_member_policy_needs_a_whole_group_source() -> None:
    # when / then
    with pytest.raises(DeclarationError, match="must be a whole Group"):

        class _ItemSource(Block):
            type = "test/item_source"
            outputs = {"x": Output(source="value", context_policy="selected")}

            class Params(BlockParams):
                value: Ref(IMAGE_KIND)

            def run(self, *, value: Any) -> Dict[str, Any]:
                return {"x": value}
