"""Real compiler/session Window pipelines with explicit camera clocks."""

from fractions import Fraction

import pytest
from roboflow_workflows.v2.blocks.models import action_recognition as action
from roboflow_workflows.v2.blocks.models.action_recognition import (
    ActionRecognitionBlock,
)
from roboflow_workflows.v2.blocks.models.cache import ModelCache, ModelRequest
from roboflow_workflows.v2.data.batch import (
    Batch,
    EntryMetadata,
    InputValue,
    SampleContext,
    TemporalContext,
    Timestamp,
)
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import BOOLEAN_KIND
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.operators.window import Window
from roboflow_workflows.v2.engine.plan import CompileOptions, PipelineOptions

from inference_models.models.base.action_recognition import VideoSampling
from tests.v2.unit.blocks.models.test_action_recognition import (
    QUANTIZED_PLANS,
    FakeActionModel,
    make_clip,
)

CAMERA = {"id": "camera", "kind": "sample", "stationary": True}
TIME = {"id": "time", "kind": "time"}


def make_plan(*, mode="run", window=True, source_fps=4, window_size=8):
    definition = {
        "version": "2.0",
        "inputs": [
            {
                "name": "frames",
                "kind": ["image"],
                "axes": [CAMERA] if window else [CAMERA, TIME],
            },
            {"name": "clear", "kind": ["boolean"], "axes": [CAMERA]},
            {"name": "model", "kind": ["string"], "axes": [CAMERA]},
            {"name": "package", "kind": ["string"], "axes": [CAMERA]},
        ],
        "steps": [
            {
                "type": ActionRecognitionBlock.type,
                "name": "action",
                "frames": "$operators.history.frames" if window else "$inputs.frames",
                "source_fps": source_fps,
                "model_id": "$operators.history.model" if window else "$inputs.model",
                "model_package_id": (
                    "$operators.history.package" if window else "$inputs.package"
                ),
            }
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "predictions",
                "selector": "$steps.action.predictions",
            }
        ],
    }
    if window:
        definition["operators"] = [
            {
                "type": "v2/window@v1",
                "name": "history",
                "size": window_size,
                "step": 1,
                "align": "position",
                "fill": "wait",
                "order": "oldest_first",
                "partial": "drop",
                "collect": {"frames": "$inputs.frames"},
                "hold": {
                    "clear": "$inputs.clear",
                    "model": "$inputs.model",
                    "package": "$inputs.package",
                },
                "clear_when": "clear",
            }
        ]
    plan = compile_workflow(
        definition,
        catalogue=Catalogue(
            [ActionRecognitionBlock],
            kinds=[IMAGE_KIND, BOOLEAN_KIND],
            operators=[Window],
        ),
        options=CompileOptions(block_execution=mode),
    )
    return plan


def arrival(rows, *, clear=(), package="v1", fps=4):
    """Each (camera position, camera-local frame number) owns fresh pixels."""
    images = []
    indices = []
    sample, temporal = {}, {}
    for camera, frame in rows:
        image = make_clip(
            1, fps=fps, camera=f"cam{camera}", offset=frame + camera * 100
        ).content[0]
        images.append(image)
        index = (camera,)
        indices.append(index)
        sample[index] = SampleContext(f"cam{camera}", "camera")
        stamp = Timestamp(frame, Fraction(1, fps), "media")
        temporal[index] = TemporalContext(stamp, media_coverage=stamp)
    inputs = {
        "frames": InputValue(
            Batch(images, indices=indices),
            metadata=EntryMetadata(sample=sample, temporal=temporal),
        ),
        "clear": Batch([camera in clear for camera, frame in rows], indices=indices),
        "model": Batch(["local"] * len(rows), indices=indices),
        "package": Batch([package] * len(rows), indices=indices),
    }
    return inputs


def prediction_metadata(result):
    (entry,) = result.selections["predictions"].values()
    return result.outputs.metadata[entry]


@pytest.mark.parametrize("mode", ["run", "phases"])
def test_interleaved_cameras_sparse_outer_arrivals_warmup_and_last_context(mode):
    model = FakeActionModel()
    session = make_plan(mode=mode).create_session(
        {"models": {ModelRequest("local", "v1"): model}}
    )
    last = {}
    for frame in range(8):
        for camera in (0, 1):
            result = session.run(arrival([(camera, frame)]))
            if frame < 7:
                assert all(row["predictions"] is None for row in result.rows())
                assert len(model.calls) == 0
            else:
                prediction = result.rows(serialize=True)[camera]["predictions"]
                assert [m["image_id"] for m in prediction["sampled_frames"]] == [
                    f"cam{camera}-{i + camera * 100}" for i in (0, 2, 4, 6)
                ]
                assert {m["source_id"] for m in prediction["sampled_frames"]} == {
                    f"cam{camera}"
                }
                metadata = prediction_metadata(result)
                assert metadata.sample_at((camera,)).source_id == f"cam{camera}"
                assert metadata.temporal_at((camera,)).media_coverage.ticks == 7
                last[camera] = prediction
    assert len(model.calls) == 2
    assert [int(frame[0, 0, 0]) for frame in model.calls[0]["frames"]] == [0, 2, 4, 6]
    assert [int(frame[0, 0, 0]) for frame in model.calls[1]["frames"]] == [
        100,
        102,
        104,
        106,
    ]
    session.close()


@pytest.mark.parametrize("mode", ["run", "phases"])
def test_pipeline_one_call_per_warm_camera_then_overlap_and_exact_cache_reuse(mode):
    loads = []
    model = FakeActionModel()

    def load(request):
        loads.append(request)
        return model

    session = make_plan(mode=mode).create_session({"models": ModelCache(load)})
    with session.pipeline(
        options=PipelineOptions(max_in_flight=2, max_workflow_parallelism=1)
    ) as pipeline:
        futures = [
            pipeline.submit(arrival([(0, frame), (1, frame)])) for frame in range(9)
        ]
        results = [future.result(timeout=10) for future in futures]
    assert all(result.rows() == [{"predictions": None}] * 2 for result in results[:7])
    assert len(model.calls) == 4
    assert loads == [ModelRequest("local", "v1")]
    for result, last_frame in zip(results[7:], (7, 8)):
        for camera in (0, 1):
            prediction = result.rows()[camera]["predictions"]
            assert (
                prediction["sampled_frames"][0]["image_id"]
                == f"cam{camera}-{last_frame - 7 + camera * 100}"
            )
            assert (
                prediction_metadata(result).temporal_at((camera,)).media_coverage.ticks
                == last_frame
            )
    session.close()


@pytest.mark.parametrize("mode", ["run", "phases"])
def test_clear_when_and_processing_and_new_session_resets_history_but_not_host_cache(
    mode,
):
    plan = make_plan(mode=mode)
    model = FakeActionModel()
    loads = []

    def load(request):
        loads.append(request)
        return model

    cache = ModelCache(load)
    session = plan.create_session({"models": cache})
    for frame in range(8):
        session.run(arrival([(0, frame), (1, frame)]))
    assert len(model.calls) == 2
    result = session.run(arrival([(0, 8), (1, 8)], clear=(0,)))
    assert result.rows()[0]["predictions"] is None
    assert result.rows()[1]["predictions"] is not None
    assert len(model.calls) == 3
    for frame in range(9, 16):
        result = session.run(arrival([(0, frame)]))
    assert result.rows()[0]["predictions"]["sampled_frames"][0]["image_id"] == "cam0-8"
    assert len(model.calls) == 4
    receipt = session.update(plan, reset=True)
    assert receipt.cleanup.wait(5)
    for frame in range(8):
        result = session.run(arrival([(0, frame)]))
        assert (result.rows()[0]["predictions"] is None) == (frame < 7)
    assert len(model.calls) == 5 and len(loads) == 1
    session.close()
    session = plan.create_session({"models": cache})
    assert session.run(arrival([(0, 0)])).rows() == [{"predictions": None}]
    assert len(model.calls) == 5 and len(loads) == 1
    session.close()


@pytest.mark.parametrize("mode", ["run", "phases"])
def test_window_does_not_compact_camera_local_missing_time(mode):
    model = FakeActionModel()
    session = make_plan(mode=mode).create_session(
        {"models": {ModelRequest("local", "v1"): model}}
    )
    for frame in (0, 1, 2, 4, 5, 6, 7):
        session.run(arrival([(0, frame)]))
    with pytest.raises(Exception, match="cadence"):
        session.run(arrival([(0, 8)]))
    assert not model.calls
    session.close()


@pytest.mark.parametrize("mode", ["run", "phases"])
def test_explicit_clip_filtered_inner_positions_remain_holes(mode):
    model = FakeActionModel()
    plan = make_plan(mode=mode, window=False)
    session = plan.create_session({"models": {ModelRequest("local", "v1"): model}})
    clip = make_clip()
    # A filtered inner child remains in known_indices; Group must preserve it.
    images = Batch(
        [image for i, image in enumerate(clip) if i != 3],
        indices=[(0, i) for i in range(8) if i != 3],
        known_indices=[(0, i) for i in range(8)],
        parent_index=(0,),
    )
    metadata = EntryMetadata(
        sample={(0,): SampleContext("a", "camera")},
        temporal={(0, i): clip.metadata.temporal_at((i,)) for i in range(8) if i != 3},
    )
    with pytest.raises(Exception, match="holes|contiguous"):
        session.run(
            {
                "frames": InputValue(Batch([images]), metadata=metadata),
                "clear": [False],
                "model": ["local"],
                "package": ["v1"],
            }
        )
    assert not model.calls
    session.close()


@pytest.mark.parametrize("mode", ["run", "phases"])
@pytest.mark.parametrize("sampling_mode", ["sliding_window", "whole_video"])
@pytest.mark.parametrize("count,fps,sample_fps,expected", QUANTIZED_PLANS)
def test_compiled_window_preserves_quantized_plan_and_last_raw_context(
    monkeypatch, mode, sampling_mode, count, fps, sample_fps, expected
):
    sampling = VideoSampling(
        mode=sampling_mode, window_seconds=2, sample_fps=sample_fps, min_frames=4
    )
    model = FakeActionModel(sampling, mutate=True)
    planner = action.plan_windows
    plans = []

    def spy(**kwargs):
        plans.append(kwargs)
        return planner(**kwargs)

    monkeypatch.setattr(action, "plan_windows", spy)
    session = make_plan(mode=mode, source_fps=fps, window_size=count).create_session(
        {"models": {ModelRequest("local", "v1"): model}}
    )
    inputs = []
    try:
        for raw in range(count):
            current = arrival([(1, 10 + raw)], fps=fps)
            inputs.append(current)
            result = session.run(current)
            if raw < count - 1:
                assert all(row["predictions"] is None for row in result.rows())
                assert not plans and not model.calls
        prediction = result.rows(serialize=True)[1]["predictions"]
        assert plans == [
            {"frame_count": count, "source_fps": fps, "sampling": sampling}
        ]
        assert len(model.calls) == 1
        assert model.calls[0]["fps"] == prediction["sample_fps"] == fps
        assert prediction["raw_frame_count"] == count
        assert [int(frame[0, 0, 0]) for frame in model.calls[0]["frames"]] == [
            110 + i for i in expected
        ]
        for sample, raw in zip(prediction["sampled_frames"], expected):
            assert sample["input_frame_idx"] == raw
            assert sample["image_id"] == f"cam1-{110 + raw}"
            assert sample["source_id"] == "cam1"
            assert sample["timestamp"] == {
                "ticks": 10 + raw,
                "time_base": [1, fps],
                "clock_id": "media",
            }
        metadata = prediction_metadata(result)
        assert metadata.sample_at((1,)).source_id == "cam1"
        assert metadata.temporal_at((1,)).media_coverage == Timestamp(
            10 + count - 1, Fraction(1, fps), "media"
        )
        assert [
            int(item["frames"].data.content[0].tensor_image[0, 0, 0]) for item in inputs
        ] == list(range(110, 110 + count))
    finally:
        session.close()
