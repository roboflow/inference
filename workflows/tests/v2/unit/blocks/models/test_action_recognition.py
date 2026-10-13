"""Raw-clip contract, one-time public sampling, ownership and backend errors."""

from dataclasses import replace
from datetime import datetime, timezone
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from roboflow_workflows.execution_engine.entities.base import VideoMetadata
from roboflow_workflows.v2.blocks.models import action_recognition as action
from roboflow_workflows.v2.blocks.models.cache import (
    ModelCache,
    ModelRequest,
    ModelRequestConflict,
)
from roboflow_workflows.v2.data.batch import (
    Batch,
    EntryMetadata,
    SampleContext,
    TemporalContext,
    Timestamp,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.engine.phases import PhaseFailure

from inference_models.models.base.action_recognition import (
    ActionRecognitionPrediction,
    VideoSampling,
)


class FakeActionModel:
    """Spy on exact selected tensors, rate and calls; never load weights."""

    def __init__(self, sampling=None, classes=("walk", "run"), raw=None, mutate=False):
        self.video_sampling = sampling or VideoSampling(
            window_seconds=2, sample_fps=2, min_frames=2
        )
        self.class_names = None if classes is None else list(classes)
        self.raw = raw
        self.mutate = mutate
        self.calls = []

    def infer(self, *, frames, class_names, fps):
        self.calls.append(
            {
                "frames": [frame.clone() for frame in frames],
                "fps": fps,
                "class_names": class_names,
            }
        )
        if self.mutate:
            for frame in frames:
                frame.fill_(255)
        if self.raw is not None:
            return self.raw
        label = "walk" if self.class_names is not None else "a person walks"
        return [ActionRecognitionPrediction(0, len(frames) - 1, label)]


def make_clip(
    count=8,
    *,
    fps=4,
    camera="a",
    offset=10,
    size=(4, 6),
    indices=None,
    known=None,
    times=None,
    sources=None,
    video=False,
):
    positions = tuple((i,) for i in range(count)) if indices is None else tuple(indices)
    images = []
    for i in range(count):
        metadata = None
        if video:
            metadata = VideoMetadata(
                video_identifier=camera,
                frame_number=offset + i,
                frame_timestamp=datetime(2026, 10, 13, tzinfo=timezone.utc),
                fps=fps,
            )
        images.append(
            ImageData.from_tensor(
                torch.full((3, *size), i + offset, dtype=torch.uint8),
                image_id=f"{camera}-{i + offset}",
                video_metadata=metadata,
            )
        )
    temporal = {}
    sample = {}
    for i, index in enumerate(positions):
        stamp = Timestamp(i if times is None else times[i], Fraction(1, fps), "media")
        temporal[index] = TemporalContext(observed_coverage=stamp, media_coverage=stamp)
        sample[index] = SampleContext(
            source_id=camera if sources is None else sources[i], source_type="camera"
        )
    clip = Batch(
        images,
        indices=positions,
        known_indices=known,
        metadata=EntryMetadata(sample=sample, temporal=temporal),
    )
    return clip


def execute(clip=None, model=None, **kwargs):
    model = model or FakeActionModel()
    block = action.ActionRecognitionBlock(models={"local": model})
    return block.run(
        frames=make_clip() if clip is None else clip,
        source_fps=kwargs.pop("source_fps", 4),
        model_id="local",
        **kwargs,
    )["predictions"]


# Exact sequences observed in the pinned public planner and original QA probes.
# Keep literal expectations: recomputing them with the planner hides regressions.
QUANTIZED_PLANS = [
    (6, 3, 4, (0, 0, 1, 3, 3, 4)),
    (8, 4, 4, (0, 1, 2, 3, 4, 5, 6, 7)),
    (
        76,
        38,
        38,
        (
            0,
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            19,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            38,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            57,
            57,
            58,
            59,
            60,
            61,
            62,
            63,
            64,
            65,
            66,
            67,
            68,
            69,
            70,
            71,
            72,
            73,
            74,
        ),
    ),
]


@pytest.mark.parametrize("count,fps,sample_fps,expected", QUANTIZED_PLANS)
@pytest.mark.parametrize("mode", ["sliding_window", "whole_video"])
def test_sufficient_raw_clip_preserves_quantized_public_plan(
    monkeypatch, count, fps, sample_fps, expected, mode
):
    sampling = VideoSampling(
        mode=mode, window_seconds=2, sample_fps=sample_fps, min_frames=4
    )
    planner = action.plan_windows
    plans = []

    def spy(**kwargs):
        plans.append(kwargs)
        return planner(**kwargs)

    class Mutator(FakeActionModel):
        def infer(self, *, frames, class_names, fps):
            assert len({frame.data_ptr() for frame in frames}) == len(expected)
            result = super().infer(frames=frames, class_names=class_names, fps=fps)
            frames[0].fill_(255)
            # A duplicate of source frame zero must own independent storage.
            if expected[1] == 0:
                assert int(frames[1][0, 0, 0]) == 10
            return result

    monkeypatch.setattr(action, "plan_windows", spy)
    clip = make_clip(count, fps=fps, video=True)
    model = Mutator(sampling)
    result = execute(clip, model, source_fps=fps)
    assert plans == [{"frame_count": count, "source_fps": fps, "sampling": sampling}]
    assert len(model.calls) == 1
    assert model.calls[0]["fps"] == result["sample_fps"] == fps
    assert result["raw_frame_count"] == count
    assert len(result["sampled_frames"]) == len(expected) <= 76
    assert [int(frame[0, 0, 0]) for frame in model.calls[0]["frames"]] == [
        10 + i for i in expected
    ]
    for position, (sample, raw) in enumerate(zip(result["sampled_frames"], expected)):
        assert sample["sampled_frame_idx"] == position
        assert sample["input_frame_idx"] == raw
        assert sample["input_position"] == [raw]
        assert sample["image_id"] == f"a-{10 + raw}"
        assert sample["source_id"] == "a"
        assert sample["source_frame_number"] == 10 + raw
        assert sample["timestamp"] == {
            "ticks": raw,
            "time_base": [1, fps],
            "clock_id": "media",
        }
    assert [int(image.tensor_image[0, 0, 0]) for image in clip] == list(
        range(10, 10 + count)
    )


@pytest.mark.parametrize("mode", ["sliding_window", "whole_video"])
@pytest.mark.parametrize("count", [2, 3])
def test_true_untrained_raw_insufficiency_rejected_before_planning(
    monkeypatch, mode, count
):
    monkeypatch.setattr(
        action,
        "plan_windows",
        lambda **kwargs: pytest.fail("insufficient clip planned"),
    )
    monkeypatch.setattr(
        action,
        "_model_tensor",
        lambda *args, **kwargs: pytest.fail("insufficient clip copied"),
    )
    model = FakeActionModel(VideoSampling(mode=mode, sample_fps=4, min_frames=4))
    with pytest.raises(
        PhaseFailure, match="insufficient raw content for untrained min_frames"
    ):
        execute(make_clip(count), model)
    assert not model.calls


@pytest.mark.parametrize("count,fps,sample_fps,expected", QUANTIZED_PLANS)
def test_quantized_selection_budget_rejected_before_planning(
    monkeypatch, count, fps, sample_fps, expected
):
    monkeypatch.setattr(
        action, "plan_windows", lambda **kwargs: pytest.fail("over-budget clip planned")
    )
    monkeypatch.setattr(
        action,
        "_model_tensor",
        lambda *args, **kwargs: pytest.fail("over-budget clip copied"),
    )
    model = FakeActionModel(VideoSampling(sample_fps=sample_fps, min_frames=4))
    with pytest.raises(PhaseFailure, match="sample budget"):
        execute(
            make_clip(count, fps=fps),
            model,
            source_fps=fps,
            max_sampled_frames=len(expected) - 1,
        )
    assert not model.calls


@pytest.mark.parametrize(
    "indices,diagnostic",
    [
        ((-1, 1), "invalid frame indices"),
        ((0, 8), "invalid frame indices"),
        ((0, True), "invalid frame indices"),
        ((0, 1.0), "invalid frame indices"),
        ((2, 0), "unordered"),
        ((1, 1), "distinct"),
    ],
)
def test_planner_index_safety_retained_before_materialization(
    monkeypatch, indices, diagnostic
):
    from inference_models.models.base.action_recognition import WindowSpec

    monkeypatch.setattr(
        action, "plan_windows", lambda **kwargs: [WindowSpec(indices, 2)]
    )
    monkeypatch.setattr(
        action,
        "_model_tensor",
        lambda *args, **kwargs: pytest.fail("invalid plan copied"),
    )
    model = FakeActionModel()
    with pytest.raises(PhaseFailure, match=diagnostic):
        execute(model=model)
    assert not model.calls


def test_public_sampling_once_exact_mapping_and_no_input_mutation(monkeypatch):
    model = FakeActionModel(mutate=True)
    clip = make_clip(video=True)
    planner = action.plan_windows
    calls = []

    def spy(**kwargs):
        calls.append(kwargs)
        return planner(**kwargs)

    monkeypatch.setattr(action, "plan_windows", spy)
    result = execute(clip, model)
    assert len(calls) == len(model.calls) == 1
    assert [int(frame[0, 0, 0]) for frame in model.calls[0]["frames"]] == [
        10,
        12,
        14,
        16,
    ]
    assert model.calls[0]["fps"] == 2
    assert [frame["input_frame_idx"] for frame in result["sampled_frames"]] == [
        0,
        2,
        4,
        6,
    ]
    assert [frame["source_frame_number"] for frame in result["sampled_frames"]] == [
        10,
        12,
        14,
        16,
    ]
    assert result["sampled_frames"][-1]["timestamp"] == {
        "ticks": 6,
        "time_base": [1, 4],
        "clock_id": "media",
    }
    assert result["segments"] == [
        {"start_frame_idx": 0, "end_frame_idx": 3, "class_name": "walk", "class_id": 0}
    ]
    assert [int(image.tensor_image[0, 0, 0]) for image in clip] == list(range(10, 18))


@pytest.mark.parametrize(
    "kwargs,diagnostic",
    [
        ({"source_fps": 0}, "source_fps"),
        ({"source_fps": float("nan")}, "source_fps"),
        ({"source_fps": float("inf")}, "source_fps"),
        ({"source_fps": True}, "source_fps"),
        ({"source_fps": 5}, "cadence"),
        ({"max_raw_frames": 7}, "raw clip"),
        ({"max_raw_frames": True}, "positive integer"),
        ({"max_frame_pixels": 23}, "max_frame_pixels"),
        ({"max_clip_bytes": 575}, "raw clip exceeds"),
        ({"max_sampled_frames": 3}, "sample budget"),
        ({"class_filter": []}, "nonempty"),
        ({"class_filter": "walk"}, "nonempty"),
        ({"class_filter": [""]}, "nonempty"),
        ({"class_filter": ["walk", "walk"]}, "duplicates"),
        ({"class_filter": ["unknown"]}, "unknown fixed classes"),
    ],
)
def test_invalid_config_never_materializes_or_infers(monkeypatch, kwargs, diagnostic):
    model = FakeActionModel()
    monkeypatch.setattr(
        action, "_model_tensor", lambda *args, **kw: pytest.fail("invalid input copied")
    )
    with pytest.raises(PhaseFailure, match=diagnostic):
        execute(model=model, **kwargs)
    assert not model.calls


@pytest.mark.parametrize(
    "clip,diagnostic",
    [
        (make_clip(1), "raw clip"),
        (make_clip(0), "raw clip"),
        (make_clip(times=[0, 1, 2, 3, 4, 5, 6, 8]), "cadence"),
        (make_clip(times=list(reversed(range(8)))), "cadence"),
        (make_clip(sources=["a"] * 7 + ["b"]), "identities"),
        (make_clip(indices=[(i,) for i in range(0, 16, 2)]), "contiguous"),
        (
            make_clip(indices=[(i,) for i in range(8)], known=[(i,) for i in range(9)]),
            "holes",
        ),
        (make_clip(indices=[(i,) for i in reversed(range(8))]), "oldest first"),
    ],
)
def test_invalid_clip_never_copies(monkeypatch, clip, diagnostic):
    model = FakeActionModel()
    monkeypatch.setattr(
        action, "_model_tensor", lambda *args, **kw: pytest.fail("invalid input copied")
    )
    with pytest.raises(PhaseFailure, match=diagnostic):
        execute(clip, model)
    assert not model.calls


@pytest.mark.parametrize(
    "change,diagnostic",
    [
        ("no_metadata", "required"),
        ("no_source", "source identity"),
        ("observed_only", "point timestamp"),
        ("different_clock", "cadence"),
        ("mixed_shape", "shape/dtype/device"),
        ("gray", "RGB"),
        ("duplicate_image", "distinct image"),
        ("video_fps", "image fps contradicts"),
        ("video_source", "video identity"),
        ("video_hole", "frame numbers"),
        ("mixed_video", "mixed video"),
        ("mixed_dtype", "uint8"),
        ("mixed_device", "shape/dtype/device"),
    ],
)
def test_metadata_and_homogeneous_tensor_contract(monkeypatch, change, diagnostic):
    clip = make_clip(video=change.startswith("video") or change == "mixed_video")
    images = list(clip.content)
    metadata = clip.metadata
    if change == "no_metadata":
        metadata = None
    elif change == "no_source":
        metadata = EntryMetadata(temporal=metadata.temporal)
    elif change == "observed_only":
        metadata = EntryMetadata(
            sample=metadata.sample,
            temporal={
                i: TemporalContext(context.observed_coverage)
                for i, context in metadata.temporal.items()
            },
        )
    elif change == "different_clock":
        temporal = dict(metadata.temporal)
        stamp = Timestamp(7, Fraction(1, 4), "other")
        temporal[(7,)] = TemporalContext(stamp, media_coverage=stamp)
        metadata = EntryMetadata(sample=metadata.sample, temporal=temporal)
    elif change == "mixed_shape":
        images[-1] = images[-1].resize((3, 4))
    elif change == "gray":
        images[-1] = ImageData.from_tensor(torch.zeros(1, 4, 6, dtype=torch.uint8))
    elif change == "duplicate_image":
        images[-1] = replace(images[-1], image_id=images[0].image_id)
    elif change in ("mixed_dtype", "mixed_device"):
        # Simulate a payload modified after boundary validation.
        pixels = (
            images[-1].tensor_image.float()
            if change == "mixed_dtype"
            else torch.empty((3, 4, 6), dtype=torch.uint8, device="meta")
        )
        object.__setattr__(images[-1], "tensor_image", pixels)
    elif change == "mixed_video":
        images[-1] = replace(images[-1], video_metadata=None)
    else:
        updates = {
            "video_fps": {"fps": 5},
            "video_source": {"video_identifier": "b"},
            "video_hole": {"frame_number": 99},
        }[change]
        images[-1] = replace(
            images[-1],
            video_metadata=images[-1].video_metadata.model_copy(update=updates),
        )
    bad = Batch(images, metadata=metadata)
    model = FakeActionModel()
    monkeypatch.setattr(
        action, "_model_tensor", lambda *args, **kw: pytest.fail("invalid input copied")
    )
    with pytest.raises(PhaseFailure, match=diagnostic):
        execute(bad, model)
    assert not model.calls


def test_capture_clock_is_explicit_alternative_and_observed_jitter_is_not_source_time():
    clip = make_clip()
    metadata = EntryMetadata(
        sample=clip.metadata.sample,
        temporal={
            i: TemporalContext(
                Timestamp(t.media_coverage.ticks * 17, Fraction(1), "wall"),
                capture_coverage=t.media_coverage,
            )
            for i, t in clip.metadata.temporal.items()
        },
    )
    result = execute(clip.with_view(layout=None, metadata=metadata))
    assert result["sampled_frames"][0]["timestamp_field"] == "capture_coverage"


def test_trained_sampling_repeats_and_effective_fps():
    sampling = VideoSampling(window_seconds=2, sample_fps=4, min_frames=4, max_frames=8)
    model = FakeActionModel(sampling)
    result = execute(make_clip(4, fps=2), model, source_fps=2)
    assert [m["input_frame_idx"] for m in result["sampled_frames"]] == [
        0,
        1,
        1,
        2,
        2,
        3,
        3,
        3,
    ]
    assert len(model.calls[0]["frames"]) == 8
    assert result["sample_fps"] == 4


def test_selected_repeat_bytes_bounded_before_allocating(monkeypatch):
    model = FakeActionModel(
        VideoSampling(window_seconds=2, sample_fps=4, min_frames=4, max_frames=8)
    )
    monkeypatch.setattr(
        action, "_model_tensor", lambda *args, **kw: pytest.fail("invalid input copied")
    )
    with pytest.raises(PhaseFailure, match="selected copies"):
        execute(make_clip(4, fps=2), model, source_fps=2, max_clip_bytes=300)
    assert not model.calls


@pytest.mark.parametrize(
    "sampling,count,fps,diagnostic",
    [
        (
            VideoSampling(window_seconds=2, sample_fps=2, min_frames=2),
            9,
            4,
            "one compatible",
        ),
        (
            VideoSampling(mode="whole_video", sample_fps=4, min_frames=2),
            77,
            4,
            "one compatible",
        ),
        (VideoSampling(sample_fps=4, min_frames=4), 2, 4, "insufficient raw"),
        (VideoSampling(sample_fps=0), 8, 4, "sample_fps"),
        (VideoSampling(window_seconds=0), 8, 4, "window_seconds"),
        (VideoSampling(mode="weird"), 8, 4, "unsupported sampling"),
        (VideoSampling(max_frames=1, min_frames=2), 8, 4, "below min_frames"),
        (VideoSampling(max_frame_side=0), 8, 4, "max_frame_side"),
    ],
)
def test_sampling_boundaries_rejected_before_materialization(
    monkeypatch, sampling, count, fps, diagnostic
):
    model = FakeActionModel(sampling)
    monkeypatch.setattr(
        action, "_model_tensor", lambda *args, **kw: pytest.fail("invalid input copied")
    )
    with pytest.raises(PhaseFailure, match=diagnostic):
        execute(make_clip(count, fps=fps), model, source_fps=fps)
    assert not model.calls


def test_whole_video_budgets_and_long_trained_clip():
    trained = FakeActionModel(
        VideoSampling(mode="whole_video", sample_fps=4, min_frames=2, max_frames=4)
    )
    result = execute(make_clip(40), trained)
    assert result["sample_fps"] == 0.4
    assert [m["input_frame_idx"] for m in result["sampled_frames"]] == [0, 10, 20, 30]
    untrained = FakeActionModel(
        VideoSampling(mode="whole_video", sample_fps=4, min_frames=2)
    )
    result = execute(make_clip(76), untrained)
    assert len(result["sampled_frames"]) == 76


@pytest.mark.parametrize(
    "classes,selected,raw,expected",
    [
        (("run", "walk"), ["walk"], [ActionRecognitionPrediction(0, 1, "walk")], 1),
        (None, None, [ActionRecognitionPrediction(0, 1, "free words")], -1),
        (
            None,
            ["free words"],
            [
                ActionRecognitionPrediction(0, 1, "free words"),
                ActionRecognitionPrediction(1, 2, "other"),
            ],
            -1,
        ),
    ],
)
def test_fixed_and_open_vocabulary_ids_and_filter(classes, selected, raw, expected):
    model = FakeActionModel(classes=classes, raw=raw)
    result = execute(model=model, class_filter=selected)
    assert len(result["segments"]) == 1
    assert result["segments"][0]["class_id"] == expected
    assert model.calls[0]["class_names"] == (selected if classes is not None else None)


@pytest.mark.parametrize(
    "raw",
    [
        {},
        [SimpleNamespace(start_frame_idx=-1, end_frame_idx=1, class_name="walk")],
        [SimpleNamespace(start_frame_idx=0, end_frame_idx=4, class_name="walk")],
        [SimpleNamespace(start_frame_idx=2, end_frame_idx=1, class_name="walk")],
        [SimpleNamespace(start_frame_idx=0.5, end_frame_idx=1, class_name="walk")],
        [SimpleNamespace(start_frame_idx=True, end_frame_idx=1, class_name="walk")],
        [SimpleNamespace(start_frame_idx=0, end_frame_idx=1, class_name="unknown")],
        [SimpleNamespace(start_frame_idx=0, end_frame_idx=1, class_name="")],
        [None],
    ],
)
def test_malformed_backend_results_are_not_clamped_or_renamed(raw):
    with pytest.raises(PhaseFailure, match="backend"):
        execute(model=FakeActionModel(raw=raw))


def test_resize_selected_only_current_device_preserves_rgb_and_is_bounded(monkeypatch):
    model = FakeActionModel(
        VideoSampling(window_seconds=2, sample_fps=2, min_frames=2, max_frame_side=3)
    )
    clip = make_clip()
    for image in clip:
        image.tensor_image[0].fill_(11)
        image.tensor_image[1].fill_(22)
        image.tensor_image[2].fill_(33)
    prepared = []
    original = action._model_tensor

    def spy(image, **kwargs):
        prepared.append(image.image_id)
        return original(image, **kwargs)

    monkeypatch.setattr(action, "_model_tensor", spy)
    execute(clip, model)
    assert prepared == ["a-10", "a-12", "a-14", "a-16"]
    for tensor in model.calls[0]["frames"]:
        assert tuple(tensor.shape) == (3, 2, 3)
        assert tensor.device.type == "cpu" and tensor.dtype == torch.uint8
        assert tensor[:, 0, 0].tolist() == [11, 22, 33]


def test_trained_no_side_preserves_detail_and_integer_area_matches_cv2_within_one_byte():
    import cv2

    torch.manual_seed(0)
    clip = make_clip()
    for image in clip:
        image.tensor_image.copy_(
            torch.randint(0, 256, image.tensor_image.shape, dtype=torch.uint8)
        )
    uncapped = FakeActionModel(
        VideoSampling(window_seconds=2, sample_fps=2, min_frames=2, max_frames=4)
    )
    execute(clip, uncapped)
    assert uncapped.calls[0]["frames"][0].shape == clip.content[0].tensor_image.shape
    capped = FakeActionModel(replace(uncapped.video_sampling, max_frame_side=3))
    execute(clip, capped)
    original = clip.content[0].tensor_image.permute(1, 2, 0).numpy()
    expected = cv2.resize(original, (3, 2), interpolation=cv2.INTER_AREA)
    actual = capped.calls[0]["frames"][0].permute(1, 2, 0).numpy()
    assert np.abs(expected.astype(int) - actual.astype(int)).max() <= 1


def test_model_package_cache_reuse_no_cross_clip_or_request_contamination():
    loaded = []
    models = {}

    def load(request):
        loaded.append(request)
        models[request] = FakeActionModel()
        return models[request]

    cache = ModelCache(load)
    block = action.ActionRecognitionBlock(models=cache)
    for package in ("one", "two", "one"):
        result = block.run(
            frames=make_clip(camera=package),
            source_fps=4,
            model_id="local",
            model_package_id=package,
        )["predictions"]
        assert {m["source_id"] for m in result["sampled_frames"]} == {package}
    assert loaded == [ModelRequest("local", "one"), ModelRequest("local", "two")]
    assert [len(models[request].calls) for request in loaded] == [2, 1]
    with pytest.raises(PhaseFailure, match="ModelRequestConflict"):
        action.ActionRecognitionBlock(models={"local": FakeActionModel()}).run(
            frames=make_clip(),
            source_fps=4,
            model_id="local",
            model_package_id="package",
        )


@pytest.mark.parametrize(
    "sampling",
    [
        VideoSampling(min_frames=10**12),
        VideoSampling(
            mode="whole_video", sample_fps=10**12, min_frames=2, max_frames=10**12
        ),
    ],
)
def test_eager_public_planner_cannot_allocate_unbounded_index_tuple(
    monkeypatch, sampling
):
    model = FakeActionModel(sampling)
    monkeypatch.setattr(
        action,
        "plan_windows",
        lambda **kwargs: pytest.fail("unbounded planner allocation"),
    )
    monkeypatch.setattr(
        action,
        "_model_tensor",
        lambda *args, **kwargs: pytest.fail("unbounded tensor allocation"),
    )
    with pytest.raises(PhaseFailure, match="sample budget"):
        execute(model=model)
    assert not model.calls
