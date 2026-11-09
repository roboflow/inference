"""Offline (whole-clip) tracking of ``SAM3TrackerVideo`` against a stand-in HF model.

The stand-in keeps the parts of ``Sam3TrackerVideoModel`` the session logic
depends on: a model call on a frame computes every object in the session;
points become a conditioning frame only on a frame the object has not been
tracked on; propagation recomputes every frame that is not a conditioning
frame, ignoring points stored there. Masks are a box around the first include
point of the frame that produced them, so a test can tell which points a mask
came from.
"""

import gc
import time
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from pycocotools import mask as mask_utils

from inference_models.errors import (
    EnvironmentConfigurationError,
    ModelInputError,
    ModelRuntimeError,
)
from inference_models.models.sam3_tracker_video import sam3_tracker_video_hf
from inference_models.models.sam3_tracker_video.clip_cache import (
    VideoClipCache,
    cache_frame_limit,
    cached_frame_bytes,
)
from inference_models.models.sam3_tracker_video.sam3_tracker_video_hf import (
    EMBED_CHUNK_FRAMES,
    SAM3TrackerVideo,
)

W, H, N_FRAMES = 64, 48, 12

# Shapes of a small model: 56 px input, feature maps of 16, 8 and 4 px.
_CONFIG = SimpleNamespace(
    image_size=56,
    vision_config=SimpleNamespace(
        backbone_feature_sizes=[[16, 16], [8, 8], [4, 4]], fpn_hidden_size=256
    ),
    mask_decoder_config=SimpleNamespace(hidden_size=256),
)


def _box(x: float, y: float) -> torch.Tensor:
    mask = torch.full((1, 1, H, W), -1.0)
    mask[0, 0, max(0, int(y) - 4) : int(y) + 4, max(0, int(x) - 4) : int(x) + 4] = 1.0
    return mask


class _FakeSession:
    def __init__(self, clip):
        self.cache = clip
        self.num_frames = clip.n_frames
        self._obj_id_to_idx = {}
        self.obj_ids = []
        self.point_inputs = {}
        self.cond = {}
        self.non_cond = {}
        self.frames_tracked = {}
        self.obj_with_new_inputs = []

    def obj_id_to_idx(self, obj_id):
        if obj_id not in self._obj_id_to_idx:
            idx = self._obj_id_to_idx[obj_id] = len(self._obj_id_to_idx)
            self.obj_ids = list(self._obj_id_to_idx)
            self.point_inputs[idx], self.cond[idx], self.non_cond[idx] = {}, {}, {}
            self.frames_tracked[idx] = set()
        return self._obj_id_to_idx[obj_id]


class _FakeProcessor:
    def video_processor(self, videos, device, return_tensors):
        pixel_values = torch.stack(
            [torch.from_numpy(v).permute(2, 0, 1).float() for v in videos]
        )
        return SimpleNamespace(
            pixel_values_videos=[pixel_values],
            original_sizes=[torch.tensor(videos[0].shape[:2])],
        )

    def add_inputs_to_inference_session(
        self, inference_session, frame_idx, obj_ids, input_points, input_labels
    ):
        idx = inference_session.obj_id_to_idx(obj_ids)
        points = [
            p for p, label in zip(input_points[0][0], input_labels[0][0]) if label == 1
        ]
        inference_session.point_inputs[idx][frame_idx] = points
        inference_session.obj_with_new_inputs.append(obj_ids)

    def post_process_masks(self, masks, original_sizes, binarize):
        return [masks[0] > 0]


class _FakeModel:
    config = _CONFIG

    def __init__(self):
        self.encoded = []

    def __call__(self, inference_session, frame_idx, reverse=False):
        s = inference_session
        out = []
        for obj_id in s.obj_ids:
            idx = s._obj_id_to_idx[obj_id]
            if obj_id not in s.obj_with_new_inputs and frame_idx in s.cond[idx]:
                out.append(s.cond[idx][frame_idx])
                continue
            points, init = None, False
            if obj_id in s.obj_with_new_inputs:
                init = frame_idx not in s.frames_tracked[idx]
                points = s.point_inputs[idx].get(frame_idx)
                if points is not None:
                    s.obj_with_new_inputs.remove(obj_id)
            if points is not None:
                mask = _box(*points[0])
            else:  # tracked: follows the nearest conditioning frame
                mask = s.cond[idx][min(s.cond[idx], key=lambda f: abs(f - frame_idx))]
            (s.cond if init else s.non_cond)[idx][frame_idx] = mask
            if not init:
                s.frames_tracked[idx].add(frame_idx)
            out.append(mask)
        return SimpleNamespace(frame_idx=frame_idx, pred_masks=torch.cat(out))

    def propagate_in_video_iterator(self, inference_session, start_frame_idx, reverse):
        last = inference_session.num_frames
        order = (
            range(start_frame_idx, -1, -1) if reverse else range(start_frame_idx, last)
        )
        for frame_idx in order:
            yield self(inference_session, frame_idx=frame_idx, reverse=reverse)

    def _prepare_vision_features(self, inference_session, frame_idx, batch_size):
        self.encoded.append(frame_idx)
        inference_session.cache._feats[frame_idx] = []


@pytest.fixture
def tracker(monkeypatch):
    tracker = SAM3TrackerVideo(
        model=_FakeModel(),
        processor=_FakeProcessor(),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    monkeypatch.setattr(tracker, "_clip_session", lambda clip: _FakeSession(clip))
    frames = [np.full((H, W, 3), i, dtype=np.uint8) for i in range(N_FRAMES)]
    tracker.embed_video(frames, video_id="v")
    return tracker


def _prompt(frame_idx, obj_id, x, y, label=1):
    return {"frame_idx": frame_idx, "obj_id": obj_id, "x": x, "y": y, "label": label}


def _center(rle):
    mask = mask_utils.decode({"size": rle["size"], "counts": rle["counts"].encode()})
    ys, xs = np.nonzero(mask)
    return float(xs.mean()), float(ys.mean())


def test_embed_video_describes_the_clip_and_can_encode_every_frame(monkeypatch):
    model = _FakeModel()
    tracker = SAM3TrackerVideo(
        model=model, processor=_FakeProcessor(), device=torch.device("cpu")
    )
    monkeypatch.setattr(tracker, "_clip_session", lambda clip: _FakeSession(clip))
    frames = [np.zeros((H, W, 3), dtype=np.uint8)] * 5

    described = tracker.embed_video(frames, video_id="v", encode_frames=True)

    assert described == {"video_id": "v", "n_frames": 5, "height": H, "width": W}
    for _ in range(200):  # encoding runs in the background
        if len(model.encoded) == 5:
            break
        time.sleep(0.01)
    assert model.encoded == [0, 1, 2, 3, 4]


def test_same_frame_clicks_on_two_objects_survive_propagation(tracker):
    prompts = [
        _prompt(0, 1, 10, 10),
        _prompt(0, 2, 50, 10),
        _prompt(5, 1, 10, 30),
        _prompt(5, 2, 50, 30),
    ]

    results = tracker.propagate_video("v", prompts=prompts)

    assert [r["frame_idx"] for r in results] == list(range(N_FRAMES))
    assert all(set(r["masks"]) == {"1", "2"} for r in results)
    assert _center(results[5]["masks"]["2"]) == pytest.approx((50, 30), abs=1)
    assert _center(results[5]["masks"]["1"]) == pytest.approx((10, 30), abs=1)


def test_each_object_is_tracked_from_its_own_first_click(tracker):
    prompts = [_prompt(0, 1, 10, 10), _prompt(8, 2, 50, 30)]

    results = tracker.propagate_video("v", prompts=prompts)

    # Object 2 appears only on frame 8, but backward propagation covers 0-7.
    assert all("2" in r["masks"] for r in results)
    assert _center(results[0]["masks"]["2"]) == pytest.approx((50, 30), abs=1)
    assert _center(results[0]["masks"]["1"]) == pytest.approx((10, 10), abs=1)


def test_propagate_returns_only_the_asked_frames(tracker):
    results = tracker.propagate_video(
        "v", prompts=[_prompt(3, 1, 10, 10)], frame_indices=[0, 6, 11]
    )

    assert [r["frame_idx"] for r in results] == [0, 6, 11]


def test_segment_returns_the_objects_clicked_on_the_frame(tracker):
    prompts = [_prompt(0, 1, 10, 10), _prompt(4, 2, 50, 30), _prompt(4, 1, 20, 20)]

    result = tracker.segment_video_frame("v", prompts=prompts, frame_idx=4)

    assert result["frame_idx"] == 4
    assert set(result["masks"]) == {"1", "2"}
    assert _center(result["masks"]["1"]) == pytest.approx((20, 20), abs=1)
    assert tracker.segment_video_frame("v", prompts=prompts, frame_idx=0)[
        "masks"
    ].keys() == {"1"}


def test_exclude_points_alone_are_refused(tracker):
    prompts = [_prompt(0, 1, 10, 10), _prompt(2, 1, 20, 20, label=0)]

    with pytest.raises(ModelInputError, match=r"\(2, 1\)"):
        tracker.propagate_video("v", prompts=prompts)


def test_unknown_or_evicted_clips_are_refused(tracker):
    with pytest.raises(ModelInputError, match="No video embedded"):
        tracker.propagate_video("other", prompts=[_prompt(0, 1, 10, 10)])


def _clip_of(n_frames):
    return SimpleNamespace(n_frames=n_frames)


def _held(cache, *video_ids):
    held = []
    for video_id in video_ids:
        try:
            cache.get(video_id)
        except ModelInputError:
            continue
        held.append(video_id)
    return held


def test_clip_cache_evicts_the_least_recently_used():
    cache = VideoClipCache(max_frames=20)
    clip_a = _clip_of(10)
    cache.put("a", clip_a)
    cache.put("b", _clip_of(10))
    cache.get("a")
    cache.put("c", _clip_of(10))

    assert cache.get("a") is clip_a
    with pytest.raises(ModelInputError):
        cache.get("b")


def test_clip_cache_keeps_short_clips_until_their_frames_exceed_the_limit():
    cache = VideoClipCache(max_frames=100)
    for name in "abcde":
        cache.put(name, _clip_of(20))

    assert _held(cache, *"abcde") == list("abcde")

    cache.put("f", _clip_of(20))

    assert _held(cache, *"abcdef") == list("bcdef")


def test_a_long_clip_evicts_as_many_old_clips_as_it_needs_oldest_first():
    cache = VideoClipCache(max_frames=100)
    for name in "abcd":
        cache.put(name, _clip_of(20))

    cache.put("long", _clip_of(55))

    assert _held(cache, "a", "b", "c", "d", "long") == ["c", "d", "long"]


def test_re_embedding_an_id_does_not_count_its_frames_twice():
    cache = VideoClipCache(max_frames=100)
    cache.put("a", _clip_of(40))
    cache.put("b", _clip_of(40))
    replacement = _clip_of(60)

    cache.put("b", replacement)

    assert cache.get("a").n_frames == 40
    assert cache.get("b") is replacement


def test_make_room_forgets_the_replaced_and_evicted_clips_before_the_put():
    cache = VideoClipCache(max_frames=100)
    cache.put("a", _clip_of(40))
    cache.put("b", _clip_of(40))
    cache.put("c", _clip_of(20))

    cache.make_room("c", n_frames=50)

    assert _held(cache, "a", "b", "c") == ["b"]


def test_a_clip_longer_than_the_cache_is_refused_and_evicts_nothing():
    cache = VideoClipCache(max_frames=100)
    cache.put("a", _clip_of(40))

    with pytest.raises(
        ModelInputError, match="101 frames.*100.*SAM3_VIDEO_CACHE_MAX_GB"
    ):
        cache.make_room("b", n_frames=101)
    with pytest.raises(ModelInputError, match="SAM3_VIDEO_CACHE_MAX_GB"):
        cache.put("b", _clip_of(101))

    assert _held(cache, "a") == ["a"]


def test_the_cache_limit_in_gb_becomes_a_number_of_frames():
    assert cache_frame_limit(16.0, frame_bytes=10_616_832) == 1507
    assert cache_frame_limit(0.0107, frame_bytes=10_616_832) == 1
    with pytest.raises(EnvironmentConfigurationError, match="at least 0.0106"):
        cache_frame_limit(0.0106, frame_bytes=10_616_832)
    with pytest.raises(EnvironmentConfigurationError):
        VideoClipCache(max_frames=0)


def test_a_cache_limit_that_holds_no_frame_fails_the_model_load(monkeypatch):
    monkeypatch.setattr(sam3_tracker_video_hf, "SAM3_VIDEO_CACHE_MAX_GB", 1e-9)

    with pytest.raises(EnvironmentConfigurationError, match="SAM3_VIDEO_CACHE_MAX_GB"):
        SAM3TrackerVideo(
            model=_FakeModel(), processor=_FakeProcessor(), device=torch.device("cpu")
        )


def test_the_model_reports_its_cache_limits(monkeypatch):
    monkeypatch.setattr(sam3_tracker_video_hf, "SAM3_VIDEO_CACHE_MAX_GB", 0.001)
    tracker = SAM3TrackerVideo(
        model=_FakeModel(),
        processor=_FakeProcessor(),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    # 16*16*32 + 8*8*64 + 4*4*256 feature values against 3*56*56 pixel values.
    frame_bytes = max(8192 + 4096 + 4096, 3 * 56 * 56) * 4

    assert tracker.video_cache_limits() == {
        "max_gb": 0.001,
        "bytes_per_frame": frame_bytes,
        "max_frames": 1_000_000 // frame_bytes,
    }


def test_embed_refuses_a_clip_longer_than_the_cache_and_keeps_the_others(monkeypatch):
    frame_bytes = cached_frame_bytes(_CONFIG, torch.float32)
    monkeypatch.setattr(
        sam3_tracker_video_hf, "SAM3_VIDEO_CACHE_MAX_GB", 10 * frame_bytes / 1e9
    )
    tracker = SAM3TrackerVideo(
        model=_FakeModel(),
        processor=_FakeProcessor(),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    monkeypatch.setattr(tracker, "_clip_session", lambda clip: _FakeSession(clip))
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    tracker.embed_video([frame] * 6, video_id="a")

    with pytest.raises(ModelInputError, match="11 frames.*10.*SAM3_VIDEO_CACHE_MAX_GB"):
        tracker.embed_video([frame] * 11, video_id="b")

    tracker.embed_video([frame] * 4, video_id="c")
    assert tracker.propagate_video("a", prompts=[_prompt(0, 1, 10, 10)])

    # "a" was used last, so the next clip evicts "c".
    tracker.embed_video([frame] * 4, video_id="d")
    with pytest.raises(ModelInputError, match="No video embedded"):
        tracker.propagate_video("c", prompts=[_prompt(0, 1, 10, 10)])
    assert tracker.propagate_video("a", prompts=[_prompt(0, 1, 10, 10)])


def test_a_call_running_on_a_clip_finishes_after_the_clip_is_evicted(tracker):
    clip = tracker._video_clip_cache.get("v")
    frames = [np.zeros((H, W, 3), dtype=np.uint8)] * 2
    evicted_during_the_call = []

    def _propagate(inference_session, start_frame_idx, reverse):
        if not evicted_during_the_call:
            tracker._video_clip_cache.make_room(
                "other", n_frames=tracker._video_clip_cache.max_frames
            )
            evicted_during_the_call.append(
                not tracker._video_clip_cache.holds("v", clip)
            )
        yield from _FakeModel.propagate_in_video_iterator(
            tracker._model, inference_session, start_frame_idx, reverse
        )

    tracker._model.propagate_in_video_iterator = _propagate

    results = tracker.propagate_video("v", prompts=[_prompt(0, 1, 10, 10)])

    assert evicted_during_the_call == [True]
    assert len(results) == N_FRAMES
    tracker.embed_video(frames, video_id="other")


class _CountingProcessor(_FakeProcessor):
    """Counts the full-precision frames alive each time a chunk is asked for."""

    def __init__(self):
        self.alive = 0
        self.alive_at_each_call = []
        self.chunk_sizes = []

    def _freed(self, n_frames):
        self.alive -= n_frames

    def video_processor(self, videos, device, return_tensors):
        gc.collect()
        self.alive_at_each_call.append(self.alive)
        self.chunk_sizes.append(len(videos))
        processed = super().video_processor(videos, device, return_tensors)
        pixel_values = processed.pixel_values_videos[0]
        # The storage dies with the last tensor or view on it.
        self.alive += len(videos)
        weakref.finalize(pixel_values.untyped_storage(), self._freed, len(videos))
        return processed


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_embed_keeps_at_most_one_chunk_of_full_precision_frames_alive(dtype):
    processor = _CountingProcessor()
    tracker = SAM3TrackerVideo(
        model=_FakeModel(), processor=processor, device=torch.device("cpu"), dtype=dtype
    )
    n_frames = 3 * EMBED_CHUNK_FRAMES + 1
    frames = [np.full((H, W, 3), i, dtype=np.uint8) for i in range(n_frames)]

    tracker.embed_video(frames, video_id="v")
    gc.collect()

    assert processor.chunk_sizes == [EMBED_CHUNK_FRAMES] * 3 + [1]
    assert processor.alive_at_each_call == [0, 0, 0, 0]
    assert processor.alive == 0
    clip = tracker._video_clip_cache.get("v")
    assert clip.n_frames == n_frames
    for i in (0, EMBED_CHUNK_FRAMES, n_frames - 1):
        assert clip.frames[i].dtype == dtype
        assert torch.equal(clip.frames[i], torch.full((3, H, W), float(i), dtype=dtype))


def test_each_frame_of_a_clip_is_its_own_allocation(tracker):
    clip = tracker._video_clip_cache.get("v")

    pointers = {clip.frames[i].untyped_storage().data_ptr() for i in range(N_FRAMES)}

    assert len(pointers) == N_FRAMES
    assert all(
        clip.frames[i].untyped_storage().nbytes() == clip.frames[i].nbytes
        for i in range(N_FRAMES)
    )


def _features():
    return {
        "vision_feats": [torch.zeros(4, 1, 2)],
        "vision_pos_embeds": [torch.ones(4, 1, 2)],
    }


def test_a_frames_pixels_are_released_when_its_features_are_cached(tracker):
    clip = tracker._video_clip_cache.get("v")
    pixels = weakref.ref(clip.frames[3])
    assert clip.frames.holds(3) and not clip.has_features(3)

    clip.cache_vision_features(3, _features())

    assert clip.has_features(3) and not clip.frames.holds(3)
    assert pixels() is None
    assert clip.frames.holds(2) and clip.frames.holds(4)
    assert clip.n_frames == N_FRAMES
    assert torch.equal(
        clip.get_vision_features(3)["vision_feats"][0], torch.zeros(4, 1, 2)
    )
    with pytest.raises(ModelRuntimeError, match="frame 3 were released"):
        clip.frames[3]


def test_cached_features_cannot_be_cleared(tracker):
    clip = tracker._video_clip_cache.get("v")
    clip.clear_all()  # nothing cached yet: nothing to refuse

    clip.cache_vision_features(0, _features())

    with pytest.raises(ModelRuntimeError, match="cannot be cleared"):
        clip.clear_all()
    assert clip.has_features(0)


def test_position_embeddings_are_shared_by_the_clips_of_a_model(tracker):
    frames = [np.zeros((H, W, 3), dtype=np.uint8)] * 2
    tracker.embed_video(frames, video_id="w")
    clip_v = tracker._video_clip_cache.get("v")
    clip_w = tracker._video_clip_cache.get("w")
    clip_v.cache_vision_features(0, _features())
    clip_w.cache_vision_features(0, _features())

    assert (
        clip_v.get_vision_features(0)["vision_pos_embeds"][0].data_ptr()
        == clip_w.get_vision_features(0)["vision_pos_embeds"][0].data_ptr()
    )
    different = dict(_features(), vision_pos_embeds=[torch.zeros(4, 1, 2)])
    with pytest.raises(ModelInputError, match="Position embeddings differ"):
        clip_w.cache_vision_features(1, different)


def test_re_embedding_an_id_replaces_the_clip_and_stops_its_encoding(tracker):
    old = tracker._video_clip_cache.get("v")
    frames = [np.zeros((H, W, 3), dtype=np.uint8)] * 3

    described = tracker.embed_video(frames, video_id="v")

    assert described["n_frames"] == 3
    assert tracker._video_clip_cache.get("v") is not old
    assert not tracker._video_clip_cache.holds("v", old)
    tracker._encode_clip_frames("v", old)
    assert tracker._model.encoded == []
