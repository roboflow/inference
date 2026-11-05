"""Offline (whole-clip) tracking of ``SAM3TrackerVideo`` against a stand-in HF model.

The stand-in keeps the parts of ``Sam3TrackerVideoModel`` the session logic
depends on: a model call on a frame computes every object in the session;
points become a conditioning frame only on a frame the object has not been
tracked on; propagation recomputes every frame that is not a conditioning
frame, ignoring points stored there. Masks are a box around the first include
point of the frame that produced them, so a test can tell which points a mask
came from.
"""

import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from pycocotools import mask as mask_utils

from inference_models.errors import ModelInputError
from inference_models.models.sam3_tracker_video.clip_cache import VideoClipCache
from inference_models.models.sam3_tracker_video.sam3_tracker_video_hf import (
    SAM3TrackerVideo,
)

W, H, N_FRAMES = 64, 48, 12


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


def test_clip_cache_evicts_the_least_recently_used():
    cache = VideoClipCache(size_limit=2)
    cache.put("a", "clip-a")
    cache.put("b", "clip-b")
    cache.get("a")
    cache.put("c", "clip-c")

    assert cache.get("a") == "clip-a"
    with pytest.raises(ModelInputError):
        cache.get("b")
