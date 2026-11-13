"""Offline tracking of ``SAM3TrackerVideo`` on the real transformers classes.

A ``Sam3TrackerVideoModel`` with a small configuration and random weights, and
the real processor, run the session code the offline path depends on: what a
session reads from a clip, what the encoder caches per frame, what happens to
a clip whose pixels were released, and how the model treats clicks on a frame
it has tracked.
"""

import threading

import numpy as np
import pytest
import torch
from pycocotools import mask as mask_utils
from transformers import (
    Sam2ImageProcessor,
    Sam2VideoVideoProcessor,
    Sam3TrackerVideoConfig,
    Sam3TrackerVideoModel,
    Sam3TrackerVideoProcessor,
)

from inference_models.errors import ModelRuntimeError
from inference_models.models.sam3_tracker_video.clip_cache import cached_frame_bytes
from inference_models.models.sam3_tracker_video.sam3_tracker_video_hf import (
    EMBED_CHUNK_FRAMES,
    SAM3TrackerVideo,
)

IMAGE_SIZE, W, H = 56, 80, 60


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    config = Sam3TrackerVideoConfig(
        image_size=IMAGE_SIZE,
        vision_config={
            "backbone_config": {
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "image_size": IMAGE_SIZE,
                "window_size": 2,
                "global_attn_indexes": [1],
                "pretrain_image_size": 28,
            },
            "backbone_feature_sizes": [[16, 16], [8, 8], [4, 4]],
        },
    )

    return Sam3TrackerVideoModel(config).eval()


@pytest.fixture
def tracker(model):
    size = {"height": IMAGE_SIZE, "width": IMAGE_SIZE}
    processor = Sam3TrackerVideoProcessor(
        image_processor=Sam2ImageProcessor(size=size),
        video_processor=Sam2VideoVideoProcessor(size=size),
        target_size=IMAGE_SIZE,
    )

    return SAM3TrackerVideo(
        model=model,
        processor=processor,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def _frames(n_frames):
    rng = np.random.default_rng(0)
    return [rng.integers(0, 255, (H, W, 3), dtype=np.uint8) for _ in range(n_frames)]


def _prompt(frame_idx, obj_id, x, y, label=1):
    return {"frame_idx": frame_idx, "obj_id": obj_id, "x": x, "y": y, "label": label}


def test_preprocessing_in_chunks_equals_preprocessing_the_whole_clip(tracker):
    frames = _frames(2 * EMBED_CHUNK_FRAMES + 1)
    whole = tracker._processor.video_processor(
        videos=frames, device="cpu", return_tensors="pt"
    ).pixel_values_videos[0]

    chunked = tracker._preprocess_frames(frames)

    assert len(chunked) == len(frames)
    assert torch.equal(torch.stack(chunked), whole)


def test_cached_frame_bytes_are_the_bytes_an_encoded_frame_holds(tracker):
    tracker.embed_video(_frames(2), video_id="v")
    clip = tracker._video_clip_cache.get("v")
    pixel_bytes = clip.frames[0].nbytes

    tracker._encode_clip_frames("v", clip)

    feature_bytes = sum(f.nbytes for f in clip._feats[0])
    assert clip._feats[0][0].untyped_storage().nbytes() == feature_bytes
    computed = cached_frame_bytes(tracker._model.config, torch.float32)
    assert feature_bytes == computed == tracker.video_cache_limits()["bytes_per_frame"]
    assert pixel_bytes <= computed
    # At the real model's shapes: 1008 px input, maps of 288, 144 and 72 px.
    real = Sam3TrackerVideoConfig()
    assert cached_frame_bytes(real, torch.bfloat16) == 10_616_832
    assert 3 * real.image_size**2 * 2 == 6_096_384


def test_embedding_again_with_keep_existing_keeps_the_clip_and_what_it_encoded(tracker):
    tracker.embed_video(_frames(3), video_id="v")
    clip = tracker._video_clip_cache.get("v")
    tracker._encode_clip_frames("v", clip)

    described = tracker.embed_video(_frames(3), video_id="v", keep_existing=True)

    assert tracker._video_clip_cache.get("v") is clip
    assert all(clip.has_features(i) for i in range(3))
    assert described == {"video_id": "v", "n_frames": 3, "height": H, "width": W}


def test_keep_existing_replaces_a_clip_of_another_length_and_plain_embeds_replace(
    tracker,
):
    tracker.embed_video(_frames(3), video_id="v")
    first = tracker._video_clip_cache.get("v")

    tracker.embed_video(_frames(4), video_id="v", keep_existing=True)
    longer = tracker._video_clip_cache.get("v")
    tracker.embed_video(_frames(4), video_id="v")

    assert longer is not first
    assert longer.n_frames == 4
    assert tracker._video_clip_cache.get("v") is not longer


def test_encoding_is_started_once_for_a_clip_that_is_kept(tracker):
    with tracker._request_lock():
        tracker.embed_video(_frames(2), video_id="v", encode_frames=True)
        clip = tracker._video_clip_cache.get("v")
        threads = threading.active_count()

        tracker.embed_video(
            _frames(2), video_id="v", encode_frames=True, keep_existing=True
        )

        assert tracker._video_clip_cache.get("v") is clip
        assert threading.active_count() == threads
        assert not any(clip.has_features(i) for i in range(2))

    for thread in threading.enumerate():
        if thread is not threading.current_thread() and thread.daemon:
            thread.join(timeout=60)
    assert all(clip.has_features(i) for i in range(2))


def test_background_encoding_starts_no_frame_while_a_request_is_open(tracker):
    tracker.embed_video(_frames(3), video_id="v")
    clip = tracker._video_clip_cache.get("v")

    with tracker._request_lock():
        encoding = threading.Thread(
            target=tracker._encode_clip_frames, args=("v", clip), daemon=True
        )
        encoding.start()
        encoding.join(timeout=0.5)
        assert encoding.is_alive()
        assert not any(clip.has_features(i) for i in range(3))

    encoding.join(timeout=60)
    assert not encoding.is_alive()
    assert all(clip.has_features(i) for i in range(3))


def test_encoding_every_frame_releases_every_frames_pixels(tracker):
    tracker.embed_video(_frames(5), video_id="v")
    clip = tracker._video_clip_cache.get("v")
    assert all(clip.frames.holds(i) for i in range(5))

    tracker._encode_clip_frames("v", clip)

    assert all(clip.has_features(i) for i in range(5))
    assert not any(clip.frames.holds(i) for i in range(5))


def test_segment_and_propagate_are_the_same_on_a_clip_without_pixels(tracker):
    frames = _frames(6)
    prompts = [_prompt(1, 1, 20, 20), _prompt(4, 2, 60, 40), _prompt(4, 2, 10, 10, 0)]
    tracker.embed_video(frames, video_id="lazy")
    tracker.embed_video(frames, video_id="encoded")
    encoded = tracker._video_clip_cache.get("encoded")
    tracker._encode_clip_frames("encoded", encoded)
    assert not any(encoded.frames.holds(i) for i in range(6))

    segmented = tracker.segment_video_frame("encoded", prompts=prompts, frame_idx=4)
    propagated = tracker.propagate_video("encoded", prompts=prompts)
    again = tracker.propagate_video("encoded", prompts=prompts)

    assert segmented["masks"]["2"]["size"] == [H, W]
    assert [frame["frame_idx"] for frame in propagated] == list(range(6))
    assert all(set(frame["masks"]) == {"1", "2"} for frame in propagated)
    assert again == propagated
    # A clip encoded frame by frame as the sessions need it gives the same masks.
    assert segmented == tracker.segment_video_frame(
        "lazy", prompts=prompts, frame_idx=4
    )
    assert propagated == tracker.propagate_video("lazy", prompts=prompts)
    lazy = tracker._video_clip_cache.get("lazy")
    assert not any(lazy.frames.holds(i) for i in range(6))


def test_segment_and_propagate_with_corrections_release_every_frames_pixels(tracker):
    frames = _frames(8)
    prompts = [
        _prompt(1, 1, 20, 20, 0),
        _prompt(3, 1, 20, 20),
        _prompt(5, 1, 60, 40, 0),
        _prompt(6, 1, 30, 30),
    ]
    tracker.embed_video(frames, video_id="lazy")
    tracker.embed_video(frames, video_id="encoded")
    encoded = tracker._video_clip_cache.get("encoded")
    tracker._encode_clip_frames("encoded", encoded)

    propagated = tracker.propagate_video("encoded", prompts=prompts)

    assert [frame["frame_idx"] for frame in propagated] == list(range(8))
    assert propagated == tracker.propagate_video("lazy", prompts=prompts)
    lazy = tracker._video_clip_cache.get("lazy")
    assert not any(lazy.frames.holds(i) for i in range(8))
    for frame_idx in (1, 3, 5, 6):
        assert (
            tracker.segment_video_frame("lazy", prompts=prompts, frame_idx=frame_idx)
            == tracker.segment_video_frame(
                "encoded", prompts=prompts, frame_idx=frame_idx
            )
            == {"frame_idx": frame_idx, "masks": propagated[frame_idx]["masks"]}
        )


def _spy_on_frames(tracker, monkeypatch):
    """Record each frame the model runs: (frame, initial conditioning frame,
    its input, reverse, the object's conditioning frames at that moment, its
    other tracked frames at that moment)."""
    model, runs = tracker._model, []
    run_frame = model._run_single_frame_inference

    def _run(**kwargs):
        session = kwargs["inference_session"]
        outputs = session.output_dict_per_obj[kwargs["obj_idx"]]
        conditioning = outputs["cond_frame_outputs"]
        tracked = set(outputs["non_cond_frame_outputs"]) - {kwargs["frame_idx"]}
        runs.append(
            (
                kwargs["frame_idx"],
                kwargs["is_init_cond_frame"],
                (
                    "points"
                    if kwargs["point_inputs"] is not None
                    else "mask" if kwargs["mask_inputs"] is not None else None
                ),
                kwargs["reverse"],
                sorted(conditioning),
                sorted(tracked),
            )
        )
        return run_frame(**kwargs)

    monkeypatch.setattr(model, "_run_single_frame_inference", _run)

    return runs


def test_clicks_off_the_anchor_correct_a_frame_the_model_has_tracked(
    tracker, monkeypatch
):
    tracker.embed_video(_frames(6), video_id="v")
    prompts = [
        _prompt(0, 1, 60, 40, 0),
        _prompt(2, 1, 20, 20),
        _prompt(4, 1, 60, 40, 0),
        _prompt(4, 1, 30, 30),
    ]
    runs = _spy_on_frames(tracker, monkeypatch)

    propagated = tracker.propagate_video("v", prompts=prompts)

    assert runs == [
        # The anchor, from its points alone.
        (2, True, "points", False, [], []),
        (3, False, None, False, [2], []),
        # Frame 4 is tracked, then run again with its points and the memory.
        (4, False, None, False, [2], [3]),
        (4, False, "points", False, [2], [3]),
        # It conditions the frames tracked after it, which no longer read
        # frame 3. The backward pass is not conditioned on it and gets the
        # forward frames back as tracked frames.
        (5, False, None, False, [2, 4], []),
        (1, False, None, True, [2], [3, 4, 5]),
        # Exclude points alone: frame 0 is tracked, then given an empty mask.
        (0, False, None, True, [2], [1, 3, 4, 5]),
        (0, False, "mask", True, [2], [1, 3, 4, 5]),
    ]
    assert not mask_utils.decode(_rle(propagated[0]["masks"]["1"])).any()


def _rle(mask):
    return {"size": mask["size"], "counts": mask["counts"].encode()}


def _tracked_before_corrections(tracker, clip, obj_id, object_prompts):
    """The tracking of the build before corrections: every prompted frame is
    an initial conditioning frame, then forward and backward from the first."""
    session = tracker._clip_session(clip)
    for frame_idx in sorted(object_prompts):
        tracker._apply_points(
            session, obj_id, frame_idx, object_prompts[frame_idx], reverse=False
        )
    masks = {}
    for reverse in (False, True):
        for output in tracker._model.propagate_in_video_iterator(
            session, start_frame_idx=min(object_prompts), reverse=reverse
        ):
            masks[int(output.frame_idx)] = output.pred_masks
    return masks


@pytest.mark.parametrize("anchor", [0, 3, 6])
def test_clicks_on_one_frame_track_as_before_corrections(tracker, anchor):
    tracker.embed_video(_frames(7), video_id="v")
    clip = tracker._video_clip_cache.get("v")
    points = {anchor: ((20.0, 20.0, 1), (60.0, 40.0, 0), (30.0, 25.0, 1))}

    with torch.inference_mode():
        before = _tracked_before_corrections(tracker, clip, 1, points)
        tracked = dict(tracker._track_object(clip, 1, points))
        segmented = dict(tracker._track_object(clip, 1, points, only_frame=anchor))

    assert sorted(tracked) == list(range(7))
    assert all(torch.equal(tracked[i], before[i]) for i in range(7))
    assert list(segmented) == [anchor]
    assert torch.equal(segmented[anchor], before[anchor])


def _window(tracker):
    """Frames a tracked frame can read back: memory frames and object pointers."""
    config = tracker._model.config
    return max(config.num_maskmem, config.max_object_pointers_in_encoder)


def test_stopping_at_a_frame_gives_the_masks_of_tracking_the_whole_clip(tracker):
    window = _window(tracker)
    n_frames, anchor = 2 * window + 8, window // 2
    tracker.embed_video(_frames(n_frames), video_id="v")
    clip = tracker._video_clip_cache.get("v")
    points = {
        1: ((60.0, 40.0, 0),),
        anchor: ((20.0, 20.0, 1),),
        anchor + 2: ((30.0, 30.0, 1),),
        n_frames - 2: ((60.0, 40.0, 0),),
    }

    with torch.inference_mode():
        tracked = dict(tracker._track_object(clip, 1, points))
        for frame_idx in (anchor, anchor + 2, n_frames - 2):
            stopped = dict(tracker._track_object(clip, 1, points, only_frame=frame_idx))
            # The masks are logits, so equality is not two empty masks.
            assert torch.equal(stopped[frame_idx], tracked[frame_idx])
            assert max(stopped) == frame_idx


def test_a_correction_changes_only_the_frames_tracked_after_it(tracker):
    window = _window(tracker)
    n_frames, anchor = 2 * window + 8, 3
    corrected = anchor + window + 2
    tracker.embed_video(_frames(n_frames), video_id="v")
    clip = tracker._video_clip_cache.get("v")
    anchor_points = {anchor: ((20.0, 20.0, 1),)}
    points = {**anchor_points, corrected: ((60.0, 40.0, 0),)}

    with torch.inference_mode():
        alone = dict(tracker._track_object(clip, 1, anchor_points))
        tracked = dict(tracker._track_object(clip, 1, points))

    same = [i for i in range(n_frames) if torch.equal(tracked[i], alone[i])]
    assert same == list(range(corrected))


def test_a_session_reset_cannot_drop_the_clips_features(tracker):
    tracker.embed_video(_frames(2), video_id="v")
    clip = tracker._video_clip_cache.get("v")
    session = tracker._clip_session(clip)
    tracker._apply_points(session, 1, 0, ((20.0, 20.0, 1),), reverse=False)

    with pytest.raises(ModelRuntimeError, match="cannot be cleared"):
        session.reset_inference_session()

    assert clip.has_features(0)
    assert tracker.segment_video_frame(
        "v", prompts=[_prompt(0, 1, 20, 20)], frame_idx=0
    )["masks"]["1"]["size"] == [H, W]


def test_a_session_cannot_add_frames_to_a_clip(tracker):
    tracker.embed_video(_frames(2), video_id="v")
    session = tracker._clip_session(tracker._video_clip_cache.get("v"))

    with pytest.raises(TypeError, match="item assignment"):
        session.add_new_frame(torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE))
