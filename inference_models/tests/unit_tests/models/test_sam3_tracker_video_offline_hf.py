"""Offline tracking of ``SAM3TrackerVideo`` on the real transformers classes.

A ``Sam3TrackerVideoModel`` with a small configuration and random weights, and
the real processor, run the session code the offline path depends on: what a
session reads from a clip, what the encoder caches per frame, and what
happens to a clip whose pixels were released.
"""

import numpy as np
import pytest
import torch
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


def test_a_session_reset_cannot_drop_the_clips_features(tracker):
    tracker.embed_video(_frames(2), video_id="v")
    clip = tracker._video_clip_cache.get("v")
    session = tracker._prompted_session(clip, 1, {0: ((20.0, 20.0, 1),)})[0]

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
