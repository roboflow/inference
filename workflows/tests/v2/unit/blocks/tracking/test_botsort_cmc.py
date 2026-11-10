"""BoT-SORT camera motion compensation (CMC) with real OpenCV estimation.

Scenes: a camera pans right over a fixed textured world; two objects stay still
in the world, so in the frame they move left by the pan each frame. Without
CMC their predicted and detected boxes barely overlap and no track confirms.

    scene    world (seeded noise)   blur    window  pan/frame  box   methods
    wide     240x800                5x5     320     24 px      30px  orb, sift,
                                                                     sparseOptFlow
    gentle   120x400, min-max       21x21   160     8 px       8px   ecc

ECC fails on "wide" and ORB finds too few features on "gentle" (verified on
trackers 2.4.0), so each method uses a scene it can estimate.
"""

from datetime import datetime

import cv2
import numpy as np
import pytest
import torch
from roboflow_workflows.execution_engine.entities.base import VideoMetadata
from roboflow_workflows.v2.blocks.tracking.trackers_botsort import v1 as botsort
from roboflow_workflows.v2.blocks.tracking.trackers_botsort.v1 import BoTSORTTrackerBlock, host_bgr_frame
from roboflow_workflows.v2.data.image import ImageData

from inference_models.models.base.object_detection import Detections

FRAMES = 6
SCENES = {
    "wide": dict(
        world_hw=(240, 800),
        blur=5,
        normalize=False,
        window=320,
        pan=24,
        boxes=[[150, 60, 180, 90], [220, 120, 250, 150]],
    ),
    "gentle": dict(
        world_hw=(120, 400),
        blur=21,
        normalize=True,
        window=160,
        pan=8,
        boxes=[[60, 30, 68, 38], [100, 70, 108, 78]],
    ),
}
METHOD_SCENES = [
    ("orb", "wide"),
    ("sift", "wide"),
    ("sparseOptFlow", "wide"),
    ("ecc", "gentle"),
]


def _world(scene):
    noise = np.random.default_rng(7).random(scene["world_hw"]) * 255
    world = cv2.GaussianBlur(noise, (scene["blur"], scene["blur"]), 0)
    if scene["normalize"]:
        world = cv2.normalize(world, None, 0, 255, cv2.NORM_MINMAX)

    return world.astype(np.uint8)


def _panning_frames(name, *, channels=3):
    """Yield (image, detections) of a camera panning right over the scene."""
    scene = SCENES[name]
    world = torch.from_numpy(_world(scene))
    for frame in range(FRAMES):
        offset = frame * scene["pan"]
        window = world[:, offset : offset + scene["window"]]
        pixels = window.unsqueeze(0).repeat(channels, 1, 1).contiguous()
        image = ImageData.from_tensor(
            pixels,
            video_metadata=VideoMetadata(
                video_identifier=name,
                fps=30,
                frame_number=frame,
                frame_timestamp=datetime(2026, 10, 13),
            ),
        )
        xyxy = torch.tensor(scene["boxes"], dtype=torch.float32)
        xyxy[:, [0, 2]] -= offset
        detections = Detections(
            xyxy=xyxy,
            confidence=torch.tensor([0.9, 0.9]),
            class_id=torch.tensor([0, 0]),
            bboxes_metadata=[{"object": "left"}, {"object": "right"}],
        )
        yield image, detections


def _track(block, frames, **overrides):
    params = dict(
        minimum_iou_threshold_first_assoc=0.2,
        minimum_iou_threshold_second_assoc=0.5,
        minimum_iou_threshold_unconfirmed_assoc=0.3,
        minimum_consecutive_frames=2,
        lost_track_buffer=30,
        track_activation_threshold=0.7,
        high_conf_det_threshold=0.6,
        enable_cmc=False,
        cmc_method="sparseOptFlow",
        cmc_downscale=2,
        instant_first_frame_activation=False,
        max_sources=4,
    )
    params.update(overrides)
    reported = []
    for image, detections in frames:
        tracked = block.run(image=image, predictions=detections, **params)[
            "tracked_detections"
        ]
        rows = tracked.bboxes_metadata or []
        reported.append({row["object"]: row["tracker_id"] for row in rows})

    return reported


def test_host_bgr_frame_reverses_channels_into_new_host_memory():
    rgb = torch.stack(
        [torch.full((4, 5), value, dtype=torch.uint8) for value in (10, 20, 30)]
    )
    image = ImageData.from_tensor(rgb)
    frame = host_bgr_frame(image)
    assert frame.shape == (4, 5, 3) and frame.dtype == np.uint8
    assert frame[0, 0].tolist() == [30, 20, 10]
    assert not np.shares_memory(frame, image.tensor_image.numpy())
    assert rgb[:, 0, 0].tolist() == [10, 20, 30]


def test_host_bgr_frame_expands_grayscale_to_three_channels():
    gray = torch.arange(20, dtype=torch.uint8).reshape(1, 4, 5)
    frame = host_bgr_frame(ImageData.from_tensor(gray))
    assert frame.shape == (4, 5, 3)
    for channel in range(3):
        assert np.array_equal(frame[:, :, channel], gray[0].numpy())


@pytest.mark.parametrize("method,scene", METHOD_SCENES)
def test_each_cmc_method_keeps_ids_on_a_panning_camera(method, scene):
    without = _track(BoTSORTTrackerBlock(), _panning_frames(scene))
    assert without == [{}] * FRAMES

    with_cmc = _track(
        BoTSORTTrackerBlock(),
        _panning_frames(scene),
        enable_cmc=True,
        cmc_method=method,
    )
    # Confirmed at the second frame (minimum_consecutive_frames=2), then kept.
    assert with_cmc[0] == {}
    confirmed = with_cmc[1]
    assert set(confirmed) == {"left", "right"}
    assert len(set(confirmed.values())) == 2
    assert all(ids == confirmed for ids in with_cmc[1:])


def test_without_cmc_pixels_never_leave_the_image(monkeypatch):
    def forbidden(image):
        raise AssertionError("host_bgr_frame called with CMC disabled")

    monkeypatch.setattr(botsort, "host_bgr_frame", forbidden)
    reported = _track(
        BoTSORTTrackerBlock(),
        _panning_frames("wide"),
        instant_first_frame_activation=True,
    )
    # Instant activation reports the first frame; motion then loses the boxes.
    assert set(reported[0]) == {"left", "right"}


def test_first_frame_fixes_cmc_per_source(monkeypatch):
    copied = []

    def counting(image):
        copied.append(image.video_metadata.video_identifier)
        return host_bgr_frame(image)

    monkeypatch.setattr(botsort, "host_bgr_frame", counting)
    block = BoTSORTTrackerBlock()
    frames = list(_panning_frames("wide"))
    _track(block, frames[:1], enable_cmc=False)
    later = _track(block, frames[1:], enable_cmc=True)
    assert copied == []
    assert later == [{}] * (FRAMES - 1)
    assert block._trackers["wide"].tracker.enable_cmc is False


def test_cmc_tracks_a_grayscale_source():
    reported = _track(
        BoTSORTTrackerBlock(),
        _panning_frames("wide", channels=1),
        enable_cmc=True,
    )
    assert set(reported[1]) == {"left", "right"}
    assert all(ids == reported[1] for ids in reported[1:])
