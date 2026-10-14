"""``roboflow_core/heatmap_visualization@v1``: heat accumulated per source.

The V1 oracle is the latest V1 block class itself, one instance kept across
frames, fed the same rows as ``sv.Detections``; V1 test modules are not
imported. Analytic cases use a flat grey image, ``kernel_size=1`` (no blur)
and ``opacity=1``, so a heated pixel shows its hue at full saturation.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
import pytest
import supervision as sv
import torch
from roboflow_workflows.core_steps.visualizations.heatmap.v1 import (
    HeatmapVisualizationBlockV1,
)
from roboflow_workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    VideoMetadata,
    WorkflowImageData,
)
from roboflow_workflows.v2.blocks.visualization.annotations.heatmap_visualization.v1 import HeatmapVisualizationBlock
from roboflow_workflows.v2.blocks.visualization.annotations.sources import (
    SourceCapacityError,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.engine.declaration import spec_of

from inference_models.models.base.object_detection import Detections
from tests.v2.unit.blocks.visualization.annotations.support import (
    detections,
    pixels_of,
    scene,
)

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
SIZE = (40, 64)
GREY = (100, 100, 100)
RED = (255, 0, 0)
PLAIN = {"kernel_size": 1, "opacity": 1.0, "radius": 3, "position": "CENTER"}
# (x1, y1, x2, y2, tracker id or None)
Row = Tuple[float, float, float, float, Optional[int]]


def frame(
    number: int,
    *,
    video: str = "cam-a",
    size: Tuple[int, int] = SIZE,
    grey: bool = False,
) -> ImageData:
    pixels = (
        np.full(size + (3,), GREY, dtype=np.uint8)
        if grey
        else scene(*size, seed=number)
    )
    metadata = VideoMetadata(
        video_identifier=video,
        frame_number=number,
        frame_timestamp=T0 + timedelta(seconds=number / 30),
    )
    image = ImageData.from_numpy_rgb(
        pixels, image_id=f"{video}-{number}", video_metadata=metadata
    )

    return image


def boxes(image: ImageData, rows: Sequence[Row]) -> Detections:
    objects = [(x1, y1, x2, y2, 0, 0.9) for x1, y1, x2, y2, _ in rows]
    metadata = None
    if rows and rows[0][4] is not None:
        metadata = [{"tracker_id": row[4]} for row in rows]
    predictions = detections(image, objects, rows=metadata)

    return predictions


def at(
    x: float, y: float, tracker_id: Optional[int] = None, *, half: float = 4.0
) -> Row:
    return (x - half, y - half, x + half, y + half, tracker_id)


def hue_rgb(hue: int) -> Tuple[int, int, int]:
    hsv = np.array([[[hue, 255, 255]]], dtype=np.uint8)
    rgb = tuple(int(value) for value in cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0])

    return rgb


def disc(
    centre: Tuple[int, int], radius: int, size: Tuple[int, int] = SIZE
) -> np.ndarray:
    mask = np.zeros(size, dtype=np.uint8)
    cv2.circle(mask, centre, radius, 1, thickness=-1)

    return mask.astype(bool)


def changed(image: ImageData) -> np.ndarray:
    mask = np.any(pixels_of(image) != GREY, axis=-1)

    return mask


class V2Heatmap:
    """One block instance across frames, parameters validated like the engine."""

    def __init__(self, **parameters: Any) -> None:
        self.block = HeatmapVisualizationBlock()
        params = spec_of(HeatmapVisualizationBlock).validate_params(
            {
                **parameters,
                "image": "$inputs.image",
                "predictions": "$inputs.predictions",
            }
        )
        self.arguments = dict(params)

    def __call__(
        self, image: ImageData, rows: Sequence[Row], **selected: Any
    ) -> ImageData:
        arguments = {
            **self.arguments,
            **selected,
            "image": image,
            "predictions": boxes(image, rows),
        }
        output = self.block.run(**arguments)["image"]

        return output


class V1Heatmap:
    """The latest V1 block, one instance across frames."""

    def __init__(self, **parameters: Any) -> None:
        self.block = HeatmapVisualizationBlockV1()
        self.parameters = {
            "position": "BOTTOM_CENTER",
            "opacity": 0.2,
            "radius": 40,
            "kernel_size": 25,
            "top_hue": 0,
            "low_hue": 125,
            "ignore_stationary": True,
            "motion_threshold": 25,
            **parameters,
        }

    def __call__(self, image: ImageData, rows: Sequence[Row]) -> np.ndarray:
        v1_image = WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="frame"),
            numpy_image=np.ascontiguousarray(pixels_of(image)[:, :, ::-1]),
        )
        tracker_ids = None
        if rows and rows[0][4] is not None:
            tracker_ids = np.array([row[4] for row in rows], dtype=int)
        v1_rows = sv.Detections(
            xyxy=np.array([row[:4] for row in rows], dtype=np.float32).reshape(-1, 4),
            class_id=np.zeros(len(rows), dtype=int),
            confidence=np.full(len(rows), 0.9, dtype=np.float32),
            tracker_id=tracker_ids,
        )
        result = self.block.run(
            image=v1_image,
            predictions=v1_rows,
            copy_image=True,
            metadata=image.video_metadata,
            **self.parameters,
        )
        output = result["image"].numpy_image[:, :, ::-1]

        return output


# Two tracks; track 2 stands still for two frames, track 5 joins on frame 3.
MOVES = (
    [at(8.3, 6.6, 1), at(55.5, 30.2, 2)],
    [at(14.9, 9.1, 1), at(55.5, 30.2, 2)],
    [at(21.6, 13.8, 1), at(55.9, 30.0, 2)],
    [at(27.2, 18.4, 1), at(48.2, 25.7, 2), at(30.0, 33.0, 5)],
    [at(27.9, 18.0, 1), at(41.1, 21.5, 2), at(37.0, 27.3, 5)],
    [at(33.7, 22.0, 1), at(35.1, 16.5, 2), at(37.3, 27.1, 5)],
)


@pytest.mark.parametrize(
    "parameters",
    [
        {"ignore_stationary": False, "opacity": 0.5, "radius": 6, "kernel_size": 5},
        {
            "motion_threshold": 3,
            "opacity": 0.75,
            "radius": 5,
            "kernel_size": 3,
            "position": "CENTER",
        },
        {
            "opacity": 0.25,
            "radius": 9,
            "top_hue": 150,
            "low_hue": 20,
            "position": "TOP_LEFT",
        },
    ],
)
def test_every_frame_equals_v1(parameters: Dict) -> None:
    v2, v1 = V2Heatmap(**parameters), V1Heatmap(**parameters)

    for number, rows in enumerate(MOVES):
        image = frame(number)
        np.testing.assert_array_equal(pixels_of(v2(image, rows)), v1(image, rows))


def test_default_opacity_is_within_one_level_of_v1() -> None:
    v2, v1 = V2Heatmap(radius=8, ignore_stationary=False), V1Heatmap(
        radius=8, ignore_stationary=False
    )

    for number, rows in enumerate(MOVES):
        image = frame(number)
        difference = np.abs(pixels_of(v2(image, rows)).astype(int) - v1(image, rows))

        assert difference.max() <= 1


def test_heat_counts_choose_the_hue_of_each_disc() -> None:
    heatmap = V2Heatmap(ignore_stationary=False, **PLAIN)
    first, second = frame(0, grey=True), frame(1, grey=True)
    heatmap(first, [at(10, 10)])
    output = heatmap(second, [at(10, 10), at(40, 20)])

    hot, warm = disc((10, 10), 3), disc((40, 20), 3)
    pixels = pixels_of(output)
    assert np.all(pixels[hot] == hue_rgb(0))
    # 125 - 1 / 2 * 125 = 62.5, truncated as V1 does.
    assert np.all(pixels[warm] == hue_rgb(62))
    np.testing.assert_array_equal(changed(output), hot | warm)


def test_overlapping_discs_of_one_frame_add_heat_once() -> None:
    heatmap = V2Heatmap(ignore_stationary=False, **PLAIN)
    image = frame(0, grey=True)
    output = heatmap(image, [at(10, 10), at(12, 10), at(40, 20)])

    assert np.all(pixels_of(output)[changed(output)] == hue_rgb(0))


def test_a_stationary_track_adds_heat_only_when_it_moves() -> None:
    heatmap = V2Heatmap(motion_threshold=10, **PLAIN)
    outputs = []
    for number, x in enumerate((10, 10, 13, 10, 25)):
        image = frame(number, grey=True)
        outputs.append(heatmap(image, [at(x, 20, 4)]))

    assert not any(changed(output).any() for output in outputs[:4])
    np.testing.assert_array_equal(changed(outputs[4]), disc((25, 20), 3))


def test_rows_without_tracker_ids_always_add_heat() -> None:
    heatmap = V2Heatmap(**PLAIN)
    image = frame(0, grey=True)

    np.testing.assert_array_equal(
        changed(heatmap(image, [at(10, 20)])), disc((10, 20), 3)
    )


def test_heat_of_256_frames_stays_visible_unlike_v1() -> None:
    parameters = {**PLAIN, "ignore_stationary": False, "radius": 1}
    v2, v1 = V2Heatmap(**parameters), V1Heatmap(**parameters)
    for number in range(256):
        image = frame(number, size=(8, 8), grey=True)
        output, reference = v2(image, [at(4, 4)]), v1(image, [at(4, 4)])

    assert tuple(pixels_of(output)[4, 4]) == RED
    assert tuple(reference[4, 4]) == GREY


def test_interleaved_sources_keep_separate_heat() -> None:
    shared = V2Heatmap(ignore_stationary=False, **PLAIN)
    alone = {
        video: V2Heatmap(ignore_stationary=False, **PLAIN)
        for video in ("cam-a", "cam-b")
    }
    x_of = {"cam-a": (8, 16, 24), "cam-b": (56, 48, 40)}
    for number in range(3):
        for video in ("cam-a", "cam-b"):
            image = frame(number, video=video, grey=True)
            rows = [at(x_of[video][number], 20)]
            output = shared(image, rows)
            np.testing.assert_array_equal(
                pixels_of(output), pixels_of(alone[video](image, rows))
            )

    assert not changed(output)[:, :32].any()


def test_restart_reset_and_a_new_size_start_a_new_heat() -> None:
    heatmap = V2Heatmap(ignore_stationary=False, **PLAIN)
    for number in (5, 6):
        image = frame(number, grey=True)
        heatmap(image, [at(10, 20)])
    single = disc((30, 20), 3)

    restarted = frame(0, grey=True)
    assert np.array_equal(changed(heatmap(restarted, [at(30, 20)])), single)
    heatmap.block.reset_state()
    after_reset = frame(1, grey=True)
    assert np.array_equal(changed(heatmap(after_reset, [at(30, 20)])), single)
    resized = frame(2, size=(48, 64), grey=True)
    assert np.array_equal(
        changed(heatmap(resized, [at(30, 20)])), disc((30, 20), 3, (48, 64))
    )


def test_one_source_too_many_fails_and_changes_no_heat() -> None:
    heatmap = V2Heatmap(ignore_stationary=False, max_sources=1, **PLAIN)
    image = frame(0, grey=True)
    heatmap(image, [at(10, 20)])

    other = frame(0, video="cam-b", grey=True)
    with pytest.raises(SourceCapacityError, match="max_sources=1"):
        heatmap(other, [at(30, 20)])
    later = frame(1, grey=True)

    assert np.array_equal(changed(heatmap(later, [])), disc((10, 20), 3))


@pytest.mark.parametrize(("max_sources", "kept"), [(None, 32), (40, 33)])
def test_the_default_keeps_32_sources_and_a_larger_limit_is_allowed(
    max_sources: Optional[int], kept: int
) -> None:
    limit = {} if max_sources is None else {"max_sources": max_sources}
    heatmap = V2Heatmap(ignore_stationary=False, **limit, **PLAIN)
    for index in range(32):
        heatmap(frame(0, video=f"cam-{index}", grey=True), [at(10, 20)])

    extra = frame(0, video="cam-32", grey=True)
    if kept == 32:
        with pytest.raises(SourceCapacityError, match="source 33 of max_sources=32"):
            heatmap(extra, [at(10, 20)])
    else:
        heatmap(extra, [at(10, 20)])

    assert len(heatmap.block._heats) == kept


def test_a_crop_is_refused_and_a_resized_whole_frame_adds_heat() -> None:
    heatmap = V2Heatmap(ignore_stationary=False, **PLAIN)
    crop = frame(0, grey=True).crop((8, 0, 64, 40))
    with pytest.raises(ValueError, match="Draw before cropping"):
        heatmap(crop, [at(10, 20)])

    for number in (1, 2):
        half = frame(number, grey=True).resize((20, 32), interpolation="nearest")
        output = heatmap(half, [at(10, 10, half=2)])

    assert len(heatmap.block._heats) == 1
    assert np.array_equal(changed(output), disc((10, 10), 3, (20, 32)))


@pytest.mark.parametrize(("unseen_frames", "drawn"), [(900, True), (1830, False)])
def test_a_kept_anchor_is_forgotten_60_seconds_after_its_track_was_seen(
    unseen_frames: int, drawn: bool
) -> None:
    heatmap = V2Heatmap(motion_threshold=10, **PLAIN)
    first = frame(0, grey=True)
    heatmap(first, [at(10, 20, 4)])

    # Frames are 1/30 s apart: 900 frames are 30 s, 1830 frames 61 s.
    later = frame(unseen_frames, grey=True)
    output = heatmap(later, [at(30, 20, 4)])

    assert changed(output).any() == drawn


def test_a_new_position_forgets_kept_anchors_but_not_heat() -> None:
    heatmap = V2Heatmap(motion_threshold=10, **PLAIN)
    for number, x in enumerate((10, 30)):
        image = frame(number, grey=True)
        heatmap(image, [at(x, 20, 4)])

    image = frame(2, grey=True)
    output = heatmap(image, [at(50, 20, 4)], position="TOP_LEFT")

    np.testing.assert_array_equal(changed(output), disc((30, 20), 3))


def test_an_empty_frame_shows_kept_heat_and_leaves_the_input_unchanged() -> None:
    heatmap = V2Heatmap(ignore_stationary=False, **PLAIN)
    first = frame(0)
    before = pixels_of(first).copy()
    nothing = heatmap(first, [])
    np.testing.assert_array_equal(pixels_of(nothing), before)

    heatmap(frame(1, grey=True), [at(10, 20)])
    empty = frame(2, grey=True)
    output = heatmap(empty, [])

    assert output is not empty and output.size_hw == empty.size_hw
    assert np.array_equal(changed(output), disc((10, 20), 3))
    np.testing.assert_array_equal(pixels_of(first), before)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("top_hue", 180, "top_hue must be an OpenCV hue from 0 to 179"),
        ("low_hue", -1, "low_hue must be an integer >= 0"),
        ("opacity", 1.5, r"opacity must be a number in \[0.0, 1.0\]"),
        ("kernel_size", 0, "kernel_size must be an integer >= 1"),
        ("radius", -2, "radius must be an integer >= 0"),
        ("motion_threshold", -1, "motion_threshold must be an integer >= 0"),
    ],
)
def test_literal_and_selected_values_fail_with_one_message(
    field: str, value: Any, message: str
) -> None:
    with pytest.raises(Exception, match=message):
        V2Heatmap(**{field: value})
    heatmap = V2Heatmap(**{field: f"$inputs.{field}"})

    with pytest.raises(ValueError, match=message):
        heatmap(frame(0), [], **{field: value})


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("metadata", "metadata is not a V2 parameter"),
        ("copy_image", "copy_image is not a V2 parameter"),
    ],
)
def test_removed_v1_fields_are_rejected_with_guidance(field: str, message: str) -> None:
    with pytest.raises(Exception, match=message):
        V2Heatmap(**{field: "$inputs.video_metadata"})


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA companion: no CUDA device"
)
def test_a_cuda_image_gets_the_same_pixels_on_its_device() -> None:
    on_host, on_device = V2Heatmap(ignore_stationary=False), V2Heatmap(
        ignore_stationary=False
    )
    for number, rows in enumerate(MOVES):
        image = frame(number)
        cuda_image = ImageData.from_tensor(
            image.tensor_image.cuda(),
            image_id=image.image_id,
            video_metadata=image.video_metadata,
        )
        expected = on_host(image, rows)
        output = on_device(cuda_image, rows)

    assert output.tensor_image.device.type == "cuda"
    np.testing.assert_array_equal(pixels_of(output), pixels_of(expected))
