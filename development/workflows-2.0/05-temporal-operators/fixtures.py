"""Deterministic camera schedules for the temporal operator examples.

Every frame is a small native CHW ``uint8`` tensor: a flat background plus a
bright marker square. The background brightness makes best-frame choices
predictable; the marker position makes crops taken after collection differ.
All media timestamps are milliseconds on one shared rig clock. A shared clock
only makes timestamps comparable; the ``v2/align`` operator decides which
samples correspond.
"""

from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, Tuple

import torch
from roboflow_workflows.execution_engine.v2.blocks.image_data import ImageData
from roboflow_workflows.execution_engine.v2.data import Timestamp

RIG_CLOCK = "rig-media"
FRAME_HEIGHT = 48
FRAME_WIDTH = 64
MARKER_SIZE = 12


@dataclass(frozen=True)
class Frame:
    """One scheduled frame of a fixture camera.

    Attributes:
        pts_ms: Media timestamp in milliseconds.
        brightness: Background value of every channel, 0..255.
        marker_x: Left column of the 12x12 marker square.
    """

    pts_ms: int
    brightness: int
    marker_x: int


@dataclass(frozen=True)
class Schedule:
    """Finite frame list of one fixture camera.

    Attributes:
        frames: Frames in emission order.
        tint: Per-channel multiplier distinguishing cameras in the gallery.
        clock: Media clock identity attached to every frame.
    """

    frames: Tuple[Frame, ...]
    tint: Tuple[float, float, float]
    clock: str = RIG_CLOCK


def _frames(*rows: Tuple[int, int, int]) -> Tuple[Frame, ...]:
    return tuple(Frame(*row) for row in rows)


# Left leads the rig at 40 ms. Right is 5 ms late on every frame. The frame
# most similar to a flat mid-grey (128) card sits at a different time per
# camera: windows of four choose left 80/240 ms, right 45/165 ms; windows of
# three choose left 80/200 ms, right 45/165 ms.
# Rows are (pts_ms, brightness, marker_x).
LEFT = _frames(
    (0, 40, 0),
    (40, 90, 12),
    (80, 150, 24),
    (120, 250, 36),
    (160, 50, 52),
    (200, 125, 40),
    (240, 180, 28),
    (280, 240, 16),
)
RIGHT = _frames(
    (5, 230, 52),
    (45, 170, 40),
    (85, 90, 28),
    (125, 30, 16),
    (165, 200, 0),
    (205, 60, 12),
    (245, 90, 24),
    (285, 20, 36),
)

SCHEDULES: Dict[str, Schedule] = {
    "left": Schedule(LEFT, tint=(1.0, 0.7, 0.5)),
    "right": Schedule(RIGHT, tint=(0.5, 0.7, 1.0)),
    # Variations: frame 85 ms is missing; a 60 ms frame arrives after 85 ms;
    # an unrelated media clock.
    "right-gap": Schedule(RIGHT[:2] + RIGHT[3:], tint=(0.5, 0.7, 1.0)),
    "right-late": Schedule(
        RIGHT[:3] + (Frame(60, 99, 8),) + RIGHT[3:], tint=(0.5, 0.7, 1.0)
    ),
    "right-other-clock": Schedule(RIGHT, tint=(0.5, 0.7, 1.0), clock="other-media"),
}


def make_image(frame: Frame, *, schedule: Schedule, image_id: str) -> ImageData:
    """Render one scheduled frame as a native tensor image.

    Args:
        frame: Scheduled brightness and marker position.
        schedule: Camera schedule providing the colour tint.
        image_id: Identity recorded in the image provenance.

    Returns:
        RGB ``uint8`` CHW image of 48x64 pixels on the CPU.
    """
    tint = torch.tensor(schedule.tint).view(3, 1, 1)
    pixels = torch.full((3, FRAME_HEIGHT, FRAME_WIDTH), float(frame.brightness))
    top = (FRAME_HEIGHT - MARKER_SIZE) // 2
    left = frame.marker_x
    pixels[:, top : top + MARKER_SIZE, left : left + MARKER_SIZE] = 255.0
    tinted = (pixels * tint).round().clamp(0, 255).to(torch.uint8)
    image = ImageData.from_tensor(tinted, image_id=image_id)

    return image


def media_timestamp(pts_ms: int, *, clock: str = RIG_CLOCK) -> Timestamp:
    """Build an exact millisecond timestamp on a named media clock.

    Args:
        pts_ms: Position in milliseconds.
        clock: Media clock identity.

    Returns:
        Timestamp with a 1/1000 s time base.
    """
    timestamp = Timestamp(ticks=pts_ms, time_base=Fraction(1, 1000), clock_id=clock)

    return timestamp
