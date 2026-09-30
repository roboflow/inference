"""Deterministic CPU images and clocks for the finite source examples."""

from fractions import Fraction

import torch
from roboflow_workflows.execution_engine.v2.blocks.image_data import ImageData
from roboflow_workflows.execution_engine.v2.data import TemporalContext, Timestamp

FRAME_BRIGHTNESS = (200, 20, 180, 30, 160)
FRAME_PTS_MS = (0, 40, 80, 120, 160)


def make_frame(index: int) -> ImageData:
    """Create one small tensor image with identifiable root geometry.

    Args:
        index: Position in the finite five-frame fixture.

    Returns:
        RGB uint8 CHW image; alternating bright/dark frames exercise gates.
    """
    pixels = torch.full((3, 12, 16), FRAME_BRIGHTNESS[index], dtype=torch.uint8)
    image = ImageData.from_tensor(pixels, image_id=f"frame-{index}")

    return image


def make_timing(*, pts_ms: int, observed_ms: int, media_clock: str) -> TemporalContext:
    """Attach exact fixture timestamps on independent media and observation clocks.

    Args:
        pts_ms: Position on the named media clock, in milliseconds.
        observed_ms: Fixture acquisition time on the observation clock.
        media_clock: Clock identity; equal positions do not imply alignment.

    Returns:
        Temporal context carried in buffer metadata, without a temporal axis.
    """
    timing = TemporalContext(
        observed_coverage=Timestamp(
            ticks=observed_ms,
            time_base=Fraction(1, 1000),
            clock_id="fixture-observation",
        ),
        media_coverage=Timestamp(
            ticks=pts_ms,
            time_base=Fraction(1, 1000),
            clock_id=media_clock,
        ),
    )

    return timing
