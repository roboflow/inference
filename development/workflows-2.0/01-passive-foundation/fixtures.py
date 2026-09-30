"""Locally generated image fixtures for the passive-foundation demo.

No files are downloaded and no camera is used. Three RGB images of different
sizes are drawn with NumPy so that the shared crop configuration
``CROP_REGIONS`` yields ``[2, 0, 1]`` crops:

| Fixture | Size (h x w) | Region 0 ``[40,40,100,100]`` | Region 1 ``[120,10,180,70]`` | Crops |
| ------- | ------------ | ---------------------------- | ---------------------------- | ----- |
| alpha   | 120 x 200    | bright square                | dark square                  | 2     |
| beta    | 32 x 32      | outside the image            | outside the image            | 0     |
| gamma   | 110 x 110    | dark square                  | outside the image            | 1     |

Brightness inside the regions is chosen so that a ``v2/has_brightness`` gate
with ``minimum`` of ``BRIGHTNESS_MINIMUM`` keeps only alpha's first crop:
alpha becomes a partially filtered group, gamma an all-filtered group and beta
stays a genuinely empty group.
"""

from dataclasses import dataclass
from fractions import Fraction
from typing import List, Tuple

import numpy as np
from roboflow_workflows.execution_engine.v2.data import (
    EntryMetadata,
    SampleContext,
    TemporalContext,
    Timestamp,
)

CROP_REGIONS: List[List[int]] = [[40, 40, 100, 100], [120, 10, 180, 70]]
EXPECTED_CROP_COUNTS: Tuple[int, ...] = (2, 0, 1)
BRIGHTNESS_MINIMUM = 100
DEMO_CLOCK_ID = "demo-clock"


@dataclass(frozen=True)
class Fixture:
    """One generated demo image with its declared source identity."""

    name: str
    image: np.ndarray
    expected_crops: int


def make_fixtures() -> List[Fixture]:
    """Generate the three demo images.

    Returns:
        Fixtures ``alpha``, ``beta`` and ``gamma`` in input order.
    """
    fixtures = [
        Fixture(name="alpha", image=_make_alpha(), expected_crops=2),
        Fixture(name="beta", image=_make_beta(), expected_crops=0),
        Fixture(name="gamma", image=_make_gamma(), expected_crops=1),
    ]

    return fixtures


def make_input_metadata(fixtures: List[Fixture]) -> EntryMetadata:
    """Build indexed source and temporal context for the ``images`` input.

    Every parent index ``(n,)`` gets a ``SampleContext`` naming its fixture.
    Only ``alpha`` and ``gamma`` get a ``TemporalContext``; ``beta`` has none,
    which lets the demo show inheritance versus absence. There is no time axis:
    timing metadata exists without temporal grouping.

    Args:
        fixtures: Generated fixtures in input order.

    Returns:
        Metadata keyed by one-component parent indices.
    """
    sample = {}
    temporal = {}
    for position, fixture in enumerate(fixtures):
        height, width = fixture.image.shape[:2]
        sample[(position,)] = SampleContext(
            source_id=f"fixture:{fixture.name}",
            source_type="static",
            source_metadata={"height": int(height), "width": int(width)},
        )
        if fixture.name != "beta":
            temporal[(position,)] = TemporalContext(
                observed_coverage=Timestamp(
                    ticks=1_000 * (position + 1),
                    time_base=Fraction(1, 1_000),
                    clock_id=DEMO_CLOCK_ID,
                )
            )

    metadata = EntryMetadata(sample=sample, temporal=temporal)

    return metadata


def _gradient(
    height: int, width: int, *, start: Tuple[int, int, int], end: Tuple[int, int, int]
) -> np.ndarray:
    ramp = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :, None]
    start_color = np.array(start, dtype=np.float32)[None, None, :]
    end_color = np.array(end, dtype=np.float32)[None, None, :]
    row = start_color + (end_color - start_color) * ramp
    image = np.repeat(row, height, axis=0)
    image_uint8 = np.clip(image, 0, 255).astype(np.uint8)

    return image_uint8


def _paint(image: np.ndarray, region: List[int], color: Tuple[int, int, int]) -> None:
    x0, y0, x1, y1 = region
    image[y0:y1, x0:x1] = np.array(color, dtype=np.uint8)


def _make_alpha() -> np.ndarray:
    image = _gradient(120, 200, start=(30, 60, 160), end=(90, 140, 220))
    _paint(image, CROP_REGIONS[0], (235, 225, 90))
    _paint(image, CROP_REGIONS[1], (25, 20, 35))
    _paint(image, [45, 45, 55, 95], (200, 40, 40))

    return image


def _make_beta() -> np.ndarray:
    tile = np.indices((32, 32)).sum(axis=0) // 8 % 2
    image = np.where(tile[..., None] == 1, (200, 60, 60), (120, 120, 120))
    image_uint8 = image.astype(np.uint8)

    return image_uint8


def _make_gamma() -> np.ndarray:
    image = _gradient(110, 110, start=(40, 150, 70), end=(180, 240, 160))
    _paint(image, CROP_REGIONS[0], (30, 35, 30))
    _paint(image, [60, 60, 80, 80], (70, 70, 70))

    return image
