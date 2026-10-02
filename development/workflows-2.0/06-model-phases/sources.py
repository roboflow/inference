"""A finite source that replays the pinned repository images as timed frames."""

from fractions import Fraction
from typing import List

from assets import IMAGES, load_image
from pydantic import Field, StrictInt
from roboflow_workflows.execution_engine.v2.blocks.image_data import ImageData
from roboflow_workflows.execution_engine.v2.blocks.kinds import IMAGE_KIND
from roboflow_workflows.execution_engine.v2.data import Timestamp
from roboflow_workflows.execution_engine.v2.sources import (
    Emission,
    Source,
    SourceOutput,
    SourceParams,
)

CLOCK = "still-frames"


class StillFrames(Source):
    """Emit each named image once, ``interval_ms`` apart on its media clock."""

    type = "model_demo/still_frames"
    outputs = {"image": SourceOutput(IMAGE_KIND)}

    class Params(SourceParams):
        images: List[str] = Field(
            min_length=1,
            description=f"Names of pinned images, any of {sorted(IMAGES)}.",
        )
        interval_ms: StrictInt = Field(
            default=40, gt=0, description="Media time between frames."
        )

    def open(self, *, images: List[str], interval_ms: int) -> None:
        """Verify and decode the images before the first read.

        Args:
            images: Pinned image names, in emission order.
            interval_ms: Media time between frames.
        """
        self.frames = [load_image(name) for name in images]
        self.interval_ms = interval_ms
        self.position = 0

    def read(self) -> Emission | None:
        """Return the next frame with its presentation timestamp.

        Returns:
            One image emission, or None after the last image or on stop.
        """
        if self.stop_event.is_set() or self.position == len(self.frames):
            return None

        pts_ms = self.position * self.interval_ms
        original: ImageData = self.frames[self.position]
        self.position += 1
        frame = ImageData.from_tensor(
            original.tensor_image, image_id=f"{original.image_id}@{pts_ms}ms"
        )
        emission = Emission(
            {"image": frame},
            media=Timestamp(ticks=pts_ms, time_base=Fraction(1, 1000), clock_id=CLOCK),
        )

        return emission
