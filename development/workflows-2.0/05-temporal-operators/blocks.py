"""Ordinary demo blocks. None of them reads or assembles temporal metadata.

The engine attaches contexts according to each output's declared policy; the
blocks only compute values from plain inputs or delivered ``Batch`` groups.
"""

import torch
from pydantic import Field, StrictInt
from roboflow_workflows.execution_engine.v2.blocks.image_data import ImageData
from roboflow_workflows.execution_engine.v2.blocks.kinds import IMAGE_KIND
from roboflow_workflows.execution_engine.v2.data import Batch
from roboflow_workflows.execution_engine.v2.declaration import (
    Block,
    BlockParams,
    Group,
    Output,
    Ref,
)
from roboflow_workflows.execution_engine.v2.kinds import (
    FLOAT_KIND,
    INTEGER_KIND,
    STRING_KIND,
)


class CelsiusToFahrenheit(Block):
    """Convert one reading; used on the unaligned sensor branch."""

    type = "temporal_demo/celsius_to_fahrenheit"
    outputs = {"fahrenheit": Output(FLOAT_KIND)}

    class Params(BlockParams):
        celsius: Ref(FLOAT_KIND) = Field(description="Temperature in Celsius.")

    def run(self, *, celsius: float) -> dict:
        """Convert a scalar temperature.

        Args:
            celsius: One sensor reading.

        Returns:
            The temperature in Fahrenheit.
        """
        result = {"fahrenheit": celsius * 9 / 5 + 32}

        return result


class DescribePair(Block):
    """Describe one aligned reading/frame pair delivered in the same pulse."""

    type = "temporal_demo/describe_pair"
    outputs = {"summary": Output(STRING_KIND)}

    class Params(BlockParams):
        celsius: Ref(FLOAT_KIND) = Field(description="Aligned sensor reading.")
        image: Ref(IMAGE_KIND) = Field(description="Aligned camera frame.")

    def run(self, *, celsius: float, image: ImageData) -> dict:
        """Combine two values the align operator declared corresponding.

        Args:
            celsius: Sensor reading.
            image: Frame chosen by the operator for this reading.

        Returns:
            A short text naming both values.
        """
        mean = image.tensor_image.float().mean().item()
        result = {"summary": f"{celsius:.1f} C with {image.image_id} (mean {mean:.0f})"}

        return result


class GreyCard(Block):
    """Render a flat grey reference in the geometry of one frame."""

    type = "temporal_demo/grey_card"
    outputs = {"image": Output(IMAGE_KIND, source="image")}

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(description="Frame whose geometry is kept.")
        value: StrictInt = Field(ge=0, le=255, description="Grey level, 0..255.")

    def run(self, *, image: ImageData, value: int) -> dict:
        """Create the reference without modifying the input tensor.

        Args:
            image: Frame of one camera.
            value: Grey level of every pixel.

        Returns:
            A grey image with the frame's identity and provenance.
        """
        pixels = torch.full_like(image.tensor_image, value)
        result = {"image": image.with_pixels(pixels)}

        return result


class ClipEnds(Block):
    """Return a clip's first and last frames and its length.

    One invocation per parent; three outputs with three declared policies.
    """

    type = "temporal_demo/clip_ends"
    outputs = {
        "first": Output(IMAGE_KIND, source="frames", context_policy="first"),
        "last": Output(IMAGE_KIND, source="frames", context_policy="last"),
        "count": Output(INTEGER_KIND, source="frames"),
    }

    class Params(BlockParams):
        frames: Group(IMAGE_KIND) = Field(description="Frames of one parent.")

    def run(self, *, frames: Batch[ImageData]) -> dict:
        """Pick the ends of the delivered group.

        Args:
            frames: Present frames of one parent, in index order.

        Returns:
            First frame, last frame and the number of frames.
        """
        result = {"first": frames[0], "last": frames[-1], "count": len(frames)}

        return result


class ExposureGuard(Block):
    """Fail loudly when any frame of a clip is darker than a minimum.

    Used by the ``failure`` example to show how a step error inside an
    operator-produced pulse is attributed.
    """

    type = "temporal_demo/exposure_guard"
    outputs = {"darkest": Output(FLOAT_KIND, source="frames")}

    class Params(BlockParams):
        frames: Group(IMAGE_KIND) = Field(description="Frames of one parent.")
        minimum: float = Field(ge=0, le=255, description="Lowest allowed frame mean.")

    def run(self, *, frames: Batch[ImageData], minimum: float) -> dict:
        """Return the darkest frame mean, or raise when it is too dark.

        Args:
            frames: Present frames of one parent.
            minimum: Lowest acceptable mean pixel value of any frame.

        Returns:
            The lowest mean pixel value in the clip.

        Raises:
            ValueError: When a frame is darker than ``minimum``.
        """
        darkest = min(frame.tensor_image.float().mean().item() for frame in frames)
        if darkest < minimum:
            raise ValueError(
                f"clip is underexposed: darkest mean {darkest:.0f} < {minimum:.0f}"
            )

        result = {"darkest": darkest}

        return result
