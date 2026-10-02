"""Native V2 blocks that choose members of a group.

These are ordinary blocks: they receive a ``Batch`` for a ``Group`` field and
return which members they chose. The engine takes each result's source and
temporal context from the chosen member, so a chosen frame keeps its own
timestamp. Blocks never build contexts or layouts themselves.

* ``v2/best_frame``: frames of one time window and the parent's reference
  image -> the ``frame`` most similar to the reference and its ``difference``.
  ``[N, T]`` frames with an ``[N]`` reference give ``[N]`` results; each
  parent may choose a different time position.
* ``v2/top_k_brightest``: a group of images -> its ``k`` brightest images,
  brightest first, as a new collection ``[..., ranked]``, even when ``k`` is 1.

The choice is written with the member's full logical index, taken from the
delivered batch::

    best = min(range(len(frames)), key=differences.__getitem__)
    return {"frame": Selected(frames.indices[best])}
"""

from typing import Any, Dict, List

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
    Selected,
    Selection,
)
from roboflow_workflows.execution_engine.v2.kinds import FLOAT_KIND, INTEGER_KIND


class BestFrameBlock(Block):
    """Choose the frame of a time window that looks most like a reference.

    Similarity is the mean absolute pixel difference after resizing the frame
    to the reference size (nearest neighbour), on the reference's device; the
    lowest difference wins and ties keep the earlier frame. ``frame`` is the
    chosen frame object itself, and both outputs carry its timestamp.

    An empty window chooses nothing: both outputs are ``None`` with no
    timestamp.
    """

    type = "v2/best_frame"
    outputs = {
        "frame": Output(
            IMAGE_KIND,
            source="frames",
            context_policy="selected",
            description="The chosen frame, unchanged.",
        ),
        "difference": Output(
            FLOAT_KIND,
            source="frames",
            context_policy="selected",
            description="Mean absolute pixel difference of the chosen frame, 0..255.",
        ),
    }

    class Params(BlockParams):
        frames: Group(IMAGE_KIND, temporal=True) = Field(
            description="Frames of one time window; the last axis must be time."
        )
        reference: Ref(IMAGE_KIND) = Field(
            description="Reference image of the same parent sample as the frames."
        )

    def run(self, *, frames: Batch[ImageData], reference: ImageData) -> Dict[str, Any]:
        """Compare every frame with ``reference`` and choose the closest one.

        Args:
            frames: Frames of one window, possibly none.
            reference: Image the frames are compared with.

        Returns:
            ``frame``: ``Selected`` of the closest frame. ``difference``: the
            same choice with its difference as the value. Both ``None`` for an
            empty window.
        """
        if not len(frames):
            result = {"frame": None, "difference": None}
            return result

        differences = [_mean_difference(frame, reference) for frame in frames]
        best = min(range(len(frames)), key=differences.__getitem__)
        chosen = frames.indices[best]
        result = {
            "frame": Selected(chosen),
            "difference": Selected(chosen, value=differences[best]),
        }

        return result


class TopKBrightestBlock(Block):
    """Choose the ``k`` brightest images of a group, brightest first.

    Brightness is the mean pixel value over all channels, computed on each
    image's device. Ties keep group order. A group with fewer than ``k``
    images yields all of them. The result is a new collection along the
    ``ranked`` axis: it keeps each chosen image object and its own context,
    and stays a collection even when it holds one image. It is not a
    reduction, so it cannot be collected over time again directly.
    """

    type = "v2/top_k_brightest"
    outputs = {
        "images": Output(
            IMAGE_KIND,
            expand="ranked",
            source="images",
            context_policy="selected",
            description="Chosen images, brightest first.",
        ),
        "brightness": Output(
            FLOAT_KIND,
            expand="ranked",
            source="images",
            context_policy="selected",
            description="Mean pixel value of each chosen image, 0..255.",
        ),
    }

    class Params(BlockParams):
        images: Group(IMAGE_KIND) = Field(
            description="Images to rank: the children of one parent, possibly none."
        )
        k: StrictInt | Ref(INTEGER_KIND) = Field(
            default=1, ge=1, description="Number of images to keep."
        )

    def run(self, *, images: Batch[ImageData], k: int) -> Dict[str, Any]:
        """Rank ``images`` by brightness and keep the first ``k``.

        Args:
            images: Images of one parent, possibly none.
            k: Number of images to keep.

        Returns:
            ``images``: ``Selection`` of the chosen images. ``brightness``: the
            same selection with each image's mean pixel value.
        """
        brightness = [_mean_brightness(image) for image in images]
        ranked = sorted(range(len(images)), key=lambda position: -brightness[position])
        kept: List[int] = ranked[:k]
        chosen = [images.indices[position] for position in kept]
        result = {
            "images": Selection(chosen),
            "brightness": Selection(
                chosen, values=[brightness[position] for position in kept]
            ),
        }

        return result


def _mean_brightness(image: ImageData) -> float:
    brightness = image.tensor_image.mean(dtype=torch.float32).item()

    return brightness


def _mean_difference(frame: ImageData, reference: ImageData) -> float:
    # A one-channel image broadcasts over an RGB one.
    resized = frame.resize(reference.size_hw, interpolation="nearest")
    frame_pixels = resized.tensor_image.to(reference.device, dtype=torch.float32)
    reference_pixels = reference.tensor_image.to(dtype=torch.float32)
    difference = (frame_pixels - reference_pixels).abs().mean().item()

    return difference
