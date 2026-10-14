"""Choose the frame of a time window that looks most like a reference.

Canonical type: ``roboflow_core/best_frame@v1``.
"""

from typing import Any, Dict
import torch
from pydantic import Field
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import FLOAT_KIND
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Group, Output, Ref, Selected


def _mean_difference(frame: ImageData, reference: ImageData) -> float:
    # A one-channel image broadcasts over an RGB one.
    resized = frame.resize(reference.size_hw, interpolation="nearest")
    frame_pixels = resized.tensor_image.to(reference.device, dtype=torch.float32)
    reference_pixels = reference.tensor_image.to(dtype=torch.float32)
    difference = (frame_pixels - reference_pixels).abs().mean().item()

    return difference


class BestFrameBlock(Block):
    """Choose the frame of a time window that looks most like a reference.

    Similarity is the mean absolute pixel difference after resizing the frame
    to the reference size (nearest neighbour), on the reference's device; the
    lowest difference wins and ties keep the earlier frame. ``frame`` is the
    chosen frame object itself, and both outputs carry its timestamp.

    An empty window chooses nothing: both outputs are ``None`` with no
    timestamp.
    """

    type = "roboflow_core/best_frame@v1"
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
