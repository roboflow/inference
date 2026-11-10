"""``roboflow_core/image_copy@v1``: an owned, ready copy of one image.

Place it before anything that keeps images beyond the current pulse, such as
a window collecting frames, so the camera or decoder buffer the image came
from is released::

    source.image ─▶ roboflow_core/image_copy@v1 (device="cpu") ─▶ window(collect=image)
"""

from typing import Any, Dict, Literal
from pydantic import Field
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref


class ImageCopyBlock(Block):
    """Copy one image into memory it owns, keeping identity and metadata.

    The copy has the same ``image_id``, mappings, video metadata and
    composite sources, and pixels that share no memory with the input. On
    CUDA it is complete when the step returns: the step waits for its own
    copy on the current stream, which costs one host wait per image (see
    ``ImageData.copy``). ``device="cpu"`` also moves the pixels to host memory.
    """

    type = "roboflow_core/image_copy@v1"
    outputs = {
        "image": Output(
            IMAGE_KIND, source="image", description="The owned copy of the image."
        )
    }
    prunable = True

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(description="Image to copy.")
        device: Literal["same", "cpu"] = Field(
            default="same",
            description="'same' keeps the image's device; 'cpu' copies to host memory.",
            examples=["same", "cpu"],
        )

    def run(self, *, image: ImageData, device: str) -> Dict[str, Any]:
        """Copy ``image``.

        Args:
            image: Image to copy.
            device: ``"same"`` or ``"cpu"``.

        Returns:
            ``image``: the owned copy.
        """
        copied = image.copy(device=None if device == "same" else "cpu")
        result = {"image": copied}

        return result
