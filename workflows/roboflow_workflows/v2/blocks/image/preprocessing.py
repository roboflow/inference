"""Resize, rotate or flip an image: ``v2/image_preprocessing@v1``.

Port of V1 ``roboflow_core/image_preprocessing@v1``. All pixel work stays on
the image's device.

=======  ===============================================  =====================
task     pixels                                           geometry of the result
=======  ===============================================  =====================
resize   OpenCV ``INTER_AREA``, as V1, to ``width`` x     child of the input:
         ``height``; one of them null keeps the aspect    scale mapping, so
         ratio (``int`` truncation, as V1); both null:    predictions map back
         input
rotate   counter-clockwise by ``rotation_degrees``, on a  new root frame
         canvas grown to hold the whole image, black      (see below)
         corners; multiples of 90 are exact
flip     vertical, horizontal or both                     new root frame
=======  ===============================================  =====================

Why a new root frame: an image's provenance (``FrameMapping``) can only scale
and shift coordinates. A rotation or a mirror cannot be written that way, so
the result does not claim a mapping it does not have. It keeps source and
time (``video_metadata``) and the detection row it was cropped for
(``parent_row``); predictions made on it stay in the rotated or flipped frame
when converted to "root" coordinates. A rotation by a multiple of 360 degrees
returns the input unchanged, with its provenance.

Differences from V1, which kept the input's provenance for every task, so
predictions on a rotated or flipped image mapped back to wrong places:

* Resize pixels follow ``ImageData.resize(interpolation="area")``: integer
  shrink and any enlargement equal OpenCV exactly; a fractional shrink may
  differ by 1 level where the area mean is an exact ``.5``.
* Multiples of 90 degrees use an exact quarter turn. V1's OpenCV warp moved
  even-sized images by one pixel and averaged neighbours of odd-sized ones.
* Other angles use bilinear sampling of the same transform OpenCV builds
  (centre ``(W // 2, H // 2)``, canvas ``int(H|sin| + W|cos|)`` wide); edge
  pixels may differ slightly from OpenCV's fixed-point arithmetic.
"""

import math
import uuid
from typing import Any, Dict, Literal, Optional

import torch
import torch.nn.functional as F
from pydantic import Field, StrictInt
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import INTEGER_KIND, STRING_KIND
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref

FlipType = Literal["vertical", "horizontal", "both"]

FLIP_DIMENSIONS = {"vertical": (-2,), "horizontal": (-1,), "both": (-2, -1)}


class ImagePreprocessingBlock(Block):
    """Resize, rotate or flip one image; see the module docstring for details.

    Only the parameters of the chosen ``task_type`` are used. Example: a
    100x60 image with ``task_type="resize"``, ``width=50`` and ``height=null``
    becomes 50x30.
    """

    type = "v2/image_preprocessing@v1"
    outputs = {"image": Output(IMAGE_KIND, source="image")}

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(description="Image to transform.")
        task_type: Literal["resize", "rotate", "flip"] = Field(
            description="Transformation to apply: resize, rotate or flip."
        )
        width: Optional[StrictInt] | Ref(INTEGER_KIND) = Field(
            default=640,
            ge=1,
            description=(
                "resize: target width in pixels; null derives it from height, "
                "keeping the aspect ratio."
            ),
        )
        height: Optional[StrictInt] | Ref(INTEGER_KIND) = Field(
            default=640,
            ge=1,
            description=(
                "resize: target height in pixels; null derives it from width, "
                "keeping the aspect ratio."
            ),
        )
        rotation_degrees: StrictInt | Ref(INTEGER_KIND) = Field(
            default=90,
            ge=-360,
            le=360,
            description=(
                "rotate: angle in degrees, positive counter-clockwise (the "
                "OpenCV convention V1 used)."
            ),
        )
        flip_type: FlipType | Ref(STRING_KIND) = Field(
            default="vertical",
            description=(
                "flip: 'vertical' (upside down), 'horizontal' (mirror) or 'both'."
            ),
        )

    def run(
        self,
        *,
        image: ImageData,
        task_type: str,
        width: Optional[int],
        height: Optional[int],
        rotation_degrees: int,
        flip_type: str,
    ) -> Dict[str, Any]:
        """Apply ``task_type`` to ``image``.

        Args:
            image: Image to transform.
            task_type: ``"resize"``, ``"rotate"`` or ``"flip"``.
            width: Target width for resize, or ``None``.
            height: Target height for resize, or ``None``.
            rotation_degrees: Counter-clockwise angle for rotate.
            flip_type: ``"vertical"``, ``"horizontal"`` or ``"both"`` for flip.

        Returns:
            ``image``: the transformed image.

        Raises:
            ValueError: On an unknown flip type, or a resize whose derived side
                would be zero pixels.
        """
        if task_type == "resize":
            transformed = resize_keeping_aspect(image, width=width, height=height)
        elif task_type == "rotate":
            transformed = rotate(image, degrees=rotation_degrees)
        else:
            transformed = flip(image, flip_type=flip_type)

        result = {"image": transformed}

        return result


def resize_keeping_aspect(
    image: ImageData, *, width: Optional[int], height: Optional[int]
) -> ImageData:
    """Resize like V1: a missing side follows the aspect ratio, truncated.

    Args:
        image: Image to resize.
        width: Target width, or ``None`` to derive it from ``height``.
        height: Target height, or ``None`` to derive it from ``width``.

    Returns:
        The resized image, a child of ``image``; ``image`` itself when both
        sides are ``None``.

    Raises:
        ValueError: When a derived side would be zero pixels.
    """
    if width is None and height is None:
        return image

    image_height, image_width = image.size_hw
    if width is None:
        width = int(image_width * (height / image_height))
    elif height is None:
        height = int(image_height * (width / image_width))
    if width < 1 or height < 1:
        raise ValueError(
            f"resizing a {image_width}x{image_height} image keeping its aspect "
            f"ratio gives {width}x{height}; choose a larger size"
        )

    resized = image.resize((height, width), interpolation="area")

    return resized


def rotate(image: ImageData, *, degrees: int) -> ImageData:
    """Rotate counter-clockwise onto a canvas that holds the whole image.

    Args:
        image: Image to rotate.
        degrees: Counter-clockwise angle in degrees.

    Returns:
        ``image`` itself for a multiple of 360 degrees; otherwise a new root
        image with the input's source and time.
    """
    if degrees % 360 == 0:
        return image

    if degrees % 90 == 0:
        pixels = torch.rot90(image.tensor_image, k=degrees // 90 % 4, dims=(-2, -1))
    else:
        pixels = _rotate_bilinear(image.tensor_image, degrees=degrees)

    rotated = _new_root(pixels, like=image, operation="rotate")

    return rotated


def flip(image: ImageData, *, flip_type: str) -> ImageData:
    """Mirror an image.

    Args:
        image: Image to flip.
        flip_type: ``"vertical"``, ``"horizontal"`` or ``"both"``.

    Returns:
        A new root image with the input's source and time.

    Raises:
        ValueError: On an unknown ``flip_type``.
    """
    dimensions = FLIP_DIMENSIONS.get(flip_type)
    if dimensions is None:
        raise ValueError(
            f"flip_type must be one of {list(FLIP_DIMENSIONS)}, got {flip_type!r}"
        )

    pixels = torch.flip(image.tensor_image, dims=dimensions)
    flipped = _new_root(pixels, like=image, operation="flip")

    return flipped


def _new_root(pixels: torch.Tensor, *, like: ImageData, operation: str) -> ImageData:
    root = ImageData.from_tensor(
        pixels,
        image_id=f"{operation}-{uuid.uuid4().hex}",
        video_metadata=like.video_metadata,
    ).with_parent_row(like.parent_row)

    return root


def _rotate_bilinear(pixels: torch.Tensor, *, degrees: int) -> torch.Tensor:
    # OpenCV's getRotationMatrix2D about (W // 2, H // 2), shifted onto the
    # grown canvas, then inverted: each output pixel samples the input there.
    height, width = int(pixels.shape[-2]), int(pixels.shape[-1])
    radians = math.radians(degrees)
    cos, sin = math.cos(radians), math.sin(radians)
    new_width = int(height * abs(sin) + width * abs(cos))
    new_height = int(height * abs(cos) + width * abs(sin))
    center_x, center_y = width // 2, height // 2
    # Forward map: x' = cos*x + sin*y + tx, y' = -sin*x + cos*y + ty.
    tx = (1 - cos) * center_x - sin * center_y + new_width / 2 - center_x
    ty = sin * center_x + (1 - cos) * center_y + new_height / 2 - center_y

    device = pixels.device
    ys, xs = torch.meshgrid(
        torch.arange(new_height, dtype=torch.float32, device=device),
        torch.arange(new_width, dtype=torch.float32, device=device),
        indexing="ij",
    )
    # Inverse of the rotation part is its transpose.
    source_x = cos * (xs - tx) - sin * (ys - ty)
    source_y = sin * (xs - tx) + cos * (ys - ty)
    # Pixel centres at integer coordinates, as OpenCV samples them.
    grid = torch.stack(
        ((2 * source_x + 1) / width - 1, (2 * source_y + 1) / height - 1), dim=-1
    )
    sampled = F.grid_sample(
        pixels.unsqueeze(0).float(),
        grid.unsqueeze(0),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    rotated = sampled.squeeze(0).round().clamp(0, 255).to(torch.uint8)

    return rotated
