"""SIFT keypoints and descriptors of an image: ``roboflow_core/sift@v1``.

Port of V1 ``roboflow_core/sift@v1``. As V1::

    gray                    = OpenCV RGB -> gray
    keypoints, descriptors  = cv2.SIFT_create().detectAndCompute(gray)
    image                   = gray as RGB with every keypoint drawn by OpenCV

Keypoints are one ``(N, 7)`` ``float32`` row per keypoint, columns in this
order (``KEYPOINT_COLUMNS``)::

    x, y, size, angle, response, octave, class_id

``x`` and ``y`` are pixels of the input image, in its own frame. ``octave`` is
OpenCV's packed octave/layer integer; it stays below ``2**24`` and is exact in
``float32``. Descriptors are ``(N, 128)`` ``float32``. An image without
keypoints gives ``(0, 7)`` and ``(0, 128)``.

CPU boundary: OpenCV converts to gray, detects, describes and draws on the
host. The image goes to the host once and the drawn ``3 x H x W`` image comes
back to the image's device. Keypoints
and descriptors are host results and stay CPU tensors.

Differences from V1: keypoints are a tensor instead of a list of dicts,
descriptors a tensor instead of an array (``(0, 128)`` instead of ``None``),
the input pixels are never written (V1 draws into the input array), and a
1-channel image is used as the gray plane (V1: OpenCV channel error).
"""

from typing import Any, Dict, NamedTuple, Sequence
import cv2
import numpy as np
import torch
from pydantic import Field
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.predictions import TENSOR_KIND
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref


KEYPOINT_COLUMNS = ("x", "y", "size", "angle", "response", "octave", "class_id")

DESCRIPTOR_SIZE = 128

class SiftFeatures(NamedTuple):
    """SIFT result of one image.

    Args:
        gray: ``(H, W)`` ``uint8`` host gray plane the features come from.
        cv_keypoints: OpenCV keypoints, for OpenCV matching and drawing.
        keypoints: ``(N, 7)`` ``float32`` CPU tensor, ``KEYPOINT_COLUMNS``.
        descriptors: ``(N, 128)`` ``float32`` CPU tensor.
    """

    gray: np.ndarray
    cv_keypoints: Sequence[cv2.KeyPoint]
    keypoints: torch.Tensor
    descriptors: torch.Tensor

class SiftBlock(Block):
    """Detect SIFT keypoints, describe them and draw them.

    Example: feed ``descriptors`` of a reference image and of a camera frame
    to ``roboflow_core/sift_comparison@v2`` to test whether the frame shows the same
    textured object.
    """

    type = "roboflow_core/sift@v1"
    outputs = {
        "image": Output(
            IMAGE_KIND,
            source="image",
            description="Gray input as RGB with the keypoints drawn.",
        ),
        "keypoints": Output(
            TENSOR_KIND,
            source="image",
            description=(
                "(N, 7) float32 CPU tensor: x, y, size, angle, response, "
                "octave, class_id."
            ),
        ),
        "descriptors": Output(
            TENSOR_KIND,
            source="image",
            description="(N, 128) float32 CPU tensor of SIFT descriptors.",
        ),
    }

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(description="Image to describe.")

    def run(self, *, image: ImageData) -> Dict[str, Any]:
        """Detect and draw the SIFT keypoints of ``image``.

        Args:
            image: Image to describe; not modified.

        Returns:
            ``image``, ``keypoints`` and ``descriptors``.
        """
        features = detect_sift(image)

        result = {
            "image": draw_keypoints(image, features),
            "keypoints": features.keypoints,
            "descriptors": features.descriptors,
        }

        return result

def detect_sift(image: ImageData) -> SiftFeatures:
    """Run OpenCV SIFT on the gray plane of ``image``.

    Args:
        image: 1- or 3-channel image on any device.

    Returns:
        The features; empty tensors when OpenCV finds no keypoint.
    """
    gray = host_gray(image)
    cv_keypoints, descriptors = cv2.SIFT_create().detectAndCompute(gray, None)
    keypoints = torch.tensor(
        [
            (
                *point.pt,
                point.size,
                point.angle,
                point.response,
                point.octave,
                point.class_id,
            )
            for point in cv_keypoints
        ],
        dtype=torch.float32,
    ).reshape(-1, len(KEYPOINT_COLUMNS))
    if descriptors is None:
        descriptors = np.zeros((0, DESCRIPTOR_SIZE), dtype=np.float32)

    features = SiftFeatures(
        gray=gray,
        cv_keypoints=cv_keypoints,
        keypoints=keypoints,
        descriptors=torch.from_numpy(descriptors.astype(np.float32, copy=False)),
    )

    return features

def host_gray(image: ImageData) -> np.ndarray:
    """Return the OpenCV gray plane of ``image`` as a host array.

    Args:
        image: 1- or 3-channel image on any device; one channel is used as is.

    Returns:
        ``(H, W)`` ``uint8`` contiguous array, equal to V1's ``BGR2GRAY`` of
        the same pixels.
    """
    pixels = image.tensor_image.cpu()
    if pixels.shape[0] == 1:
        return np.ascontiguousarray(pixels[0].numpy())

    rgb = np.ascontiguousarray(pixels.permute(1, 2, 0).numpy())

    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

def draw_keypoints(image: ImageData, features: SiftFeatures) -> ImageData:
    """Draw the keypoints on the gray plane as OpenCV does, keeping ``image``'s frame.

    Args:
        image: Image the features were detected on.
        features: Its SIFT features.

    Returns:
        A 3-channel image with ``image``'s identity, on its device.
    """
    drawn_bgr = cv2.drawKeypoints(features.gray, features.cv_keypoints, None)
    drawn = host_rgb_tensor(drawn_bgr).to(image.device)

    return image.with_pixels(drawn)

def host_rgb_tensor(bgr: np.ndarray) -> torch.Tensor:
    """Convert an OpenCV ``(H, W, 3)`` BGR drawing to a ``(3, H, W)`` RGB tensor.

    Args:
        bgr: ``uint8`` host image as OpenCV draws it.

    Returns:
        A CPU tensor that owns its memory.
    """
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    return torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
