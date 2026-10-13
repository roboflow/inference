"""Compare two images or their SIFT descriptors: ``v2/sift_comparison@v2``.

Port of V1 ``roboflow_core/sift_comparison@v2``. As V1::

    descriptors_i  = input_i if it is a descriptor tensor, else SIFT of image i
    if either side has fewer than 2 descriptors:  good_matches_count = 0
    else:
        for each descriptor of input_1, its two nearest in input_2 (m, n)
        good  = matches with m.distance < ratio_threshold * n.distance
    images_match = good_matches_count >= good_matches_threshold

``BFMatcher`` is exact L2 search; ``FlannBasedMatcher`` is OpenCV's
approximate kd-tree search with V1's settings (5 trees, 50 checks).

Keypoints and descriptors use the layout of ``v2/sift@v1``. A side given as
descriptors has no keypoints and no visualization (``None``), as in V1.

Visualizations (``visualize``): each image side gets its keypoints drawn,
keeping its frame. ``visualization_matches`` draws both images side by side
with the good matches (OpenCV ``drawMatchesKnn``, V1 colors) when both
inputs are images with at least 2 descriptors each; it is a new composite
image whose two tiles map back to the inputs.

CPU boundary: OpenCV matches and draws on the host; descriptor tensors on
another device are copied to the host for matching.

Differences from V1: keypoints and descriptors are tensors; a descriptor input
must be a ``(N, 128)`` floating tensor (V1 also matches ``float32``
descriptors of another width, e.g. ``(N, 64)``, and, with ``BFMatcher``,
``uint8`` descriptors; V1's ``FlannBasedMatcher`` fails inside OpenCV on
``uint8``); keypoint visualizations are images even when a side has
too few descriptors (V1 returns raw arrays there); ``visualization_matches`` is
a composite of both inputs (V1 labels it with ``input_1``'s frame, whose size
differs).
"""

import re
from typing import Any, Dict, List, Literal, Optional, Tuple

import cv2
import numpy as np
import torch
from pydantic import Field, StrictInt, field_validator
from roboflow_workflows.v2.blocks.geometry.sift import (
    DESCRIPTOR_SIZE,
    SiftFeatures,
    detect_sift,
    draw_keypoints,
)
from roboflow_workflows.v2.data.errors import ContractError
from roboflow_workflows.v2.data.image import CompositeSource, ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import (
    BOOLEAN_KIND,
    FLOAT_KIND,
    INTEGER_KIND,
    STRING_KIND,
)
from roboflow_workflows.v2.data.predictions import TENSOR_KIND
from roboflow_workflows.v2.engine.declaration import (
    DATA_SELECTOR_PATTERN,
    Block,
    BlockParams,
    Output,
    Ref,
)

MATCHERS = ("FlannBasedMatcher", "BFMatcher")
FLANN_INDEX_PARAMS = {"algorithm": 1, "trees": 5}  # 1 = kd-tree
FLANN_SEARCH_PARAMS = {"checks": 50}
# V1 colors, given as RGB because the canvas is RGB: green matches, red points.
FLANN_DRAW_PARAMS = {
    "matchColor": (0, 255, 0),
    "singlePointColor": (255, 0, 0),
    "flags": cv2.DrawMatchesFlags_DEFAULT,
}
BF_DRAW_PARAMS = {"flags": cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS}


class SiftComparisonBlock(Block):
    """Count good SIFT matches between two images and decide if they match.

    Example: a product photo against a shelf frame with the default ratio 0.7
    and threshold 50; ``images_match`` is true when at least 50 descriptors of
    the photo have a clearly closest partner in the frame.
    """

    type = "v2/sift_comparison@v2"
    outputs = {
        "images_match": Output(
            BOOLEAN_KIND, description="good_matches_count >= good_matches_threshold."
        ),
        "good_matches_count": Output(
            INTEGER_KIND, description="Matches that pass the ratio test."
        ),
        "keypoints_1": Output(
            TENSOR_KIND,
            source="input_1",
            description="(N, 7) keypoints of image 1; None for descriptors.",
        ),
        "descriptors_1": Output(
            TENSOR_KIND,
            source="input_1",
            description="(N, 128) descriptors of input 1.",
        ),
        "keypoints_2": Output(
            TENSOR_KIND,
            source="input_2",
            description="(N, 7) keypoints of image 2; None for descriptors.",
        ),
        "descriptors_2": Output(
            TENSOR_KIND,
            source="input_2",
            description="(N, 128) descriptors of input 2.",
        ),
        "visualization_1": Output(
            IMAGE_KIND,
            source="input_1",
            description="Image 1 with keypoints, when visualize is on.",
        ),
        "visualization_2": Output(
            IMAGE_KIND,
            source="input_2",
            description="Image 2 with keypoints, when visualize is on.",
        ),
        "visualization_matches": Output(
            IMAGE_KIND,
            description="Both images side by side with the good matches.",
        ),
    }

    class Params(BlockParams):
        input_1: Ref(IMAGE_KIND, TENSOR_KIND) = Field(
            description="Image, or (N, 128) SIFT descriptors, to compare."
        )
        input_2: Ref(IMAGE_KIND, TENSOR_KIND) = Field(
            description="Image, or (N, 128) SIFT descriptors, to compare against."
        )
        good_matches_threshold: StrictInt | Ref(INTEGER_KIND) = Field(
            default=50,
            ge=1,
            description="Good matches needed for images_match.",
        )
        ratio_threshold: float | Ref(FLOAT_KIND) = Field(
            default=0.7,
            ge=0,
            le=1,
            description="Lowe's ratio test: keep m if m < ratio * n.",
        )
        matcher: Literal["FlannBasedMatcher", "BFMatcher"] | Ref(STRING_KIND) = Field(
            default="FlannBasedMatcher",
            description="FlannBasedMatcher (approximate) or BFMatcher (exact).",
            examples=["FlannBasedMatcher", "BFMatcher", "$inputs.matcher"],
        )
        visualize: bool | Ref(BOOLEAN_KIND) = Field(
            default=False,
            description="Draw keypoints and matches of image inputs.",
        )

        @field_validator("matcher", mode="before")
        @classmethod
        def _matcher_is_known(cls, value: Any) -> Any:
            # The engine reruns this check on a selected matcher name.
            if isinstance(value, str) and (
                value not in MATCHERS and not re.fullmatch(DATA_SELECTOR_PATTERN, value)
            ):
                raise ValueError(
                    f"unknown matcher {value!r}; use one of {list(MATCHERS)}"
                )

            return value

    def run(
        self,
        *,
        input_1: Any,
        input_2: Any,
        good_matches_threshold: int,
        ratio_threshold: float,
        matcher: str,
        visualize: bool,
    ) -> Dict[str, Any]:
        """Match the descriptors of ``input_1`` against those of ``input_2``.

        Args:
            input_1: ``ImageData`` or ``(N, 128)`` descriptor tensor.
            input_2: ``ImageData`` or ``(N, 128)`` descriptor tensor.
            good_matches_threshold: Good matches needed for a match.
            ratio_threshold: Lowe's ratio.
            matcher: ``"FlannBasedMatcher"`` or ``"BFMatcher"``.
            visualize: Draw keypoints and matches of image inputs.

        Returns:
            The nine outputs; unavailable ones are ``None``.

        Raises:
            ContractError: When a descriptor tensor is not ``(N, 128)`` floating.
        """
        features_1, descriptors_1 = _features_and_descriptors(input_1, name="input_1")
        features_2, descriptors_2 = _features_and_descriptors(input_2, name="input_2")
        good_matches = []
        compared = len(descriptors_1) >= 2 and len(descriptors_2) >= 2
        if compared:
            good_matches = _ratio_test_matches(
                descriptors_1, descriptors_2, matcher=matcher, ratio=ratio_threshold
            )

        result = {
            "images_match": len(good_matches) >= good_matches_threshold,
            "good_matches_count": len(good_matches),
            "keypoints_1": _keypoints(features_1),
            "descriptors_1": descriptors_1,
            "keypoints_2": _keypoints(features_2),
            "descriptors_2": descriptors_2,
            "visualization_1": None,
            "visualization_2": None,
            "visualization_matches": None,
        }
        if visualize:
            if features_1 is not None:
                result["visualization_1"] = draw_keypoints(input_1, features_1)
            if features_2 is not None:
                result["visualization_2"] = draw_keypoints(input_2, features_2)
            if compared and features_1 is not None and features_2 is not None:
                result["visualization_matches"] = _drawn_matches(
                    (input_1, features_1), (input_2, features_2), good_matches, matcher
                )

        return result


def _features_and_descriptors(
    value: Any, *, name: str
) -> Tuple[Optional[SiftFeatures], torch.Tensor]:
    if isinstance(value, ImageData):
        features = detect_sift(value)
        return features, features.descriptors

    if value.dim() != 2 or value.shape[1] != DESCRIPTOR_SIZE:
        raise ContractError(
            f"v2/sift_comparison@v2: {name} descriptors must have shape "
            f"(N, {DESCRIPTOR_SIZE}), got {tuple(value.shape)}"
        )
    if not value.is_floating_point():
        raise ContractError(
            f"v2/sift_comparison@v2: {name} descriptors must be floating, "
            f"got {value.dtype}"
        )

    return None, value


def _keypoints(features: Optional[SiftFeatures]) -> Optional[torch.Tensor]:
    return None if features is None else features.keypoints


def _ratio_test_matches(
    descriptors_1: torch.Tensor,
    descriptors_2: torch.Tensor,
    *,
    matcher: str,
    ratio: float,
) -> List[cv2.DMatch]:
    # Best matches of input_1 descriptors that are clearly closer than the
    # second best (Lowe's ratio test, strict as V1).
    if matcher == "BFMatcher":
        cv_matcher = cv2.BFMatcher(cv2.NORM_L2)
    else:
        cv_matcher = cv2.FlannBasedMatcher(FLANN_INDEX_PARAMS, FLANN_SEARCH_PARAMS)
    pairs = cv_matcher.knnMatch(
        _host_float32(descriptors_1), _host_float32(descriptors_2), k=2
    )

    # FLANN can return fewer than two neighbours for a query; such a query has
    # no second distance to compare with and is skipped.
    good = [
        pair[0]
        for pair in pairs
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance
    ]

    return good


def _host_float32(descriptors: torch.Tensor) -> np.ndarray:
    return np.ascontiguousarray(
        descriptors.detach().to(device="cpu", dtype=torch.float32).numpy()
    )


def _drawn_matches(
    side_1: Tuple[ImageData, SiftFeatures],
    side_2: Tuple[ImageData, SiftFeatures],
    good_matches: List[cv2.DMatch],
    matcher: str,
) -> ImageData:
    # Both images side by side, image 1 at the left, as one composite canvas.
    (image_1, features_1), (image_2, features_2) = side_1, side_2
    draw_params = BF_DRAW_PARAMS if matcher == "BFMatcher" else FLANN_DRAW_PARAMS
    canvas = cv2.drawMatchesKnn(
        _host_pixels(image_1),
        features_1.cv_keypoints,
        _host_pixels(image_2),
        features_2.cv_keypoints,
        [[match] for match in good_matches],
        None,
        **draw_params,
    )
    sources = [
        CompositeSource.place(
            image_1, index=(0,), canvas_xyxy=(0, 0, image_1.width, image_1.height)
        ),
        CompositeSource.place(
            image_2,
            index=(1,),
            canvas_xyxy=(
                image_1.width,
                0,
                image_1.width + image_2.width,
                image_2.height,
            ),
        ),
    ]
    pixels = torch.from_numpy(canvas).permute(2, 0, 1).contiguous()

    return ImageData.composite(pixels.to(image_1.device), sources=sources)


def _host_pixels(image: ImageData) -> np.ndarray:
    # (H, W, 3) RGB or (H, W) gray host array, as OpenCV draws on it.
    pixels = image.tensor_image.cpu()
    if pixels.shape[0] == 1:
        return np.ascontiguousarray(pixels[0].numpy())

    return np.ascontiguousarray(pixels.permute(1, 2, 0).numpy())
