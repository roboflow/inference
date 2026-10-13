"""A painted scene and deterministic fake models for annotation workflows.

The real V2 model blocks load these fakes from a ``ModelCache``; nothing is
downloaded and this is not model inference. Each fake reads exact paint
colors, so a test knows every box, mask and class in advance::

    model id          fake                    reads
    shapes-seg        ShapesInstanceModel     one row per painted object color
    shapes-semantic   ShapesSemanticModel     road paint -> road, car paints -> car
    shapes-keypoints  ShapesKeypointModel     object top/bottom centre per row
    colors            ColorClassifier         most frequent paint: red/blue/yellow

Objects are not rectangles (ellipse, triangle, ring), so a mask differs
from its box. ``test/synthetic_compact_segmentation@v1`` is the real
instance segmentation block with only ``_segmentations`` replaced: it returns
a stride-4 logit grid built from the dense masks. It is a synthetic compact
view for comparing mask_quality values, not a real compact model.
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from roboflow_workflows.v2.blocks import create_catalogue, geometry, models
from roboflow_workflows.v2.blocks.image.dynamic_crop import DynamicCropBlock
from roboflow_workflows.v2.blocks.models.cache import ModelRequest
from roboflow_workflows.v2.blocks.models.common import ModelGroup
from roboflow_workflows.v2.blocks.models.instance_segmentation import (
    InstanceSegmentationBlock,
)
from roboflow_workflows.v2.blocks.visualization import annotations
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.masks import MaskGridGeometry, SegmentationView
from roboflow_workflows.v2.engine.catalogue import Catalogue

from inference_models.models.base.classification import ClassificationPrediction
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.semantic_segmentation import (
    SemanticSegmentationResult,
)

BACKGROUND = (90, 90, 90)
ROAD = (50, 50, 50)
RED, BLUE, YELLOW = (230, 60, 60), (60, 90, 230), (230, 200, 40)
CLASS_NAMES = ("person", "car")
SEMANTIC_CLASS_NAMES = ("background", "road", "car")
COLOR_NAMES = ("red", "blue", "yellow")
STRIDE = 4


@dataclass(frozen=True)
class Shape:
    """One painted object: its color decides class and confidence."""

    kind: str
    xyxy: Tuple[int, int, int, int]
    color: Tuple[int, int, int]


# Paint color -> (class id, confidence). Both cars share class id 1.
OBJECT_COLORS = {RED: (0, 0.9), BLUE: (1, 0.8), YELLOW: (1, 0.7)}

STREET_SIZE = (72, 112)
STREET = (
    Shape("ellipse", (8, 8, 30, 50), RED),
    Shape("triangle", (40, 30, 72, 62), BLUE),
    Shape("ring", (80, 20, 106, 52), YELLOW),
)
ALLEY_SIZE = (48, 64)
ALLEY = (Shape("triangle", (20, 8, 52, 40), BLUE),)


def shape_mask(shape: Shape, size_hw: Tuple[int, int]) -> np.ndarray:
    """Return the ``(H, W)`` bool pixels of one shape."""
    height, width = size_hw
    x0, y0, x1, y1 = shape.xyxy
    ys, xs = np.mgrid[0:height, 0:width]
    inside = (xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)
    if shape.kind == "ellipse":
        cx, cy = (x0 + x1 - 1) / 2, (y0 + y1 - 1) / 2
        rx, ry = (x1 - x0) / 2, (y1 - y0) / 2
        inside &= ((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2 <= 1.0
    elif shape.kind == "triangle":
        # Apex at the top centre, base on the bottom edge.
        half = (x1 - x0) / 2 * (ys - y0 + 1) / (y1 - y0)
        inside &= np.abs(xs - (x0 + x1 - 1) / 2) <= half
    elif shape.kind == "ring":
        hole = (xs >= x0 + 8) & (xs < x1 - 8) & (ys >= y0 + 8) & (ys < y1 - 8)
        inside &= ~hole

    return inside


def scene_pixels(
    size_hw: Tuple[int, int], shapes: Sequence[Shape], *, road: bool = True
) -> np.ndarray:
    """Return the ``(H, W, 3)`` RGB scene: background, road band, shapes."""
    height, width = size_hw
    pixels = np.empty((height, width, 3), dtype=np.uint8)
    pixels[:] = BACKGROUND
    if road:
        pixels[height - 12 :] = ROAD
    for shape in shapes:
        pixels[shape_mask(shape, size_hw)] = shape.color

    return pixels


def scene_image(
    image_id: str,
    size_hw: Tuple[int, int],
    shapes: Sequence[Shape],
    *,
    road: bool = True,
) -> ImageData:
    """Return the scene as an RGB ``ImageData``."""
    pixels = scene_pixels(size_hw, shapes, road=road)
    pixels = torch.from_numpy(pixels).permute(2, 0, 1)
    image = ImageData.from_tensor(pixels.contiguous(), image_id=image_id)

    return image


def street_image() -> ImageData:
    """Three objects on a 72x112 street."""
    image = scene_image("street", STREET_SIZE, STREET)

    return image


def alley_image() -> ImageData:
    """One car on a 48x64 alley: a smaller image with fewer rows."""
    image = scene_image("alley", ALLEY_SIZE, ALLEY)

    return image


def color_masks(pixels: torch.Tensor, colors: Sequence[Tuple[int, int, int]]):
    """Return ``(len(colors), H, W)`` bool masks of exact paint colors."""
    paint = torch.tensor(colors, dtype=pixels.dtype, device=pixels.device)
    masks = (pixels[None] == paint[:, :, None, None]).all(dim=1)

    return masks


class FakeModel:
    """Phase API shared by the fakes: images pass through unchanged."""

    class_names: Sequence[str] = CLASS_NAMES

    def __init__(self, request: ModelRequest):
        self.request = request
        self.batch_sizes: List[int] = []

    def pre_process(self, images: List[torch.Tensor], input_color_format: str):
        assert input_color_format == "rgb"
        self.batch_sizes.append(len(images))
        sizes = [tuple(image.shape[1:]) for image in images]

        return list(images), sizes

    def forward(self, batch: Any) -> Any:
        return batch


def _object_rows(pixels: torch.Tensor, *, threshold: float):
    colors = [
        color for color, (_, score) in OBJECT_COLORS.items() if score >= threshold
    ]
    masks = color_masks(pixels, colors)
    present = masks.flatten(1).any(dim=1)
    masks = masks[present]
    kept = [color for color, keep in zip(colors, present.tolist()) if keep]
    boxes = []
    for mask in masks:
        ys, xs = torch.nonzero(mask, as_tuple=True)
        boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
    xyxy = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    class_id = torch.tensor([OBJECT_COLORS[c][0] for c in kept], dtype=torch.int64)
    confidence = torch.tensor([OBJECT_COLORS[c][1] for c in kept])

    return xyxy, class_id, confidence, masks


def _threshold(confidence: Any) -> float:
    threshold = {"best": 0.5, "default": 0.35}.get(confidence, confidence)

    return threshold


class ShapesInstanceModel(FakeModel):
    """Instance segmentation: one dense mask per painted object color."""

    def post_process(
        self,
        images: List[torch.Tensor],
        sizes: list,
        confidence: Any = "default",
        iou_threshold: float = 0.3,
        max_detections: int = 300,
        class_agnostic_nms: bool = False,
        mask_format: str = "rle",
        **kwargs: Any,
    ) -> List[InstanceDetections]:
        assert mask_format == "dense"
        predictions = []
        for pixels in images:
            xyxy, class_id, scores, masks = _object_rows(
                pixels, threshold=_threshold(confidence)
            )
            predictions.append(
                InstanceDetections(
                    xyxy=xyxy, class_id=class_id, confidence=scores, mask=masks
                )
            )

        return predictions


class ShapesKeypointModel(FakeModel):
    """Keypoints: the top and bottom centre of each object's box."""

    key_points_classes = [["top", "bottom"]] * len(CLASS_NAMES)
    skeletons = [[(0, 1)]] * len(CLASS_NAMES)

    def post_process(
        self,
        images: List[torch.Tensor],
        sizes: list,
        confidence: Any = "default",
        iou_threshold: float = 0.3,
        max_detections: int = 300,
        class_agnostic_nms: bool = False,
        key_points_threshold: float = 0.0,
        **kwargs: Any,
    ) -> Tuple[List[KeyPoints], List[Detections]]:
        key_points, boxes = [], []
        for pixels in images:
            xyxy, class_id, scores, _ = _object_rows(
                pixels, threshold=_threshold(confidence)
            )
            centre = (xyxy[:, 0] + xyxy[:, 2]) / 2
            top = torch.stack([centre, xyxy[:, 1]], dim=1)
            bottom = torch.stack([centre, xyxy[:, 3] - 1], dim=1)
            key_points.append(
                KeyPoints(
                    xy=torch.stack([top, bottom], dim=1),
                    class_id=class_id,
                    confidence=scores[:, None].repeat(1, 2),
                )
            )
            boxes.append(Detections(xyxy=xyxy, class_id=class_id, confidence=scores))

        return key_points, boxes


class ShapesSemanticModel(FakeModel):
    """Semantic segmentation: road paint is road, car paints are car."""

    class_names = SEMANTIC_CLASS_NAMES

    def post_process(
        self, images: List[torch.Tensor], sizes: list, **kwargs: Any
    ) -> List[SemanticSegmentationResult]:
        results = []
        for pixels in images:
            labels = torch.zeros(pixels.shape[1:], dtype=torch.long)
            labels[color_masks(pixels, [ROAD])[0]] = 1
            labels[color_masks(pixels, [BLUE, YELLOW]).any(dim=0)] = 2
            results.append(
                SemanticSegmentationResult(
                    segmentation_map=labels,
                    confidence=torch.ones(labels.shape),
                )
            )

        return results


class ColorClassifier(FakeModel):
    """Single-label classifier: the most frequent of red, blue and yellow paint."""

    class_names = COLOR_NAMES

    def pre_process(self, images: List[torch.Tensor], input_color_format: str):
        assert input_color_format == "rgb"
        self.batch_sizes.append(len(images))
        counts = [
            color_masks(image, [RED, BLUE, YELLOW]).flatten(1).sum(dim=1).float()
            for image in images
        ]

        return torch.stack(counts)

    def post_process(self, counts: torch.Tensor, **kwargs: Any):
        confidence = counts / counts.sum(dim=1, keepdim=True).clamp(min=1.0)
        prediction = ClassificationPrediction(
            class_id=confidence.argmax(dim=1), confidence=confidence
        )

        return prediction


MODEL_TYPES = {
    "shapes-seg": ShapesInstanceModel,
    "shapes-semantic": ShapesSemanticModel,
    "shapes-keypoints": ShapesKeypointModel,
    "colors": ColorClassifier,
}


class SceneLoader:
    """``ModelCache`` loader: builds the fake named by the model id."""

    def __init__(self) -> None:
        self.loaded: List[ModelRequest] = []

    def __call__(self, request: ModelRequest) -> FakeModel:
        self.loaded.append(request)
        model = MODEL_TYPES[request.model_id](request)

        return model


class SyntheticCompactSegmentationBlock(InstanceSegmentationBlock):
    """The instance segmentation block returning a synthetic stride-4 grid view.

    Only ``_segmentations`` changes: each dense mask becomes a logit grid,
    ``16 * (cell coverage - 0.5)``. ``accurate`` then upsamples the logits
    smoothly; ``coarse`` keeps whole cells, so edges differ.
    """

    type = "test/synthetic_compact_segmentation@v1"

    def _segmentations(
        self, group: ModelGroup, group_output: Any
    ) -> List[SegmentationView]:
        dense_views = super()._segmentations(group, group_output)
        views = []
        for dense in dense_views:
            predictions = dense.full_res()
            views.append(compact_view(predictions))

        return views


def compact_view(predictions: InstanceDetections) -> SegmentationView:
    """Return a stride-4 logit grid view of dense predictions."""
    height, width = predictions.mask.shape[1:]
    coverage = F.avg_pool2d(predictions.mask.float().unsqueeze(1), STRIDE)
    detections = Detections(
        xyxy=predictions.xyxy,
        class_id=predictions.class_id,
        confidence=predictions.confidence,
        image_metadata=predictions.image_metadata,
    )
    view = SegmentationView.from_selected_rows(
        (coverage.squeeze(1) - 0.5) * 16.0,
        rows=torch.arange(len(predictions)),
        detections=detections,
        score_type="logits",
        threshold=0.0,
        geometry=MaskGridGeometry(
            grid_size_hw=(height // STRIDE, width // STRIDE),
            inference_size_hw=(height, width),
            padding_ltrb=(0, 0, 0, 0),
            scale_xy=(1.0, 1.0),
            crop_size_hw=(height, width),
            original_size_hw=(height, width),
            static_crop_xywh=(0, 0, width, height),
        ),
    )

    return view


def annotation_catalogue() -> Catalogue:
    """The V2 catalogue plus model, crop, relabel and the ten annotation blocks."""
    extra = [block for block in models.BLOCKS if block.type != "v2/object_detection@v1"]
    catalogue = create_catalogue().with_blocks(
        [
            *extra,
            *annotations.BLOCKS,
            DynamicCropBlock,
            *geometry.BLOCKS,
            SyntheticCompactSegmentationBlock,
        ]
    )

    return catalogue


def pixels_of(image: Optional[ImageData]) -> np.ndarray:
    """Return an image's ``(H, W, 3)`` RGB pixels."""
    pixels = image.tensor_image.permute(1, 2, 0).cpu().numpy()

    return pixels


def changed(before: ImageData, after: ImageData) -> np.ndarray:
    """Return the ``(H, W)`` bool pixels that differ between two images."""
    difference = (pixels_of(before) != pixels_of(after)).any(axis=2)

    return difference
