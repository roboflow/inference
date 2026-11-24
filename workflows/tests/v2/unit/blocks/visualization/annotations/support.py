"""Fixtures of the annotation block tests: native predictions and V1 oracles.

Images are small synthetic RGB arrays. Predictions are the native carriers V2
model blocks produce, with a frame record and ``class_names``. The V1 oracle
runs the latest V1 block class itself on the same data converted to
``sv.Detections`` and a BGR ``WorkflowImageData``; V1 test modules are not
imported.
"""

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pycocotools.mask as mask_utils
import supervision as sv
import torch
import torch.nn.functional as F
from roboflow_workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)
from roboflow_workflows.v2.data.frames import FrameRef
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.masks import MaskGridGeometry, SegmentationView
from roboflow_workflows.v2.engine.declaration import spec_of

from inference_models.models.base.classification import (
    ClassificationPrediction,
    MultiLabelClassificationPrediction,
)
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.semantic_segmentation import (
    SemanticSegmentationResult,
)
from inference_models.models.base.types import InstancesRLEMasks

CLASS_NAMES = {0: "car", 1: "person", 2: "dog"}
HEIGHT, WIDTH = 48, 64
# Three overlapping objects: (x1, y1, x2, y2, class id, confidence).
OBJECTS = (
    (4.6, 6.2, 30.4, 30.9, 0, 0.91),
    (20.0, 14.0, 52.7, 44.0, 1, 0.5),
    (40.0, 2.0, 60.0, 20.0, 2, 0.333),
)


def scene(height: int = HEIGHT, width: int = WIDTH, *, seed: int = 0) -> np.ndarray:
    """Return a smooth-plus-noise RGB uint8 image."""
    generator = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:height, 0:width]
    base = np.stack([xs * 3, ys * 4, (xs + ys) * 2], axis=-1) % 256
    noise = generator.integers(0, 40, (height, width, 3))
    pixels = ((base + noise) % 256).astype(np.uint8)

    return pixels


def image_of(pixels: np.ndarray, *, image_id: str = "frame") -> ImageData:
    """Wrap RGB pixels as a workflow input image."""
    image = ImageData.from_numpy_rgb(pixels, image_id=image_id)

    return image


def object_masks(height: int = HEIGHT, width: int = WIDTH) -> np.ndarray:
    """Return ``(3, H, W)`` bool masks: an ellipse inside each of ``OBJECTS``."""
    ys, xs = np.mgrid[0:height, 0:width]
    masks = []
    for x1, y1, x2, y2, _, _ in OBJECTS:
        centre_x, centre_y = (x1 + x2) / 2, (y1 + y2) / 2
        radius_x, radius_y = (x2 - x1) / 2, (y2 - y1) / 2
        masks.append(
            ((xs - centre_x) / radius_x) ** 2 + ((ys - centre_y) / radius_y) ** 2 <= 1
        )
    masks[1][28:32, 30:36] = False  # a hole: inner contour for polygons

    return np.stack(masks)


def metadata_for(image: ImageData, **extra: Any) -> Dict[str, Any]:
    """Image metadata a V2 model block writes: frame record and class names."""
    metadata = {
        **FrameRef.from_image(image).prediction_metadata(),
        "class_names": dict(CLASS_NAMES),
        **extra,
    }

    return metadata


def detections(
    image: ImageData,
    objects: Sequence[Tuple] = OBJECTS,
    *,
    rows: Optional[Sequence[dict]] = None,
    device: str = "cpu",
) -> Detections:
    """Native ``Detections`` of ``objects`` in ``image``."""
    boxes = torch.tensor([list(obj[:4]) for obj in objects], dtype=torch.float32)
    predictions = Detections(
        xyxy=boxes.reshape(-1, 4).to(device),
        class_id=torch.tensor([obj[4] for obj in objects], dtype=torch.int64).to(
            device
        ),
        confidence=torch.tensor([obj[5] for obj in objects], dtype=torch.float32).to(
            device
        ),
        image_metadata=metadata_for(image),
        bboxes_metadata=list(rows) if rows is not None else None,
    )

    return predictions


def instance_detections(
    image: ImageData, *, masks: Optional[np.ndarray] = None, device: str = "cpu"
) -> InstanceDetections:
    """Native dense ``InstanceDetections`` of ``OBJECTS``."""
    boxes = detections(image, device=device)
    masks = object_masks(*image.size_hw) if masks is None else masks
    predictions = InstanceDetections(
        xyxy=boxes.xyxy,
        class_id=boxes.class_id,
        confidence=boxes.confidence,
        mask=torch.from_numpy(masks).to(device),
        image_metadata=boxes.image_metadata,
    )

    return predictions


def rle_detections(image: ImageData) -> InstanceDetections:
    """Native RLE ``InstanceDetections`` of ``OBJECTS``."""
    dense = instance_detections(image)
    encoded = [
        mask_utils.encode(np.asfortranarray(mask.numpy().astype(np.uint8)))
        for mask in dense.mask
    ]
    predictions = InstanceDetections(
        xyxy=dense.xyxy,
        class_id=dense.class_id,
        confidence=dense.confidence,
        mask=InstancesRLEMasks.from_coco_rle_masks(image.size_hw, encoded),
        image_metadata=dense.image_metadata,
    )

    return predictions


def grid_view(image: ImageData, *, stride: int = 4) -> SegmentationView:
    """A low-resolution mask view of ``OBJECTS``: logits per ``stride`` cell."""
    dense = instance_detections(image)
    height, width = image.size_hw
    touched = F.avg_pool2d(dense.mask.float().unsqueeze(1), stride).squeeze(1)
    view = SegmentationView(
        detections=Detections(
            xyxy=dense.xyxy,
            class_id=dense.class_id,
            confidence=dense.confidence,
            image_metadata=dense.image_metadata,
        ),
        scores=touched * 16.0 - 8.0,
        score_type="logits",
        threshold=0.0,
        geometry=MaskGridGeometry(
            grid_size_hw=(height // stride, width // stride),
            inference_size_hw=(height, width),
            padding_ltrb=(0, 0, 0, 0),
            scale_xy=(1.0, 1.0),
            crop_size_hw=(height, width),
            original_size_hw=(height, width),
            static_crop_xywh=(0, 0, width, height),
        ),
        take_ownership=True,
    )

    return view


def key_points(
    image: ImageData, *, with_boxes: bool = True
) -> Tuple[KeyPoints, Optional[Detections]]:
    """Two 17-keypoint (COCO) skeletons; a few keypoints are below threshold."""
    generator = np.random.default_rng(7)
    xy = generator.uniform(2, [image.width - 2, image.height - 2], (2, 17, 2))
    xy = np.round(xy).astype(np.float32)
    confidence = generator.uniform(0.4, 1.0, (2, 17)).astype(np.float32)
    for skeleton, point in ((0, 3), (0, 9), (1, 0)):
        xy[skeleton, point] = 0.0
        confidence[skeleton, point] = 0.0
    boxes = detections(image, OBJECTS[:2]) if with_boxes else None
    points = KeyPoints(
        xy=torch.from_numpy(xy),
        class_id=torch.tensor([1, 1]),
        confidence=torch.from_numpy(confidence),
        image_metadata=metadata_for(image),
    )

    return points, boxes


def classification(image: ImageData) -> ClassificationPrediction:
    """Single-label classification: ``person`` wins."""
    prediction = ClassificationPrediction(
        class_id=torch.tensor([1]),
        confidence=torch.tensor([[0.2, 0.7, 0.1]]),
        images_metadata=[metadata_for(image)],
    )

    return prediction


def multi_label(image: ImageData) -> MultiLabelClassificationPrediction:
    """Multi-label classification: ``dog`` and ``car`` predicted."""
    prediction = MultiLabelClassificationPrediction(
        class_ids=torch.tensor([0, 2]),
        confidence=torch.tensor([0.61, 0.2, 0.83]),
        image_metadata=metadata_for(image),
    )

    return prediction


def semantic(
    image: ImageData, *, names: Optional[Dict[int, str]] = None
) -> SemanticSegmentationResult:
    """A class grid: 0 background, 1 road in the lower half, 2 car in a box."""
    height, width = image.size_hw
    grid = torch.zeros((height, width), dtype=torch.int64)
    grid[height // 2 :, :] = 1
    grid[10:30, 20:44] = 2
    result = SemanticSegmentationResult(
        segmentation_map=grid,
        confidence=torch.full((height, width), 0.9),
        image_metadata={
            **FrameRef.from_image(image).prediction_metadata(),
            "class_names": names or {0: "background", 1: "road", 2: "car"},
        },
    )

    return result


def run_v2(
    block_class: type, *, image: ImageData, predictions: Any, **parameters: Any
) -> ImageData:
    """Run one invocation as the engine does: validate parameters, then ``run``."""
    params = spec_of(block_class).validate_params(
        {**parameters, "image": "$inputs.image", "predictions": "$inputs.predictions"}
    )
    arguments = {**dict(params), "image": image, "predictions": predictions}
    result = (getattr(block_class, "implementations", ()) or (block_class,))[0]().run(
        **arguments
    )

    return result["image"]


def pixels_of(image: ImageData) -> np.ndarray:
    """RGB ``(H, W, 3)`` host pixels of an image."""
    pixels = image.tensor_image.permute(1, 2, 0).cpu().numpy()

    return pixels


def v1_detections(
    objects: Sequence[Tuple] = OBJECTS,
    *,
    masks: Optional[np.ndarray] = None,
    tracker_id: Optional[Sequence[int]] = None,
    data: Optional[Dict[str, Any]] = None,
) -> sv.Detections:
    """The V1 form of the same rows (class names in ``data``)."""
    predictions = sv.Detections(
        xyxy=np.array([list(obj[:4]) for obj in objects], dtype=np.float32).reshape(
            -1, 4
        ),
        class_id=np.array([obj[4] for obj in objects], dtype=int),
        confidence=np.array([obj[5] for obj in objects], dtype=np.float32),
        mask=masks,
        tracker_id=None if tracker_id is None else np.asarray(tracker_id),
        data={
            "class_name": np.array([CLASS_NAMES[obj[4]] for obj in objects]),
            **(data or {}),
        },
    )

    return predictions


def run_v1(block_class: type, pixels: np.ndarray, **arguments: Any) -> np.ndarray:
    """Run the latest V1 block on RGB ``pixels``; return its RGB output."""
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="frame"),
        numpy_image=np.ascontiguousarray(pixels[:, :, ::-1]),
    )
    result = (getattr(block_class, "implementations", ()) or (block_class,))[0]().run(
        image=image, copy_image=True, **arguments
    )
    output = result["image"].numpy_image[:, :, ::-1]

    return output


V1_COLOR_DEFAULTS = {
    "color_palette": "DEFAULT",
    "palette_size": 10,
    "custom_colors": [],
    "color_axis": "CLASS",
}
