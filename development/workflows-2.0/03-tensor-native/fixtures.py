"""Small, deterministic native carriers; no model weights or remote inputs."""

from copy import deepcopy
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from pycocotools import mask as mask_codec

from inference_models.models.base.classification import (
    ClassificationPrediction,
    MultiLabelClassificationPrediction,
)
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.types import InstancesRLEMasks

FIRST_REGIONS = [[30, 70, 230, 230], [500, 500, 550, 550], [150, 70, 350, 230]]
LOCAL_BOX = [2.0, 4.0, 6.0, 8.0]
EXPECTED_ROOT_BOX = [44.0, 98.0, 52.0, 114.0]


def make_root_pixels() -> List[torch.Tensor]:
    """Build two 300x400 roots and one small root with no surviving crops.

    Returns:
        CHW uint8 tensors. The first root has one bright and one dark crop;
        the second is dark everywhere; all regions miss the third root.
    """
    bright = torch.full((3, 300, 400), 210, dtype=torch.uint8)
    bright[:, 70:230, 150:350] = 20
    # Mark the numerical oracle's root-space box inside the bright crop.
    bright[0, 98:114, 44:52] = 255
    bright[1:, 98:114, 44:52] = 150
    dark = torch.full((3, 300, 400), 20, dtype=torch.uint8)
    empty = torch.full((3, 32, 32), 100, dtype=torch.uint8)
    pixels = [bright, dark, empty]

    return pixels


def make_native_fixtures(metadata: Dict[str, Any]) -> List[Tuple[str, str, Any]]:
    """Construct all eleven native families, including difficult empty cases.

    Args:
        metadata: Provenance for an 8x10 image, attached without host mirrors.

    Returns:
        Tuples of descriptive fixture name, public kind name, and real native
        payload. Both classification forms and dense/RLE masks are represented.
    """
    boxes = torch.tensor([[1, 1, 8, 7]], dtype=torch.float32)
    classes = torch.tensor([0], dtype=torch.int64)
    confidence = torch.tensor([0.9], dtype=torch.float32)
    rows = [{"detection_id": "gallery:0", "class_name": "marker"}]
    detections = Detections(
        xyxy=boxes,
        class_id=classes,
        confidence=confidence,
        image_metadata=deepcopy(metadata),
        bboxes_metadata=deepcopy(rows),
    )
    dense = torch.zeros((1, 8, 10), dtype=torch.bool)
    dense[0, 1:3, 1:3] = True
    dense[0, 5:7, 6:8] = True
    dense_predictions = InstanceDetections(
        xyxy=boxes,
        class_id=classes,
        confidence=confidence,
        mask=dense,
        image_metadata=deepcopy(metadata),
        bboxes_metadata=deepcopy(rows),
    )
    # Encoding is fixture construction, an explicit host boundary. The payload
    # carried through the engine stays compressed, including both components.
    encoded = mask_codec.encode(np.asfortranarray(dense[0].numpy(), dtype=np.uint8))
    rle_predictions = InstanceDetections(
        xyxy=boxes,
        class_id=classes,
        confidence=confidence,
        mask=InstancesRLEMasks(image_size=(8, 10), masks=[encoded["counts"]]),
        image_metadata=deepcopy(metadata),
        bboxes_metadata=deepcopy(rows),
    )
    semantic = InstanceDetections(
        xyxy=boxes,
        class_id=classes,
        confidence=confidence,
        mask=rle_predictions.mask,
        image_metadata={
            **deepcopy(metadata),
            "prediction_type": "semantic-segmentation",
        },
        bboxes_metadata=deepcopy(rows),
    )
    keypoints = KeyPoints(
        xy=torch.tensor([[[2, 2], [0, 0], [7, 6]]], dtype=torch.float32),
        class_id=classes,
        confidence=torch.tensor([[0.9, 0.0, 0.8]], dtype=torch.float32),
        covariance=torch.eye(2).repeat(1, 3, 1, 1),
        detection_confidence=confidence,
        image_metadata=deepcopy(metadata),
        key_points_metadata=[{"keypoint_ids": [0, 4, 9], "detection_id": "skeleton:0"}],
    )
    single_label = ClassificationPrediction(
        class_id=torch.tensor([1], dtype=torch.int64),
        confidence=torch.tensor([[0.2, 0.8]], dtype=torch.float16),
        images_metadata=[deepcopy(metadata)],
    )
    multi_label = MultiLabelClassificationPrediction(
        class_ids=torch.tensor([0, 2], dtype=torch.int64),
        confidence=torch.tensor([0.9, 0.1, 0.8], dtype=torch.float16),
        image_metadata=deepcopy(metadata),
    )
    qr = Detections(
        xyxy=boxes,
        class_id=classes,
        confidence=confidence,
        image_metadata=deepcopy(metadata),
        bboxes_metadata=[{**rows[0], "data": "offline:qr:17"}],
    )
    barcode = Detections(
        xyxy=boxes,
        class_id=classes,
        confidence=confidence,
        image_metadata=deepcopy(metadata),
        bboxes_metadata=[{**rows[0], "data": "0123456789"}],
    )
    empty_detections = Detections(
        xyxy=torch.empty((0, 4), dtype=torch.float16),
        class_id=torch.empty((0,), dtype=torch.int64),
        confidence=torch.empty((0,), dtype=torch.float16),
        image_metadata=deepcopy(metadata),
        bboxes_metadata=[],
    )
    fixtures = [
        ("boxes", "object_detection_prediction", detections),
        ("empty-boxes", "object_detection_prediction", empty_detections),
        ("dense-mask", "instance_segmentation_prediction", dense_predictions),
        ("disconnected-rle", "rle_instance_segmentation_prediction", rle_predictions),
        ("disconnected-semantic", "semantic_segmentation_prediction", semantic),
        ("keypoints-without-boxes", "keypoint_detection_prediction", (keypoints, None)),
        ("single-label", "classification_prediction", single_label),
        ("multi-label", "classification_prediction", multi_label),
        ("single-detection-row", "detection", next(iter(detections))),
        ("qr", "qr_code_detection", qr),
        ("barcode", "bar_code_detection", barcode),
        (
            "embedding",
            "embedding",
            torch.tensor([0.25, 0.5, 0.75], dtype=torch.float16),
        ),
        ("empty-float16", "tensor", torch.empty((0, 4), dtype=torch.float16)),
        ("dense-map", "tensor", torch.arange(80, dtype=torch.float32).reshape(8, 10)),
    ]

    return fixtures
