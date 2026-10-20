"""Merge predictions of slices or crops back into one image: ``roboflow_core/detections_stitch@v1``.

Port of V1 ``roboflow_core/detections_stitch@v1``, the last step of SAHI::

    roboflow_core/image_slicer@v2 -> detect on every slice -> this block (one per image)

Every child prediction moves by whole pixels into the reference image (see
``geometry.frames``): boxes, keypoint/polygon metadata and masks. The reference
can be the workflow input or any image the children were cut from, such as a
crop that was sliced again. Rows are concatenated in group order with all
fields aligned; then ``overlap_filtering_strategy`` removes duplicates from
overlapping slices (see ``geometry.overlap``).

The result describes the reference image (frame record and V1 keys) and maps
class ids to the union of the children's ``class_names``; a later child wins
when two name the same id differently, as in V1.

``overlap_by`` chooses what is compared: ``auto`` (default) uses masks when the
predictions have them, as V1's standard implementation does; ``box`` uses
boxes, as V1's tensor implementation did for masks; ``mask`` requires masks.
Mask overlap costs more: RLE masks are decoded on the host, and dense masks of
all rows on the reference canvas are multiplied pairwise on their device.
It uses full-resolution masks where supervision resized them to 640 pixels. NMM keeps the winning row's metadata, as the standard
implementation does; the tensor implementation replaced it.

The stitched rows are a new result: they get a fresh row identity (decision
015), whatever identities the children had. Children may also be
``SegmentationView`` values; stitching pastes masks, so a view gives its
reference masks (``geometry.mask_inputs``).

Empty and absent inputs (decision 016): an absent reference image gives an
absent output; a present reference with no children, an empty group or an
absent group gives an empty result in the reference frame.
"""

from typing import Any, Dict, List, Literal, Optional
import numpy as np
import torch
from pycocotools import mask as mask_utils
from pydantic import Field, StrictFloat, StrictInt
from roboflow_workflows.v2.blocks.geometry.frames import (
    offset_between,
    translate_prediction,
)
from roboflow_workflows.v2.blocks.geometry.mask_inputs import (
    INSTANCE_SEGMENTATION_VIEW_KIND,
    instance_predictions_of,
)
from roboflow_workflows.v2.blocks.geometry.overlap import (
    non_max_merge,
    non_max_suppression,
)
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.frames import frame_of
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import FLOAT_KIND, STRING_KIND
from roboflow_workflows.v2.data.predictions import (
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
)
from roboflow_workflows.v2.data.row_identity import with_row_identity
from roboflow_workflows.v2.engine.declaration import (
    Block,
    BlockParams,
    Group,
    Output,
    Ref,
)
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.types import InstancesRLEMasks


PREDICTION_KINDS = (
    OBJECT_DETECTION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
)

STRATEGIES = ("none", "nms", "nmm")

OVERLAP_SOURCES = ("auto", "mask", "box")

CLASS_NAMES_KEY = "class_names"

class DetectionsStitchBlock(Block):
    """Combine the predictions of an image's slices into one prediction.

    Example: slices at x offsets 0 and 360 each find the same object, at
    ``[600, 10, 640, 50]`` and ``[240, 10, 280, 50]`` in slice coordinates.
    Both become ``[600, 10, 640, 50]`` in the image; ``nms`` keeps the more
    confident one, ``nmm`` merges them, ``none`` keeps both.

    An image whose group is empty, or whose slices were all filtered, gets an
    empty prediction.
    """

    type = "roboflow_core/detections_stitch@v1"
    accepts_empty = True
    outputs = {
        "predictions": Output(
            *PREDICTION_KINDS,
            source="reference_image",
            description=(
                "All children's rows in the reference image, deduplicated. "
                "'none' and 'nms' keep group order; 'nmm' orders rows by class "
                "id, then confidence."
            ),
        ),
    }

    class Params(BlockParams):
        reference_image: Ref(IMAGE_KIND) = Field(
            description="Image the slices or crops were cut from."
        )
        predictions: Group(*PREDICTION_KINDS, INSTANCE_SEGMENTATION_VIEW_KIND) = Field(
            description="Predictions made on the slices or crops of that image."
        )
        overlap_filtering_strategy: Literal["none", "nms", "nmm"] | Ref(STRING_KIND) = (
            Field(
                default="nms",
                description=(
                    "Duplicate handling for overlapping slices: 'none' keeps all rows, "
                    "'nms' keeps the most confident, 'nmm' merges them."
                ),
            )
        )
        iou_threshold: StrictFloat | StrictInt | Ref(FLOAT_KIND) = Field(
            default=0.3,
            ge=0,
            le=1,
            description="Overlap (IoU) at which two rows of one class are duplicates.",
        )
        overlap_by: Literal["auto", "mask", "box"] | Ref(STRING_KIND) = Field(
            default="auto",
            description=(
                "What the IoU compares: 'auto' masks when present else boxes, "
                "'mask' (instance segmentation only) or 'box'."
            ),
        )

    def run(
        self,
        *,
        reference_image: Optional[ImageData],
        predictions: Optional[Batch],
        overlap_filtering_strategy: str,
        iou_threshold: float,
        overlap_by: str,
    ) -> Dict[str, Any]:
        """Stitch ``predictions`` into ``reference_image``.

        Args:
            reference_image: Image the children were cut from; ``None`` when
                the step has no image.
            predictions: The children's predictions, possibly none; ``None``
                when the group is absent.
            overlap_filtering_strategy: ``"none"``, ``"nms"`` or ``"nmm"``.
            iou_threshold: Duplicate threshold in ``[0, 1]``.
            overlap_by: ``"auto"``, ``"mask"`` or ``"box"``.

        Returns:
            ``predictions``: one prediction in the reference image's frame,
            with a new row identity; ``None`` without a reference image.

        Raises:
            ValueError: On an unknown strategy or overlap source, ``"mask"``
                without masks, children that cannot be moved
                into the reference by whole pixels, or a mix of object
                detection and instance segmentation children.
        """
        if overlap_filtering_strategy not in STRATEGIES:
            raise ValueError(
                f"overlap_filtering_strategy must be one of {list(STRATEGIES)}, got "
                f"{overlap_filtering_strategy!r}"
            )
        if overlap_by not in OVERLAP_SOURCES:
            raise ValueError(
                f"overlap_by must be one of {list(OVERLAP_SOURCES)}, got {overlap_by!r}"
            )
        if reference_image is None:
            return {"predictions": None}

        children = [instance_predictions_of(child) for child in predictions or ()]
        moved = [
            translate_prediction(
                child,
                offset=offset_between(frame_of(child), target=reference_image),
                target=reference_image,
            )
            for child in children
            if len(child.xyxy) > 0
        ]
        with_masks = any(isinstance(child, InstanceDetections) for child in children)
        if overlap_by == "mask" and not with_masks and children:
            raise ValueError(
                "overlap_by='mask' needs instance segmentation predictions; "
                "use 'auto' or 'box' for object detection"
            )
        merged = concatenate(
            moved, reference_image=reference_image, with_masks=with_masks
        )
        use_masks = with_masks and overlap_by != "box"
        if overlap_filtering_strategy == "nms":
            merged = non_max_suppression(
                merged, threshold=iou_threshold, use_masks=use_masks
            )
        elif overlap_filtering_strategy == "nmm":
            merged = non_max_merge(merged, threshold=iou_threshold, use_masks=use_masks)

        result = {"predictions": with_row_identity(merged)}

        return result

def concatenate(
    predictions: List[Any], *, reference_image: ImageData, with_masks: bool
) -> Any:
    """Concatenate predictions that already describe ``reference_image``.

    Args:
        predictions: ``Detections`` or ``InstanceDetections``, all of one type.
        reference_image: Image they describe; gives the frame metadata.
        with_masks: Return ``InstanceDetections`` even when no rows remain.

    Returns:
        One prediction with every row in order; ``class_names`` is the union.

    Raises:
        ValueError: When the predictions mix detections with and without masks.
    """
    image_metadata = reference_image.prediction_metadata()
    class_names: Dict[int, str] = {}
    for prediction in predictions:
        class_names.update(_class_names(prediction.image_metadata))
    image_metadata[CLASS_NAMES_KEY] = class_names
    if any(isinstance(p, InstanceDetections) != with_masks for p in predictions):
        raise ValueError(
            "roboflow_core/detections_stitch@v1 cannot mix object detection and instance "
            "segmentation predictions in one group"
        )

    device = reference_image.device
    if predictions:
        xyxy = torch.cat([p.xyxy for p in predictions])
        class_id = torch.cat([p.class_id for p in predictions])
        confidence = torch.cat([p.confidence for p in predictions])
    else:
        xyxy = torch.zeros((0, 4), dtype=torch.float32, device=device)
        class_id = torch.zeros((0,), dtype=torch.int64, device=device)
        confidence = torch.zeros((0,), dtype=torch.float32, device=device)
    rows = [
        dict(row)
        for p in predictions
        for row in (p.bboxes_metadata or [{} for _ in range(len(p.xyxy))])
    ]
    fields = {
        "xyxy": xyxy,
        "class_id": class_id,
        "confidence": confidence,
        "image_metadata": image_metadata,
        "bboxes_metadata": rows if predictions else None,
    }
    if not with_masks:
        return Detections(**fields)

    stacked = InstanceDetections(
        mask=_concatenate_masks(predictions, reference_image=reference_image), **fields
    )

    return stacked

def _concatenate_masks(predictions: List[Any], *, reference_image: ImageData) -> Any:
    masks = [p.mask for p in predictions]
    if not any(isinstance(mask, InstancesRLEMasks) for mask in masks):
        if masks:
            return torch.cat(masks)
        return torch.zeros(
            (0, *reference_image.size_hw),
            dtype=torch.bool,
            device=reference_image.device,
        )

    # Mixed dense and RLE children: dense masks are encoded on the host.
    encoded: List[bytes] = []
    for mask in masks:
        if isinstance(mask, InstancesRLEMasks):
            encoded.extend(mask.masks)
            continue
        for one in mask.cpu().numpy().astype(np.uint8):
            encoded.append(mask_utils.encode(np.asfortranarray(one))["counts"])
    combined = InstancesRLEMasks(image_size=reference_image.size_hw, masks=encoded)

    return combined

def _class_names(image_metadata: Optional[Dict[str, Any]]) -> Dict[int, str]:
    names = (image_metadata or {}).get(CLASS_NAMES_KEY) or {}
    if not isinstance(names, dict):
        names = dict(enumerate(names))

    return {int(class_id): name for class_id, name in names.items()}
