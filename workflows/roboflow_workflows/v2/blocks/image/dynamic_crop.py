"""Crop every detected box out of its image: ``v2/dynamic_crop@v1``.

Port of V1 ``roboflow_core/dynamic_crop@v1``. Like every V2 block it answers
only to its V2 type; the V1 aliases ``Crop`` and ``DynamicCrop`` are not
accepted::

    for detection row k:
        box = round(xyxy[k]) (half to even), clamped to the image
        crop k = image[box]; None when empty
        prediction k = row k alone, moved into crop k (geometry.frames)
        mask_opacity > 0 with instance masks: pixels outside the row's mask
            fade to background_color: op * colour + (1 - op) * pixel

Both outputs share a new ``regions`` axis with one slot per detection row, so
slot ``k`` belongs to row ``k`` (decision 016). An empty crop is a ``None``
slot, also at the end, so the axis keeps the row count; steps on the crops skip
``None`` slots, and ``v2/dimension_collapse@v1`` reports them as ``None``. The
slot position is lineage, not row identity: class replacement joins results to
rows by their row keys, not by position.

Each crop records the detection it was cut for as ``ImageData.parent_row``, the
row key ``(namespace, id)`` of the input predictions (decision 015). Predictions
without row identity give crops without a parent row; class replacement then
refuses their classifications instead of guessing.

Boxes are read in the frame of the image to crop (decision 021). Row identity
does not prove that; the prediction's frame metadata does::

    prediction metadata                       boxes used
    no frame metadata at all                  as given: assumed made on image
    made on image (same image id)             as given
    made on another image of the same         mapped into image:
      workflow input (resize, static crop)      data.coordinates.prediction_to_frame
    other workflow input, or a root id or     ContractError naming both frames
      placement the metadata does not state

Equal sizes alone never count as the same frame: a static crop has an offset
and a rotation starts a new root. Legacy metadata without a root id cannot
prove a shared input, so only its same-image case is accepted.

Instance segmentation may arrive as ``InstanceDetections`` or as a
``SegmentationView`` (decision 018). Cropping needs only boxes, so a view gives
box-only per-crop predictions and no masks are materialized, also when the
boxes are mapped. Only the fade needs mask pixels; then the view's reference
masks are computed once (``geometry.mask_inputs``) and mapped with the boxes.

Keypoint predictions need their boxes: a ``(KeyPoints, None)`` tuple is a
``ContractError`` (decision 018), unlike detections with zero rows, which give
no crops.

A crop's provenance places it in the image and its root; source and time are
the image's. The per-crop prediction keeps all of the row's fields: box, mask
cut to the crop, keypoints and row metadata, shifted by ``(-x0, -y0)``. Its box
is not clipped: a box overhanging the image keeps its overhang in crop
coordinates, as in V1.

CPU boundary: the rounded boxes and row ids go to the host together, once per
image, to size and label the crops. Mapping boxes reads no tensor values. RLE
masks are decoded on the host only when the fade is used: one row per crop,
plus every row once when the masks are mapped from another frame.
"""

import re
from typing import Annotated, Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from pydantic import AfterValidator, Field, StrictFloat, StrictInt
from roboflow_workflows.execution_engine.constants import PARENT_ID_KEY
from roboflow_workflows.v2.blocks.geometry.frames import translate_prediction
from roboflow_workflows.v2.blocks.geometry.mask_inputs import (
    INSTANCE_SEGMENTATION_VIEW_KIND,
    boxes_of_view,
    instance_predictions_of,
)
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.coordinates import prediction_to_frame
from roboflow_workflows.v2.data.errors import ContractError
from roboflow_workflows.v2.data.frames import (
    FRAME_RECORD_KEY,
    FrameRef,
    frame_from_metadata,
    root_placement_from_metadata,
)
from roboflow_workflows.v2.data.image import FrameMapping, ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import FLOAT_KIND, STRING_KIND
from roboflow_workflows.v2.data.prediction_ops import select_predictions
from roboflow_workflows.v2.data.predictions import (
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
)
from roboflow_workflows.v2.data.rle import decode_rle_masks
from roboflow_workflows.v2.data.row_identity import RowRef, row_record
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.types import InstancesRLEMasks

PREDICTION_KINDS = (
    OBJECT_DETECTION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
)
RGB = Tuple[int, int, int]

_HEX6 = re.compile(r"^#([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})$")
_HEX3 = re.compile(r"^#([0-9a-fA-F])([0-9a-fA-F])([0-9a-fA-F])$")
_TUPLE = re.compile(r"^\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)$")


def parse_rgb(color: Union[str, Sequence[int]]) -> RGB:
    """Read a colour in one of V1's notations.

    Args:
        color: ``"#RRGGBB"``, ``"#RGB"``, ``"(r, g, b)"`` or a sequence of three
            integers, all in RGB order and ``0..255``.

    Returns:
        ``(r, g, b)``.

    Raises:
        ValueError: On any other value.
    """
    if isinstance(color, str):
        for pattern, base in ((_HEX6, 16), (_HEX3, 16), (_TUPLE, 10)):
            match = pattern.match(color.strip())
            if match:
                parts = match.groups()
                if pattern is _HEX3:
                    parts = tuple(part * 2 for part in parts)
                rgb = tuple(int(part, base) for part in parts)
                break
        else:
            raise ValueError(
                f"Invalid colour {color!r}; use '#RRGGBB', '#RGB' or '(r, g, b)'"
            )
    elif isinstance(color, (list, tuple)) and len(color) == 3:
        rgb = tuple(color)
    else:
        raise ValueError(f"Invalid colour {color!r}; expected three integers")

    if not all(isinstance(v, int) and not isinstance(v, bool) for v in rgb) or not all(
        0 <= v <= 255 for v in rgb
    ):
        raise ValueError(f"Colour {color!r} must have three integers in 0..255")

    return rgb


def _check_color(value: Any) -> Any:
    # Literal colours are checked at compile time, selected ones per call.
    if isinstance(value, (str, list, tuple)):
        parse_rgb(value)

    return value


class DynamicCropBlock(Block):
    """Crop each detection's box out of its image, with that detection alone.

    Example: on a 100x80 image the box ``[-4.6, 10.5, 30.2, 90]`` rounds to
    ``[-5, 10, 30, 90]`` and clamps to ``[0, 10, 30, 80]``: a 30x70 crop at
    offset ``(0, 10)``; its prediction's box, unrounded as in V1, becomes
    ``[-4.6, 0.5, 30.2, 80]``.
    """

    type = "v2/dynamic_crop@v1"
    outputs = {
        "crops": Output(
            IMAGE_KIND,
            expand="regions",
            source="image",
            description="Crop of detection row k; None when the box is empty.",
        ),
        "predictions": Output(
            *PREDICTION_KINDS,
            expand="regions",
            source="image",
            description="Detection row k alone, in crop k coordinates; None with it.",
        ),
    }

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(description="Image to crop.")
        predictions: Ref(*PREDICTION_KINDS, INSTANCE_SEGMENTATION_VIEW_KIND) = Field(
            description="Detections on that image; every row gives one crop."
        )
        mask_opacity: StrictFloat | StrictInt | Ref(FLOAT_KIND) = Field(
            default=0.0,
            ge=0,
            le=1,
            description=(
                "Instance segmentation only: how strongly pixels outside the "
                "row's mask fade to background_color; 0 keeps them."
            ),
        )
        background_color: Annotated[
            str | Tuple[StrictInt, StrictInt, StrictInt] | Ref(STRING_KIND),
            AfterValidator(_check_color),
        ] = Field(
            default=(0, 0, 0),
            description=(
                "Fade colour, RGB: '#RRGGBB', '#RGB', '(r, g, b)' or [r, g, b]."
            ),
            examples=["#431112", [18, 17, 67]],
        )

    def run(
        self,
        *,
        image: ImageData,
        predictions: Any,
        mask_opacity: float,
        background_color: Union[str, Sequence[int]],
    ) -> Dict[str, Any]:
        """Crop every row of ``predictions`` out of ``image``.

        Args:
            image: Image to crop: the one the predictions were made on, or
                another crop or resize of the same workflow input.
            predictions: ``Detections``, ``InstanceDetections``, a
                ``SegmentationView`` or a ``(KeyPoints, Detections)`` tuple.
            mask_opacity: Fade strength in ``[0, 1]``.
            background_color: Fade colour.

        Returns:
            ``crops`` and ``predictions``: ``Batch`` values with one slot per
            row, ``None`` where the box is empty.

        Raises:
            ContractError: When the predictions were made on an image of
                another workflow input, or their frame metadata cannot place
                them in ``image``; or keypoints come without boxes.
        """
        boxes_input = boxes_of_view(predictions)
        detections = _boxes_of(boxes_input)
        if detections is None or len(detections.xyxy) == 0:
            return {"crops": Batch.of([]), "predictions": Batch.of([])}

        mask_source = instance_predictions_of(predictions) if mask_opacity > 0 else None
        fade_color = _fade_color(
            mask_source, opacity=mask_opacity, color=background_color
        )
        predictions = boxes_input
        into_image = _mapping_into(image, detections=detections)
        if into_image is not None:
            frame = FrameRef.from_image(image)
            predictions = prediction_to_frame(
                boxes_input, mapping=into_image, frame=frame
            )
            detections = _boxes_of(predictions)
            if fade_color is not None:
                # InstanceDetections moved with their masks; a view's
                # reference masks are separate predictions and move here.
                mask_source = (
                    predictions
                    if mask_source is boxes_input
                    else prediction_to_frame(
                        mask_source, mapping=into_image, frame=frame
                    )
                )

        parent_rows, boxes = _parent_rows_and_boxes(
            predictions, rounded=detections.xyxy.round().to(torch.int64)
        )
        crops: List[Optional[ImageData]] = []
        rows: List[Any] = []
        for row, box in enumerate(boxes):
            x0, y0, x1, y1 = _clamped(box, size_hw=image.size_hw)
            crop = image.crop((x0, y0, x1, y1))
            if crop is None:
                crops.append(None)
                rows.append(None)
                continue

            crop = crop.with_parent_row(parent_rows[row])
            if fade_color is not None:
                inside = _mask_window(
                    mask_source.mask, row=row, window=(x0, y0, x1, y1)
                )
                crop = _faded(
                    crop, inside=inside, opacity=mask_opacity, color=fade_color
                )
            single = select_predictions(predictions, indices=[row])
            crops.append(crop)
            rows.append(translate_prediction(single, offset=(-x0, -y0), target=crop))

        result = {"crops": Batch.of(crops), "predictions": Batch.of(rows)}

        return result


def _boxes_of(predictions: Any) -> Any:
    """The box ``Detections``; ``None`` only for absent predictions."""
    if not isinstance(predictions, tuple):
        return predictions

    key_points, detections = predictions
    if detections is None:
        raise ContractError(
            f"{DynamicCropBlock.type}: the keypoint prediction has no bounding "
            f"boxes ({len(key_points)} skeletons, Detections None); crops are cut "
            "from boxes, so use a keypoint model that also returns boxes"
        )

    return detections


def _mapping_into(image: ImageData, *, detections: Any) -> Optional[FrameMapping]:
    """Placement of the boxes' pixels in ``image``; ``None`` when already there."""
    metadata = detections.image_metadata or {}
    predicted_on = _predicted_image_id(metadata)
    if predicted_on == image.image_id:
        return None

    try:
        placement = root_placement_from_metadata(metadata)
    except ContractError as error:
        raise ContractError(
            f"{DynamicCropBlock.type}: cannot place predictions of image "
            f"{predicted_on!r} in image {image.image_id!r}: {error}"
        ) from error
    if placement is None and predicted_on is None:
        # No frame metadata at all: the documented local-frame assumption.
        return None

    boxes_root = placement.root if placement is not None else None
    image_root = image.root
    if (
        boxes_root is None
        or boxes_root.frame_id is None
        or boxes_root.frame_id != image_root.frame_id
        or boxes_root.frame_size_hw != image_root.frame_size_hw
    ):
        raise ContractError(
            _mismatch_message(predicted_on, boxes_root=boxes_root, image=image)
        )

    # Through the shared root, per axis:
    #   image_xy = (local_xy * S_boxes + o_boxes - o_image) / S_image
    scale_xy = [
        boxes_scale / image_scale
        for boxes_scale, image_scale in zip(boxes_root.scale_xy, image_root.scale_xy)
    ]
    offset_xy = [
        (boxes_offset - image_offset) / image_scale
        for boxes_offset, image_offset, image_scale in zip(
            boxes_root.offset_xy, image_root.offset_xy, image_root.scale_xy
        )
    ]
    mapping = FrameMapping(
        frame_id=image.image_id,
        frame_size_hw=image.size_hw,
        scale_xy=scale_xy,
        offset_xy=offset_xy,
    )

    return mapping


def _predicted_image_id(metadata: Dict[str, Any]) -> Optional[str]:
    if metadata.get(FRAME_RECORD_KEY) is not None:
        return frame_from_metadata(metadata).image_id

    return metadata.get(PARENT_ID_KEY)


def _mismatch_message(
    predicted_on: Optional[str],
    *,
    boxes_root: Optional[FrameMapping],
    image: ImageData,
) -> str:
    if boxes_root is None or boxes_root.frame_id is None:
        known = "its metadata does not name the workflow input it belongs to"
    else:
        known = (
            f"it belongs to workflow input {boxes_root.frame_id!r} "
            f"(h, w)={boxes_root.frame_size_hw}"
        )
    message = (
        f"{DynamicCropBlock.type}: predictions were made on image "
        f"{predicted_on!r}, not on image {image.image_id!r} (workflow input "
        f"{image.root.frame_id!r} (h, w)={image.root.frame_size_hw}), and "
        f"{known}. Bind 'image' to the image the predictions were made on, or "
        "to another crop or resize of the same workflow input"
    )

    return message


def _parent_rows_and_boxes(
    predictions: Any, *, rounded: torch.Tensor
) -> Tuple[List[Optional[RowRef]], List[List[int]]]:
    """Row key and rounded box of every row, read to the host together."""
    record = row_record(predictions)
    if record is None:
        boxes = rounded.cpu().tolist()
        return [None] * len(boxes), boxes

    ids, boxes = record.host_ids_with(rounded)
    parent_rows = [RowRef(namespace=record.namespace, id=row_id) for row_id in ids]

    return parent_rows, boxes


def _clamped(box: Sequence[int], *, size_hw: Tuple[int, int]) -> Tuple[int, ...]:
    height, width = size_hw
    x0, y0, x1, y1 = box
    clamped = (
        min(max(x0, 0), width),
        min(max(y0, 0), height),
        min(max(x1, 0), width),
        min(max(y1, 0), height),
    )

    return clamped


def _fade_color(mask_source: Any, *, opacity: float, color: Any) -> Optional[RGB]:
    # The fade applies to instance masks with a positive opacity only.
    has_masks = (
        isinstance(mask_source, InstanceDetections) and mask_source.mask is not None
    )
    if opacity <= 0 or not has_masks:
        return None

    rgb = parse_rgb(color)

    return rgb


def _mask_window(mask: Any, *, row: int, window: Tuple[int, ...]) -> torch.Tensor:
    x0, y0, x1, y1 = window
    if isinstance(mask, InstancesRLEMasks):
        # Host boundary: one row's RLE mask is validated and decoded.
        dense = decode_rle_masks(mask, rows=[row])[0][y0:y1, x0:x1]
        return torch.from_numpy(np.ascontiguousarray(dense)).bool()

    return mask[row, y0:y1, x0:x1].bool()


def _faded(
    crop: ImageData, *, inside: torch.Tensor, opacity: float, color: RGB
) -> ImageData:
    pixels = crop.tensor_image.to(torch.float32)
    background = torch.tensor(color, dtype=torch.float32, device=crop.device)
    if crop.channels == 1:
        # One-channel images fade to the colour's mean grey.
        background = background.mean(dim=0, keepdim=True)
    faded = opacity * background.reshape(-1, 1, 1) + (1 - opacity) * pixels
    blended = torch.where(inside.to(crop.device).unsqueeze(0), pixels, faded)
    result = crop.with_pixels(blended.round().clamp(0, 255).to(torch.uint8))

    return result
