"""Live detection blocks: a phased detector, a box painter and a label painter.

Data flow of ``workflows/live_detection.json``::

    image (CHW RGB uint8) ─┬─ detector ── predictions (native Detections) ─┐
                           └──────────── boxes (clone + paint) ── labels (paint in place)
                                                                     └─ annotated

The detector wraps an already loaded ``inference_models`` object detection
model, passed by the host as the ``detection_model`` resource. The model owns
backend and device selection; nothing here picks CPU, MPS or a runtime.

The painters call the V1 tensor primitives directly: ``gpu_draw_boxes`` and
the V1 label sprite path (``_measure_label``, ``_render_label_sprite``,
``gpu_paste_label_sprites``). Despite the ``gpu_`` names they run on the
tensor's own device, the CPU here. There is no supervision/NumPy fallback: a
painter that cannot draw raises.

Every block also returns its own elapsed wall time in milliseconds, measured
inside the call (``pre_ms``, ``model_ms``, ``post_ms``, ``draw_ms``). The
calls are synchronous on the CPU, so these are compute times. On an
asynchronous device they would only time the host side of each call.
"""

from collections import OrderedDict
from time import perf_counter
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import supervision as sv
import torch
from pydantic import Field, StrictFloat
from roboflow_workflows.core_steps.visualizations.bounding_box.v1_tensor import (
    gpu_draw_boxes,
)
from roboflow_workflows.core_steps.visualizations.label.v1_tensor import (
    _LabelMeasurement,
    _LabelSprite,
    _measure_label,
    _render_label_sprite,
    gpu_paste_label_sprites,
)
from roboflow_workflows.execution_engine.v2.blocks.image_data import ImageData
from roboflow_workflows.execution_engine.v2.blocks.kinds import IMAGE_KIND
from roboflow_workflows.execution_engine.v2.blocks.predictions import (
    OBJECT_DETECTION_PREDICTION_KIND,
)
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.declaration import (
    Block,
    BlockParams,
    Output,
    Ref,
)
from roboflow_workflows.execution_engine.v2.kinds import FLOAT_KIND
from roboflow_workflows.execution_engine.v2.phases import phase
from supervision.annotators.utils import resolve_text_background_xyxy

from inference_models.models.base.object_detection import (
    Detections,
    ObjectDetectionModel,
)

PREDICTION_TYPE = "object-detection"
DEFAULT_CONFIDENCE = 0.4

# V1 bounding box and label defaults: palette by class, 2 px boxes, white
# "<class> <confidence>" text on a box-colored patch above the top-left corner.
PALETTE = sv.ColorPalette.DEFAULT
BOX_THICKNESS = 2
TEXT_COLOR_BGR = (255, 255, 255)
TEXT_SCALE = 1.0
TEXT_THICKNESS = 1
TEXT_PADDING = 10
BORDER_RADIUS = 0
LABEL_POSITION = sv.Position.TOP_LEFT

# Sprites are keyed by label text, color and clip window. A stream keeps a
# small vocabulary (class x two-digit confidence), so this bound is rarely hit.
SPRITE_CACHE_SIZE = 256


class _NetworkInput(NamedTuple):
    batch: Any
    metadata: Any
    pre_ms: float


class _RawOutput(NamedTuple):
    output: Any
    model_ms: float


def _elapsed_ms(started: float) -> float:
    elapsed = (perf_counter() - started) * 1000.0

    return elapsed


def _require_rgb(image: ImageData, *, block: str) -> None:
    if image.channels != 3:
        raise ValueError(f"{block} needs a 3-channel RGB image, got {image!r}")


class ObjectDetector(Block):
    """Detect objects with an ``inference_models`` object detection model.

    The phases are the model's own ``pre_process``, ``forward`` and
    ``post_process``; ``run`` composes the same three. The predictions are the
    model's native ``Detections`` with this image's ``prediction_metadata``,
    ``class_names`` and ``prediction_type`` in ``image_metadata``. Boxes are in
    this image's pixel coordinates.
    """

    type = "live_detection/object_detector@v1"
    outputs = {
        "predictions": Output(
            OBJECT_DETECTION_PREDICTION_KIND,
            source="image",
            description="Native Detections in this image's pixel coordinates.",
        ),
        "pre_ms": Output(
            FLOAT_KIND, source="image", description="pre_process wall time, ms."
        ),
        "model_ms": Output(
            FLOAT_KIND, source="image", description="forward wall time, ms."
        ),
        "post_ms": Output(
            FLOAT_KIND,
            source="image",
            description="post_process and metadata wall time, ms.",
        ),
    }

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(description="RGB image to detect objects in.")
        confidence: StrictFloat | Ref(FLOAT_KIND) = Field(
            default=DEFAULT_CONFIDENCE,
            ge=0,
            le=1,
            description="Minimum detection confidence, 0..1.",
        )

    def __init__(self, *, detection_model: ObjectDetectionModel):
        self.model = detection_model

    @phase
    def network_input(self, image: ImageData) -> _NetworkInput:
        """Resize and normalize the image for the network (``pre_process``)."""
        _require_rgb(image, block=self.type)

        started = perf_counter()
        batch, metadata = self.model.pre_process(
            image.tensor_image, input_color_format="rgb"
        )
        network_input = _NetworkInput(batch, metadata, _elapsed_ms(started))

        return network_input

    @phase
    def raw_output(self, network_input: _NetworkInput) -> _RawOutput:
        """Run the network (``forward``)."""
        started = perf_counter()
        output = self.model.forward(network_input.batch)
        raw_output = _RawOutput(output, _elapsed_ms(started))

        return raw_output

    @phase
    def result(
        self,
        raw_output: _RawOutput,
        network_input: _NetworkInput,
        image: ImageData,
        confidence: float,
    ) -> Dict[str, Any]:
        """Decode boxes (``post_process``) and attach the image metadata."""
        started = perf_counter()
        batch = self.model.post_process(
            raw_output.output, network_input.metadata, confidence=confidence
        )
        if len(batch) != 1:
            raise ValueError(
                f"post_process returned {len(batch)} predictions for one image"
            )

        (predictions,) = batch
        predictions.image_metadata = {
            **(predictions.image_metadata or {}),
            **image.prediction_metadata(),
            "class_names": dict(enumerate(self.model.class_names)),
            "prediction_type": PREDICTION_TYPE,
        }
        result = {
            "predictions": predictions,
            "pre_ms": network_input.pre_ms,
            "model_ms": raw_output.model_ms,
            "post_ms": _elapsed_ms(started),
        }

        return result

    def run(self, *, image: ImageData, confidence: float) -> Dict[str, Any]:
        """Detect objects in one image.

        Args:
            image: RGB image.
            confidence: Minimum detection confidence.

        Returns:
            ``predictions`` (native ``Detections``) and the wall times
            ``pre_ms``, ``model_ms`` and ``post_ms``.

        Raises:
            PhaseFailure: Phase ``network_input`` on a non-RGB image; phase
                ``result`` when the model returns other than one prediction.
        """
        network_input = self.network_input(image)
        raw_output = self.raw_output(network_input)
        result = self.result(raw_output, network_input, image, confidence)

        return result


class BoxVisualization(Block):
    """Draw class-colored box borders on a copy of the image.

    The copy is a contiguous CHW tensor owned by the output, also with no
    detections, so a later painter may draw on it in place. The input image
    is never changed.
    """

    type = "live_detection/box_visualization@v1"
    outputs = {
        "image": Output(IMAGE_KIND, source="image", description="New annotated image."),
        "draw_ms": Output(
            FLOAT_KIND, source="image", description="Copy and paint wall time, ms."
        ),
    }

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(description="RGB image to annotate.")
        predictions: Ref(OBJECT_DETECTION_PREDICTION_KIND) = Field(
            description="Boxes in the image's pixel coordinates."
        )

    def run(self, *, image: ImageData, predictions: Detections) -> Dict[str, Any]:
        """Paint the box borders of ``predictions`` onto a copy of ``image``.

        Args:
            image: RGB image; not modified.
            predictions: Detections to draw.

        Returns:
            ``image``: the copy with boxes, same identity and provenance as
            ``image``; ``draw_ms``: wall time of the copy and the paint.

        Raises:
            ValueError: When the image is not RGB.
        """
        _require_rgb(image, block=self.type)

        started = perf_counter()
        scene = image.tensor_image.clone(memory_format=torch.contiguous_format)
        if len(predictions) > 0:
            class_ids = predictions.class_id.cpu().numpy()
            colors_rgb = np.asarray(
                [PALETTE.by_idx(int(class_id)).as_rgb() for class_id in class_ids],
                dtype=np.uint8,
            )
            xyxy = predictions.xyxy.cpu().numpy().astype(int)
            gpu_draw_boxes(scene, xyxy, colors_rgb, BOX_THICKNESS)

        result = {"image": image.with_pixels(scene), "draw_ms": _elapsed_ms(started)}

        return result


class LabelVisualization(Block):
    """Paste "<class> <confidence>" labels into the image, in place.

    Each label is rendered once with supervision's exact cv2 calls into a
    cached sprite, then all labels are pasted with one indexed tensor store
    (the V1 sprite path). Bind an image this workflow created, such as the
    box painter's output. Bound directly to a workflow input, this block
    overwrites the caller's pixels; the compiler flags conflicting readers,
    not caller ownership. The image tensor must be contiguous.
    """

    type = "live_detection/label_visualization@v1"
    mutates = ("image",)
    outputs = {
        "image": Output(
            IMAGE_KIND, source="image", description="The same image, now labelled."
        ),
        "draw_ms": Output(
            FLOAT_KIND, source="image", description="Paint wall time, ms."
        ),
    }

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(description="RGB image changed in place.")
        predictions: Ref(OBJECT_DETECTION_PREDICTION_KIND) = Field(
            description="Detections with class_names in image_metadata."
        )

    def __init__(self):
        # The pipeline serializes this block, so the sprite cache needs no lock.
        self._sprites: "OrderedDict[tuple, _LabelSprite]" = OrderedDict()

    def run(self, *, image: ImageData, predictions: Detections) -> Dict[str, Any]:
        """Paste one label per detection above its box's top-left corner.

        Args:
            image: Contiguous RGB image; modified in place.
            predictions: Detections whose ``image_metadata["class_names"]``
                names every class id.

        Returns:
            ``image``: the same image object; ``draw_ms``: wall time.

        Raises:
            ValueError: On a non-RGB or non-contiguous image, or a class id
                without a name.
        """
        _require_rgb(image, block=self.type)
        if not image.tensor_image.is_contiguous():
            raise ValueError(
                f"{self.type} paints in place and needs a contiguous image tensor; "
                "bind the box painter's output or another owned contiguous image"
            )

        started = perf_counter()
        labels = _class_and_confidence_labels(predictions)
        scene = image.tensor_image
        frame_hw = (int(scene.shape[1]), int(scene.shape[2]))
        corners = predictions.xyxy[:, :2].cpu().numpy().astype(int)
        class_ids = predictions.class_id.cpu().numpy()
        sprites: List[_LabelSprite] = []
        origins: List[Tuple[int, int]] = []
        for label, (x, y), class_id in zip(labels, corners, class_ids):
            background_bgr = PALETTE.by_idx(int(class_id)).as_bgr()
            placed = self._place(
                label,
                anchor=(int(x), int(y)),
                background_bgr=background_bgr,
                frame_hw=frame_hw,
                device=scene.device,
            )
            if placed is not None:
                sprites.append(placed[0])
                origins.append(placed[1])
        gpu_paste_label_sprites(scene, sprites, origins)

        result = {"image": image, "draw_ms": _elapsed_ms(started)}

        return result

    def _place(
        self,
        label: str,
        *,
        anchor: Tuple[int, int],
        background_bgr: Tuple[int, int, int],
        frame_hw: Tuple[int, int],
        device: torch.device,
    ) -> Optional[Tuple[_LabelSprite, Tuple[int, int]]]:
        # Port of LabelVisualizationBlockV1.run's per-label geometry: an
        # interior sprite with a margin on every side, or, for a label that
        # crosses the frame edge, a variant whose canvas ends at that edge
        # (cv2 anti-aliases clipped strokes differently). None: fully outside.
        frame_h, frame_w = frame_hw
        measurement = _measure_label(label, TEXT_SCALE, TEXT_THICKNESS, TEXT_PADDING)
        background_xyxy = resolve_text_background_xyxy(
            center_coordinates=anchor,
            text_wh=(measurement.width_padded, measurement.height_padded),
            position=LABEL_POSITION,
        )
        bx1, by1, bx2, by2 = (
            int(value) for value in np.asarray(background_xyxy, dtype=np.float32)
        )
        margin = measurement.margin

        sprite = self._sprite(
            label,
            measurement=measurement,
            background_bgr=background_bgr,
            device=device,
            box_in_canvas=(
                margin,
                margin,
                margin + measurement.width_padded,
                margin + measurement.height_padded,
            ),
            canvas_hw=(
                measurement.height_padded + 1 + 2 * margin,
                measurement.width_padded + 1 + 2 * margin,
            ),
            frame_edge_sides=(False, False, False, False),
        )
        origin = (bx1 - sprite.offset_x, by1 - sprite.offset_y)
        inside = (
            origin[0] + sprite.col_min >= 0
            and origin[1] + sprite.row_min >= 0
            and origin[0] + sprite.col_max < frame_w
            and origin[1] + sprite.row_max < frame_h
        )
        if inside:
            return sprite, origin

        window_x1, window_y1 = max(bx1 - margin, 0), max(by1 - margin, 0)
        window_x2 = min(bx2 + 1 + margin, frame_w)
        window_y2 = min(by2 + 1 + margin, frame_h)
        if window_x2 <= window_x1 or window_y2 <= window_y1:
            return None

        sprite = self._sprite(
            label,
            measurement=measurement,
            background_bgr=background_bgr,
            device=device,
            box_in_canvas=(
                bx1 - window_x1,
                by1 - window_y1,
                bx2 - window_x1,
                by2 - window_y1,
            ),
            canvas_hw=(window_y2 - window_y1, window_x2 - window_x1),
            frame_edge_sides=(
                bx1 - margin < 0,
                by1 - margin < 0,
                bx2 + 1 + margin > frame_w,
                by2 + 1 + margin > frame_h,
            ),
        )
        origin = (bx1 - sprite.offset_x, by1 - sprite.offset_y)

        return sprite, origin

    def _sprite(
        self,
        label: str,
        *,
        measurement: _LabelMeasurement,
        background_bgr: Tuple[int, int, int],
        device: torch.device,
        box_in_canvas: Tuple[int, int, int, int],
        canvas_hw: Tuple[int, int],
        frame_edge_sides: Tuple[bool, bool, bool, bool],
    ) -> _LabelSprite:
        # LRU lookup; typography is fixed, so the key is what varies per label.
        key = (
            label,
            background_bgr,
            str(device),
            box_in_canvas,
            canvas_hw,
            frame_edge_sides,
        )
        sprite = self._sprites.get(key)
        if sprite is not None:
            self._sprites.move_to_end(key)
            return sprite

        sprite = _render_label_sprite(
            measurement=measurement,
            text_color_bgr=TEXT_COLOR_BGR,
            background_color_bgr=background_bgr,
            text_scale=TEXT_SCALE,
            text_thickness=TEXT_THICKNESS,
            text_padding=TEXT_PADDING,
            border_radius=BORDER_RADIUS,
            device=device,
            box_in_canvas=box_in_canvas,
            canvas_hw=canvas_hw,
            frame_edge_sides=frame_edge_sides,
        )
        self._sprites[key] = sprite
        if len(self._sprites) > SPRITE_CACHE_SIZE:
            self._sprites.popitem(last=False)

        return sprite


def _class_and_confidence_labels(predictions: Detections) -> List[str]:
    # V1 "Class and Confidence" text; a missing name is an error, not a guess.
    if len(predictions) == 0:
        return []

    class_names = (predictions.image_metadata or {}).get("class_names")
    if class_names is None:
        raise ValueError("predictions.image_metadata has no class_names")

    labels = []
    class_ids = predictions.class_id.cpu().numpy()
    confidences = predictions.confidence.cpu().numpy()
    for class_id, confidence in zip(class_ids, confidences):
        if int(class_id) not in class_names:
            raise ValueError(f"class id {int(class_id)} has no name in class_names")
        labels.append(f"{class_names[int(class_id)]} {confidence:.2f}")

    return labels


def create_catalogue() -> Catalogue:
    """Collect the three live detection blocks and the kinds they exchange.

    Returns:
        A catalogue with the detector, box and label blocks only.
    """
    catalogue = Catalogue(
        [ObjectDetector, BoxVisualization, LabelVisualization],
        kinds=[IMAGE_KIND, OBJECT_DETECTION_PREDICTION_KIND, FLOAT_KIND],
    )

    return catalogue
