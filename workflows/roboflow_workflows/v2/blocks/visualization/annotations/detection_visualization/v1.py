"""Boxes then labels: ``roboflow_core/detection_visualization@v1``.

One step for the common pair ``bounding_box_visualization@v1`` followed by
``label_visualization@v2`` on the same predictions. Each implementation runs
the two existing implementations of its quality in sequence::

    image ──▶ boxes (all rows, row order) ──▶ labels (all rows, row order) ──▶ image

Occlusion is global: every box is drawn before any label, so a later row's
box never covers an earlier row's label. Labels, palette, text style, anchors
and frame mapping are exactly those of the label block; box widths and
roundness are those of the box block. Inputs and predictions are not changed.
The output is a new image; the intermediate boxed image is an owned copy.
CUDA writes on the caller stream finish before return, including empty inputs
and failures after work has been queued.
"""

from typing import Any, Dict

import torch
from pydantic import Field, StrictInt, model_validator
from roboflow_workflows.v2.blocks.visualization.annotations.bounding_box_visualization.v1 import (
    _CHECKS as _BOX_CHECKS,
)
from roboflow_workflows.v2.blocks.visualization.annotations.bounding_box_visualization.v1 import (
    AccurateBoundingBoxes,
    FastBoundingBoxes,
)
from roboflow_workflows.v2.blocks.visualization.annotations.label_visualization.v2 import (
    _CHECKS as _LABEL_CHECKS,
)
from roboflow_workflows.v2.blocks.visualization.annotations.label_visualization.v2 import (
    AccurateLabels,
    FastLabels,
    LabelVisualizationBlock,
)
from roboflow_workflows.v2.blocks.visualization.annotations.params import (
    Number,
    check_literals,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import FLOAT_KIND, INTEGER_KIND
from roboflow_workflows.v2.engine.declaration import Block, Output, Ref
from roboflow_workflows.v2.engine.implementations import Implementation

BLOCK_TYPE = "roboflow_core/detection_visualization@v1"


class AccurateDetections(Implementation):
    """Run the accurate box implementation, then the accurate label one.

    Pixels equal ``bounding_box_visualization@v1`` chained into
    ``label_visualization@v2`` with the same parameters, both accurate.
    """

    name = "accurate"
    quality = ("accurate",)

    def __init__(self) -> None:
        self._boxes = AccurateBoundingBoxes()
        self._labels = AccurateLabels()

    def run(
        self,
        *,
        image: ImageData,
        predictions: Any,
        color_palette: str,
        palette_size: int,
        custom_colors: Any,
        color_axis: str,
        text: str,
        text_position: str,
        text_color: str,
        text_size_mode: str,
        text_scale: float,
        text_thickness: int,
        text_padding: int,
        border_radius: int,
        mask_quality: str,
        thickness: int,
        roundness: float,
    ) -> Dict[str, Any]:
        """Draw every box, then every label of ``predictions`` on a copy of ``image``.

        Args:
            image: RGB image; not modified.
            predictions: A box carrier (see ``rows``).
            color_palette: Palette name; shared by boxes and labels.
            palette_size: Matplotlib sample count.
            custom_colors: ``CUSTOM`` palette colors.
            color_axis: ``CLASS``, ``INDEX`` or ``TRACK``.
            text: Text option or ``bboxes_metadata`` key (label block).
            text_position: Anchor of each label.
            text_color: Text color.
            text_size_mode: ``Manual`` or ``Automatic``.
            text_scale: Font scale.
            text_thickness: Text stroke width.
            text_padding: Background padding.
            border_radius: Background corner radius.
            mask_quality: Mask quality for ``Area`` and ``CENTER_OF_MASS``.
            thickness: Box outline width.
            roundness: Box corner roundness in [0, 1].

        Returns:
            ``image``: the new annotated image.

        Raises:
            ValueError: On invalid parameters, before anything is drawn, or
                on any error of the box or label block for these predictions.
        """
        values = dict(
            text_position=text_position,
            text_color=text_color,
            text_size_mode=text_size_mode,
            text_scale=text_scale,
            text_thickness=text_thickness,
            text_padding=text_padding,
            border_radius=border_radius,
            mask_quality=mask_quality,
            thickness=thickness,
            roundness=roundness,
        )
        for checks in (_BOX_CHECKS, _LABEL_CHECKS):
            for name, check in checks.items():
                check(values[name])
        palette = dict(
            color_palette=color_palette,
            palette_size=palette_size,
            custom_colors=custom_colors,
            color_axis=color_axis,
        )

        # The reference primitives queue writes on the caller stream. Retain
        # their intermediates and finish those writes even on a later error.
        stream = (
            torch.cuda.current_stream(image.device)
            if image.device.type == "cuda"
            else None
        )
        try:
            boxed = self._boxes.run(
                image=image,
                predictions=predictions,
                thickness=thickness,
                roundness=roundness,
                **palette,
            )
            result = self._labels.run(
                image=boxed["image"],
                predictions=predictions,
                text=text,
                text_position=text_position,
                text_color=text_color,
                text_size_mode=text_size_mode,
                text_scale=text_scale,
                text_thickness=text_thickness,
                text_padding=text_padding,
                border_radius=border_radius,
                mask_quality=mask_quality,
                **palette,
            )
        finally:
            if stream is not None:
                stream.synchronize()

        return result


class FastDetections(AccurateDetections):
    """Run the fast box implementation, then the fast label one.

    Same order, parameters and frame mapping as the accurate implementation.
    Pixels can differ from it (see ``DetectionVisualizationBlock``).
    """

    name = "fast"
    quality = ("fast",)

    def __init__(self) -> None:
        self._boxes = FastBoundingBoxes()
        self._labels = FastLabels()


class DetectionVisualizationBlock(Block):
    """Draw each detection's box, then its label, onto a copy of the image.

    Equivalent to ``bounding_box_visualization@v1`` followed by
    ``label_visualization@v2`` with the same palette, in one step. All boxes
    are drawn in row order first, then all labels in row order.

    The default accurate implementation equals that chain of accurate
    implementations pixel for pixel. The fast implementation chains the fast
    ones: square boxes are drawn as tensor border bands whose corner joins,
    borders and pixel rounding can rasterize differently from OpenCV
    (rounded boxes use the accurate renderer); opaque single-line labels are
    pasted from cached sprites. Label texts, row order, palette, text style,
    anchors and frame mapping are the same in both. Fast output is not
    guaranteed to equal accurate output, nor to be faster on every input.

    Example: ``text="Class and Confidence"`` draws 2 px class-colored boxes
    and writes ``"car 0.93"`` above each box's top-left corner.
    """

    type = BLOCK_TYPE
    implementations = (AccurateDetections, FastDetections)
    prunable = True
    metadata = {
        "name": "Detection Visualization",
        "section": "visualization",
    }
    outputs = {
        "image": Output(
            IMAGE_KIND,
            source="image",
            description=(
                "New annotated image with the input's identity, frame, source, "
                "time and parent row."
            ),
        ),
    }

    class Params(LabelVisualizationBlock.Params):
        thickness: StrictInt | Ref(INTEGER_KIND) = Field(
            default=2,
            description="Box outline width in pixels; >= 1.",
            examples=[2, "$inputs.thickness"],
        )
        roundness: Number | Ref(FLOAT_KIND) = Field(
            default=0.0,
            description="Box corner roundness in [0, 1]; 0 draws square corners.",
            examples=[0.0, 0.5, "$inputs.roundness"],
        )

        # The label block's copy_image, palette and label validators are
        # inherited; this one adds the box parameters.
        @model_validator(mode="after")
        def _check_box_literals(self) -> "DetectionVisualizationBlock.Params":
            check_literals(self.__dict__, _BOX_CHECKS)

            return self
