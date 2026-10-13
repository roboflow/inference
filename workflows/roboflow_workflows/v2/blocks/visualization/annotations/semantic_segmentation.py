"""Semantic class grids: ``v2/semantic_segmentation_visualization@v1``.

The direct consumer of native ``SemanticSegmentationResult`` (decision 018
F3): the label grid is colored as it is, without instance rows::

    colored = image; colored[grid == c] = palette[c]   for every painted class c
    output  = round(colored * opacity + image * (1 - opacity))

This equals ``v2/mask_visualization@v1`` given one dense mask per painted
class (color axis CLASS). Pixels of the class named exactly ``background``
(case-insensitive) stay unpainted unless ``skip_background`` is false; other
names such as ``unlabeled`` need ``skip_classes``. Everything runs on the
grid's device; no host reads.

The grid must describe this image: the image the model saw, at its size.
Grids of crops or resized images are not resampled (no lossless mapping
exists); a resampling option would be a separate decision. Known gap: no
class legend or class-name labels yet.
"""

from typing import Any, Dict, List, Mapping, Set

import torch
from pydantic import Field, StrictBool, model_validator
from roboflow_workflows.v2.blocks.visualization.annotations.base import (
    AnnotationBlock,
    annotated,
    canvas_of,
)
from roboflow_workflows.v2.blocks.visualization.annotations.blend import (
    blend,
    device_colors,
)
from roboflow_workflows.v2.blocks.visualization.annotations.colors import (
    resolve_palette,
)
from roboflow_workflows.v2.blocks.visualization.annotations.params import (
    Number,
    PaletteParams,
    check_literals,
    checked_opacity,
)
from roboflow_workflows.v2.data.frames import FRAME_RECORD_KEY, frame_of
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.kinds import (
    BOOLEAN_KIND,
    FLOAT_KIND,
    LIST_OF_VALUES_KIND,
)
from roboflow_workflows.v2.data.predictions import NATIVE_SEMANTIC_SEGMENTATION_KIND
from roboflow_workflows.v2.engine.declaration import Ref

from inference_models.models.base.semantic_segmentation import (
    SemanticSegmentationResult,
)

BACKGROUND_NAME = "background"


class SemanticSegmentationVisualizationBlock(AnnotationBlock):
    """Overlay a semantic segmentation's class grid on a copy of the image.

    Example: a road/car/background model with defaults paints road and car
    pixels half-transparent in their class colors and leaves background
    pixels as they are.
    """

    type = "v2/semantic_segmentation_visualization@v1"
    metadata = {
        "name": "Semantic Segmentation Visualization",
        "section": "visualization",
    }

    class Params(PaletteParams):
        predictions: Ref(NATIVE_SEMANTIC_SEGMENTATION_KIND) = Field(
            description=(
                "Native semantic segmentation of this exact image: a class id per "
                "pixel, with image_metadata['class_names']. Draw on the image "
                "the model saw (for example the resized image, not the "
                "original); grids are not resampled. Colors only: no class "
                "legend or labels yet."
            ),
            examples=["$steps.semantic_segmentation.predictions"],
        )
        opacity: Number | Ref(FLOAT_KIND) = Field(
            default=0.5,
            description="Weight of the class color, in [0, 1].",
            examples=[0.5, "$inputs.opacity"],
        )
        skip_background: StrictBool | Ref(BOOLEAN_KIND) = Field(
            default=True,
            description=(
                "Leave pixels of the class named exactly 'background' "
                "(case-insensitive) unpainted. Other names such as 'unlabeled' "
                "need skip_classes. No effect when the model has no such class."
            ),
            examples=[True, False],
        )
        skip_classes: List[str] | Ref(LIST_OF_VALUES_KIND) = Field(
            default_factory=list,
            description="More class names whose pixels stay unpainted.",
            examples=[["sky"], "$inputs.skip_classes"],
        )

        @model_validator(mode="after")
        def _check_literals(self) -> "SemanticSegmentationVisualizationBlock.Params":
            check_literals(self.__dict__, _CHECKS)

            return self

    def run(
        self,
        *,
        image: ImageData,
        predictions: SemanticSegmentationResult,
        color_palette: str,
        palette_size: int,
        custom_colors: Any,
        opacity: float,
        skip_background: bool,
        skip_classes: Any,
    ) -> Dict[str, Any]:
        """Color the class grid of ``predictions`` over a copy of ``image``.

        Args:
            image: RGB image the grid describes; not modified.
            predictions: Native semantic segmentation.
            color_palette: Palette name; classes pick colors by class id.
            palette_size: Matplotlib sample count.
            custom_colors: ``CUSTOM`` palette colors.
            opacity: Class color weight.
            skip_background: Leave ``background`` pixels unpainted.
            skip_classes: More class names to leave unpainted.

        Returns:
            ``image``: the new annotated image.

        Raises:
            ValueError: On invalid parameters, unknown class names, or a grid
                of another image or size.
        """
        values = dict(opacity=opacity)
        for name, check in _CHECKS.items():
            check(values[name])
        if not isinstance(skip_background, bool):
            raise ValueError(
                f"skip_background must be true or false, got {skip_background!r}"
            )
        palette = resolve_palette(
            color_palette, palette_size=palette_size, custom_colors=custom_colors
        )
        grid = _grid_of(predictions, image=image, block=self.type)
        skipped = _skipped_ids(
            predictions.image_metadata or {},
            skip_background=skip_background,
            skip_classes=skip_classes,
            block=self.type,
        )

        canvas = canvas_of(image, block=self.type)
        grid = grid.to(canvas.device)
        colors = device_colors(palette, grid).permute(2, 0, 1)
        painted = ~torch.isin(
            grid, torch.tensor(sorted(skipped), dtype=grid.dtype, device=grid.device)
        )
        colored = torch.where(painted, colors, canvas)
        result = annotated(image, blend(canvas, colored, opacity=opacity))

        return result


def _grid_of(
    predictions: SemanticSegmentationResult, *, image: ImageData, block: str
) -> torch.Tensor:
    grid = predictions.segmentation_map
    remedy = (
        "Draw a semantic segmentation on the image the model saw (the "
        "segmentation step's image input); class grids are not resampled."
    )
    if FRAME_RECORD_KEY in (predictions.image_metadata or {}):
        frame = frame_of(predictions)
        if frame.image_id != image.image_id:
            raise ValueError(
                f"{block}: the class grid describes image {frame.image_id!r} "
                f"of (h, w)={tuple(frame.size_hw)}, not image {image.image_id!r} "
                f"of {tuple(image.size_hw)}. {remedy}"
            )
    if tuple(grid.shape) != tuple(image.size_hw):
        raise ValueError(
            f"{block}: the class grid is {tuple(grid.shape)} but the image is "
            f"{tuple(image.size_hw)}. {remedy}"
        )

    return grid


def _skipped_ids(
    image_metadata: Mapping[str, Any],
    *,
    skip_background: bool,
    skip_classes: Any,
    block: str,
) -> Set[int]:
    if isinstance(skip_classes, str) or not isinstance(skip_classes, (list, tuple)):
        raise ValueError(
            f"skip_classes must be a list of class names, got {skip_classes!r}"
        )
    names = image_metadata.get("class_names")
    if names is not None and not isinstance(names, Mapping):
        names = dict(enumerate(names))
    if names is None:
        if skip_classes:
            raise ValueError(
                f"{block}: skip_classes needs image_metadata['class_names'], which "
                "these predictions do not have"
            )
        return set()

    ids_by_name = {str(name): int(class_id) for class_id, name in names.items()}
    unknown = [name for name in skip_classes if name not in ids_by_name]
    if unknown:
        raise ValueError(
            f"{block}: skip_classes {unknown} are not classes of this model; "
            f"classes: {sorted(ids_by_name)}"
        )

    skipped = {ids_by_name[name] for name in skip_classes}
    if skip_background:
        skipped |= {
            class_id
            for name, class_id in ids_by_name.items()
            if name.lower() == BACKGROUND_NAME
        }

    return skipped


_CHECKS = {"opacity": checked_opacity}
