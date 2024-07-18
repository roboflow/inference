from abc import ABC, abstractmethod
from typing import List, Literal, Optional, Type, Union

import supervision as sv
from pydantic import AliasChoices, ConfigDict, Field

from inference.core.workflows.core_steps.visualizations.utils import str_to_color
from inference.core.workflows.entities.base import OutputDefinition, WorkflowImageData
from inference.core.workflows.entities.types import (
    BATCH_OF_IMAGES_KIND,
    BATCH_OF_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    BATCH_OF_KEYPOINT_DETECTION_PREDICTION_KIND,
    BATCH_OF_OBJECT_DETECTION_PREDICTION_KIND,
    BOOLEAN_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
    StepOutputImageSelector,
    StepOutputSelector,
    WorkflowImageSelector,
    WorkflowParameterSelector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

OUTPUT_IMAGE_KEY: str = "image"


class VisualizationManifest(WorkflowBlockManifest, ABC):
    model_config = ConfigDict(
        json_schema_extra={
            "license": "Apache-2.0",
            "block_type": "visualization",
        }
    )
    predictions: StepOutputSelector(
        kind=[
            BATCH_OF_OBJECT_DETECTION_PREDICTION_KIND,
            BATCH_OF_INSTANCE_SEGMENTATION_PREDICTION_KIND,
            BATCH_OF_KEYPOINT_DETECTION_PREDICTION_KIND,
        ]
    ) = Field(  # type: ignore
        description="Predictions",
        examples=["$steps.object_detection_model.predictions"],
    )
    image: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
        title="Input Image",
        description="The input image for this step.",
        examples=["$inputs.image", "$steps.cropping.crops"],
        validation_alias=AliasChoices("image", "images"),
    )

    copy_image: Union[bool, WorkflowParameterSelector(kind=[BOOLEAN_KIND])] = Field(  # type: ignore
        description="Duplicate the image contents (vs overwriting the image in place). Deselect for chained visualizations that should stack on previous ones where the intermediate state is not needed.",
        default=True,
    )

    color_palette: Union[
        Literal[
            "DEFAULT",
            "CUSTOM",
            "ROBOFLOW",
            "Matplotlib Viridis",
            "Matplotlib Plasma",
            "Matplotlib Inferno",
            "Matplotlib Magma",
            "Matplotlib Cividis",
            # TODO: Re-enable once supervision 0.23 is released with a fix
            # "Matplotlib Twilight",
            # "Matplotlib Twilight_Shifted",
            # "Matplotlib HSV",
            # "Matplotlib Jet",
            # "Matplotlib Turbo",
            # "Matplotlib Rainbow",
            # "Matplotlib gist_rainbow",
            # "Matplotlib nipy_spectral",
            # "Matplotlib gist_ncar",
            "Matplotlib Pastel1",
            "Matplotlib Pastel2",
            "Matplotlib Paired",
            "Matplotlib Accent",
            "Matplotlib Dark2",
            "Matplotlib Set1",
            "Matplotlib Set2",
            "Matplotlib Set3",
            "Matplotlib Tab10",
            "Matplotlib Tab20",
            "Matplotlib Tab20b",
            "Matplotlib Tab20c",
            # TODO: Re-enable once supervision 0.23 is released with a fix
            # "Matplotlib Ocean",
            # "Matplotlib Gist_Earth",
            # "Matplotlib Terrain",
            # "Matplotlib Stern",
            # "Matplotlib gnuplot",
            # "Matplotlib gnuplot2",
            # "Matplotlib Spring",
            # "Matplotlib Summer",
            # "Matplotlib Autumn",
            # "Matplotlib Winter",
            # "Matplotlib Cool",
            # "Matplotlib Hot",
            # "Matplotlib Copper",
            # "Matplotlib Bone",
            # "Matplotlib Greys_R",
            # "Matplotlib Purples_R",
            # "Matplotlib Blues_R",
            # "Matplotlib Greens_R",
            # "Matplotlib Oranges_R",
            # "Matplotlib Reds_R",
        ],
        WorkflowParameterSelector(kind=[STRING_KIND]),
    ] = Field(  # type: ignore
        default="DEFAULT",
        description="Color palette to use for annotations.",
        examples=["DEFAULT", "$inputs.color_palette"],
    )

    palette_size: Union[
        int,
        WorkflowParameterSelector(kind=[INTEGER_KIND]),
    ] = Field(  # type: ignore
        default=10,
        description="Number of colors in the color palette. Applies when using a matplotlib `color_palette`.",
        examples=[10, "$inputs.palette_size"],
    )

    custom_colors: Union[
        List[str], WorkflowParameterSelector(kind=[LIST_OF_VALUES_KIND])
    ] = Field(  # type: ignore
        default=[],
        description='List of colors to use for annotations when `color_palette` is set to "CUSTOM".',
        examples=[["#FF0000", "#00FF00", "#0000FF"], "$inputs.custom_colors"],
    )

    color_axis: Union[
        Literal["INDEX", "CLASS", "TRACK"],
        WorkflowParameterSelector(kind=[STRING_KIND]),
    ] = Field(  # type: ignore
        default="CLASS",
        description="Strategy to use for mapping colors to annotations.",
        examples=["CLASS", "$inputs.color_axis"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name=OUTPUT_IMAGE_KEY,
                kind=[
                    BATCH_OF_IMAGES_KIND,
                ],
            ),
        ]


class VisualizationBlock(WorkflowBlock, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    @abstractmethod
    def get_manifest(cls) -> Type[VisualizationManifest]:
        pass

    @abstractmethod
    def getAnnotator(self, *args, **kwargs) -> sv.annotators.base.BaseAnnotator:
        pass

    @classmethod
    def getPalette(self, color_palette, palette_size, custom_colors):
        if color_palette == "CUSTOM":
            return sv.ColorPalette(
                colors=[str_to_color(color) for color in custom_colors]
            )
        elif hasattr(sv.ColorPalette, color_palette):
            return getattr(sv.ColorPalette, color_palette)
        else:
            palette_name = color_palette.replace("Matplotlib ", "")

            if palette_name in [
                "Greys_R",
                "Purples_R",
                "Blues_R",
                "Greens_R",
                "Oranges_R",
                "Reds_R",
                "Wistia",
                "Pastel1",
                "Pastel2",
                "Paired",
                "Accent",
                "Dark2",
                "Set1",
                "Set2",
                "Set3",
            ]:
                palette_name = palette_name.capitalize()
            else:
                palette_name = palette_name.lower()

            return sv.ColorPalette.from_matplotlib(palette_name, int(palette_size))

    @abstractmethod
    async def run(
        self,
        image: WorkflowImageData,
        predictions: sv.Detections,
        copy_image: bool,
        color_palette: Optional[str],
        palette_size: Optional[int],
        custom_colors: Optional[List[str]],
        color_axis: Optional[str],
        *args,
        **kwargs
    ) -> BlockResult:
        pass
