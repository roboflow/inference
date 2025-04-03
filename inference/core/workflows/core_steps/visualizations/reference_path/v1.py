from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import cv2
import numpy as np
from pydantic import ConfigDict, Field, field_validator

from inference.core.workflows.core_steps.visualizations.common.base import (
    OUTPUT_IMAGE_KEY,
    VisualizationManifest,
)
from inference.core.workflows.core_steps.visualizations.common.utils import str_to_color
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData
from inference.core.workflows.execution_engine.entities.types import (
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

SHORT_DESCRIPTION = "Draw a reference path in the image."
LONG_DESCRIPTION = """
The **Reference Path Visualization** block draws reference path in the image.
To be used in combination with **Path deviation** block - to display the reference
path.
"""


class ReferencePathVisualizationManifest(VisualizationManifest):
    type: Literal["roboflow_core/reference_path_visualization@v1"]
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Reference Path Visualization",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "visualization",
            "search_keywords": ["annotator"],
            "ui_manifest": {
                "section": "visualization",
                "icon": "fas fa-road",
                "blockPriority": 18,
                "supervision": True,
                "warnings": [
                    {
                        "property": "copy_image",
                        "value": False,
                        "message": "This setting will mutate its input image. If the input is used by other blocks, it may cause unexpected behavior.",
                    }
                ],
            },
        }
    )
    reference_path: Union[
        list,
        Selector(kind=[LIST_OF_VALUES_KIND]),
        Selector(kind=[LIST_OF_VALUES_KIND]),
    ] = Field(  # type: ignore
        description="Reference path in a format [(x1, y1), (x2, y2), (x3, y3), ...]",
        examples=["$inputs.expected_path"],
    )
    color: Union[str, Selector(kind=[STRING_KIND])] = Field(  # type: ignore
        description="Color of the zone.",
        default="#5bb573",
        examples=["WHITE", "#FFFFFF", "rgb(255, 255, 255)" "$inputs.background_color"],
    )
    thickness: Union[int, Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        description="Thickness of the lines in pixels.",
        default=2,
        examples=[2, "$inputs.thickness"],
    )

    @field_validator("thickness")
    @classmethod
    def validate_thickness_greater_than_zero(
        cls, value: Union[int, str]
    ) -> Union[int, str]:
        if isinstance(value, int) and value <= 0:
            raise ValueError("Thickness must be greater or equal to zero")
        return value

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class ReferencePathVisualizationBlockV1(WorkflowBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache: Dict[str, np.ndarray] = {}

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return ReferencePathVisualizationManifest

    def run(
        self,
        image: WorkflowImageData,
        reference_path: List[Union[Tuple[int, int], List[int]]],
        copy_image: bool,
        color: str,
        thickness: int,
    ) -> BlockResult:
        reference_path_array = np.array(reference_path)[:, :2].astype(np.int32)
        numpy_image = image.numpy_image
        result_image = cv2.polylines(
            numpy_image if not copy_image else numpy_image.copy(),
            [reference_path_array],
            False,
            str_to_color(color).as_bgr(),
            thickness,
        )
        return {
            OUTPUT_IMAGE_KEY: WorkflowImageData.copy_and_replace(
                origin_image_data=image,
                numpy_image=result_image,
            )
        }
