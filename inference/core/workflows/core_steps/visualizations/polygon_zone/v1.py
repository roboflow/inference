import hashlib
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import cv2 as cv
import numpy as np
import supervision as sv
from pydantic import ConfigDict, Field

from inference.core.workflows.core_steps.visualizations.common.base import (
    OUTPUT_IMAGE_KEY,
    VisualizationBlock,
    VisualizationManifest,
)
from inference.core.workflows.core_steps.visualizations.common.utils import str_to_color
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData
from inference.core.workflows.execution_engine.entities.types import (
    FLOAT_ZERO_TO_ONE_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
    FloatZeroToOne,
    Selector,
)
from inference.core.workflows.prototypes.block import BlockResult, WorkflowBlockManifest

TYPE: str = "roboflow_core/polygon_zone_visualization@v1"
SHORT_DESCRIPTION = "Apply a mask over a polygon zone in an image."
LONG_DESCRIPTION = """
The `PolygonZoneVisualization` block draws polygon zone
in an image with a specified color and opacity.
Please note: zones will be drawn on top of image passed to this block,
this block should be placed before other visualization blocks in the workflow.
"""


class PolygonZoneVisualizationManifest(VisualizationManifest):
    type: Literal[f"{TYPE}"]
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Polygon Zone Visualization",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "visualization",
            "search_keywords": ["annotator"],
            "ui_manifest": {
                "section": "visualization",
                "icon": "far fa-hexagon",
                "blockPriority": 15,
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
    zone: Union[list, Selector(kind=[LIST_OF_VALUES_KIND]), Selector(kind=[LIST_OF_VALUES_KIND])] = Field(  # type: ignore
        description="Polygon zones (one for each batch) in a format [[(x1, y1), (x2, y2), (x3, y3), ...], ...];"
        " each zone must consist of more than 2 points",
        examples=["$inputs.zones"],
    )
    color: Union[str, Selector(kind=[STRING_KIND])] = Field(  # type: ignore
        description="Color of the zone.",
        default="#5bb573",
        examples=["WHITE", "#FFFFFF", "rgb(255, 255, 255)" "$inputs.background_color"],
    )
    opacity: Union[FloatZeroToOne, Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])] = Field(  # type: ignore
        description="Transparency of the Mask overlay.",
        default=0.3,
        examples=[0.3, "$inputs.opacity"],
    )

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class PolygonZoneVisualizationBlockV1(VisualizationBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache: Dict[str, np.ndarray] = {}

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return PolygonZoneVisualizationManifest

    def getAnnotator(
        self,
        **kwargs,
    ) -> sv.PolygonZoneAnnotator:
        pass

    def run(
        self,
        image: WorkflowImageData,
        zone: List[Tuple[int, int]],
        copy_image: bool,
        color: str,
        opacity: float,
    ) -> BlockResult:
        h, w, *_ = image.numpy_image.shape
        zone_fingerprint = hashlib.md5(str(zone).encode()).hexdigest()
        key = f"{zone_fingerprint}_{color}_{opacity}_{w}_{h}"
        if key not in self._cache:
            mask = np.zeros(
                shape=image.numpy_image.shape,
                dtype=image.numpy_image.dtype,
            )
            mask = cv.fillPoly(
                img=mask,
                pts=[np.array(zone)],
                color=str_to_color(color).as_bgr(),
            )
            self._cache[key] = mask
        mask = self._cache[key]

        np_image = image.numpy_image
        if copy_image:
            np_image = np_image.copy()
        annotated_image = cv.addWeighted(
            src1=mask,
            alpha=opacity,
            src2=np_image,
            beta=1,
            gamma=0,
        )
        return {
            OUTPUT_IMAGE_KEY: WorkflowImageData.copy_and_replace(
                origin_image_data=image, numpy_image=annotated_image
            )
        }
