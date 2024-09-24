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
    FLOAT_KIND,
    FLOAT_ZERO_TO_ONE_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
    FloatZeroToOne,
    StepOutputSelector,
    WorkflowParameterSelector,
)
from inference.core.workflows.prototypes.block import BlockResult, WorkflowBlockManifest

TYPE: str = "roboflow_core/line_counter_visualization@v1"
SHORT_DESCRIPTION = "Paints a mask over line zone in an image."
LONG_DESCRIPTION = """
The `LineCounterZoneVisualization` block draws line
in an image with a specified color and opacity.
Please note: line zone will be drawn on top of image passed to this block,
this block should be placed before other visualization blocks in the workflow.
"""


class LineCounterZoneVisualizationManifest(VisualizationManifest):
    type: Literal[f"{TYPE}"]
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Line Counter Visualization",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "visualization",
        }
    )
    zone: Union[list, StepOutputSelector(kind=[LIST_OF_VALUES_KIND]), WorkflowParameterSelector(kind=[LIST_OF_VALUES_KIND])] = Field(  # type: ignore
        description="Line in the format [[x1, y1], [x2, y2]] consisting of exactly two points.",
        examples=[[[0, 50], [500, 50]], "$inputs.zones"],
    )
    color: Union[str, WorkflowParameterSelector(kind=[STRING_KIND])] = Field(  # type: ignore
        description="Color of the zone.",
        default="#5bb573",
        examples=["WHITE", "#FFFFFF", "rgb(255, 255, 255)" "$inputs.background_color"],
    )
    thickness: Union[int, WorkflowParameterSelector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        description="Thickness of the lines in pixels.",
        default=2,
        examples=[2, "$inputs.thickness"],
    )
    text_thickness: Union[int, WorkflowParameterSelector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        description="Thickness of the text in pixels.",
        default=1,
        examples=[1, "$inputs.text_thickness"],
    )
    text_scale: Union[float, WorkflowParameterSelector(kind=[FLOAT_KIND])] = Field(  # type: ignore
        description="Scale of the text.",
        default=1.0,
        examples=[1.0, "$inputs.text_scale"],
    )
    count_in: Union[int, WorkflowParameterSelector(kind=[INTEGER_KIND]), StepOutputSelector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        description="Reference to the number of objects that crossed into the line zone.",
        default=0,
        examples=["$steps.line_counter.count_in"],
        json_schema_extra={"always_visible": True},
    )
    count_out: Union[int, WorkflowParameterSelector(kind=[INTEGER_KIND]), StepOutputSelector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        description="Reference to the number of objects that crossed out of the line zone.",
        default=0,
        examples=["$steps.line_counter.count_out"],
        json_schema_extra={"always_visible": True},
    )
    opacity: Union[FloatZeroToOne, WorkflowParameterSelector(kind=[FLOAT_ZERO_TO_ONE_KIND])] = Field(  # type: ignore
        description="Transparency of the Mask overlay.",
        default=0.3,
        examples=[0.3, "$inputs.opacity"],
    )

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


class LineCounterZoneVisualizationBlockV1(VisualizationBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache: Dict[str, np.ndarray] = {}

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return LineCounterZoneVisualizationManifest

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
        thickness: int,
        text_thickness: int,
        text_scale: int,
        count_in: int,
        count_out: int,
        opacity: float,
    ) -> BlockResult:
        h, w, *_ = image.numpy_image.shape
        zone_fingerprint = hashlib.md5(str(zone).encode()).hexdigest()
        key = f"{zone_fingerprint}_{color}_{opacity}_{w}_{h}"
        x1, y1 = zone[0]
        x2, y2 = zone[1]
        if key not in self._cache:
            mask = np.zeros(
                shape=image.numpy_image.shape,
                dtype=image.numpy_image.dtype,
            )
            mask = cv.line(
                img=mask,
                pt1=(x1, y1),
                pt2=(x2, y2),
                color=str_to_color(color).as_bgr(),
                thickness=thickness,
            )
            self._cache[key] = mask
        mask = self._cache[key].copy()

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
        annotated_image = sv.draw_text(
            scene=annotated_image,
            text=f"in: {count_in}, out: {count_out}",
            text_anchor=sv.Point(x1, y1),
            text_thickness=text_thickness,
            text_scale=text_scale,
            background_color=sv.Color.WHITE,
            text_padding=0,
        )

        output = WorkflowImageData(
            parent_metadata=image.parent_metadata,
            workflow_root_ancestor_metadata=image.workflow_root_ancestor_metadata,
            numpy_image=annotated_image,
        )

        return {OUTPUT_IMAGE_KEY: output}
