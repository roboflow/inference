import hashlib
from collections import OrderedDict
from typing import List, Literal, Optional, Tuple, Type, Union

import cv2 as cv
import numpy as np
import supervision as sv
from pydantic import ConfigDict, Field

from inference.core.cache.lru_cache import LRUCache
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
Draw polygon zones on an image to visualize monitoring areas, displaying colored polygon overlays for zone-based detection and counting workflows that track objects within irregular, custom-defined regions.

## How This Block Works

This block takes an image and polygon zone coordinates (a list of points defining a polygon shape) and draws a filled polygon overlay to visualize the monitoring zone. The block:

1. Takes an image and polygon zone coordinates (a list of points: [(x1, y1), (x2, y2), (x3, y3), ...]) as input
2. Creates a filled polygon mask from the zone coordinates using the specified color
3. Overlays the filled polygon onto the image with the specified opacity, creating a semi-transparent zone visualization
4. Returns an annotated image with the polygon zone overlay on the original image

The block visualizes polygon zones used to define irregular monitoring areas for detection, counting, or tracking workflows. The polygon is drawn as a filled shape between the specified points, creating a closed region that can represent any custom area shape (unlike rectangular bounding boxes). This allows for flexible zone definitions that match real-world boundaries, such as specific floor areas, irregular regions of interest, or complex monitoring zones. The zone overlay is semi-transparent, allowing the underlying image details to remain visible while clearly indicating the monitoring area. Note: This block should typically be placed before other visualization blocks in the workflow, as the polygon zone provides a background reference layer for object detection visualizations.

## Common Use Cases

- **Zone Detection Visualization**: Visualize polygon zones for object detection or counting workflows where objects are tracked within irregular, custom-defined areas, displaying the monitoring boundaries clearly
- **Area-Based Monitoring**: Display polygon zones for area-based monitoring applications such as occupancy tracking, people counting in specific regions, or object presence detection within defined spaces
- **Custom Region Visualization**: Visualize custom monitoring regions that don't fit rectangular boundaries, such as irregular floor areas, complex room layouts, or specific zones within larger spaces
- **Security and Surveillance**: Display polygon zones for security monitoring, access control, or surveillance workflows where specific areas need to be visually marked and monitored
- **Retail and Business Analytics**: Visualize polygon zones for foot traffic analysis, customer movement tracking, or space utilization monitoring in retail, hospitality, or business intelligence applications
- **Real-Time Zone Monitoring**: Create visual overlays for real-time monitoring dashboards, live video feeds, or monitoring interfaces where polygon zones need to be clearly visible to indicate monitored areas

## Connecting to Other Blocks

The annotated image from this block can be connected to:

- **Zone detection or counting blocks** to receive polygon zone coordinates that are visualized
- **Other visualization blocks** (e.g., Bounding Box Visualization, Label Visualization, Polygon Visualization) to add object detection annotations on top of the polygon zone visualization
- **Data storage blocks** (e.g., Local File Sink, CSV Formatter, Roboflow Dataset Upload) to save images with polygon zone visualizations for documentation, reporting, or analysis
- **Webhook blocks** to send visualized results with polygon zones to external systems, APIs, or web applications for display in dashboards or monitoring tools
- **Notification blocks** (e.g., Email Notification, Slack Notification) to send annotated images with polygon zones as visual evidence in alerts or reports
- **Video output blocks** to create annotated video streams or recordings with polygon zone visualizations for live monitoring, zone visualization, or post-processing analysis
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
    zone: Union[
        list, Selector(kind=[LIST_OF_VALUES_KIND]), Selector(kind=[LIST_OF_VALUES_KIND])
    ] = Field(  # type: ignore
        description="Polygon zone coordinates in the format [[(x1, y1), (x2, y2), (x3, y3), ...], ...] defining one or more polygon shapes. Each zone must consist of more than 2 points to form a valid polygon. The polygon is drawn as a filled shape connecting these points in order, creating a closed region. Typically connected from zone detection or counting blocks that define monitoring areas.",
        examples=["$inputs.zones"],
    )
    color: Union[str, Selector(kind=[STRING_KIND])] = Field(  # type: ignore
        description="Color of the polygon zone overlay. Can be specified as a color name (e.g., 'WHITE', 'RED'), hex color code (e.g., '#5bb573', '#FFFFFF'), or RGB format (e.g., 'rgb(255, 255, 255)'). The polygon is filled with this color and overlaid with the specified opacity.",
        default="#5bb573",
        examples=["WHITE", "#FFFFFF", "rgb(255, 255, 255)", "$inputs.background_color"],
    )
    opacity: Union[FloatZeroToOne, Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])] = Field(  # type: ignore
        description="Opacity of the polygon zone overlay, ranging from 0.0 (fully transparent) to 1.0 (fully opaque). Controls how transparent the polygon zone appears over the image. Lower values create more transparent zones that blend with the background, while higher values create more opaque, visible zones. Typical values range from 0.2 to 0.5 for balanced visibility where both the zone and underlying image are visible.",
        default=0.3,
        examples=[0.3, "$inputs.opacity"],
    )

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class PolygonZoneVisualizationBlockV1(VisualizationBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache: LRUCache[str, np.ndarray] = LRUCache()

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
        np_image = image.numpy_image
        if copy_image:
            np_image = np_image.copy()

        if not zone or len(zone) == 0:
            return {
                OUTPUT_IMAGE_KEY: WorkflowImageData.copy_and_replace(
                    origin_image_data=image, numpy_image=np_image
                )
            }

        h, w, *_ = np_image.shape
        zone_fingerprint = hashlib.md5(str(zone).encode()).hexdigest()
        key = f"{zone_fingerprint}_{color}_{opacity}_{w}_{h}"
        mask = self._cache.get(key)
        if mask is None:
            mask = np.zeros(
                shape=np_image.shape,
                dtype=np_image.dtype,
            )
            pts = []
            if zone and zone[0] and isinstance(zone[0][0], (int, float, np.int32)):
                pts = [np.array(zone, dtype=np.int32)]
            else:
                pts = [np.array(z, dtype=np.int32) for z in zone]
            mask = cv.fillPoly(
                img=mask,
                pts=pts,
                color=str_to_color(color).as_bgr(),
            )
            self._cache.set(key, mask)

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
