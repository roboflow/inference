"""Zone drawings: polygon zones, line counters and reference paths.

Ports of V1 ``roboflow_core/polygon_zone_visualization@v1``,
``roboflow_core/line_counter_visualization@v1`` and
``roboflow_core/reference_path_visualization@v1``. Pixels equal V1::

    polygon zone    one cv2.fillPoly of every polygon (OpenCV's fill rule);
                    covered pixels: pixel + opacity * color, rounded, saturated
    line counter    cv2.line of ``thickness``, tinted the same way; then
                    "in: N, out: M", black on white, centred on the first point
    reference path  open cv2.polylines in the solid color

"pixel + opacity * color" is V1's ``cv2.addWeighted(mask, opacity, image, 1,
0)``: a zone brightens towards the color rather than blending with it.

Polygons and lines are read by the zone blocks' own parsers
(``analytics.zones.parse_polygons`` and ``parse_line``), so one workflow value
such as ``$inputs.zone`` draws exactly when the counters accept it. Points are
image pixels; fractional coordinates are truncated as V1's int32 cast does.
A reference path is read more leniently than ``roboflow_core/path_deviation_analytics@v2``
reads it, as V1 drew it (``path_points``).

OpenCV rasterizes on the host into a frame-sized scratch mask, so it clips at
the frame edges as V1. Only the shapes' window (bounds plus the stroke's
reach) goes to the device, as one bool mask; the tint is computed there.
Reference paths and the counter text use the shared ``strokes`` and ``text``
painters.
"""


from roboflow_workflows.v2.blocks.visualization.annotations.line_counter_visualization.v1 import LineCounterVisualizationBlock
from roboflow_workflows.v2.blocks.visualization.annotations.polygon_zone_visualization.v1 import PolygonZoneVisualizationBlock
from roboflow_workflows.v2.blocks.visualization.annotations.reference_path_visualization.v1 import ReferencePathVisualizationBlock


BLOCKS = (
    PolygonZoneVisualizationBlock,
    LineCounterVisualizationBlock,
    ReferencePathVisualizationBlock,
)
