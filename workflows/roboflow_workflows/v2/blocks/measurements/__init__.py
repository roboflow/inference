"""Measurements from detections and images.

``distance`` measures the gap between two detected boxes, ``size`` measures
objects against a reference of known size, ``mask_area`` adds each
detection's area, and ``pixel_color`` counts pixels near one colour.

Pixel results are in pixels of the image each prediction (or image)
describes. Physical units are only as good as the calibration the workflow
supplies; the blocks cannot detect perspective or lens distortion.

``BLOCKS`` lists the blocks to register in the catalogue.
"""

from roboflow_workflows.v2.blocks.measurements.distance import DistanceMeasurementBlock
from roboflow_workflows.v2.blocks.measurements.mask_area import MaskAreaMeasurementBlock
from roboflow_workflows.v2.blocks.measurements.pixel_color import PixelColorCountBlock
from roboflow_workflows.v2.blocks.measurements.size import SizeMeasurementBlock

BLOCKS = (
    DistanceMeasurementBlock,
    SizeMeasurementBlock,
    MaskAreaMeasurementBlock,
    PixelColorCountBlock,
)

__all__ = [
    "BLOCKS",
    "DistanceMeasurementBlock",
    "MaskAreaMeasurementBlock",
    "PixelColorCountBlock",
    "SizeMeasurementBlock",
]
