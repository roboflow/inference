"""Tensor-native sibling of ``geotag_detection/v1.py``.

Boxes / confidences / class ids are pulled to host in one DtoH copy per tensor.
The per-box class name resolves from ``bboxes_metadata[i][CLASS_NAME_KEY]`` when
present, else from the per-image ``image_metadata[CLASS_NAMES_KEY]``
class_id -> name map, else degrades to ``"unknown"``. Image dimensions are read
via ``_read_shape_without_materialization()`` so a tensor-only image is not
copied to host just for its shape.
"""

import math
from typing import List, Literal, Optional, Tuple, Type, Union

import numpy as np
from pydantic import ConfigDict, Field

from inference.core.workflows.execution_engine.constants import (
    CLASS_NAME_KEY,
    CLASS_NAMES_KEY,
)
from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.tensor_native_types import (
    TENSOR_NATIVE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    TENSOR_NATIVE_OBJECT_DETECTION_PREDICTION_KIND,
)
from inference.core.workflows.execution_engine.entities.types import (
    DICTIONARY_KIND,
    FLOAT_KIND,
    IMAGE_KIND,
    LIST_OF_VALUES_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.object_detection import Detections

SHORT_DESCRIPTION = (
    "Project detections from pixel coordinates to real-world GPS positions."
)
LONG_DESCRIPTION = """
Convert object detection bounding boxes to real-world GPS coordinates using camera position metadata.

## How This Block Works

This block takes object detection predictions and camera GPS metadata (latitude, longitude, altitude) and projects each detection's pixel position to a real-world ground coordinate. The projection uses the camera's field of view and altitude to compute a ground footprint, then maps pixel offsets from image center to geographic offsets from the camera position.

1. **Receives detection predictions** from an upstream object detection block (any model that outputs bounding boxes)
2. **Takes camera GPS metadata** as inputs: latitude, longitude, altitude above ground, and optionally horizontal field of view
3. **Computes ground footprint** from altitude and FOV using basic trigonometry
4. **Projects each detection center** from pixel coordinates to lat/lon offset from camera position
5. **Outputs GeoJSON-ready records** with class, confidence, and geographic coordinates for each detection

The projection assumes a nadir (straight-down) camera orientation, which is accurate for most drone survey flights. For oblique angles, accuracy decreases with distance from image center.

## Common Use Cases

- **Drone Survey Analysis**: Process drone footage to map detected objects (vehicles, people, animals, structures) at their real-world locations for survey, inspection, or monitoring applications
- **Agricultural Monitoring**: Map crop damage, equipment, or livestock positions from aerial imagery for precision agriculture workflows
- **Security and Surveillance**: Create geospatial awareness from aerial camera feeds, mapping detected activity to real-world coordinates for situational awareness
- **Wildlife Conservation**: Track and map animal detections from drone surveys to monitor populations, migration patterns, or habitat usage
- **Construction Site Monitoring**: Map equipment, materials, and personnel positions from aerial imagery for site management and safety compliance
- **Search and Rescue**: Rapidly map detected persons or objects across large areas from drone footage to coordinate response efforts

## Connecting to Other Blocks

This block receives detections and produces geospatial data:

- **After object detection blocks** (YOLO, RF-DETR, etc.) to geotag their predictions with real-world coordinates
- **After tracking blocks** (ByteTrack, OC-SORT) to produce geotagged tracks with movement paths
- **Before data sink blocks** (CSV, JSON, Webhook) to export detection locations for GIS analysis
- **Before visualization blocks** to annotate frames with GPS coordinate labels
- **In video processing pipelines** where each frame's GPS comes from drone telemetry or EXIF metadata
"""

OUTPUT_KEY_GEO_DETECTIONS = "geo_detections"
OUTPUT_KEY_GEOJSON = "geojson"

METERS_PER_DEG_LAT = 111320.0


class BlockManifest(WorkflowBlockManifest):
    type: Literal["roboflow_core/geotag_detection@v1"]
    model_config = ConfigDict(
        json_schema_extra={
            "name": "GeoTag Detection",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "transformation",
            "ui_manifest": {
                "section": "transformation",
                "icon": "far fa-map-pin",
                "blockPriority": 5,
            },
        }
    )

    image: Selector(kind=[IMAGE_KIND]) = Field(
        title="Input Image",
        description="The image that detections were generated from. Used to determine image dimensions for coordinate projection.",
        examples=["$inputs.image", "$steps.detection.image"],
    )

    predictions: Selector(
        kind=[
            TENSOR_NATIVE_OBJECT_DETECTION_PREDICTION_KIND,
            TENSOR_NATIVE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
        ]
    ) = Field(
        title="Detections",
        description="Object detection predictions to geotag. Each detection's bounding box center will be projected to a GPS coordinate.",
        examples=["$steps.detection.predictions"],
    )

    latitude: Union[Selector(kind=[FLOAT_KIND]), float] = Field(
        title="Camera Latitude",
        description="GPS latitude of the camera position in decimal degrees. For drone imagery, this comes from the flight controller GPS. Positive values are North, negative are South.",
        examples=[47.428681, "$inputs.latitude"],
    )

    longitude: Union[Selector(kind=[FLOAT_KIND]), float] = Field(
        title="Camera Longitude",
        description="GPS longitude of the camera position in decimal degrees. For drone imagery, this comes from the flight controller GPS. Positive values are East, negative are West.",
        examples=[-105.279125, "$inputs.longitude"],
    )

    altitude: Union[Selector(kind=[FLOAT_KIND]), float] = Field(
        title="Altitude (meters AGL)",
        description="Camera altitude above ground level in meters. Used with field of view to compute the ground footprint. For drones, this is the relative altitude reported by the flight controller, not absolute altitude.",
        examples=[69.0, "$inputs.altitude"],
    )

    horizontal_fov: Union[Selector(kind=[FLOAT_KIND]), float] = Field(
        title="Horizontal Field of View (degrees)",
        description="Horizontal field of view of the camera in degrees. Default 73.7 covers most DJI consumer drones (Mini, Air, Mavic series). Adjust for other cameras. Wider FOV = larger ground footprint per frame.",
        examples=[73.7, 84.0],
        default=73.7,
    )

    heading: Union[Selector(kind=[FLOAT_KIND]), float] = Field(
        title="Camera Heading (degrees clockwise from North)",
        description="Compass bearing that the top of the image points toward, in degrees clockwise from true north. 0 means image-up is North (the default). When the gimbal does not report yaw, derive this from the flight course (bearing between successive GPS fixes) for nose-forward flight. Rotates the ground footprint so detections land on the correct real-world bearing instead of being pinned to North.",
        examples=[0.0, 90.0, "$inputs.heading"],
        default=0.0,
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name=OUTPUT_KEY_GEO_DETECTIONS,
                kind=[LIST_OF_VALUES_KIND],
                description="List of GeoJSON-compatible detection records. Each record contains: class, confidence, lat, lon, pixel_x, pixel_y, width, height. Coordinates are in WGS-84 decimal degrees.",
            ),
            OutputDefinition(
                name=OUTPUT_KEY_GEOJSON,
                kind=[DICTIONARY_KIND],
                description="Complete GeoJSON FeatureCollection with all detections as Point features. Ready for use with Mapbox, Leaflet, QGIS, or any GIS tool.",
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class GeoTagDetectionBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        image: WorkflowImageData,
        predictions: Union[Detections, InstanceDetections],
        latitude: float,
        longitude: float,
        altitude: float,
        horizontal_fov: float = 73.7,
        heading: float = 0.0,
    ) -> BlockResult:
        image_h, image_w = image._read_shape_without_materialization()
        geo_detections, features = project_detections(
            predictions,
            image_w,
            image_h,
            latitude,
            longitude,
            altitude,
            horizontal_fov,
            heading,
        )
        return {
            OUTPUT_KEY_GEO_DETECTIONS: geo_detections,
            OUTPUT_KEY_GEOJSON: {"type": "FeatureCollection", "features": features},
        }


def project_detections(
    predictions: Union[Detections, InstanceDetections],
    image_w: int,
    image_h: int,
    latitude: float,
    longitude: float,
    altitude: float,
    horizontal_fov: float = 73.7,
    heading: float = 0.0,
) -> Tuple[List[dict], List[dict]]:
    """Project tensor-native detections to ground GPS coordinates.

    Returns (geo_detections records, GeoJSON features).
    """
    geo_detections, features = [], []
    detections_number = int(predictions.xyxy.shape[0])
    if detections_number == 0:
        return geo_detections, features

    image_metadata = predictions.image_metadata or {}
    class_names_mapping = image_metadata.get(CLASS_NAMES_KEY) or {}
    bboxes_metadata = predictions.bboxes_metadata or [
        {} for _ in range(detections_number)
    ]
    # Keep the boxes as float32 numpy for the projection arithmetic — unpacking
    # through Python floats changes the lat/lon low-order decimals.
    boxes = predictions.xyxy.detach().to("cpu").numpy()
    confidences = (
        predictions.confidence.detach().to("cpu").numpy()
        if predictions.confidence is not None
        else np.zeros(detections_number)
    )
    class_ids = [
        int(value) for value in predictions.class_id.detach().to("cpu").tolist()
    ]

    for i in range(detections_number):
        x1, y1, x2, y2 = boxes[i]
        cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
        det_lat, det_lon = _pixel_to_gps(
            cx,
            cy,
            image_w,
            image_h,
            latitude,
            longitude,
            altitude,
            horizontal_fov,
            heading,
        )
        data = bboxes_metadata[i]
        if CLASS_NAME_KEY in data:
            class_name = str(data[CLASS_NAME_KEY])
        else:
            class_name = str(class_names_mapping.get(class_ids[i], "unknown"))
        record = {
            "class": class_name,
            "confidence": round(float(confidences[i]), 4),
            "lat": round(det_lat, 7),
            "lon": round(det_lon, 7),
            "pixel_x": round(float(cx), 1),
            "pixel_y": round(float(cy), 1),
            "width": round(float(w), 1),
            "height": round(float(h), 1),
        }
        geo_detections.append(record)
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [record["lon"], record["lat"]],
                },
                "properties": {
                    "class": record["class"],
                    "confidence": record["confidence"],
                },
            }
        )
    return geo_detections, features


def _pixel_to_gps(
    px: float,
    py: float,
    image_w: int,
    image_h: int,
    camera_lat: float,
    camera_lon: float,
    camera_alt: float,
    fov_h: float = 73.7,
    heading_deg: float = 0.0,
) -> Tuple[float, float]:
    """Project a pixel to ground GPS for a nadir camera at a given heading.

    Similar triangles give the target's offset in the image's own right/up axes
    (exact for a pinhole nadir camera over flat ground). The heading then rotates
    those axes onto East/North before the lat/lon conversion. heading_deg = 0
    means image-up points true North, i.e. the original north-up behavior.
    """
    if camera_alt <= 0:
        return camera_lat, camera_lon

    fov_rad = math.radians(fov_h)
    ground_width = 2 * camera_alt * math.tan(fov_rad / 2)
    ground_height = ground_width * (image_h / image_w)

    # Offset in the image's own axes: +right toward frame-right, +up toward top.
    right = (px - image_w / 2) / image_w * ground_width
    up = -(py - image_h / 2) / image_h * ground_height

    # Rotate image axes onto compass axes by the camera heading (CW from North).
    psi = math.radians(heading_deg)
    east = right * math.cos(psi) + up * math.sin(psi)
    north = -right * math.sin(psi) + up * math.cos(psi)

    meters_per_deg_lon = METERS_PER_DEG_LAT * math.cos(math.radians(camera_lat))
    det_lat = camera_lat + north / METERS_PER_DEG_LAT
    det_lon = camera_lon + east / meters_per_deg_lon
    return det_lat, det_lon
