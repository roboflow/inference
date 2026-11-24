"""Project detections to GPS positions: ``roboflow_core/geotag_detection@v1``.

Port of V1 ``roboflow_core/geotag_detection@v1``, with V1's nadir-camera math
for each box centre ``(cx, cy)`` in a ``W x H`` camera image::

    ground_width  = 2 * altitude * tan(horizontal_fov / 2)
    ground_height = ground_width * H / W
    right = (cx - W / 2) / W * ground_width
    up    = -(cy - H / 2) / H * ground_height
    east  =  right * cos(heading) + up * sin(heading)
    north = -right * sin(heading) + up * cos(heading)
    lat   = latitude  + north / 111320
    lon   = longitude + east / (111320 * cos(latitude))

An altitude of zero or less puts every detection at the camera position (V1).

The camera image is the workflow input: box centres are first mapped through
the prediction's root frame, and ``W x H`` is the size of that input.

Differences from V1: V1 projects box centres in the pixels of the image the
detections were made on. Detections on a crop, or on a resize that changes
the aspect ratio, land at wrong positions in V1; after any resize, V1's
``pixel_x``, ``pixel_y``, ``width`` and ``height`` are resized-image pixels.
Here detections land where they are in the workflow input, and those four
fields are workflow-input pixels. The math runs in float64. For detections on
the full input image, the result equalled V1 in every tested case where V1's
boxes were float64. V1 NumPy mode alone does not give float64: with float32
boxes V1 computes in float32, and V1 tensor mode always computes in float32.
A float32 V1 result differs from V2 by float32 rounding. The difference
depends on latitude, longitude and the boxes, and is larger far from latitude
0 and longitude 0. It has no fixed bound: probes measured 0.015 m near
latitude 1, longitude 2, 0.61 m near latitude 47.4, longitude -105.3, and
1.33 m near latitude -33.9, longitude 151.2. None of these is a tolerance.
Predictions from a composite image (stitched panorama or mosaic), or from
another workflow input than ``image``, are a ``ContractError``. Parameters
are checked: latitude in (-90, 90) (``cos(latitude)`` is zero at +-90),
longitude in [-180, 180], ``horizontal_fov`` in (0, 180); latitude,
longitude, altitude, ``horizontal_fov`` and heading must be finite. V1
checks none of these; an infinite heading fails there with a ``ValueError``.
"""

import math
from typing import Any, Dict, List, Tuple
import torch
from pydantic import Field
from roboflow_workflows.v2.data.errors import ContractError
from roboflow_workflows.v2.data.frames import FrameRef, frame_of
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import (
    DICTIONARY_KIND,
    FLOAT_KIND,
    LIST_OF_VALUES_KIND,
)
from roboflow_workflows.v2.data.prediction_ops import class_names_by_id, row_class_names
from roboflow_workflows.v2.data.predictions import (
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
)
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref


METERS_PER_DEG_LAT = 111320.0

PREDICTION_KINDS = (
    OBJECT_DETECTION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
)

class GeotagDetectionBlock(Block):
    """Give every detection the GPS position of its box centre.

    Example: a camera at latitude 0, 100 m up, with a 90 degree field of view
    sees a ground strip 200 m wide. A box centred on the right edge of the
    image is 100 m east: longitude ``+100 / 111320`` degrees. With heading 90
    (image top points east) the same box is 100 m south.
    """

    type = "roboflow_core/geotag_detection@v1"
    outputs = {
        "geo_detections": Output(
            LIST_OF_VALUES_KIND,
            source="predictions",
            description=(
                "One record per detection: class, confidence, lat, lon, "
                "pixel_x, pixel_y, width, height (WGS-84 degrees, input pixels)."
            ),
        ),
        "geojson": Output(
            DICTIONARY_KIND,
            source="predictions",
            description="GeoJSON FeatureCollection with one Point per detection.",
        ),
    }

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(
            description="Image the detections come from, or one derived from it."
        )
        predictions: Ref(*PREDICTION_KINDS) = Field(
            description="Object detection or instance segmentation prediction."
        )
        latitude: float | Ref(FLOAT_KIND) = Field(
            gt=-90,
            lt=90,
            allow_inf_nan=False,
            description="Camera latitude in decimal degrees; north is positive.",
            examples=[47.428681, "$inputs.latitude"],
        )
        longitude: float | Ref(FLOAT_KIND) = Field(
            ge=-180,
            le=180,
            allow_inf_nan=False,
            description="Camera longitude in decimal degrees; east is positive.",
            examples=[-105.279125, "$inputs.longitude"],
        )
        altitude: float | Ref(FLOAT_KIND) = Field(
            allow_inf_nan=False,
            description="Camera height above ground in meters.",
            examples=[69.0, "$inputs.altitude"],
        )
        horizontal_fov: float | Ref(FLOAT_KIND) = Field(
            default=73.7,
            gt=0,
            lt=180,
            allow_inf_nan=False,
            description="Horizontal field of view in degrees.",
        )
        heading: float | Ref(FLOAT_KIND) = Field(
            default=0.0,
            allow_inf_nan=False,
            description="Bearing of the image top, degrees clockwise from north.",
            examples=[0.0, 90.0, "$inputs.heading"],
        )

    def run(
        self,
        *,
        image: ImageData,
        predictions: Any,
        latitude: float,
        longitude: float,
        altitude: float,
        horizontal_fov: float,
        heading: float,
    ) -> Dict[str, Any]:
        """Project the box centres of ``predictions`` to GPS.

        Args:
            image: Image naming the workflow input the camera saw.
            predictions: ``Detections`` or ``InstanceDetections``.
            latitude: Camera latitude in degrees.
            longitude: Camera longitude in degrees.
            altitude: Camera height above ground in meters.
            horizontal_fov: Horizontal field of view in degrees.
            heading: Bearing of the image top, degrees clockwise from north.

        Returns:
            ``geo_detections`` records and the ``geojson`` FeatureCollection.

        Raises:
            ContractError: When the predictions come from a composite image or
                from another workflow input than ``image``.
        """
        frame = frame_of(predictions)
        _check_same_input(self.type, frame=frame, image=image)

        root = frame.root
        xyxy = predictions.xyxy.to(torch.float64)
        scale = xyxy.new_tensor(root.scale_xy).repeat(2)
        offset = xyxy.new_tensor(root.offset_xy).repeat(2)
        corners = xyxy * scale + offset
        centres = (corners[:, :2] + corners[:, 2:]) / 2
        sizes = corners[:, 2:] - corners[:, :2]
        lat, lon = ground_positions(
            centres,
            camera_size_hw=root.frame_size_hw,
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
            horizontal_fov=horizontal_fov,
            heading=heading,
        )

        columns = torch.stack(
            [lat, lon, predictions.confidence.to(torch.float64), *centres.T, *sizes.T],
            dim=1,
        ).tolist()
        class_names = row_class_names(
            predictions.class_id.tolist(),
            row_metadata=predictions.bboxes_metadata or [{}] * len(columns),
            names_by_id=class_names_by_id(predictions.image_metadata),
        )
        geo_detections = [
            _record(name, values) for name, values in zip(class_names, columns)
        ]
        features = [_feature(record) for record in geo_detections]
        result = {
            "geo_detections": geo_detections,
            "geojson": {"type": "FeatureCollection", "features": features},
        }

        return result

def ground_positions(
    centres: torch.Tensor,
    *,
    camera_size_hw: Tuple[int, int],
    latitude: float,
    longitude: float,
    altitude: float,
    horizontal_fov: float,
    heading: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the latitude and longitude seen at each camera pixel.

    Args:
        centres: ``(N, 2)`` float64 ``(x, y)`` pixels of the camera image.
        camera_size_hw: ``(H, W)`` of the camera image.
        latitude: Camera latitude in degrees.
        longitude: Camera longitude in degrees.
        altitude: Camera height above ground in meters.
        horizontal_fov: Horizontal field of view in degrees.
        heading: Bearing of the image top, degrees clockwise from north.

    Returns:
        ``(lat, lon)``: two ``(N,)`` float64 tensors in degrees.
    """
    if altitude <= 0:
        return (
            torch.full_like(centres[:, 0], latitude),
            torch.full_like(centres[:, 0], longitude),
        )

    height, width = camera_size_hw
    ground_width = 2 * altitude * math.tan(math.radians(horizontal_fov) / 2)
    ground_height = ground_width * height / width
    right = (centres[:, 0] - width / 2) / width * ground_width
    up = -(centres[:, 1] - height / 2) / height * ground_height

    psi = math.radians(heading)
    east = right * math.cos(psi) + up * math.sin(psi)
    north = -right * math.sin(psi) + up * math.cos(psi)
    meters_per_deg_lon = METERS_PER_DEG_LAT * math.cos(math.radians(latitude))

    return latitude + north / METERS_PER_DEG_LAT, longitude + east / meters_per_deg_lon

def _check_same_input(block: str, *, frame: FrameRef, image: ImageData) -> None:
    if frame.is_composite or image.is_composite:
        raise ContractError(
            f"{block}: predictions of image {frame.image_id!r} or image "
            f"{image.image_id!r} come from a composite image (stitched "
            "panorama or mosaic), which has no single camera position; "
            "geotag detections of a camera image"
        )
    prediction_input, image_input = frame.root, image.root
    if (prediction_input.frame_id, tuple(prediction_input.frame_size_hw)) != (
        image_input.frame_id,
        tuple(image_input.frame_size_hw),
    ):
        raise ContractError(
            f"{block}: predictions of image {frame.image_id!r} come from "
            f"workflow input {prediction_input.frame_id!r} "
            f"(h, w)={prediction_input.frame_size_hw}, but image "
            f"{image.image_id!r} from {image_input.frame_id!r} "
            f"(h, w)={image_input.frame_size_hw}; pass the image the camera "
            "position belongs to"
        )

def _record(class_name: Any, values: List[float]) -> Dict[str, Any]:
    lat, lon, confidence, pixel_x, pixel_y, width, height = values
    record = {
        "class": "unknown" if class_name is None else str(class_name),
        "confidence": round(confidence, 4),
        "lat": round(lat, 7),
        "lon": round(lon, 7),
        "pixel_x": round(pixel_x, 1),
        "pixel_y": round(pixel_y, 1),
        "width": round(width, 1),
        "height": round(height, 1),
    }

    return record

def _feature(record: Dict[str, Any]) -> Dict[str, Any]:
    feature = {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [record["lon"], record["lat"]]},
        "properties": {"class": record["class"], "confidence": record["confidence"]},
    }

    return feature
