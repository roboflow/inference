import math
from typing import Any

import numpy as np
import pytest
import supervision as sv
from pydantic import ValidationError

from inference.core.workflows.core_steps.transformations.geotag_detection.v1 import (
    BlockManifest,
    GeoTagDetectionBlockV1,
    _pixel_to_gps,
    project_detections,
)

METERS_PER_DEG_LAT = 111320.0

VALID_MANIFEST_DATA = {
    "type": "roboflow_core/geotag_detection@v1",
    "name": "geotag",
    "image": "$steps.detection.image",
    "predictions": "$steps.detection.predictions",
    "latitude": "$inputs.latitude",
    "longitude": "$inputs.longitude",
    "altitude": "$inputs.altitude",
}


def test_manifest_parsing_when_valid_data_with_selectors() -> None:
    result = BlockManifest.model_validate(VALID_MANIFEST_DATA)

    assert result.type == "roboflow_core/geotag_detection@v1"
    assert result.latitude == "$inputs.latitude"
    assert result.longitude == "$inputs.longitude"
    assert result.altitude == "$inputs.altitude"
    assert result.horizontal_fov == 73.7
    assert result.heading == 0.0


def test_manifest_parsing_when_valid_data_with_literals() -> None:
    data = {
        **VALID_MANIFEST_DATA,
        "latitude": 47.428681,
        "longitude": -105.279125,
        "altitude": 69.0,
        "horizontal_fov": 84.0,
        "heading": 90.0,
    }

    result = BlockManifest.model_validate(data)

    assert result.latitude == 47.428681
    assert result.longitude == -105.279125
    assert result.altitude == 69.0
    assert result.horizontal_fov == 84.0
    assert result.heading == 90.0


@pytest.mark.parametrize(
    "field_name, value",
    [
        ("predictions", "invalid"),
        ("image", "invalid"),
    ],
)
def test_manifest_parsing_when_invalid_selector_provided(
    field_name: str,
    value: Any,
) -> None:
    data = {**VALID_MANIFEST_DATA, field_name: value}

    with pytest.raises(ValidationError):
        BlockManifest.model_validate(data)


def test_manifest_describes_two_outputs() -> None:
    outputs = BlockManifest.describe_outputs()

    names = {o.name for o in outputs}
    assert names == {"geo_detections", "geojson"}


def test_manifest_execution_engine_compatibility() -> None:
    compat = BlockManifest.get_execution_engine_compatibility()

    assert compat == ">=1.3.0,<2.0.0"


# --- Projection math: _pixel_to_gps ---


def test_center_pixel_returns_camera_position() -> None:
    lat, lon = _pixel_to_gps(
        px=640,
        py=360,
        image_w=1280,
        image_h=720,
        camera_lat=47.0,
        camera_lon=-105.0,
        camera_alt=100.0,
    )

    assert lat == pytest.approx(47.0, abs=1e-7)
    assert lon == pytest.approx(-105.0, abs=1e-7)


def test_zero_altitude_returns_camera_position() -> None:
    lat, lon = _pixel_to_gps(
        px=0,
        py=0,
        image_w=1280,
        image_h=720,
        camera_lat=47.0,
        camera_lon=-105.0,
        camera_alt=0.0,
    )

    assert lat == 47.0
    assert lon == -105.0


def test_negative_altitude_returns_camera_position() -> None:
    lat, lon = _pixel_to_gps(
        px=100,
        py=100,
        image_w=1280,
        image_h=720,
        camera_lat=47.0,
        camera_lon=-105.0,
        camera_alt=-10.0,
    )

    assert lat == 47.0
    assert lon == -105.0


def test_pixel_right_of_center_increases_longitude_heading_north() -> None:
    center_lat, center_lon = _pixel_to_gps(
        px=640,
        py=360,
        image_w=1280,
        image_h=720,
        camera_lat=47.0,
        camera_lon=-105.0,
        camera_alt=100.0,
    )
    right_lat, right_lon = _pixel_to_gps(
        px=960,
        py=360,
        image_w=1280,
        image_h=720,
        camera_lat=47.0,
        camera_lon=-105.0,
        camera_alt=100.0,
    )

    assert right_lon > center_lon, "Pixel right of center should be east (higher lon)"
    assert right_lat == pytest.approx(
        center_lat, abs=1e-7
    ), "Same row should have same latitude"


def test_pixel_above_center_increases_latitude_heading_north() -> None:
    center_lat, _ = _pixel_to_gps(
        px=640,
        py=360,
        image_w=1280,
        image_h=720,
        camera_lat=47.0,
        camera_lon=-105.0,
        camera_alt=100.0,
    )
    top_lat, _ = _pixel_to_gps(
        px=640,
        py=0,
        image_w=1280,
        image_h=720,
        camera_lat=47.0,
        camera_lon=-105.0,
        camera_alt=100.0,
    )

    assert top_lat > center_lat, "Pixel above center should be north (higher lat)"


def test_heading_90_rotates_axes() -> None:
    """With heading=90 (image-up points East), a pixel right of center should
    move south (decreasing lat), not east."""
    center_lat, center_lon = _pixel_to_gps(
        px=640,
        py=360,
        image_w=1280,
        image_h=720,
        camera_lat=47.0,
        camera_lon=-105.0,
        camera_alt=100.0,
        heading_deg=90.0,
    )
    right_lat, right_lon = _pixel_to_gps(
        px=960,
        py=360,
        image_w=1280,
        image_h=720,
        camera_lat=47.0,
        camera_lon=-105.0,
        camera_alt=100.0,
        heading_deg=90.0,
    )

    assert right_lat < center_lat, "Heading 90: pixel-right should map to south"
    assert right_lon == pytest.approx(center_lon, abs=1e-6)


def test_ground_footprint_scales_with_altitude() -> None:
    """Higher altitude = larger ground footprint = bigger coordinate offset for the same pixel."""
    _, lon_low = _pixel_to_gps(
        px=1280,
        py=360,
        image_w=1280,
        image_h=720,
        camera_lat=47.0,
        camera_lon=-105.0,
        camera_alt=50.0,
    )
    _, lon_high = _pixel_to_gps(
        px=1280,
        py=360,
        image_w=1280,
        image_h=720,
        camera_lat=47.0,
        camera_lon=-105.0,
        camera_alt=200.0,
    )

    offset_low = abs(lon_low - (-105.0))
    offset_high = abs(lon_high - (-105.0))
    assert offset_high > offset_low, "Higher altitude should produce larger lon offset"
    assert offset_high == pytest.approx(
        offset_low * 4, rel=0.01
    ), "4x altitude = 4x offset"


def test_equator_vs_high_latitude_lon_scaling() -> None:
    """At the equator, 1 degree of longitude ~ 111km. At 60N, ~ 55km.
    Same pixel offset should yield a larger longitude delta at higher latitudes."""
    _, lon_equator = _pixel_to_gps(
        px=1280,
        py=360,
        image_w=1280,
        image_h=720,
        camera_lat=0.0,
        camera_lon=0.0,
        camera_alt=100.0,
    )
    _, lon_60n = _pixel_to_gps(
        px=1280,
        py=360,
        image_w=1280,
        image_h=720,
        camera_lat=60.0,
        camera_lon=0.0,
        camera_alt=100.0,
    )

    delta_equator = abs(lon_equator)
    delta_60n = abs(lon_60n)
    assert (
        delta_60n > delta_equator
    ), "Same meter offset at 60N needs larger degree delta"
    assert delta_60n == pytest.approx(delta_equator * 2.0, rel=0.05)


def test_known_offset_value() -> None:
    """At the equator, heading=0, altitude=100m, FOV=90deg:
    ground_width = 2*100*tan(45) = 200m.
    Right edge pixel (px=1280) is half the ground width (100m) east of center.
    100m / 111320 m/deg = ~0.000898 deg lon offset."""
    lat, lon = _pixel_to_gps(
        px=1280,
        py=360,
        image_w=1280,
        image_h=720,
        camera_lat=0.0,
        camera_lon=0.0,
        camera_alt=100.0,
        fov_h=90.0,
    )

    expected_lon = 100.0 / METERS_PER_DEG_LAT
    assert lon == pytest.approx(expected_lon, abs=1e-6)
    assert lat == pytest.approx(0.0, abs=1e-7)


# --- project_detections ---


def test_project_detections_empty() -> None:
    detections = sv.Detections.empty()

    geo, features = project_detections(
        detections,
        1280,
        720,
        47.0,
        -105.0,
        100.0,
    )

    assert geo == []
    assert features == []


def test_project_detections_single() -> None:
    detections = sv.Detections(
        xyxy=np.array([[600, 340, 680, 380]], dtype=np.float64),
        class_id=np.array([0]),
        confidence=np.array([0.95], dtype=np.float64),
        data={"class_name": np.array(["person"])},
    )

    geo, features = project_detections(
        detections,
        1280,
        720,
        47.0,
        -105.0,
        100.0,
    )

    assert len(geo) == 1
    assert geo[0]["class"] == "person"
    assert geo[0]["confidence"] == 0.95
    assert "lat" in geo[0]
    assert "lon" in geo[0]
    assert geo[0]["pixel_x"] == 640.0
    assert geo[0]["pixel_y"] == 360.0

    assert len(features) == 1
    assert features[0]["type"] == "Feature"
    assert features[0]["geometry"]["type"] == "Point"
    coords = features[0]["geometry"]["coordinates"]
    assert len(coords) == 2
    assert coords[0] == geo[0]["lon"], "GeoJSON coordinates are [lon, lat]"
    assert coords[1] == geo[0]["lat"]


def test_project_detections_multiple() -> None:
    detections = sv.Detections(
        xyxy=np.array(
            [
                [0, 0, 100, 100],
                [1180, 620, 1280, 720],
            ],
            dtype=np.float64,
        ),
        class_id=np.array([0, 1]),
        confidence=np.array([0.9, 0.8], dtype=np.float64),
        data={"class_name": np.array(["car", "truck"])},
    )

    geo, features = project_detections(
        detections,
        1280,
        720,
        47.0,
        -105.0,
        100.0,
    )

    assert len(geo) == 2
    assert geo[0]["class"] == "car"
    assert geo[1]["class"] == "truck"
    assert geo[0]["lat"] != geo[1]["lat"] or geo[0]["lon"] != geo[1]["lon"]


def test_project_detections_no_class_name() -> None:
    detections = sv.Detections(
        xyxy=np.array([[600, 340, 680, 380]], dtype=np.float64),
        class_id=np.array([0]),
        confidence=np.array([0.5], dtype=np.float64),
    )

    geo, _ = project_detections(
        detections,
        1280,
        720,
        47.0,
        -105.0,
        100.0,
    )

    assert geo[0]["class"] == "unknown"


def test_project_detections_no_confidence() -> None:
    detections = sv.Detections(
        xyxy=np.array([[600, 340, 680, 380]], dtype=np.float64),
    )

    geo, _ = project_detections(
        detections,
        1280,
        720,
        47.0,
        -105.0,
        100.0,
    )

    assert geo[0]["confidence"] == 0.0


def test_project_detections_geojson_structure() -> None:
    detections = sv.Detections(
        xyxy=np.array([[0, 0, 100, 100]], dtype=np.float64),
        class_id=np.array([0]),
        confidence=np.array([0.9], dtype=np.float64),
        data={"class_name": np.array(["dog"])},
    )

    geo, features = project_detections(
        detections,
        1280,
        720,
        47.0,
        -105.0,
        100.0,
    )

    feature = features[0]
    assert feature["properties"]["class"] == "dog"
    assert feature["properties"]["confidence"] == 0.9


def test_get_manifest_returns_block_manifest() -> None:
    assert GeoTagDetectionBlockV1.get_manifest() is BlockManifest
