"""Unit tests of ``roboflow_core/geotag_detection@v1`` with hand-computed positions.

Camera for most tests: latitude 0, longitude 0, 100 m up, 90 degree field of
view over a 200 x 100 image. The ground strip is ``2 * 100 * tan(45) = 200`` m
wide and 100 m high, so one pixel is one meter and one meter east is
``1 / 111320`` degrees of longitude.
"""

import pytest
import torch
from roboflow_workflows.v2.blocks.geometry.geotag_detection.v1 import GeotagDetectionBlock
from roboflow_workflows.v2.data.errors import ContractError
from roboflow_workflows.v2.data.image import ImageData

from inference_models.models.base.object_detection import Detections

DEGREES_PER_METER = 1 / 111320


def camera_image(image_id="camera"):
    return ImageData.from_tensor(
        torch.zeros((3, 100, 200), dtype=torch.uint8), image_id=image_id
    )


def boxes(image, xyxy, class_id=None, confidence=None, rows=None):
    count = len(xyxy)
    return Detections(
        xyxy=torch.tensor(xyxy, dtype=torch.float32).reshape(count, 4),
        class_id=torch.tensor(class_id or [0] * count, dtype=torch.int64),
        confidence=torch.tensor(confidence or [0.9] * count),
        image_metadata={**image.prediction_metadata(), "class_names": {0: "car"}},
        bboxes_metadata=rows,
    )


def geotag(image, predictions, **options):
    arguments = {
        "latitude": 0.0,
        "longitude": 0.0,
        "altitude": 100.0,
        "horizontal_fov": 90.0,
        "heading": 0.0,
    }
    arguments.update(options)

    return GeotagDetectionBlock().run(image=image, predictions=predictions, **arguments)


def positions(result):
    return [(record["lat"], record["lon"]) for record in result["geo_detections"]]


def test_box_on_the_right_edge_is_one_hundred_meters_east():
    image = camera_image()
    predictions = boxes(image, [[190, 45, 210, 55]])

    result = geotag(image, predictions)

    assert positions(result) == [(0.0, round(100 * DEGREES_PER_METER, 7))]
    assert result["geo_detections"][0] == {
        "class": "car",
        "confidence": 0.9,
        "lat": 0.0,
        "lon": round(100 * DEGREES_PER_METER, 7),
        "pixel_x": 200.0,
        "pixel_y": 50.0,
        "width": 20.0,
        "height": 10.0,
    }


def test_box_on_the_top_edge_is_fifty_meters_north():
    image = camera_image()

    result = geotag(image, boxes(image, [[95, -5, 105, 5]]))

    assert positions(result) == [(round(50 * DEGREES_PER_METER, 7), 0.0)]


def test_heading_ninety_turns_image_right_to_south():
    image = camera_image()

    result = geotag(image, boxes(image, [[190, 45, 210, 55]]), heading=90.0)

    assert positions(result) == [(round(-100 * DEGREES_PER_METER, 7), 0.0)]


def test_longitude_degrees_widen_with_latitude():
    image = camera_image()

    result = geotag(image, boxes(image, [[190, 45, 210, 55]]), latitude=60.0)

    # cos(60) = 0.5: one degree of longitude is half as many meters.
    assert positions(result) == [(60.0, round(200 * DEGREES_PER_METER, 7))]


@pytest.mark.parametrize("altitude", [0.0, -5.0])
def test_altitude_zero_or_less_puts_detections_at_the_camera(altitude):
    image = camera_image()

    result = geotag(
        image,
        boxes(image, [[190, 45, 210, 55]]),
        latitude=47.5,
        longitude=-105.25,
        altitude=altitude,
    )

    assert positions(result) == [(47.5, -105.25)]


def test_detections_on_a_crop_land_where_they_are_in_the_camera_image():
    image = camera_image()
    crop = image.crop((50, 20, 150, 80), image_id="crop")
    on_image = geotag(image, boxes(image, [[190, 45, 210, 55], [0, 0, 10, 10]]))

    on_crop = geotag(crop, boxes(crop, [[140, 25, 160, 35], [-50, -20, -40, -10]]))

    assert on_crop == on_image


def test_detections_on_a_resized_image_are_scaled_back_to_the_camera_image():
    image = camera_image()
    half = image.resize((50, 100), image_id="half")
    on_image = geotag(image, boxes(image, [[180, 40, 200, 60]]))

    on_half = geotag(image, boxes(half, [[90, 20, 100, 30]]))

    assert on_half == on_image


def test_class_names_come_from_rows_then_the_class_map():
    image = camera_image()
    predictions = boxes(
        image,
        [[0, 0, 10, 10]] * 3,
        class_id=[0, 0, 7],
        confidence=[0.123456, 0.5, 0.25],
        rows=[{}, {"class": "truck"}, {}],
    )

    records = geotag(image, predictions)["geo_detections"]

    assert [record["class"] for record in records] == ["car", "truck", "unknown"]
    assert [record["confidence"] for record in records] == [0.1235, 0.5, 0.25]


def test_geojson_has_one_point_per_detection_in_lon_lat_order():
    image = camera_image()

    result = geotag(image, boxes(image, [[190, 45, 210, 55]]), latitude=10.0)

    record = result["geo_detections"][0]
    assert result["geojson"] == {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [record["lon"], record["lat"]],
                },
                "properties": {"class": "car", "confidence": 0.9},
            }
        ],
    }


def test_no_detections_give_an_empty_collection():
    image = camera_image()

    result = geotag(image, boxes(image, []))

    assert result == {
        "geo_detections": [],
        "geojson": {"type": "FeatureCollection", "features": []},
    }


def test_detections_of_another_workflow_input_are_a_contract_error():
    other = camera_image(image_id="other")

    with pytest.raises(ContractError, match="workflow input 'other'"):
        geotag(camera_image(), boxes(other, [[0, 0, 10, 10]]))


def test_detections_on_a_composite_image_are_a_contract_error():
    mosaic = ImageData.composite(
        torch.zeros((3, 100, 200), dtype=torch.uint8), sources=[]
    )

    with pytest.raises(ContractError, match="composite"):
        geotag(mosaic, boxes(mosaic, [[0, 0, 10, 10]]))


@pytest.mark.parametrize(
    "field, value",
    [
        ("latitude", 90.0),
        ("latitude", -90.0),
        ("longitude", 180.5),
        ("horizontal_fov", 0.0),
        ("horizontal_fov", 180.0),
        ("altitude", float("nan")),
        ("heading", float("inf")),
    ],
)
def test_out_of_range_camera_values_are_rejected(field, value):
    params = {
        "image": "$inputs.image",
        "predictions": "$steps.model.predictions",
        "latitude": 0.0,
        "longitude": 0.0,
        "altitude": 100.0,
    }
    params[field] = value

    with pytest.raises(Exception, match=field):
        GeotagDetectionBlock.Params.model_validate(params)
