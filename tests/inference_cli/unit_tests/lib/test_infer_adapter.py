import json
import os.path

import cv2
import numpy as np

from inference_cli.lib.infer_adapter import (
    prepare_target_path,
    save_visualisation_image,
    save_prediction,
)


def test_prepare_target_path_when_reference_is_integer() -> None:
    # when
    result = prepare_target_path(
        reference=39, output_location="/some/location", extension="jpg"
    )

    # then
    assert result == "/some/location/frame_000039.jpg"


def test_prepare_target_path_when_reference_is_file_name() -> None:
    # when
    result = prepare_target_path(
        reference="some.jpg", output_location="/some/location", extension="json"
    )

    # then
    assert result == "/some/location/some_prediction.json"


def test_save_visualisation_image(empty_directory: str) -> None:
    # given
    image = np.zeros((192, 168, 3), dtype=np.uint8)

    # when
    save_visualisation_image(
        reference=30,
        visualisation=image,
        output_location=empty_directory,
    )

    # then
    result = cv2.imread(os.path.join(empty_directory, "frame_000030.jpg"))
    assert np.allclose(image, result)


def test_save_prediction(empty_directory: str) -> None:
    # given
    prediction = {"predictions": []}

    # when
    save_prediction(
        reference=30,
        prediction=prediction,
        output_location=empty_directory,
    )

    # then
    with open(os.path.join(empty_directory, "frame_000030.json"), "r") as f:
        result = json.load(f)
    assert result == {"predictions": []}
