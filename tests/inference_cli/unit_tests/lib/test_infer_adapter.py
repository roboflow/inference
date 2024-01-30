import json
import os.path

import cv2
import numpy as np
import pytest

from inference_cli.lib.infer_adapter import (
    is_something_to_do,
    prepare_target_path,
    save_prediction,
    save_visualisation_image,
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


@pytest.mark.parametrize(
    "display, visualise",
    [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ],
)
def test_is_something_to_do_when_output_location_is_set(
    display: bool, visualise: bool
) -> None:
    # when
    result = is_something_to_do(
        output_location="/some", display=display, visualise=visualise
    )

    # then
    assert result is True


def test_is_something_to_do_when_output_location_is_not_given_but_both_other_flags_are_true() -> (
    None
):
    # when
    result = is_something_to_do(output_location=None, display=True, visualise=True)

    # then
    assert result is True


@pytest.mark.parametrize(
    "display, visualise",
    [
        (True, False),
        (False, True),
        (False, False),
    ],
)
def test_is_something_to_do_when_output_location_is_not_given_and_at_least_one_other_flag_is_disabled(
    display: bool,
    visualise: bool,
) -> None:
    # when
    result = is_something_to_do(
        output_location=None, display=display, visualise=visualise
    )

    # then
    assert result is False
