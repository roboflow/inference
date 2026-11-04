import copy
import json
import os.path
from pathlib import Path
from typing import Optional
from unittest import mock
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
from supervision import MaskAnnotator

from inference_cli.lib import infer_adapter
from inference_cli.lib.infer_adapter import (
    create_visualisation,
    infer,
    is_something_to_do,
    prepare_target_path,
    save_prediction,
    save_visualisation_image,
)
from inference_sdk.http.client import InferenceHTTPClient


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


def _triangle_prediction() -> dict:
    return {
        "x": 2,
        "y": 2,
        "width": 4,
        "height": 4,
        "confidence": 0.9,
        "class": "valid",
        "class_id": 0,
        "points": [{"x": 0, "y": 0}, {"x": 4, "y": 0}, {"x": 2, "y": 4}],
    }


def _two_point_prediction() -> dict:
    return {
        "x": 2,
        "y": 2,
        "width": 4,
        "height": 4,
        "confidence": 0.8,
        "class": "invalid",
        "class_id": 1,
        "points": [{"x": 0, "y": 0}, {"x": 4, "y": 4}],
    }


def _rle_prediction(points: Optional[list]) -> dict:
    return {
        "x": 2,
        "y": 2,
        "width": 4,
        "height": 4,
        "confidence": 0.7,
        "class": "rle",
        "class_id": 2,
        "points": points,
        "rle": {"size": [4, 4], "counts": [0, 16]},
    }


def _annotator() -> MagicMock:
    annotator = MagicMock(spec=MaskAnnotator)
    annotator.annotate.side_effect = lambda scene, detections: scene
    return annotator


def test_create_visualisation_drops_polygons_with_too_few_points() -> None:
    # given
    prediction = {
        "image": {"width": 4, "height": 4},
        "predictions": [_triangle_prediction(), _two_point_prediction()],
    }
    annotator = _annotator()

    # when
    result = create_visualisation(
        frame=np.zeros((4, 4, 3), dtype=np.uint8),
        prediction=prediction,
        annotators=[annotator],
        tracker=None,
    )

    # then
    assert result is not None
    detections = annotator.annotate.call_args.kwargs["detections"]
    assert len(detections) == 1
    assert detections.mask is not None
    assert detections.mask.shape == (1, 4, 4)


def test_create_visualisation_keeps_rle_predictions_with_short_empty_and_null_points() -> (
    None
):
    # given
    prediction = {
        "image": {"width": 4, "height": 4},
        "predictions": [
            _rle_prediction(points=[{"x": 0, "y": 0}]),
            _rle_prediction(points=[]),
            _rle_prediction(points=None),
        ],
    }
    annotator = _annotator()

    # when
    result = create_visualisation(
        frame=np.zeros((4, 4, 3), dtype=np.uint8),
        prediction=prediction,
        annotators=[annotator],
        tracker=None,
    )

    # then
    assert result is not None
    detections = annotator.annotate.call_args.kwargs["detections"]
    assert len(detections) == 3
    assert detections.mask is not None
    assert detections.mask.shape == (3, 4, 4)


def test_create_visualisation_does_not_modify_prediction() -> None:
    # given
    prediction = {
        "image": {"width": 4, "height": 4},
        "predictions": [
            _triangle_prediction(),
            _two_point_prediction(),
            _rle_prediction(points=None),
        ],
    }
    original_prediction = copy.deepcopy(prediction)

    # when
    create_visualisation(
        frame=np.zeros((4, 4, 3), dtype=np.uint8),
        prediction=prediction,
        annotators=[_annotator()],
        tracker=None,
    )

    # then
    assert prediction == original_prediction


def test_create_visualisation_returns_none_when_prediction_has_no_detections() -> None:
    # given
    annotator = _annotator()

    # when
    result = create_visualisation(
        frame=np.zeros((4, 4, 3), dtype=np.uint8),
        prediction={"top": "cat"},
        annotators=[annotator],
        tracker=None,
    )

    # then
    assert result is None
    annotator.annotate.assert_not_called()


def test_create_visualisation_returns_frame_when_predictions_is_none() -> None:
    # given
    annotator = _annotator()

    # when
    result = create_visualisation(
        frame=np.zeros((4, 4, 3), dtype=np.uint8),
        prediction={"predictions": None, "image": {"width": 4, "height": 4}},
        annotators=[annotator],
        tracker=None,
    )

    # then
    assert result is not None
    assert len(annotator.annotate.call_args.kwargs["detections"]) == 0


def test_create_visualisation_returns_none_when_image_is_missing_and_points_is_none() -> (
    None
):
    # given
    annotator = _annotator()

    # when
    result = create_visualisation(
        frame=np.zeros((4, 4, 3), dtype=np.uint8),
        prediction={"predictions": [{"points": None}]},
        annotators=[annotator],
        tracker=None,
    )

    # then
    assert result is None
    annotator.annotate.assert_not_called()


def test_create_visualisation_keeps_rle_mask_predictions_with_short_points() -> None:
    # given
    rle_mask_prediction = _rle_prediction(points=[{"x": 0, "y": 0}])
    rle_mask_prediction["rle_mask"] = rle_mask_prediction.pop("rle")
    prediction = {
        "image": {"width": 4, "height": 4},
        "predictions": [_triangle_prediction(), rle_mask_prediction],
    }
    annotator = _annotator()

    # when
    result = create_visualisation(
        frame=np.zeros((4, 4, 3), dtype=np.uint8),
        prediction=prediction,
        annotators=[annotator],
        tracker=None,
    )

    # then
    assert result is not None
    detections = annotator.annotate.call_args.kwargs["detections"]
    assert len(detections) == 2
    assert detections.mask is not None
    assert detections.mask.shape == (2, 4, 4)


def _run_infer(input_reference: str) -> None:
    infer(
        input_reference=input_reference,
        model_id="some/1",
        api_key="some-key",
        host="http://127.0.0.1:9001",
        output_location=None,
        display=False,
        visualise=False,
        visualisation_config=None,
        model_configuration=None,
    )


@pytest.mark.parametrize(
    "input_reference",
    ["clip.mp4", "CLIP.MP4", "clip.MOV", "clip.Mp4", "/some/dir/clip.AVI"],
)
@mock.patch.object(infer_adapter, "infer_on_image")
@mock.patch.object(infer_adapter, "infer_on_video")
def test_infer_treats_video_extension_case_insensitively(
    infer_on_video_mock: MagicMock,
    infer_on_image_mock: MagicMock,
    input_reference: str,
) -> None:
    # when
    _run_infer(input_reference=input_reference)

    # then
    infer_on_video_mock.assert_called_once()
    assert infer_on_video_mock.call_args.kwargs["input_reference"] == input_reference
    infer_on_image_mock.assert_not_called()


@pytest.mark.parametrize("input_reference", ["photo.jpg", "PHOTO.JPG"])
@mock.patch.object(infer_adapter, "infer_on_image")
@mock.patch.object(infer_adapter, "infer_on_video")
def test_infer_keeps_routing_images_to_image_inference(
    infer_on_video_mock: MagicMock,
    infer_on_image_mock: MagicMock,
    input_reference: str,
) -> None:
    # when
    _run_infer(input_reference=input_reference)

    # then
    infer_on_image_mock.assert_called_once()
    infer_on_video_mock.assert_not_called()


@pytest.mark.parametrize(
    "directory_name", ["MP4", "mp4", "clips.MOV", "clips.mov", "./clips.MOV"]
)
@mock.patch.object(infer_adapter, "infer_on_directory")
@mock.patch.object(infer_adapter, "infer_on_image")
@mock.patch.object(infer_adapter, "infer_on_video")
def test_infer_routes_directory_named_like_video_to_directory_inference(
    infer_on_video_mock: MagicMock,
    infer_on_image_mock: MagicMock,
    infer_on_directory_mock: MagicMock,
    directory_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # given
    (tmp_path / directory_name).mkdir()
    monkeypatch.chdir(tmp_path)

    # when
    _run_infer(input_reference=directory_name)

    # then
    infer_on_directory_mock.assert_called_once()
    infer_on_video_mock.assert_not_called()
    infer_on_image_mock.assert_not_called()


@pytest.mark.parametrize("video_name", ["clip.avi", "CLIP.AVI"])
@mock.patch.object(infer_adapter, "initialise_client")
def test_infer_sends_each_frame_of_real_video_to_inference(
    initialise_client_mock: MagicMock,
    video_name: str,
    tmp_path: Path,
) -> None:
    # given
    video_path = tmp_path / video_name
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 64)
    )
    assert writer.isOpened()
    for value in (0, 40, 80):
        writer.write(np.full((64, 64, 3), value, dtype=np.uint8))
    writer.release()
    client = InferenceHTTPClient(api_url="http://127.0.0.1:9001", api_key="some-key")
    client.infer = MagicMock(return_value={"predictions": []})
    initialise_client_mock.return_value = client
    output_location = tmp_path / "output"
    output_location.mkdir()

    # when
    infer(
        input_reference=str(video_path),
        model_id="some/1",
        api_key="some-key",
        host="http://127.0.0.1:9001",
        output_location=str(output_location),
        display=False,
        visualise=False,
        visualisation_config=None,
        model_configuration=None,
    )

    # then
    assert client.infer.call_count == 3
    for call in client.infer.call_args_list:
        assert call.kwargs["inference_input"].shape == (64, 64, 3)
    assert sorted(os.listdir(output_location)) == [
        "frame_000000.json",
        "frame_000001.json",
        "frame_000002.json",
    ]
