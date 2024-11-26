import numpy as np
import pytest
import supervision as sv
from pydantic import ValidationError

from inference.core.workflows.core_steps.visualizations.keypoint.v1 import (
    KeypointManifest,
    KeypointVisualizationBlockV1,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)

TEST_PREDICTIONS = sv.Detections(
    xyxy=np.array(
        [[615, 16, 1438, 960], [145, 203, 760, 744]],
        dtype=np.float64,
    ),
    class_id=np.array([0, 0]),
    data={
        "keypoints_xy": [
            [
                [1135.0, 301.0],
                [1217.0, 246.0],
                [1072.0, 246.0],
                [1326.0, 291.0],
                [991.0, 290.0],
                [1440.0, 623.0],
                [836.0, 642.0],
                [1440.0, 848.0],
                [714.0, 924.0],
                [1273.0, 624.0],
                [785.0, 863.0],
                [1303.0, 960.0],
                [957.0, 960.0],
                [1098.0, 940.0],
                [798.0, 900.0],
                [1145.0, 926.0],
                [1131.0, 877.0],
            ],
            [
                [547.0, 356.0],
                [589.0, 334.0],
                [504.0, 329.0],
                [636.0, 372.0],
                [444.0, 362.0],
                [690.0, 574.0],
                [358.0, 537.0],
                [722.0, 776.0],
                [196.0, 677.0],
                [689.0, 759.0],
                [325.0, 532.0],
                [623.0, 871.0],
                [419.0, 859.0],
                [575.0, 813.0],
                [365.0, 809.0],
                [484.0, 835.0],
                [406.0, 826.0],
            ],
        ],
        "keypoints_confidence": [
            [
                0.9955374002456665,
                0.9850325584411621,
                0.9924459457397461,
                0.6771311163902283,
                0.8257092237472534,
                0.6847628355026245,
                0.988980770111084,
                0.020470470190048218,
                0.4994047284126282,
                0.10626623034477234,
                0.4919512867927551,
                0.002728283405303955,
                0.013417482376098633,
                0.00026804208755493164,
                0.0010205507278442383,
                0.00011846423149108887,
                0.0002935826778411865,
            ],
            [
                0.9964120388031006,
                0.9867579340934753,
                0.9912893772125244,
                0.7188618779182434,
                0.8569645881652832,
                0.9544534683227539,
                0.9891356229782104,
                0.503325343132019,
                0.8857028484344482,
                0.6204462647438049,
                0.9193166494369507,
                0.13720226287841797,
                0.26782700419425964,
                0.0065190792083740234,
                0.014030814170837402,
                0.001272827386856079,
                0.0018590092658996582,
            ],
        ],
        "keypoints_class_name": [
            [
                "nose",
                "left_eye",
                "right_eye",
                "left_ear",
                "right_ear",
                "left_shoulder",
                "right_shoulder",
                "left_elbow",
                "right_elbow",
                "left_wrist",
                "right_wrist",
                "left_hip",
                "right_hip",
                "left_knee",
                "right_knee",
                "left_ankle",
                "right_ankle",
            ],
            [
                "nose",
                "left_eye",
                "right_eye",
                "left_ear",
                "right_ear",
                "left_shoulder",
                "right_shoulder",
                "left_elbow",
                "right_elbow",
                "left_wrist",
                "right_wrist",
                "left_hip",
                "right_hip",
                "left_knee",
                "right_knee",
                "left_ankle",
                "right_ankle",
            ],
        ],
    },
)


@pytest.mark.parametrize(
    "type_alias",
    ["roboflow_core/keypoint_visualization@v1", "KeypointVisualization"],
)
@pytest.mark.parametrize("images_field_alias", ["images", "image"])
def test_keypoint_validation_when_valid_manifest_is_given(
    type_alias: str,
    images_field_alias: str,
) -> None:
    # given
    data = {
        "type": type_alias,
        "name": "keypoints1",
        "predictions": "$steps.kp_model.predictions",
        images_field_alias: "$inputs.image",
        "annotator_type": "edge",
        "color": "#A351FB",
        "thickness": 2,
        "radius": 10,
    }

    # when
    result = KeypointManifest.model_validate(data)

    # then
    assert result == KeypointManifest(
        type=type_alias,
        name="keypoints1",
        images="$inputs.image",
        predictions="$steps.kp_model.predictions",
        annotator_type="edge",
        color="#A351FB",
        text_color="black",
        text_scale=0.5,
        text_thickness=1,
        text_padding=10,
        thickness=2,
        radius=10,
    )


def test_keypoint_validation_when_invalid_image_is_given() -> None:
    # given
    data = {
        "type": "KeypointVisualization",
        "name": "keypoints1",
        "images": "invalid",
        "predictions": "$steps.kp_model.predictions",
        "annotator_type": "edge",
        "color": "#A351FB",
    }

    # when
    with pytest.raises(ValidationError):
        _ = KeypointManifest.model_validate(data)


def test_keypoint_visualization_block_edge() -> None:
    # given
    block = KeypointVisualizationBlockV1()

    start_image = np.zeros((1000, 1000, 3), dtype=np.uint8)
    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=start_image,
        ),
        predictions=TEST_PREDICTIONS,
        copy_image=True,
        annotator_type="edge",
        color="#A351FB",
        text_color="black",
        text_scale=0.5,
        text_thickness=1,
        text_padding=10,
        thickness=2,
        radius=10,
    )

    assert output is not None
    assert "image" in output
    assert hasattr(output.get("image"), "numpy_image")

    # dimensions of output match input
    assert output.get("image").numpy_image.shape == (1000, 1000, 3)
    # check if the image is modified
    assert not np.array_equal(
        output.get("image").numpy_image, np.zeros((1000, 1000, 3), dtype=np.uint8)
    )

    # check that the image is copied
    assert (
        output.get("image").numpy_image.__array_interface__["data"][0]
        != start_image.__array_interface__["data"][0]
    )


def test_keypoint_visualization_block_vertex() -> None:
    # given
    block = KeypointVisualizationBlockV1()

    start_image = np.zeros((1000, 1000, 3), dtype=np.uint8)
    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=start_image,
        ),
        predictions=TEST_PREDICTIONS,
        copy_image=True,
        annotator_type="vertex",
        color="#A351FB",
        text_color="black",
        text_scale=0.5,
        text_thickness=1,
        text_padding=10,
        thickness=2,
        radius=10,
    )

    assert output is not None
    assert "image" in output
    assert hasattr(output.get("image"), "numpy_image")

    # dimensions of output match input
    assert output.get("image").numpy_image.shape == (1000, 1000, 3)
    # check if the image is modified
    assert not np.array_equal(
        output.get("image").numpy_image, np.zeros((1000, 1000, 3), dtype=np.uint8)
    )

    # check that the image is copied
    assert (
        output.get("image").numpy_image.__array_interface__["data"][0]
        != start_image.__array_interface__["data"][0]
    )


def test_keypoint_visualization_block_vertex_label() -> None:
    # given
    block = KeypointVisualizationBlockV1()

    start_image = np.zeros((1000, 1000, 3), dtype=np.uint8)
    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=start_image,
        ),
        predictions=TEST_PREDICTIONS,
        copy_image=True,
        annotator_type="vertex",
        color="#A351FB",
        text_color="black",
        text_scale=0.5,
        text_thickness=1,
        text_padding=10,
        thickness=2,
        radius=10,
    )

    assert output is not None
    assert "image" in output
    assert hasattr(output.get("image"), "numpy_image")

    # dimensions of output match input
    assert output.get("image").numpy_image.shape == (1000, 1000, 3)
    # check if the image is modified
    assert not np.array_equal(
        output.get("image").numpy_image, np.zeros((1000, 1000, 3), dtype=np.uint8)
    )

    # check that the image is copied
    assert (
        output.get("image").numpy_image.__array_interface__["data"][0]
        != start_image.__array_interface__["data"][0]
    )


def test_keypoint_visualization_block_nocopy() -> None:
    # given
    block = KeypointVisualizationBlockV1()

    start_image = np.zeros((1000, 1000, 3), dtype=np.uint8)
    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=start_image,
        ),
        predictions=TEST_PREDICTIONS,
        copy_image=False,
        annotator_type="edge",
        color="#A351FB",
        text_color="black",
        text_scale=0.5,
        text_thickness=1,
        text_padding=10,
        thickness=2,
        radius=10,
    )

    assert output is not None
    assert "image" in output
    assert hasattr(output.get("image"), "numpy_image")

    # dimensions of output match input
    assert output.get("image").numpy_image.shape == (1000, 1000, 3)
    # check if the image is modified
    assert not np.array_equal(
        output.get("image").numpy_image, np.zeros((1000, 1000, 3), dtype=np.uint8)
    )

    # check if the image reference references the same memory space as the start_image
    assert (
        output.get("image").numpy_image.__array_interface__["data"][0]
        == start_image.__array_interface__["data"][0]
    )


def test_keypoint_visualization_block_no_predictions() -> None:
    # given
    block = KeypointVisualizationBlockV1()
    start_image = np.zeros((1000, 1000, 3), dtype=np.uint8)

    empty_predictions = sv.Detections.empty()

    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=start_image,
        ),
        predictions=empty_predictions,
        copy_image=True,
        annotator_type="edge",
        color="#A351FB",
        text_color="black",
        text_scale=0.5,
        text_thickness=1,
        text_padding=10,
        thickness=2,
        radius=10,
    )

    assert output is not None
    assert "image" in output
    assert hasattr(output.get("image"), "numpy_image")

    # dimensions of output match input
    assert output.get("image").numpy_image.shape == (1000, 1000, 3)

    # check if the image is unchanged since there were no predictions
    assert np.array_equal(
        output.get("image").numpy_image, np.zeros((1000, 1000, 3), dtype=np.uint8)
    )

    # check that the image is copied
    assert (
        output.get("image").numpy_image.__array_interface__["data"][0]
        != start_image.__array_interface__["data"][0]
    )
