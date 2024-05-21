import base64

import cv2
import numpy as np
import supervision as sv

from inference.core.interfaces.http.orjson_utils import (
    serialise_image,
    serialise_list,
    serialise_sv_detections,
    serialise_workflow_result,
)
from inference.core.workflows.entities.base import (
    OriginCoordinatesSystem,
    ParentImageMetadata,
    WorkflowImageData,
)


def test_serialise_image() -> None:
    # given
    np_image = np.zeros((192, 168, 3), dtype=np.uint8)
    image = WorkflowImageData(
        parent_metadata=ParentImageMetadata(
            parent_id="some",
            origin_coordinates=OriginCoordinatesSystem(
                left_top_x=0,
                left_top_y=0,
                origin_width=192,
                origin_height=168,
            ),
        ),
        numpy_image=np_image,
    )

    # when
    result = serialise_image(image=image)

    # then
    assert result["type"] == "base64", "Type of image must point base64"
    decoded = base64.b64decode(result["value"])
    recovered_image = cv2.imdecode(
        np.fromstring(decoded, dtype=np.uint8),
        cv2.IMREAD_UNCHANGED,
    )
    assert (
        recovered_image == np_image
    ).all(), "Recovered image should be equal to input image"


def test_serialise_list() -> None:
    # given
    np_image = np.zeros((192, 168, 3), dtype=np.uint8)
    elements = [
        3,
        "some",
        WorkflowImageData(
            parent_metadata=ParentImageMetadata(
                parent_id="some",
                origin_coordinates=OriginCoordinatesSystem(
                    left_top_x=0,
                    left_top_y=0,
                    origin_width=192,
                    origin_height=168,
                ),
            ),
            numpy_image=np_image,
        ),
    ]

    # when
    result = serialise_list(elements=elements)

    # then
    assert len(result) == 3, "The same number of elements must be returned"
    assert result[0] == 3, "First element of list must be untouched"
    assert result[1] == "some", "Second element of list must be untouched"
    assert (
        result[2]["type"] == "base64"
    ), "Type of third element must be changed into base64"
    decoded = base64.b64decode(result[2]["value"])
    recovered_image = cv2.imdecode(
        np.fromstring(decoded, dtype=np.uint8),
        cv2.IMREAD_UNCHANGED,
    )
    assert (
        recovered_image == np_image
    ).all(), "Recovered image should be equal to input image"


def test_serialise_workflow_result() -> None:
    # given
    np_image = np.zeros((192, 168, 3), dtype=np.uint8)
    workflow_result = {
        "some": [{"detection": 1}],
        "other": WorkflowImageData(
            parent_metadata=ParentImageMetadata(
                parent_id="some",
                origin_coordinates=OriginCoordinatesSystem(
                    left_top_x=0,
                    left_top_y=0,
                    origin_width=192,
                    origin_height=168,
                ),
            ),
            numpy_image=np_image,
        ),
        "third": [
            "some",
            WorkflowImageData(
                parent_metadata=ParentImageMetadata(
                    parent_id="some",
                    origin_coordinates=OriginCoordinatesSystem(
                        left_top_x=0,
                        left_top_y=0,
                        origin_width=192,
                        origin_height=168,
                    ),
                ),
                numpy_image=np_image,
            ),
        ],
        "fourth": "to_be_excluded",
        "fifth": {
            "some": "value",
            "my_image": WorkflowImageData(
                parent_metadata=ParentImageMetadata(
                    parent_id="some",
                    origin_coordinates=OriginCoordinatesSystem(
                        left_top_x=0,
                        left_top_y=0,
                        origin_width=192,
                        origin_height=168,
                    ),
                ),
                numpy_image=np_image,
            ),
        },
        "sixth": [
            "some",
            1,
            [
                2,
                WorkflowImageData(
                    parent_metadata=ParentImageMetadata(
                        parent_id="some",
                        origin_coordinates=OriginCoordinatesSystem(
                            left_top_x=0,
                            left_top_y=0,
                            origin_width=192,
                            origin_height=168,
                        ),
                    ),
                    numpy_image=np_image,
                ),
            ],
        ],
    }

    # when
    result = serialise_workflow_result(
        result=workflow_result, excluded_fields=["fourth"]
    )

    # then
    assert (
        len(result) == 5
    ), "Size of dictionary must be 5, one field should be excluded"
    assert result["some"] == [{"detection": 1}], "Element must not change"
    assert (
        result["other"]["type"] == "base64"
    ), "Type of this element must change due to serialisation"
    assert (
        result["third"][0] == "some"
    ), "This element must not be touched by serialistaion"
    assert (
        result["third"][1]["type"] == "base64"
    ), "Type of this element must change due to serialisation"
    assert (
        result["fifth"]["some"] == "value"
    ), "some key of fifth element is not to be changed"
    assert (
        result["fifth"]["my_image"]["type"] == "base64"
    ), "my_image key of fifth element is to be serialised"
    assert len(result["sixth"]) == 3, "Number of element in sixth list to be preserved"
    assert result["sixth"][:2] == ["some", 1], "First two elements not to be changed"
    assert result["sixth"][2][0] == 2, "First element of nested list not to be changed"
    assert (
        result["sixth"][2][1]["type"] == "base64"
    ), "Second element of nested list to be serialised"


def test_serialise_sv_detections() -> None:
    # given
    np_image = np.zeros((192, 168, 3), dtype=np.uint8)
    detections = sv.Detections(
        xyxy=np.array([[1, 1, 2, 2], [3, 3, 4, 4]], dtype=np.float64),
        class_id=np.array([1, 2]),
        confidence=np.array([0.1, 0.9], dtype=np.float64),
        tracker_id=np.array([1, 2]),
        mask=np.array(
            [
                sv.polygon_to_mask(
                    np.array([[1, 1], [1, 10], [10, 10], [10, 1]]),
                    resolution_wh=(15, 15),
                ),
                sv.polygon_to_mask(
                    np.array([[1, 1], [1, 10], [10, 10], [10, 1]]),
                    resolution_wh=(15, 15),
                ),
            ],
            dtype=bool,
        ),
        data={
            "class_name": np.array(["cat", "dog"]),
            "detection_id": np.array(["first", "second"]),
            "parent_id": np.array(["image", "image"]),
            "keypoints_xy": np.array(
                [
                    np.array([[11, 11], [12, 13], [14, 15]], dtype=np.float64),
                    np.array(
                        [[16, 16], [17, 17], [18, 18], [19, 19]], dtype=np.float64
                    ),
                ],
                dtype="object",
            ),
            "keypoints_class_id": np.array(
                [
                    np.array([1, 2, 3]),
                    np.array([1, 2, 3, 4]),
                ],
                dtype="object",
            ),
            "keypoints_class_name": np.array(
                [
                    np.array(["nose", "ear", "eye"]),
                    np.array(["nose", "ear", "eye", "tail"]),
                ],
                dtype="object",
            ),
            "keypoints_confidence": np.array(
                [
                    np.array([0.1, 0.2, 0.3], dtype=np.float64),
                    np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64),
                ],
                dtype="object",
            ),
        },
    )

    # when
    result = serialise_sv_detections(detections=detections)

    # then
    assert result == [
        {
            "width": 1.0,
            "height": 1.0,
            "x": 1.5,
            "y": 1.5,
            "confidence": 0.1,
            "class_id": 1,
            "points": [
                {"x": 1.0, "y": 1.0},
                {"x": 1.0, "y": 10.0},
                {"x": 10.0, "y": 10.0},
                {"x": 10.0, "y": 1.0},
            ],
            "tracker_id": 1,
            "class": "cat",
            "detection_id": "first",
            "parent_id": "image",
            "keypoints": [
                {
                    "keypoints_class_id": 1,
                    "keypoints_class_name": "nose",
                    "keypoints_confidence": 0.1,
                    "x": 11.0,
                    "y": 11.0,
                },
                {
                    "keypoints_class_id": 2,
                    "keypoints_class_name": "ear",
                    "keypoints_confidence": 0.2,
                    "x": 12.0,
                    "y": 13.0,
                },
                {
                    "keypoints_class_id": 3,
                    "keypoints_class_name": "eye",
                    "keypoints_confidence": 0.3,
                    "x": 14.0,
                    "y": 15.0,
                },
            ],
        },
        {
            "width": 1.0,
            "height": 1.0,
            "x": 3.5,
            "y": 3.5,
            "confidence": 0.9,
            "class_id": 2,
            "points": [
                {"x": 1.0, "y": 1.0},
                {"x": 1.0, "y": 10.0},
                {"x": 10.0, "y": 10.0},
                {"x": 10.0, "y": 1.0},
            ],
            "tracker_id": 2,
            "class": "dog",
            "detection_id": "second",
            "parent_id": "image",
            "keypoints": [
                {
                    "keypoints_class_id": 1,
                    "keypoints_class_name": "nose",
                    "keypoints_confidence": 0.1,
                    "x": 16.0,
                    "y": 16.0,
                },
                {
                    "keypoints_class_id": 2,
                    "keypoints_class_name": "ear",
                    "keypoints_confidence": 0.2,
                    "x": 17.0,
                    "y": 17.0,
                },
                {
                    "keypoints_class_id": 3,
                    "keypoints_class_name": "eye",
                    "keypoints_confidence": 0.3,
                    "x": 18.0,
                    "y": 18.0,
                },
                {
                    "keypoints_class_id": 4,
                    "keypoints_class_name": "tail",
                    "keypoints_confidence": 0.4,
                    "x": 19.0,
                    "y": 19.0,
                },
            ],
        },
    ]
