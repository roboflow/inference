import numpy as np

from inference.core.interfaces.http.orjson_utils import serialise_workflow_result
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


def test_serialise_workflow_result() -> None:
    # given
    np_image = np.zeros((192, 168, 3), dtype=np.uint8)
    workflow_result = [
        {
            "some": [{"detection": 1}],
            "other": WorkflowImageData(
                parent_metadata=ImageParentMetadata(parent_id="some"),
                numpy_image=np_image,
            ),
            "third": [
                "some",
                WorkflowImageData(
                    parent_metadata=ImageParentMetadata(parent_id="some"),
                    numpy_image=np_image,
                ),
            ],
            "fourth": "to_be_excluded",
            "fifth": {
                "some": "value",
                "my_image": WorkflowImageData(
                    parent_metadata=ImageParentMetadata(parent_id="some"),
                    numpy_image=np_image,
                ),
            },
            "sixth": [
                "some",
                1,
                [
                    2,
                    WorkflowImageData(
                        parent_metadata=ImageParentMetadata(parent_id="some"),
                        numpy_image=np_image,
                    ),
                ],
            ],
        }
    ]

    # when
    result = serialise_workflow_result(
        result=workflow_result, excluded_fields=["fourth"]
    )

    # then
    assert len(result) == 1, "Expected to get list of one element"
    result_element = result[0]
    assert (
        len(result_element) == 5
    ), "Size of dictionary must be 5, one field should be excluded"
    assert result_element["some"] == [{"detection": 1}], "Element must not change"
    assert (
        result_element["other"]["type"] == "base64"
    ), "Type of this element must change due to serialisation"
    assert (
        result_element["third"][0] == "some"
    ), "This element must not be touched by serialistaion"
    assert (
        result_element["third"][1]["type"] == "base64"
    ), "Type of this element must change due to serialisation"
    assert (
        result_element["fifth"]["some"] == "value"
    ), "some key of fifth element is not to be changed"
    assert (
        result_element["fifth"]["my_image"]["type"] == "base64"
    ), "my_image key of fifth element is to be serialised"
    assert (
        len(result_element["sixth"]) == 3
    ), "Number of element in sixth list to be preserved"
    assert result_element["sixth"][:2] == [
        "some",
        1,
    ], "First two elements not to be changed"
    assert (
        result_element["sixth"][2][0] == 2
    ), "First element of nested list not to be changed"
    assert (
        result_element["sixth"][2][1]["type"] == "base64"
    ), "Second element of nested list to be serialised"
