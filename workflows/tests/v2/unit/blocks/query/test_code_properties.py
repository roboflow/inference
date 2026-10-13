"""UQL reads the decoded text of code detections as ``decoded_text`` or ``data``.

The fixture is built by the code detectors' own ``code_detections``, so rows
carry what a real detector returns: the text in ``bboxes_metadata[i]["data"]``,
the image frame, ``class_names`` and a row identity::

    row  text            box
    0    LOT-42          (10, 10, 60, 60)
    1    https://x.org   (100, 10, 150, 60)
    2    LOT-42-SPARE    (10, 100, 60, 150)
"""

import numpy as np
import pytest
from roboflow_workflows.v2.blocks.codes.detectors import code_detections
from roboflow_workflows.v2.blocks.query import (
    DETECTION_PROPERTIES,
    QueryDefinitionError,
    QueryRuntimeError,
    compile_operations,
)
from roboflow_workflows.v2.data.frames import frame_of
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.row_identity import row_record

from inference_models.models.base.object_detection import Detections

TEXTS = ["LOT-42", "https://x.org", "LOT-42-SPARE"]
BOXES = [(10, 10, 60, 60), (100, 10, 150, 60), (10, 100, 60, 150)]


def _codes(texts=TEXTS, boxes=BOXES) -> Detections:
    image = ImageData.from_numpy_rgb(np.zeros((200, 200, 3), dtype=np.uint8))
    corners = [[(x0, y0), (x1, y0), (x1, y1), (x0, y1)] for x0, y0, x1, y1 in boxes]
    detections = code_detections(
        image,
        corners=corners,
        texts=list(texts),
        class_name="qr_code",
        prediction_type="qrcode-detection",
    )

    return detections


def _text_filter(comparator: str, right: dict, *, name: str = "decoded_text") -> dict:
    operand = {
        "type": "DynamicOperand",
        "operations": [{"type": "ExtractDetectionProperty", "property_name": name}],
    }
    detection_filter = {
        "type": "DetectionsFilter",
        "filter_operation": {
            "type": "StatementGroup",
            "statements": [
                {
                    "type": "BinaryStatement",
                    "left_operand": operand,
                    "comparator": {"type": comparator},
                    "right_operand": right,
                }
            ],
        },
    }

    return detection_filter


def test_decoded_text_and_its_data_alias_are_detection_properties() -> None:
    assert {"decoded_text", "data"} <= set(DETECTION_PROPERTIES)


@pytest.mark.parametrize("name", ["decoded_text", "data"])
def test_extract_reads_each_rows_decoded_text_in_row_order(name) -> None:
    # given
    chain = compile_operations(
        [{"type": "DetectionsPropertyExtract", "property_name": name}]
    )

    # when
    texts = chain.apply(_codes())

    # then
    assert texts == TEXTS


def test_filter_on_decoded_text_keeps_rows_with_geometry_metadata_and_identity() -> (
    None
):
    # given
    codes = _codes()
    chain = compile_operations(
        [_text_filter("==", {"type": "StaticOperand", "value": "LOT-42"})]
    )

    # when
    kept = chain.apply(codes)

    # then
    assert [row["data"] for row in kept.bboxes_metadata] == ["LOT-42"]
    assert kept.xyxy.tolist() == [[10.0, 10.0, 60.0, 60.0]]
    assert kept.image_metadata["class_names"] == {0: "qr_code"}
    assert kept.image_metadata["prediction_type"] == "qrcode-detection"
    assert frame_of(kept) == frame_of(codes)
    assert row_record(kept).namespace == row_record(codes).namespace
    assert row_record(kept).host_ids() == [0]


def test_filter_text_can_come_from_a_selector_parameter() -> None:
    # given
    chain = compile_operations(
        [
            _text_filter(
                "(String) startsWith",
                {"type": "DynamicOperand", "operand_name": "prefix"},
                name="data",
            )
        ],
        declared_names={"prefix"},
    )

    # when
    kept = chain.apply(_codes(), parameters={"prefix": "LOT-42"})
    none_kept = chain.apply(_codes(), parameters={"prefix": "nothing"})

    # then
    assert [row["data"] for row in kept.bboxes_metadata] == ["LOT-42", "LOT-42-SPARE"]
    assert len(none_kept) == 0


def test_no_codes_give_an_empty_list_and_an_empty_filter_result() -> None:
    # given
    extract = compile_operations(
        [{"type": "DetectionsPropertyExtract", "property_name": "decoded_text"}]
    )
    keep = compile_operations(
        [_text_filter("==", {"type": "StaticOperand", "value": "LOT-42"})]
    )
    empty = _codes(texts=[], boxes=[])

    # when
    texts = extract.apply(empty)
    kept = keep.apply(empty)

    # then
    assert texts == []
    assert len(kept) == 0


def test_rows_without_text_keep_the_missing_property_semantics() -> None:
    # given
    no_text = _codes()
    no_text.bboxes_metadata[1].pop("data")
    extract = compile_operations(
        [{"type": "DetectionsPropertyExtract", "property_name": "decoded_text"}]
    )
    without_text = compile_operations(
        [
            {
                "type": "DetectionsFilter",
                "filter_operation": {
                    "type": "StatementGroup",
                    "statements": [
                        {
                            "type": "UnaryStatement",
                            "operand": {
                                "type": "DynamicOperand",
                                "operations": [
                                    {
                                        "type": "ExtractDetectionProperty",
                                        "property_name": "decoded_text",
                                    }
                                ],
                            },
                            "operator": {"type": "DoesNotExist"},
                        }
                    ],
                },
            }
        ]
    )

    # when
    texts = extract.apply(no_text)
    kept = without_text.apply(no_text)

    # then
    assert texts == ["LOT-42", None, "LOT-42-SPARE"]
    assert kept.xyxy.tolist() == [[100.0, 10.0, 150.0, 60.0]]


def test_text_no_row_carries_fails_with_the_requested_name() -> None:
    # given
    no_text = _codes()
    for row in no_text.bboxes_metadata:
        row.pop("data")
    chain = compile_operations(
        [{"type": "DetectionsPropertyExtract", "property_name": "decoded_text"}]
    )

    # when
    with pytest.raises(QueryRuntimeError) as error:
        chain.apply(no_text)

    # then
    assert "property `decoded_text`" in error.value.detail


def test_unknown_property_names_still_fail_at_definition() -> None:
    # when
    with pytest.raises(QueryDefinitionError) as error:
        compile_operations(
            [{"type": "DetectionsPropertyExtract", "property_name": "decoded"}]
        )

    # then
    assert error.value.parameter == "property_name"
