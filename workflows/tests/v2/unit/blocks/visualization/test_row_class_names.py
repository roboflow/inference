"""Label names of ``prepare_drawings``: a row's own name wins (decision 025).

Name of row ``i``::

    bboxes_metadata[i]["class"], if it is a string
    else image_metadata["class_names"][class_id[i]]
    else ValueError naming the row, its id and why neither source names it
"""

from typing import Any, List, Optional

import numpy as np
import pytest
import torch
from roboflow_workflows.v2.blocks.visualization import drawing
from roboflow_workflows.v2.blocks.visualization.drawing import prepare_drawings

from inference_models.models.base.object_detection import Detections


def detections(
    class_ids: List[int],
    *,
    class_names: Any = None,
    rows: Optional[list] = None,
    confidences: Optional[List[float]] = None,
) -> Detections:
    count = len(class_ids)
    image_metadata = {} if class_names is None else {"class_names": class_names}
    result = Detections(
        xyxy=torch.arange(count * 4, dtype=torch.float32).reshape(count, 4),
        class_id=torch.tensor(class_ids, dtype=torch.int64),
        confidence=torch.tensor(confidences or [0.5] * count, dtype=torch.float32),
        image_metadata=image_metadata,
        bboxes_metadata=rows,
    )

    return result


def labels_of(prediction: Detections) -> tuple:
    (prepared,) = prepare_drawings([prediction])

    return prepared.labels


def test_without_row_names_labels_come_from_the_class_map() -> None:
    # given
    by_map = detections(
        [1, 0], class_names={0: "box", 1: "disc"}, confidences=[0.9, 0.25]
    )
    by_list = detections([1, 0], class_names=["box", "disc"], confidences=[0.9, 0.25])
    other_row_keys = detections(
        [1, 0],
        class_names={0: "box", 1: "disc"},
        rows=[{"marker": 1}, {"detection_id": "a"}],
        confidences=[0.9, 0.25],
    )

    # then
    assert labels_of(by_map) == ("disc 0.90", "box 0.25")
    assert labels_of(by_list) == ("disc 0.90", "box 0.25")
    assert labels_of(other_row_keys) == ("disc 0.90", "box 0.25")


def test_ocr_texts_with_one_class_id_keep_their_own_names() -> None:
    # given: OCR replacement gives every text class id 0; the map names the first
    prediction = detections(
        [0, 0, 0],
        class_names={0: "ABC123"},
        rows=[{"class": "ABC123"}, {"class": "XYZ789"}, {"class": ""}],
        confidences=[1.0, 1.0, 1.0],
    )

    # when
    (prepared,) = prepare_drawings([prediction])

    # then: names per row; colours still follow the shared class id
    assert prepared.labels == ("ABC123 1.00", "XYZ789 1.00", " 1.00")
    assert prepared.class_id.tolist() == [0, 0, 0]


def test_combined_models_with_clashing_ids_label_each_row_by_its_model() -> None:
    # given: combine keeps ids; the losing model's rows carry their own name
    prediction = detections(
        [0, 0, 1],
        class_names={0: "person", 1: "bus"},
        rows=[{}, {"class": "car"}, {}],
        confidences=[0.9, 0.8, 0.7],
    )

    # then
    assert labels_of(prediction) == ("person 0.90", "car 0.80", "bus 0.70")


def test_row_names_alone_need_no_class_map() -> None:
    # given
    prediction = detections([3, 7], rows=[{"class": "plate"}, {"class": "sign"}])

    # then
    assert labels_of(prediction) == ("plate 0.50", "sign 0.50")


def test_malformed_row_names_fall_back_to_the_class_map() -> None:
    # given: names that are no strings, and rows that are no mappings
    prediction = detections(
        [0, 1, 0, 1],
        class_names={0: "box", 1: "disc"},
        rows=[{"class": None}, {"class": 7}, None, "not a row"],
    )

    # then
    assert labels_of(prediction) == ("box 0.50", "disc 0.50", "box 0.50", "disc 0.50")


@pytest.mark.parametrize(
    "class_names, rows, message",
    [
        (None, None, "row 0 with class id 0 has no name: image_metadata has no class_names"),
        ({1: "disc"}, [{}], r"row 0 with class id 0 has no name: class_names does not name it \(ids: \[1\]\)"),
        (None, [{"class": 5}], "image_metadata has no class_names; bboxes_metadata\\['class'\\] is 5, not a string"),
        ({1: "disc"}, [{"class": None}], "bboxes_metadata\\['class'\\] is None, not a string"),
    ],
)  # fmt: skip
def test_a_row_without_any_name_is_an_error_saying_why(
    class_names: Any, rows: Optional[list], message: str
) -> None:
    # given
    prediction = detections([0], class_names=class_names, rows=rows)

    # then
    with pytest.raises(ValueError, match=message):
        prepare_drawings([prediction])


def test_row_metadata_of_another_length_is_an_error() -> None:
    # given
    prediction = detections([0, 0], class_names={0: "box"}, rows=[{"class": "a"}])

    # then
    with pytest.raises(ValueError, match="bboxes_metadata has 1 rows for 2 boxes"):
        prepare_drawings([prediction])


def test_empty_detections_need_no_names() -> None:
    # given
    prediction = detections([], rows=[])

    # then
    assert labels_of(prediction) == ()


def test_a_mixed_batch_still_makes_one_host_transfer(monkeypatch) -> None:
    # given
    transfers = []
    to_host = drawing._to_host

    def counted(tensors):
        transfers.append(len(tensors))
        return to_host(tensors)

    monkeypatch.setattr(drawing, "_to_host", counted)
    named = detections(
        [0, 0], class_names={0: "a"}, rows=[{"class": "a"}, {"class": "b"}]
    )
    mapped = detections([1], class_names={1: "disc"})

    # when
    first, second = prepare_drawings([named, mapped])

    # then
    assert transfers == [6]
    assert first.labels == ("a 0.50", "b 0.50")
    assert second.labels == ("disc 0.50",)
    for prepared in (first, second):
        assert not any(
            array.flags.writeable
            for array in (prepared.xyxy, prepared.class_id, prepared.confidence)
        )
    np.testing.assert_array_equal(first.xyxy, named.xyxy.numpy())
