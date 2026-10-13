"""Decoding of VLM answers into native predictions.

Expected boxes are computed by hand from each format's contract on a
400 x 200 (width x height) image.
"""

import json
import math

import numpy as np
import pytest
import torch
from pycocotools import mask as mask_utils
from roboflow_workflows.v2.blocks.language.decoding import (
    DETECTION_BOX_FORMATS,
    anthropic_upload_size,
    build_instance_segmentation_prompt,
    build_object_detection_prompt,
    decode_classification,
    decode_instance_segmentations,
    decode_object_detections,
    decode_vlm_output,
    extract_json,
    max_edge_upload_size,
    vlm_prediction_outputs,
)
from roboflow_workflows.v2.data.frames import frame_of
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.language import LANGUAGE_MODEL_OUTPUT_KIND
from roboflow_workflows.v2.data.predictions import (
    CLASSIFICATION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
)
from roboflow_workflows.v2.data.row_identity import (
    RowRef,
    classification_parent,
    row_record,
)

from inference_models.models.base.classification import (
    ClassificationPrediction,
    MultiLabelClassificationPrediction,
)

CLASSES = ["dog", "cat"]


def _image(height: int = 200, width: int = 400) -> ImageData:
    image = ImageData.from_numpy_rgb(np.zeros((height, width, 3), dtype=np.uint8))

    return image


def _detect(answer, box_format: str, **kwargs):
    decoded = decode_object_detections(
        answer,
        image=kwargs.pop("image", _image()),
        classes=kwargs.pop("classes", CLASSES),
        box_format=box_format,
        inference_id="answer-1",
        **kwargs,
    )

    return decoded


# --- language_model_output kind -----------------------------------------------


@pytest.mark.parametrize("payload", ["text", {"detections": []}])
def test_language_model_output_kind_accepts_text_and_dicts_unchanged(payload) -> None:
    LANGUAGE_MODEL_OUTPUT_KIND.check(payload)

    assert LANGUAGE_MODEL_OUTPUT_KIND.to_payload(payload) is payload
    assert LANGUAGE_MODEL_OUTPUT_KIND.to_serialized(payload) is payload


@pytest.mark.parametrize("payload", [None, 3, ["a"], b"text"])
def test_language_model_output_kind_rejects_other_payloads(payload) -> None:
    with pytest.raises(Exception, match="language_model_output"):
        LANGUAGE_MODEL_OUTPUT_KIND.check(payload)


# --- JSON extraction ------------------------------------------------------------


@pytest.mark.parametrize(
    "answer, expected",
    [
        ('Sure:\n```json\n{"a": 1}\n```\nand ```json\n{"a": 2}\n```', {"a": 1}),
        ("```python\n[1, 2]\n```", [1, 2]),
        ('{"a": 1}\n```', {"a": 1}),
        ('{"a": 1}\n{"b": 2}\n', [{"a": 1}, {"b": 2}]),
        ('[{"a": 1}], [{"b": 2}]]', [{"a": 1}, {"b": 2}]),
        ('I found these: [{"a": 1}] - done', [{"a": 1}]),
        ('Answer: {"a": {"b": 1}} ok', {"a": {"b": 1}}),
    ],
)
def test_extract_json_reads_fences_sequences_and_embedded_json(
    answer, expected
) -> None:
    assert extract_json(answer) == (False, expected)


@pytest.mark.parametrize("answer", ["no json here", "", "[1, 2", 42, None])
def test_extract_json_reports_unreadable_answers(answer) -> None:
    assert extract_json(answer) == (True, None)


def test_extract_json_returns_a_decoded_answer_as_is() -> None:
    answer = {"class_name": "dog", "confidence": 1}

    assert extract_json(answer)[1] is answer


def test_extract_json_survives_hostile_nesting_and_huge_numbers() -> None:
    assert extract_json("[" * 100_000 + "]" * 100_000) == (True, None)
    assert extract_json('{"a": 1' + "0" * 5000 + "}")[0] is True


def test_truncated_named_boxes_are_salvaged_only_when_asked() -> None:
    first = {"label": "a", "x_min": 1, "y_min": 2, "x_max": 3, "y_max": 4}
    second = {"label": "b", "x_min": 5, "y_min": 6, "x_max": 7, "y_max": 8}
    answer = f'[{json.dumps(first)}, {{"note": 1}}, {json.dumps(second)}, {{"lab'

    assert extract_json(answer) == (True, None)
    assert extract_json(answer, salvage_truncated_detections=True) == (
        False,
        [first, second],
    )


# --- classification ---------------------------------------------------------------


def test_single_label_answer_keeps_the_full_confidence_vector() -> None:
    error_status, prediction = decode_classification(
        '```json\n{"class_name": "cat", "confidence": 0.8}\n```',
        image=_image(),
        classes=CLASSES,
    )

    assert error_status is False
    assert isinstance(prediction, ClassificationPrediction)
    CLASSIFICATION_PREDICTION_KIND.check(prediction)
    assert prediction.class_id.tolist() == [1]
    assert prediction.confidence[0].tolist() == pytest.approx([0.0, 0.8])
    metadata = prediction.images_metadata[0]
    assert metadata["class_names"] == {0: "dog", 1: "cat"}
    assert metadata["prediction_type"] == "classification"
    assert frame_of(prediction).size_hw == (200, 400)


def test_label_outside_classes_is_appended_after_them() -> None:
    _, prediction = decode_classification(
        {"class_name": "bird", "confidence": 3}, image=_image(), classes=CLASSES
    )

    assert prediction.class_id.tolist() == [2]
    assert prediction.confidence.tolist() == [[0.0, 0.0, 1.0]]
    assert prediction.images_metadata[0]["class_names"][2] == "bird"


def test_multi_label_answer_keeps_highest_confidence_per_label() -> None:
    answer = {
        "predicted_classes": [
            {"class": "bird", "confidence": 0.2},
            {"class": "dog", "confidence": 0.4},
            {"class": "bird", "confidence": 0.7},
        ]
    }

    error_status, prediction = decode_classification(
        answer, image=_image(), classes=CLASSES
    )

    assert error_status is False
    assert isinstance(prediction, MultiLabelClassificationPrediction)
    CLASSIFICATION_PREDICTION_KIND.check(prediction)
    assert prediction.class_ids.tolist() == [2, 0]
    assert prediction.confidence.tolist() == pytest.approx([0.4, 0.0, 0.7])
    assert prediction.image_metadata["class_names"] == {0: "dog", 1: "cat", 2: "bird"}


def test_bare_list_of_class_entries_is_a_multi_label_answer() -> None:
    answer = '[{"label": "cat"}, {"class_name": "dog", "confidence": 0.5}]'

    _, prediction = decode_classification(answer, image=_image(), classes=CLASSES)

    assert prediction.class_ids.tolist() == [1, 0]
    assert prediction.confidence.tolist() == pytest.approx([0.5, 1.0])


def test_empty_predicted_classes_is_a_valid_answer_with_no_label() -> None:
    _, prediction = decode_classification(
        {"predicted_classes": []}, image=_image(), classes=CLASSES
    )

    assert prediction.class_ids.tolist() == []
    assert prediction.confidence.tolist() == [0.0, 0.0]


@pytest.mark.parametrize(
    "answer",
    [
        '{"class_name": "dog"}',
        '{"class_name": "dog", "confidence": NaN}',
        '{"class_name": "dog", "confidence": "high"}',
        '{"predicted_classes": [{"class": "dog"}]}',
        '{"predicted_classes": ["dog"]}',
        "[]",
        '[{"confidence": 0.3}]',
        "the image shows a dog",
    ],
)
def test_unreadable_classification_answers_set_error_status(answer) -> None:
    assert decode_classification(answer, image=_image(), classes=CLASSES) == (
        True,
        None,
    )


def test_classification_of_a_crop_records_the_detection_row() -> None:
    crop = _image().with_parent_row(RowRef(namespace="boxes", id=3))

    _, single = decode_classification(
        '{"class_name": "dog", "confidence": 1}', image=crop, classes=CLASSES
    )
    _, multi = decode_classification(
        '{"predicted_classes": []}', image=crop, classes=CLASSES
    )

    assert classification_parent(single) == RowRef(namespace="boxes", id=3)
    assert classification_parent(multi) == RowRef(namespace="boxes", id=3)


@pytest.mark.parametrize(
    "classes, message",
    [("dog", "list of strings"), (["dog", 1], "strings"), (["a", "a"], "repeat")],
)
def test_invalid_classes_are_a_call_error(classes, message) -> None:
    with pytest.raises(ValueError, match=message):
        decode_classification("{}", image=_image(), classes=classes)


# --- detection ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "box_format, entry, upload_size_wh, expected",
    [
        ("xyxy_0_1000", {"box_2d": [100, 500, 600, 1000]}, None, [40, 100, 240, 200]),
        ("yxyx_0_1000", {"box_2d": [100, 500, 600, 1000]}, None, [200, 20, 400, 120]),
        ("xyxy_percent", {"box_2d": [10, 50, 60, 100]}, None, [40, 100, 240, 200]),
        (
            "named_0_1000",
            {"x_min": 100, "y_min": 500, "x_max": 600, "y_max": 1000},
            None,
            [40, 100, 240, 200],
        ),
        (
            "named_normalized",
            {"x_min": 0.1, "y_min": 0.5, "x_max": 0.6, "y_max": 1},
            None,
            [40, 100, 240, 200],
        ),
        (
            "xyxy_absolute",
            {"box_2d": [20, 50, 120, 100]},
            (200, 100),
            [40, 100, 240, 200],
        ),
        # Legacy 0-1 named fields when box_2d is missing (V1 OpenAI, Claude, Gemini).
        (
            "xyxy_absolute",
            {"x_min": 0.1, "y_min": 0.5, "x_max": 0.6, "y_max": 1},
            (200, 100),
            [40, 100, 240, 200],
        ),
        (
            "yxyx_0_1000",
            {"x_min": 0.1, "y_min": 0.5, "x_max": 0.6, "y_max": 1},
            None,
            [40, 100, 240, 200],
        ),
    ],
)
def test_each_box_format_maps_its_contract_to_image_pixels(
    box_format, entry, upload_size_wh, expected
) -> None:
    error_status, detections = _detect(
        [{**entry, "label": "cat"}], box_format, upload_size_wh=upload_size_wh
    )

    assert error_status is False
    OBJECT_DETECTION_PREDICTION_KIND.check(detections)
    assert detections.xyxy.tolist() == [expected]
    assert detections.class_id.tolist() == [1]
    assert detections.bboxes_metadata == [{"class": "cat"}]


def test_boxes_are_clamped_reordered_and_rounded() -> None:
    answer = [{"box_2d": [1100, 333, -50, 0], "label": "dog"}]

    _, detections = _detect(answer, "xyxy_0_1000")

    # x: 1000 -> 400, -50 -> 0; y: 333 -> 66.6 -> 67, 0 -> 0; corners swapped.
    assert detections.xyxy.tolist() == [[0, 0, 400, 67]]
    assert detections.xyxy.dtype == torch.float32


def test_detections_carry_frame_class_names_and_row_identity() -> None:
    _, detections = _detect(
        '[{"box_2d": [0, 0, 10, 10], "label": "fox"}]', "xyxy_percent"
    )

    assert detections.class_id.tolist() == [-1]
    assert detections.bboxes_metadata == [{"class": "fox"}]
    assert detections.image_metadata["class_names"] == {0: "dog", 1: "cat"}
    assert detections.image_metadata["prediction_type"] == "object-detection"
    assert frame_of(detections).size_hw == (200, 400)
    record = row_record(detections)
    assert (record.namespace, record.rows) == ("answer-1", 1)


@pytest.mark.parametrize(
    "box_format, entry, expected_confidence",
    [
        ("xyxy_percent", {"box_2d": [0, 0, 1, 1], "confidence": 0.25}, 0.25),
        ("xyxy_percent", {"box_2d": [0, 0, 1, 1], "confidence": 7}, 1.0),
        ("xyxy_percent", {"box_2d": [0, 0, 1, 1], "confidence": "n/a"}, 1.0),
        ("xyxy_percent", {"box_2d": [0, 0, 1, 1]}, 1.0),
        ("xyxy_0_1000", {"box_2d": [0, 0, 1, 1], "confidence": 0.25}, 1.0),
        (
            "named_0_1000",
            {"x_min": 0, "y_min": 0, "x_max": 1, "y_max": 1, "confidence": 0.25},
            1.0,
        ),
    ],
)
def test_confidence_is_read_only_by_formats_that_honour_it(
    box_format, entry, expected_confidence
) -> None:
    _, detections = _detect([entry], box_format)

    assert detections.confidence.tolist() == [expected_confidence]


def test_labels_fall_back_through_aliases_to_unknown() -> None:
    answer = [
        {"box_2d": [0, 0, 1, 1], "class_name": "cat"},
        {"box_2d": [0, 0, 1, 1], "description": "red car"},
        {"box_2d": [0, 0, 1, 1], "label": ""},
    ]

    _, detections = _detect(answer, "xyxy_percent")

    assert [row["class"] for row in detections.bboxes_metadata] == [
        "cat",
        "red car",
        "unknown",
    ]


def test_malformed_entries_are_skipped_when_another_entry_matches() -> None:
    answer = [
        {"box_2d": [0, 0, 10, 10], "label": "dog"},
        {"box_2d": [0, 0, "x", 10], "label": "bad"},
        {"box_2d": [0, 0, float("nan"), 10], "label": "nan"},
        {"box_2d": [True, 0, 1, 1], "label": "bool"},
        {"box_2d": ["5", "5", "10", "10"], "label": "quoted"},
        "not an entry",
    ]

    _, detections = _detect(answer, "xyxy_percent")

    assert [row["class"] for row in detections.bboxes_metadata] == ["dog", "quoted"]


@pytest.mark.parametrize(
    "answer",
    [
        '[{"x_min": 0.1, "y_min": 0.1, "x_max": 0.2, "y_max": 0.2}]',
        '{"objects": []}',
        "[1, 2, 3, 4]",
        "no boxes",
    ],
)
def test_answer_in_another_contract_is_unreadable_not_empty(answer) -> None:
    assert _detect(answer, "xyxy_percent") == (True, None)


@pytest.mark.parametrize("answer", ["[]", '{"detections": []}'])
def test_empty_answer_is_an_empty_prediction(answer) -> None:
    error_status, detections = _detect(answer, "xyxy_0_1000")

    assert error_status is False
    OBJECT_DETECTION_PREDICTION_KIND.check(detections)
    assert detections.xyxy.shape == (0, 4)
    assert row_record(detections).rows == 0


def test_detection_wrapper_and_single_entry_shapes_are_read() -> None:
    wrapped = {"detections": [{"box_2d": [0, 0, 10, 10], "label": "dog"}]}
    single = {"box_2d": [0, 0, 10, 10], "label": "dog"}

    assert _detect(wrapped, "xyxy_percent")[1].xyxy.tolist() == [[0, 0, 40, 20]]
    assert _detect(single, "xyxy_percent")[1].xyxy.tolist() == [[0, 0, 40, 20]]


@pytest.mark.parametrize(
    "box_format, kwargs, message",
    [
        ("xyxy_cm", {}, "Unknown detection box format"),
        ("xyxy_absolute", {}, "upload_size_wh"),
        ("xyxy_absolute", {"upload_size_wh": (0, 10)}, "upload_size_wh"),
        ("xyxy_0_1000", {"classes": ["a", "a"]}, "repeat"),
    ],
)
def test_wrong_detection_calls_raise(box_format, kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        _detect("[]", box_format, **kwargs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_predictions_follow_the_image_device() -> None:
    image = ImageData.from_tensor(_image().tensor_image.cuda())

    _, detections = _detect("[]", "xyxy_0_1000", image=image)
    _, classification = decode_classification(
        '{"class_name": "dog", "confidence": 1}', image=image, classes=CLASSES
    )

    assert detections.xyxy.device.type == "cuda"
    assert classification.confidence.device.type == "cuda"


# --- segmentation ------------------------------------------------------------------


def _segment(answer, **kwargs):
    decoded = decode_instance_segmentations(
        answer,
        image=_image(),
        classes=CLASSES,
        inference_id="seg-1",
        upload_size_wh=kwargs.pop("upload_size_wh", (200, 100)),
        **kwargs,
    )

    return decoded


@pytest.mark.parametrize(
    "polygon",
    [
        [10, 10, 60, 10, 60, 30, 10, 30],
        [[10, 10], [60, 10], [60, 30], [10, 30]],
        [
            {"x": 10, "y": 10},
            {"x": 60, "y": 10},
            {"x": 60, "y": 30},
            {"x": 10, "y": 30},
        ],
    ],
)
def test_polygons_are_scaled_from_upload_and_encoded_as_rle(polygon) -> None:
    answer = {"segmentations": [{"polygon": polygon, "label": "cat"}]}

    error_status, instances = _segment(answer)

    assert error_status is False
    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND.check(instances)
    assert instances.mask.image_size == (200, 400)
    dense = mask_utils.decode(instances.mask.to_coco_rle_masks()[0])
    # Upload 200 x 100 -> image 400 x 200: rectangle x 20..120, y 20..60.
    assert dense.shape == (200, 400)
    assert dense[20:60, 20:120].all() and dense.sum() == pytest.approx(
        100 * 40, rel=0.05
    )
    assert instances.xyxy[0].tolist() == pytest.approx([20, 20, 120, 60], abs=1)
    assert instances.class_id.tolist() == [1]
    assert instances.confidence.tolist() == [1.0]
    assert instances.image_metadata["prediction_type"] == "instance-segmentation"
    assert row_record(instances).namespace == "seg-1"


def test_unusable_polygons_are_skipped() -> None:
    answer = [
        {"polygon": [0, 0, 50, 0, 50, 50, 0, 50], "label": "dog"},
        {"polygon": [0, 0, 10, 10], "label": "two vertices"},
        {"polygon": [0, 0, 10, 10, 20, 20], "label": "collinear"},
        {"polygon": [0, 0, 10, 0, 10], "label": "odd"},
        {"polygon": [300, 300, 400, 300, 400, 400], "label": "outside upload"},
        {"points": [0, 0, 20, 0, 20, 20], "label": "alias"},
    ]

    _, instances = _segment(answer)

    assert [row["class"] for row in instances.bboxes_metadata] == ["dog", "alias"]
    assert len(instances.mask.masks) == 2


def test_oversized_polygon_is_skipped_before_encoding() -> None:
    circle = [
        coordinate
        for angle in np.linspace(0, 2 * math.pi, 40, endpoint=False)
        for coordinate in (50 + 40 * math.cos(angle), 50 + 40 * math.sin(angle))
    ]
    answer = [{"polygon": circle, "label": "big"}, {"polygon": [0, 0, 9, 0, 9, 9]}]

    _, instances = _segment(answer, max_polygon_vertices=39)

    assert [row["class"] for row in instances.bboxes_metadata] == ["unknown"]


@pytest.mark.parametrize(
    "answer", ['[{"polygon": [0, 0, 1, 1]}]', '{"masks": []}', "no polygons"]
)
def test_unreadable_segmentation_answers_set_error_status(answer) -> None:
    assert _segment(answer) == (True, None)


def test_empty_segmentation_answer_is_an_empty_prediction() -> None:
    error_status, instances = _segment('{"segmentations": []}')

    assert error_status is False
    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND.check(instances)
    assert instances.xyxy.shape == (0, 4)


def test_segmentation_needs_the_upload_size() -> None:
    with pytest.raises(ValueError, match="upload_size_wh"):
        _segment("[]", upload_size_wh=None)


# --- dispatch, prompts and upload sizes --------------------------------------------


def test_decode_vlm_output_dispatches_by_task() -> None:
    image = _image()
    kwargs = {"image": image, "classes": CLASSES, "inference_id": "t"}

    single = decode_vlm_output(
        "classification", '{"class_name": "dog", "confidence": 1}', **kwargs
    )
    boxes = decode_vlm_output(
        "object-detection", "[]", box_format="xyxy_0_1000", **kwargs
    )
    masks = decode_vlm_output(
        "instance-segmentation", "[]", upload_size_wh=(10, 10), **kwargs
    )

    assert isinstance(single[1], ClassificationPrediction)
    assert boxes[1].xyxy.shape == (0, 4)
    assert masks[1].mask.masks == []
    assert decode_vlm_output("caption", "A dog.", **kwargs) == (False, None)


@pytest.mark.parametrize(
    "task_type, kwargs, message",
    [
        ("object-detection", {"classes": CLASSES}, "box_format"),
        ("classification", {"classes": None}, "classes"),
    ],
)
def test_decode_vlm_output_rejects_incomplete_calls(task_type, kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        decode_vlm_output(task_type, "[]", image=_image(), inference_id="t", **kwargs)


def test_prompts_name_the_classes_and_upload_size() -> None:
    absolute = build_object_detection_prompt(
        "xyxy_absolute", classes=CLASSES, upload_size_wh=(640, 480)
    )
    normalized = build_object_detection_prompt("named_0_1000", classes=CLASSES)
    segmentation = build_instance_segmentation_prompt(
        classes=CLASSES, upload_size_wh=(640, 480)
    )

    assert "640x480 pixel image" in absolute and absolute.endswith("dog, cat")
    assert '{"label": "<name>", "x_min": <int>' in normalized
    assert "640x480 pixel image" in segmentation


@pytest.mark.parametrize(
    "box_format, kwargs, message",
    [
        ("named_normalized", {}, "no prompt template"),
        ("xyxy_absolute", {}, "upload_size_wh"),
    ],
)
def test_prompts_that_cannot_be_rendered_raise(box_format, kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        build_object_detection_prompt(box_format, classes=CLASSES, **kwargs)


def test_every_format_with_a_template_renders() -> None:
    for name, box_format in DETECTION_BOX_FORMATS.items():
        if box_format.prompt_template is None:
            continue
        prompt = build_object_detection_prompt(
            name, classes=["a"], upload_size_wh=(10, 10)
        )
        assert "{" not in prompt.replace('{"label"', "")


@pytest.mark.parametrize(
    "size_wh, expected",
    [
        ((1000, 500), (1000, 500)),
        ((4096, 1024), (2048, 512)),
        ((1000, 3000), (683, 2048)),
    ],
)
def test_max_edge_upload_size_downscales_the_longest_edge(size_wh, expected) -> None:
    assert max_edge_upload_size(size_wh, max_edge=2048) == expected


@pytest.mark.parametrize(
    "size_wh", [(4000, 3000), (3000, 4000), (2576, 100), (10000, 50)]
)
def test_anthropic_upload_size_is_the_largest_size_claude_reads(size_wh) -> None:
    def fits(width: int, height: int) -> bool:
        tiles_x, tiles_y = math.ceil(width / 28), math.ceil(height / 28)
        return max(tiles_x, tiles_y) * 28 <= 2576 and tiles_x * tiles_y <= 4784

    width, height = anthropic_upload_size(size_wh)
    long_side = max(width, height) + 1
    ratio = max(size_wh) / min(size_wh)
    grown = (
        (long_side, round(long_side / ratio))
        if size_wh[0] >= size_wh[1]
        else (round(long_side / ratio), long_side)
    )

    assert fits(width, height)
    assert not fits(*grown)
    assert anthropic_upload_size((800, 600)) == (800, 600)


@pytest.mark.parametrize(
    "task_type, kind_names",
    [
        ("object-detection", ("object_detection_prediction",)),
        (
            "instance-segmentation",
            (
                "rle_instance_segmentation_prediction",
                "instance_segmentation_prediction",
            ),
        ),
        ("multi-label-classification", ("classification_prediction",)),
        ("caption", ("object_detection_prediction", "classification_prediction")),
    ],
)
def test_vlm_prediction_outputs_type_predictions_by_task(task_type, kind_names) -> None:
    outputs = vlm_prediction_outputs(task_type)

    assert list(outputs) == ["predictions", "error_status", "inference_id"]
    assert outputs["predictions"].kind_names == kind_names
    assert outputs["error_status"].kind_names == ("boolean",)
    assert {output.source for output in outputs.values()} == {"image"}
