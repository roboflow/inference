"""The six task-model blocks called directly on fake models.

The fakes (``tests/v2/support/models``) make outputs a fixed function of each
image's size and brightness, so every expectation below is exact. They test
the blocks' one-model-per-call phases, options, metadata, row identity and
output shapes, not real inference. Compiled workflows are in
``integration/blocks/models/test_task_model_workflows.py``.
"""

from typing import Any, Dict, List

import pytest
import torch
from roboflow_workflows.v2.blocks.models.cache import ModelCache, ModelRequest
from roboflow_workflows.v2.blocks.models.depth_estimation.v1 import DepthEstimationBlock
from roboflow_workflows.v2.blocks.models.easy_ocr.v1 import EasyOCRBlock
from roboflow_workflows.v2.blocks.models.ocr_model.v1 import OCRModelBlock
from roboflow_workflows.v2.blocks.models.roboflow_classification_model.v3 import ClassificationBlock
from roboflow_workflows.v2.blocks.models.roboflow_multi_label_classification_model.v3 import MultiLabelClassificationBlock
from roboflow_workflows.v2.blocks.models.common import (
    ModelOption,
    checked_confidence,
    confidence_option,
    post_process_options,
)
from roboflow_workflows.v2.blocks.models.roboflow_object_detection_model.v3 import ObjectDetectionBlock
from roboflow_workflows.v2.blocks.models.roboflow_instance_segmentation_model.v4 import InstanceSegmentationBlock
from roboflow_workflows.v2.blocks.models.roboflow_keypoint_detection_model.v3 import KeypointDetectionBlock
from roboflow_workflows.v2.blocks.models.roboflow_semantic_segmentation_model.v2 import SemanticSegmentationBlock
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.frames import frame_of
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.masks import mask_provider_of
from roboflow_workflows.v2.data.masks.grid import SegmentationView
from roboflow_workflows.v2.data.predictions import (
    CLASSIFICATION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    NATIVE_SEMANTIC_SEGMENTATION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
)
from roboflow_workflows.v2.data.row_identity import (
    ROW_RECORD_KEY,
    RowRef,
    classification_parent,
    row_record,
)
from roboflow_workflows.v2.engine.phases import PhaseFailure
from roboflow_workflows.v2.engine.reactions.testing import block_call

from tests.v2.support.models.fake_task_models import (
    CLASS_NAMES,
    Classifier,
    ForwardThresholdKeypointModel,
    IgnoringNmsDetector,
    InstanceModel,
    KeypointModel,
    MultiLabelClassifier,
    NmsDetector,
    NmsFreeDetector,
    SemanticModel,
    box_masks,
)
from tests.v2.support.models.grid_segmentation import (
    GridInstanceModel,
    GridInstanceSegmentationBlock,
)

# Mean brightness -> rows: 20 -> 0, 100 -> 2, 200 -> 3 (fake_task_models).
DARK, MID, BRIGHT = 20, 100, 200


def _image(value: int, image_id: str, *, height: int = 20, width: int = 30):
    pixels = torch.full((3, height, width), value, dtype=torch.uint8)
    image = ImageData.from_tensor(pixels, image_id=image_id)

    return image


def _ragged_batch() -> Batch:
    batch = Batch(
        [_image(BRIGHT, "a"), _image(DARK, "b", height=12), _image(MID, "c", width=16)]
    )

    return batch


def _predictions(results: List[Dict[str, Any]]) -> List[Any]:
    predictions = [result["predictions"] for result in results]

    return predictions


def _segment(block: Any, *, wanted=None, **params: Any) -> List[Dict[str, Any]]:
    # The block asks wants() for its mask outputs: run it as one engine call.
    with block_call(block, wanted=wanted):
        results = block.run(**params)

    return results


@pytest.fixture
def materializations(monkeypatch) -> List[int]:
    """Every full-resolution upscale of any SegmentationView in the test."""
    calls: List[int] = []
    materialize = SegmentationView._materialize

    def counted(view):
        calls.append(len(view))
        return materialize(view)

    monkeypatch.setattr(SegmentationView, "_materialize", counted)

    return calls


# --- option binding -----------------------------------------------------------


def test_options_bind_only_to_named_post_process_keywords() -> None:
    def post_process(raw, metadata, confidence, iou_threshold=0.5, **kwargs):
        return None

    options = (
        ModelOption("confidence", "confidence", "best"),
        ModelOption("iou_threshold", "iou_threshold", None, 0.3),
        ModelOption("max_detections", "max_detections", None, 300),
    )

    bound = post_process_options(
        post_process, options, block="t", request=ModelRequest("m")
    )

    # max_detections is only collected by **kwargs: not named, not passed.
    assert bound == {"confidence": "best", "iou_threshold": 0.3}


def test_a_set_option_the_model_does_not_name_is_an_error() -> None:
    def post_process(raw, metadata, confidence, **kwargs):
        return None

    options = (ModelOption("iou_threshold", "iou_threshold", 0.5, 0.3),)

    with pytest.raises(ValueError, match="does not read 'iou_threshold'"):
        post_process_options(
            post_process, options, block="t", request=ModelRequest("m")
        )


def test_a_renamed_option_reports_the_step_parameter() -> None:
    def post_process(raw, metadata, confidence):
        return None

    options = (ModelOption("keypoint_confidence", "key_points_threshold", 0.2),)

    with pytest.raises(
        ValueError, match="'keypoint_confidence'.*'key_points_threshold'"
    ):
        post_process_options(
            post_process, options, block="t", request=ModelRequest("m")
        )


@pytest.mark.parametrize("mode", ["best", "default"])
def test_a_confidence_mode_is_honoured_by_a_model_without_a_confidence_keyword(
    mode,
) -> None:
    def post_process(raw, metadata, **kwargs):
        return None

    bound = post_process_options(
        post_process, (confidence_option(mode),), block="t", request=ModelRequest("m")
    )

    # The model applies its own threshold, which is what both modes ask for.
    assert bound == {}


def test_a_numeric_confidence_for_a_model_without_the_keyword_is_an_error() -> None:
    def post_process(raw, metadata, **kwargs):
        return None

    with pytest.raises(ValueError, match="does not read 'confidence'"):
        post_process_options(
            post_process,
            (confidence_option(0.4),),
            block="t",
            request=ModelRequest("m"),
        )


def test_an_unread_option_error_names_the_backend_value_and_the_fix() -> None:
    model = NmsFreeDetector()
    options = (
        confidence_option(0.4),
        ModelOption("iou_threshold", "iou_threshold", 0.5, 0.3),
    )

    with pytest.raises(ValueError) as error:
        post_process_options(
            model.post_process, options, block="t", request=ModelRequest("rfdetr")
        )

    message = str(error.value)
    assert "model 'rfdetr' (backend NmsFreeDetector)" in message
    assert "iou_threshold=0.5 would have no effect" in message
    assert "Options this model reads: ['confidence']" in message
    assert "Remove 'iou_threshold' or set it to null" in message
    assert "before any model runs" in message


def test_a_named_but_ignored_keyword_is_not_passed_and_cannot_be_set() -> None:
    model = IgnoringNmsDetector()
    unset = (ModelOption("class_agnostic_nms", "class_agnostic_nms", None, False),)
    explicit = (ModelOption("class_agnostic_nms", "class_agnostic_nms", True, False),)

    bound = post_process_options(
        model.post_process, unset, block="t", request=ModelRequest("nas-trt")
    )
    with pytest.raises(ValueError, match="names 'class_agnostic_nms' but never uses"):
        post_process_options(
            model.post_process, explicit, block="t", request=ModelRequest("nas-trt")
        )

    assert bound == {}


@pytest.mark.parametrize(
    "value, expected", [(0.25, 0.25), (1, 1.0), (0, 0.0), ("best", "best")]
)
def test_confidence_is_a_unit_number_or_an_allowed_mode(value, expected) -> None:
    assert checked_confidence(value, modes=("best",), block="t") == expected


@pytest.mark.parametrize("value", [1.5, -0.1, True, "default", "BEST"])
def test_confidence_outside_unit_interval_or_unknown_mode_is_rejected(value) -> None:
    with pytest.raises(ValueError, match="confidence"):
        checked_confidence(value, modes=("best",), block="t")


# --- object detection ---------------------------------------------------------


def test_unset_nms_options_pass_v1_defaults_to_a_model_with_nms() -> None:
    model = NmsDetector()
    block = ObjectDetectionBlock(models={"od": model})

    block.run(image=Batch([_image(BRIGHT, "a")]), model_id="od", confidence="best")

    assert model.received == [
        {
            "confidence": "best",
            "iou_threshold": 0.3,
            "max_detections": 300,
            "class_agnostic_nms": False,
        }
    ]


def test_set_nms_options_reach_the_model() -> None:
    model = NmsDetector()
    block = ObjectDetectionBlock(models={"od": model})

    (result,) = block.run(
        image=Batch([_image(BRIGHT, "a")]),
        model_id="od",
        confidence=0.6,
        iou_threshold=0.7,
        max_detections=1,
        class_agnostic_nms=True,
    )

    assert model.received[0] == {
        "confidence": 0.6,
        "iou_threshold": 0.7,
        "max_detections": 1,
        "class_agnostic_nms": True,
    }
    assert len(result["predictions"]) == 1


def test_an_nms_free_model_gets_only_confidence() -> None:
    model = NmsFreeDetector()
    block = ObjectDetectionBlock(models={"rfdetr": model})

    block.run(image=Batch([_image(BRIGHT, "a")]), model_id="rfdetr", confidence=0.4)

    assert model.received == [{"confidence": 0.4}]


def test_an_iou_threshold_for_an_nms_free_model_fails_before_any_model_call() -> None:
    model = NmsFreeDetector()
    block = ObjectDetectionBlock(models={"rfdetr": model})

    with pytest.raises(PhaseFailure, match="does not read 'iou_threshold'"):
        block.run(
            image=Batch([_image(BRIGHT, "a")]), model_id="rfdetr", iou_threshold=0.5
        )

    assert model.calls == []


def test_ragged_and_empty_results_keep_row_order_and_get_their_own_identity() -> None:
    block = ObjectDetectionBlock(models={"od": NmsDetector()})

    results = block.run(image=_ragged_batch(), model_id="od", confidence=0.0)

    predictions = _predictions(results)
    assert [len(p) for p in predictions] == [3, 0, 2]
    assert [frame_of(p).image_id for p in predictions] == ["a", "b", "c"]
    records = [row_record(p) for p in predictions]
    assert [record.rows for record in records] == [3, 0, 2]
    assert all(record.ids is None for record in records)
    assert [record.namespace for record in records] == [
        result["inference_id"] for result in results
    ]
    assert len({record.namespace for record in records}) == 3
    assert [result["model_id"] for result in results] == ["od"] * 3
    for prediction in predictions:
        OBJECT_DETECTION_PREDICTION_KIND.validate(prediction)


def test_class_filter_keeps_named_classes_on_their_rows() -> None:
    block = ObjectDetectionBlock(models={"od": NmsDetector()})

    (result,) = block.run(
        image=Batch([_image(BRIGHT, "a")]),
        model_id="od",
        confidence=0.0,
        class_filter=["person", "dog"],
    )

    predictions = result["predictions"]
    assert predictions.class_id.tolist() == [0, 2]
    # Rows 0 and 2 of the fake: box [i, i, w - i, h - i].
    assert predictions.xyxy.tolist() == [[0, 0, 30, 20], [2, 2, 28, 18]]
    assert row_record(predictions).rows == 2


def test_a_class_the_model_does_not_have_fails_before_any_model_call() -> None:
    model = NmsDetector()
    block = ObjectDetectionBlock(models={"od": model})

    with pytest.raises(PhaseFailure, match=r"class_filter names \['cat'\]"):
        block.run(
            image=Batch([_image(BRIGHT, "a")]), model_id="od", class_filter=["cat"]
        )

    assert model.calls == []


def test_an_empty_selected_class_filter_is_rejected() -> None:
    block = ObjectDetectionBlock(models={"od": NmsDetector()})

    with pytest.raises(
        PhaseFailure, match="omit class_filter or set it to null to keep every class"
    ):
        block.run(image=Batch([_image(BRIGHT, "a")]), model_id="od", class_filter=[])


def test_object_detection_defaults_to_the_best_confidence_mode() -> None:
    detector, segmenter = NmsDetector(), InstanceModel()

    (detected,) = ObjectDetectionBlock(models={"od": detector}).run(
        image=Batch([_image(BRIGHT, "a")]), model_id="od"
    )
    (segmented,) = _segment(
        InstanceSegmentationBlock(models={"seg": segmenter}),
        image=Batch([_image(BRIGHT, "a")]),
        model_id="seg",
    )

    assert detector.received[0]["confidence"] == "best"
    assert segmenter.received[0]["confidence"] == "best"
    # The fakes resolve "best" to 0.5; the bright image's 3 rows score 0.9..0.7.
    assert detected["predictions"].confidence.tolist() == pytest.approx([0.9, 0.8, 0.7])
    assert torch.equal(detected["predictions"].xyxy, segmented["predictions"].xyxy)


def test_a_misspelled_class_gets_a_suggestion_and_the_way_to_keep_all() -> None:
    block = ObjectDetectionBlock(models={"od": NmsDetector()})

    with pytest.raises(PhaseFailure) as error:
        block.run(
            image=Batch([_image(BRIGHT, "a")]), model_id="od", class_filter=["persn"]
        )

    message = str(error.value)
    assert "model 'od' does not have" in message
    assert "its classes are ['person', 'car', 'dog']" in message
    assert "'persn' -> 'person'" in message
    assert "omit class_filter (or set it to null)" in message


def test_one_class_filter_applies_to_every_image_in_one_forward() -> None:
    model = NmsDetector()
    block = ObjectDetectionBlock(models={"od": model})
    images = Batch([_image(BRIGHT, str(row)) for row in range(4)])

    results = block.run(
        image=images, model_id="od", confidence=0.0, class_filter=["dog", "car"]
    )

    assert model.calls == ["pre_process", "forward", "post_process"]
    assert model.batch_sizes == [4]
    class_ids = [result["predictions"].class_id.tolist() for result in results]
    assert class_ids == [[1, 2]] * 4
    assert [frame_of(p).image_id for p in _predictions(results)] == list("0123")


@pytest.mark.parametrize(
    "per_image",
    [
        {"model_id": Batch(["od", "od"])},
        {"model_package_id": Batch([None, None])},
        {"class_filter": Batch([["person"], ["car"]])},
    ],
    ids=["model_id", "model_package_id", "class_filter"],
)
def test_a_per_image_batch_of_configuration_fails_before_any_model_call(
    per_image,
) -> None:
    model = NmsDetector()
    block = ObjectDetectionBlock(models={"od": model})
    (name,) = per_image

    with pytest.raises(
        PhaseFailure, match=f"{name} is one value for every image of a call"
    ):
        block.run(
            image=Batch([_image(BRIGHT, "a"), _image(BRIGHT, "b")]),
            **{"model_id": "od", **per_image},
        )

    assert model.calls == []


def test_the_model_and_filter_may_change_between_calls() -> None:
    first, second = NmsDetector(), NmsDetector()
    block = ObjectDetectionBlock(models={"od-1": first, "od-2": second})
    images = Batch([_image(BRIGHT, "a"), _image(MID, "b")])

    before = block.run(
        image=images, model_id="od-1", confidence=0.0, class_filter=["person"]
    )
    after = block.run(image=images, model_id="od-2", confidence=0.0)

    assert first.batch_sizes == second.batch_sizes == [2]
    assert [result["model_id"] for result in before + after] == ["od-1"] * 2 + [
        "od-2"
    ] * 2
    assert [len(p) for p in _predictions(before)] == [1, 1]
    assert [len(p) for p in _predictions(after)] == [3, 2]


# Block, run() selection, and network_input arguments after the image.
PHASED_BLOCKS = {
    "object_detection": (
        ObjectDetectionBlock,
        {"model_id": "m"},
        ("m", None, "best", None, None, None, None),
    ),
    "instance_segmentation": (
        InstanceSegmentationBlock,
        {"model_id": "m"},
        ("m", None, "best", None, None, None, None),
    ),
    "keypoints": (
        KeypointDetectionBlock,
        {"model_id": "m"},
        ("m", None, "default", None, None, None, None, None),
    ),
    "semantic_segmentation": (
        SemanticSegmentationBlock,
        {"model_id": "m"},
        ("m", None, "best"),
    ),
    "classification": (ClassificationBlock, {"model_id": "m"}, ("m", None)),
    "multi_label": (
        MultiLabelClassificationBlock,
        {"model_id": "m"},
        ("m", None, "best"),
    ),
    "doctr": (OCRModelBlock, {"model_id": "m"}, ("m", None)),
    "easy_ocr": (EasyOCRBlock, {"language": "English"}, ("English", None)),
    "depth": (DepthEstimationBlock, {"model_id": "m"}, ("m", None)),
}


@pytest.mark.parametrize("name", PHASED_BLOCKS)
def test_an_empty_batch_loads_no_model_and_runs_no_phase_work(name) -> None:
    block_type, selection, phase_arguments = PHASED_BLOCKS[name]
    loaded = []
    block = block_type(models=ModelCache(lambda request: loaded.append(request)))
    empty = Batch([])

    with block_call(block):
        results = block.run(image=empty, **selection)
        # The same phases called one by one, as a pipelined run does.
        network_input = block.network_input(empty, *phase_arguments)
        raw_output = block.raw_output(network_input)
        phased = block.result(raw_output, network_input, empty)

    assert results == phased == []
    assert network_input is None
    assert loaded == []


def test_overlapping_phased_calls_keep_their_own_model_and_options() -> None:
    first, second = NmsDetector(), NmsDetector()
    block = ObjectDetectionBlock(models={"od-1": first, "od-2": second})
    early = Batch([_image(BRIGHT, "early")])
    late = Batch([_image(BRIGHT, "late-0"), _image(MID, "late-1")])

    # Both calls are in flight before either finishes; nothing is on the block.
    early_input = block.network_input(
        early, "od-1", None, 0.0, None, None, None, ["dog"]
    )
    late_input = block.network_input(late, "od-2", None, 0.0, None, 1, None, None)
    late_raw = block.raw_output(late_input)
    early_raw = block.raw_output(early_input)
    late_results = block.result(late_raw, late_input, late)
    early_results = block.result(early_raw, early_input, early)

    assert (early_input.model, late_input.model) == (first, second)
    assert [r["model_id"] for r in early_results] == ["od-1"]
    assert [r["model_id"] for r in late_results] == ["od-2", "od-2"]
    assert early_results[0]["predictions"].class_id.tolist() == [2]
    assert [len(p) for p in _predictions(late_results)] == [1, 1]
    assert second.received[0]["max_detections"] == 1
    assert first.received[0]["max_detections"] == 300
    assert [frame_of(p).image_id for p in _predictions(late_results)] == [
        "late-0",
        "late-1",
    ]


def test_overlapping_classifier_calls_keep_their_own_model_without_metadata() -> None:
    first, second = Classifier(), Classifier()
    block = ClassificationBlock(models={"cls-1": first, "cls-2": second})
    early = Batch([_image(BRIGHT, "early")])
    late = Batch([_image(DARK, "late-0"), _image(MID, "late-1")])

    early_input = block.network_input(early, "cls-1", None)
    late_input = block.network_input(late, "cls-2", None)
    late_results = block.result(block.raw_output(late_input), late_input, late)
    early_results = block.result(block.raw_output(early_input), early_input, early)

    # Classifiers' pre_process returns no metadata; none is invented.
    assert early_input.metadata is None and late_input.metadata is None
    assert (first.batch_sizes, second.batch_sizes) == ([1], [2])
    assert [r["model_id"] for r in early_results + late_results] == [
        "cls-1",
        "cls-2",
        "cls-2",
    ]
    assert [
        frame_of(r["predictions"]).image_id for r in early_results + late_results
    ] == ["early", "late-0", "late-1"]


def test_an_explicit_value_for_an_ignored_keyword_runs_nothing() -> None:
    model = IgnoringNmsDetector()
    block = ObjectDetectionBlock(models={"nas-trt": model})

    with pytest.raises(PhaseFailure, match="never uses it"):
        block.run(
            image=Batch([_image(BRIGHT, "a")]),
            model_id="nas-trt",
            class_agnostic_nms=True,
        )

    assert model.calls == []


# --- classification -----------------------------------------------------------


def test_single_label_batch_is_split_into_one_image_predictions() -> None:
    model = Classifier()
    block = ClassificationBlock(models={"cls": model})

    results = block.run(image=_ragged_batch(), model_id="cls")

    predictions = _predictions(results)
    assert model.batch_sizes == [3]
    assert model.received == [{}]  # no confidence: top-1 does not depend on it
    assert [p.class_id.tolist() for p in predictions] == [[2], [0], [1]]
    assert all(p.confidence.shape == (1, len(CLASS_NAMES)) for p in predictions)
    for prediction, image_id in zip(predictions, ["a", "b", "c"]):
        metadata = prediction.images_metadata[0]
        assert metadata["class_names"] == dict(enumerate(CLASS_NAMES))
        assert metadata["prediction_type"] == "classification"
        assert frame_of(prediction).image_id == image_id
        assert ROW_RECORD_KEY not in metadata
        assert classification_parent(prediction) is None
        CLASSIFICATION_PREDICTION_KIND.validate(prediction)


def test_single_label_rejects_a_confidence_option() -> None:
    with pytest.raises(TypeError):
        ClassificationBlock(models={"cls": Classifier()}).run(
            image=Batch([_image(MID, "a")]), model_id="cls", confidence=0.5
        )


def test_a_crop_classification_carries_its_parent_row() -> None:
    parent = RowRef("detections-1", 4)
    crop = _image(MID, "crop").with_parent_row(parent)
    block = ClassificationBlock(models={"cls": Classifier()})

    (result,) = block.run(image=Batch([crop]), model_id="cls")

    assert classification_parent(result["predictions"]) == parent


@pytest.mark.parametrize(
    "confidence, expected_ids", [("best", [0, 2]), (0.1, [0, 1, 2]), (0.95, [])]
)
def test_multi_label_threshold_reaches_the_model(confidence, expected_ids) -> None:
    model = MultiLabelClassifier()
    block = MultiLabelClassificationBlock(models={"ml": model})

    (result,) = block.run(
        image=Batch([_image(BRIGHT, "a")]), model_id="ml", confidence=confidence
    )

    prediction = result["predictions"]
    assert model.received == [{"confidence": confidence}]
    assert prediction.class_ids.tolist() == expected_ids
    assert prediction.image_metadata["class_names"] == dict(enumerate(CLASS_NAMES))
    assert frame_of(prediction).image_id == "a"
    CLASSIFICATION_PREDICTION_KIND.validate(prediction)


# --- keypoints ----------------------------------------------------------------


def test_keypoint_threshold_is_passed_under_the_model_keyword() -> None:
    model = KeypointModel()
    block = KeypointDetectionBlock(models={"kp": model})

    block.run(image=Batch([_image(BRIGHT, "a")]), model_id="kp")
    block.run(
        image=Batch([_image(BRIGHT, "a")]), model_id="kp", keypoint_confidence=0.3
    )

    assert [r["key_points_threshold"] for r in model.received] == [0.0, 0.3]
    assert [r["confidence"] for r in model.received] == ["default", "default"]


def test_keypoint_class_filter_selects_skeletons_and_boxes_together() -> None:
    block = KeypointDetectionBlock(models={"kp": KeypointModel()})

    (result,) = block.run(
        image=Batch([_image(BRIGHT, "a")]),
        model_id="kp",
        confidence=0.0,
        class_filter=["car"],
    )

    key_points, boxes = result["predictions"]
    assert key_points.class_id.tolist() == boxes.class_id.tolist() == [1]
    assert key_points.xy.tolist() == [[[1, 1], [29, 19]]]
    assert boxes.xyxy.tolist() == [[1, 1, 29, 19]]
    record = row_record(result["predictions"])
    assert (record.rows, record.namespace) == (1, result["inference_id"])
    assert ROW_RECORD_KEY in boxes.image_metadata
    assert key_points.image_metadata["key_points_classes"][1] == ["nose", "tail"]
    KEYPOINT_DETECTION_PREDICTION_KIND.validate(result["predictions"])


def test_keypoints_without_boxes_carry_identity_themselves() -> None:
    block = KeypointDetectionBlock(models={"kp": KeypointModel(with_boxes=False)})

    results = block.run(image=_ragged_batch(), model_id="kp", confidence=0.0)

    for result, rows in zip(results, [3, 0, 2]):
        key_points, boxes = result["predictions"]
        assert boxes is None
        assert row_record(result["predictions"]).rows == rows == len(key_points)
        KEYPOINT_DETECTION_PREDICTION_KIND.validate(result["predictions"])


def test_a_forward_thresholded_keypoint_model_keeps_its_own_thresholds() -> None:
    model = ForwardThresholdKeypointModel()
    block = KeypointDetectionBlock(models={"two-stage": model})

    (result,) = block.run(image=Batch([_image(BRIGHT, "a")]), model_id="two-stage")

    # Neither threshold is a post_process keyword: nothing is passed.
    assert model.received == [{}]
    assert len(result["predictions"][0]) == 3


@pytest.mark.parametrize(
    "option, value", [("confidence", 0.5), ("keypoint_confidence", 0.3)]
)
def test_a_numeric_threshold_for_a_forward_thresholded_model_runs_nothing(
    option, value
) -> None:
    model = ForwardThresholdKeypointModel()
    block = KeypointDetectionBlock(models={"two-stage": model})

    with pytest.raises(PhaseFailure, match=f"does not read '{option}'"):
        block.run(
            image=Batch([_image(BRIGHT, "a")]), model_id="two-stage", **{option: value}
        )

    assert model.calls == []


# --- instance segmentation ----------------------------------------------------


def test_instance_masks_are_requested_dense_and_viewed_without_a_copy() -> None:
    model = InstanceModel()
    block = InstanceSegmentationBlock(models={"seg": model})

    results = _segment(block, image=_ragged_batch(), model_id="seg", confidence=0.0)

    assert model.received[0]["mask_format"] == "dense"
    assert model.received[0]["confidence"] == 0.0
    for result, rows in zip(results, [3, 0, 2]):
        predictions, view = result["predictions"], result["segmentation"]
        INSTANCE_SEGMENTATION_PREDICTION_KIND.validate(predictions)
        assert len(predictions) == len(view) == rows
        assert view.is_dense_fallback
        assert view.full_res() is predictions
        assert view.materialization_count == 0
        # Same rows, same row record: the view shares the metadata.
        assert view.detections.image_metadata is predictions.image_metadata
        assert row_record(view.detections).namespace == result["inference_id"]


def test_instance_class_filter_keeps_masks_aligned_with_boxes() -> None:
    block = InstanceSegmentationBlock(models={"seg": InstanceModel()})

    (result,) = _segment(
        block,
        image=Batch([_image(BRIGHT, "a")]),
        model_id="seg",
        confidence=0.0,
        class_filter=["car", "dog"],
    )

    predictions = result["predictions"]
    assert predictions.class_id.tolist() == [1, 2]
    assert torch.equal(predictions.mask, box_masks(predictions.xyxy, size_hw=(20, 30)))
    from_view = mask_provider_of(result["segmentation"]).binary(fidelity="reference")
    from_predictions = mask_provider_of(predictions).binary(fidelity="reference")
    assert from_view.fidelity == from_predictions.fidelity == "reference"
    assert torch.equal(from_view.predictions.mask, predictions.mask)
    assert torch.equal(from_predictions.predictions.mask, predictions.mask)


@pytest.mark.parametrize(
    "block_class, model_class",
    [
        (InstanceSegmentationBlock, InstanceModel),
        (GridInstanceSegmentationBlock, GridInstanceModel),
    ],
)
def test_filtered_instance_outputs_carry_merged_metadata_and_fresh_identity(
    block_class, model_class
) -> None:
    block = block_class(models={"seg": model_class()})

    (result,) = _segment(
        block,
        image=Batch([_image(BRIGHT, "a")]),
        model_id="seg",
        confidence=0.0,
        class_filter=["car", "dog"],
    )

    for detections in (result["segmentation"].detections, result["predictions"]):
        metadata = detections.image_metadata
        assert "served_by" in metadata  # the model's own metadata is kept
        assert metadata["class_names"] == dict(enumerate(CLASS_NAMES))
        assert metadata["prediction_type"] == "instance-segmentation"
        assert frame_of(detections).image_id == "a"
        record = row_record(detections)
        assert (record.namespace, record.rows, record.ids) == (
            result["inference_id"],
            2,
            None,
        )


def test_a_model_returning_rle_masks_fails_the_step() -> None:
    block = InstanceSegmentationBlock(models={"seg": InstanceModel(rle_only=True)})

    with pytest.raises(PhaseFailure, match="dense"):
        _segment(block, image=Batch([_image(BRIGHT, "a")]), model_id="seg")


def test_the_reference_leaves_out_predictions_nobody_wants() -> None:
    block = InstanceSegmentationBlock(models={"seg": InstanceModel()})

    with block_call(block, wanted=("segmentation", "model_id")) as call:
        (result,) = block.run(
            image=Batch([_image(BRIGHT, "a")]), model_id="seg", confidence=0.0
        )

    assert {"segmentation", "predictions"} <= call.queried
    assert "predictions" not in result
    # The reference decoded dense masks anyway: its view is a dense fallback.
    assert result["segmentation"].is_dense_fallback
    assert row_record(result["segmentation"].detections).rows == 3


def test_a_grid_producer_builds_no_full_resolution_mask_for_box_readers(
    materializations,
) -> None:
    model = GridInstanceModel()
    block = GridInstanceSegmentationBlock(models={"seg": model})

    results = _segment(
        block,
        wanted=("segmentation", "model_id", "inference_id"),
        image=_ragged_batch(),
        model_id="seg",
        confidence=0.0,
    )

    assert block.asked == {"segmentation": True, "predictions": False}
    assert model.received[0] == {
        "confidence": 0.0,
        "iou_threshold": 0.3,
        "max_detections": 300,
        "class_agnostic_nms": False,
    }
    for result, rows in zip(results, [3, 0, 2]):
        view = result["segmentation"]
        assert "predictions" not in result
        assert not view.is_dense_fallback
        assert len(view) == rows == row_record(view.detections).rows
        assert view.detections.xyxy.shape == (rows, 4)
    assert materializations == []


def test_a_grid_producer_serves_both_outputs_from_one_view(materializations) -> None:
    block = GridInstanceSegmentationBlock(models={"seg": GridInstanceModel()})

    (result,) = _segment(
        block,
        image=Batch([_image(BRIGHT, "a")]),
        model_id="seg",
        confidence=0.0,
        class_filter=["car", "dog"],
    )

    view, predictions = result["segmentation"], result["predictions"]
    assert block.asked == {"segmentation": True, "predictions": True}
    assert view.full_res() is predictions
    assert materializations == [2]
    assert view.detections.class_id.tolist() == predictions.class_id.tolist() == [1, 2]
    assert torch.equal(view.detections.xyxy, predictions.xyxy)
    # One row record on both: selections downstream keep them aligned.
    assert row_record(view.detections) == row_record(predictions)
    assert row_record(predictions).namespace == result["inference_id"]
    INSTANCE_SEGMENTATION_PREDICTION_KIND.validate(predictions)
    assert predictions.mask.shape == (2, 20, 30)


# --- semantic segmentation ----------------------------------------------------


def test_semantic_result_is_a_class_grid_without_rows() -> None:
    model = SemanticModel()
    block = SemanticSegmentationBlock(models={"sem": model})

    results = block.run(image=_ragged_batch(), model_id="sem", confidence="default")

    assert model.received == [{"confidence": "default"}]
    maps = [result["predictions"].segmentation_map for result in results]
    assert [tuple(m.shape) for m in maps] == [(20, 30), (12, 30), (20, 16)]
    # Bright: right half class 2; dark: below the 0.35 default, all background.
    assert maps[0][:, 15:].unique().tolist() == [2]
    assert maps[1].unique().tolist() == [0]
    assert maps[2][:, 8:].unique().tolist() == [1]
    for result, image_id in zip(results, ["a", "b", "c"]):
        prediction = result["predictions"]
        NATIVE_SEMANTIC_SEGMENTATION_KIND.validate(prediction)
        assert ROW_RECORD_KEY not in prediction.image_metadata
        assert prediction.image_metadata["class_names"] == {
            0: "background",
            1: "road",
            2: "car",
        }
        assert frame_of(prediction).image_id == image_id


# --- configuration is checked before any model loads (r2: C1, C3) --------------


class _RefusingLoader:
    """A loader that records and refuses every request: nothing may load."""

    def __init__(self):
        self.requests: List[ModelRequest] = []

    def __call__(self, request: ModelRequest) -> Any:
        self.requests.append(request)
        raise AssertionError(f"model {request} was resolved")


def _refusing(block_type: type) -> tuple:
    loader = _RefusingLoader()
    block = block_type(models=ModelCache(loader))

    return block, loader


@pytest.mark.parametrize(
    "block_type, field, values",
    [
        (ObjectDetectionBlock, "confidence", [0.5, 0.5]),
        (ObjectDetectionBlock, "iou_threshold", [0.5, 0.5]),
        (ObjectDetectionBlock, "max_detections", [10, 10]),
        (ObjectDetectionBlock, "class_agnostic_nms", [True, True]),
        (InstanceSegmentationBlock, "iou_threshold", [0.5, 0.5]),
        (KeypointDetectionBlock, "keypoint_confidence", [0.2, 0.2]),
        (SemanticSegmentationBlock, "confidence", [0.5, 0.5]),
        (MultiLabelClassificationBlock, "confidence", [0.5, 0.5]),
    ],
)
def test_a_batch_of_equal_option_values_fails_before_any_load(
    block_type, field, values
) -> None:
    block, loader = _refusing(block_type)

    with pytest.raises(PhaseFailure, match=f"{field} is one value for every image"):
        block.run(
            image=Batch([_image(BRIGHT, "a"), _image(MID, "b")]),
            model_id="m",
            **{field: Batch(values)},
        )

    assert loader.requests == []


def test_a_batch_language_fails_before_any_load() -> None:
    block, loader = _refusing(EasyOCRBlock)

    with pytest.raises(PhaseFailure, match="language is one value for every image"):
        block.run(
            image=Batch([_image(BRIGHT, "a"), _image(MID, "b")]),
            language=Batch(["English", "English"]),
        )

    assert loader.requests == []


MALFORMED_CLASS_FILTERS = {
    "batch": (Batch([["person"], ["person"]]), "class_filter is one value"),
    "string": ("person", "class_filter must be a list of class names"),
    "number": ([1], "class_filter holds class names, got 1"),
    "none-item": (["person", None], "class_filter holds class names, got None"),
    "empty": ([], "omit class_filter or set it to null"),
}


@pytest.mark.parametrize("name", MALFORMED_CLASS_FILTERS)
@pytest.mark.parametrize(
    "block_type",
    [ObjectDetectionBlock, InstanceSegmentationBlock, KeypointDetectionBlock],
)
@pytest.mark.parametrize("images", [2, 0], ids=["images", "empty-batch"])
def test_a_malformed_class_filter_fails_before_any_load(
    block_type, name, images
) -> None:
    class_filter, message = MALFORMED_CLASS_FILTERS[name]
    block, loader = _refusing(block_type)
    image = Batch([_image(BRIGHT, "a"), _image(MID, "b")][:images])

    with pytest.raises(PhaseFailure, match=message):
        block.run(image=image, model_id="m", class_filter=class_filter)

    assert loader.requests == []


def test_an_unknown_class_name_is_still_checked_against_the_loaded_model() -> None:
    model = NmsDetector()
    block = ObjectDetectionBlock(models={"od": model})

    with pytest.raises(PhaseFailure, match=r"class_filter names \['cat'\]"):
        block.run(
            image=Batch([_image(BRIGHT, "a")]), model_id="od", class_filter=["cat"]
        )

    assert model.calls == []

