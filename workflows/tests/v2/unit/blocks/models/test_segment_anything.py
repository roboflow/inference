"""SAM2 and SAM3 blocks called directly on fake models.

The fakes (``tests/v2/support/models/fake_sam_models.py``) return masks that are
a fixed function of each prompt, so every expected mask, box and score below
is written out by hand. They test the blocks' prompts, lineage, thresholds,
NMS, class names and embedding reuse, not real inference. Compiled workflows
are in ``integration/blocks/models/test_segment_anything_workflows.py``.
"""

import copy
import dataclasses
import weakref
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pydantic
import pytest
import torch
from roboflow_workflows.execution_engine.entities.base import VideoMetadata
from roboflow_workflows.v2.blocks.models.segment_anything import (
    SAM2_DEFAULT_MODEL_ID,
    SAM3_DEFAULT_MODEL_ID,
    SegmentAnything2Block,
    SegmentAnything3Block,
    _mask_boxes,
    _mask_nms,
    _most_confident,
    _Row,
)
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.frames import FrameRef, frame_of
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.masks.grid import SegmentationView
from roboflow_workflows.v2.data.prediction_ops import CLASS_NAMES_KEY
from roboflow_workflows.v2.data.row_identity import row_record, with_row_identity
from roboflow_workflows.v2.engine.reactions.testing import block_call

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections
from inference_models.models.sam2.entities import SAM2Prediction
from tests.v2.support.models.fake_sam_models import FakeSam2, FakeSam3, box_mask

SIZE_HW = (20, 30)
PROMPT_NAMES = {0: "person", 1: "car", 2: "dog"}
# Prompt rows on a 20x30 image. FakeSam2 makes each box's best mask the box
# itself, scored 0.9, 0.7 and 0.5 by row.
PROMPT_XYXY = [[2, 3, 10, 8], [12, 4, 20, 15], [5, 5, 25, 18]]
PROMPT_CLASSES = [1, 0, 2]
# Unprompted, FakeSam2's best mask is the centre rectangle of a 20x30 image:
# x 30 // 4 .. 3 * 30 // 4 - 1, y 20 // 4 .. 3 * 20 // 4 - 1.
CENTRE_XYXY = [7, 5, 21, 14]
# Expected output extents cover every pixel of the inclusive fake masks.
PROMPT_EXTENTS = [[2, 3, 11, 9], [12, 4, 21, 16], [5, 5, 26, 19]]


def _image(image_id: str, *, size_hw=SIZE_HW, channels: int = 3) -> ImageData:
    pixels = torch.full((channels, *size_hw), 100, dtype=torch.uint8)
    image = ImageData.from_tensor(pixels, image_id=image_id)

    return image


def _prompt(
    image: ImageData,
    xyxy: Optional[List[List[int]]] = None,
    classes: Optional[List[int]] = None,
    *,
    namespace: str = "detector",
) -> Detections:
    """Detector-like prompt rows on ``image``, with frame and row identity."""
    xyxy = PROMPT_XYXY if xyxy is None else xyxy
    classes = PROMPT_CLASSES if classes is None else classes
    detections = Detections(
        xyxy=torch.tensor(xyxy, dtype=torch.float32).reshape(-1, 4),
        class_id=torch.tensor(classes, dtype=torch.int64),
        confidence=torch.full((len(xyxy),), 0.8),
        image_metadata={
            **FrameRef.from_image(image).prediction_metadata(),
            CLASS_NAMES_KEY: dict(PROMPT_NAMES),
        },
        bboxes_metadata=[{"detection_id": f"d{row}"} for row in range(len(xyxy))],
    )
    identified = with_row_identity(detections, namespace=namespace)

    return identified


def _sam2(model: Any = None) -> SegmentAnything2Block:
    block = SegmentAnything2Block(
        models={SAM2_DEFAULT_MODEL_ID: model if model is not None else FakeSam2()}
    )

    return block


def _sam3(model: Any = None) -> SegmentAnything3Block:
    block = SegmentAnything3Block(
        models={SAM3_DEFAULT_MODEL_ID: model if model is not None else FakeSam3()}
    )

    return block


def _call(block: Any, *, wanted=None, **params: Any) -> List[Dict[str, Any]]:
    # The blocks ask wants() for their mask outputs: run them as one engine call.
    with block_call(block, wanted=wanted):
        results = block.run(**params)

    return results


def _masks_of(xyxy: List[List[int]]) -> torch.Tensor:
    masks = torch.stack([box_mask(tuple(box), size_hw=SIZE_HW) for box in xyxy])

    return masks


# --- SAM2: whole image ---------------------------------------------------------


def test_sam2_without_boxes_gives_one_foreground_mask_per_image() -> None:
    model = FakeSam2()
    images = [_image("a"), _image("b", size_hw=(8, 12))]

    results = _call(_sam2(model), image=Batch(images))

    assert model.prompts == [None, None]
    first, second = (result["predictions"] for result in results)
    assert first.xyxy.tolist() == [[7, 5, 22, 15]]
    assert torch.equal(first.mask, _masks_of([CENTRE_XYXY]))
    assert first.class_id.tolist() == [0]
    assert first.confidence.tolist() == pytest.approx([0.8])
    assert first.image_metadata[CLASS_NAMES_KEY] == {0: "foreground"}
    assert first.image_metadata["prediction_type"] == "instance-segmentation"
    # 8x12 image: x 3 .. 8, y 2 .. 5.
    assert second.xyxy.tolist() == [[3, 2, 9, 6]]
    assert [frame_of(r["predictions"]).image_id for r in results] == ["a", "b"]
    assert frame_of(second).size_hw == (8, 12)


def test_sam2_whole_image_rows_start_a_fresh_identity_per_result() -> None:
    results = _call(_sam2(), image=Batch([_image("a"), _image("b")]))

    records = [row_record(result["predictions"]) for result in results]
    namespaces = [record.namespace for record in records]
    assert namespaces == [result["inference_id"] for result in results]
    assert len(set(namespaces)) == 2
    assert [record.host_ids() for record in records] == [[0], [0]]
    assert [result["model_id"] for result in results] == [SAM2_DEFAULT_MODEL_ID] * 2


def test_sam2_below_confidence_whole_image_mask_is_dropped() -> None:
    (result,) = _call(_sam2(), image=Batch([_image("a")]), confidence=0.85)

    predictions = result["predictions"]
    assert len(predictions) == 0
    assert tuple(predictions.mask.shape) == (0, *SIZE_HW)
    assert row_record(predictions).rows == 0


# --- SAM2: box prompts -----------------------------------------------------------


def test_sam2_box_prompts_give_one_mask_per_kept_row_with_its_lineage() -> None:
    model = FakeSam2()
    image = _image("a")
    prompt = _prompt(image)

    (result,) = _call(
        _sam2(model), image=Batch([image]), boxes=Batch([prompt]), confidence=0.6
    )

    assert model.prompts[0].tolist() == PROMPT_XYXY
    predictions = result["predictions"]
    # Scores 0.9, 0.7, 0.5: the third row is below 0.6.
    assert predictions.xyxy.tolist() == PROMPT_EXTENTS[:2]
    assert torch.equal(predictions.mask, _masks_of(PROMPT_XYXY[:2]))
    assert predictions.confidence.tolist() == pytest.approx([0.9, 0.7])
    assert predictions.class_id.tolist() == [1, 0]
    assert predictions.image_metadata[CLASS_NAMES_KEY] == PROMPT_NAMES
    assert predictions.bboxes_metadata == [
        {"detection_id": "d0"},
        {"detection_id": "d1"},
    ]
    record = row_record(predictions)
    assert (record.namespace, record.host_ids()) == ("detector", [0, 1])
    assert frame_of(predictions).image_id == "a"


def test_sam2_mask_boxes_are_the_inclusive_mask_bounds_not_the_prompt() -> None:
    # A prompt reaching past the image: the mask is clipped, and so is its box.
    image = _image("a")
    prompt = _prompt(image, [[0.0, 0.0, 30.0, 20.0]], [0])

    (result,) = _call(_sam2(), image=Batch([image]), boxes=Batch([prompt]))

    assert result["predictions"].xyxy.tolist() == [[0, 0, 30, 20]]
    assert bool(result["predictions"].mask.all())


def test_sam2_empty_prompt_gives_no_masks_and_calls_no_model() -> None:
    model = FakeSam2()
    image = _image("a")
    empty = _prompt(image, [], [])

    (result,) = _call(_sam2(model), image=Batch([image]), boxes=Batch([empty]))

    assert (model.embedded, model.prompts) == ([], [])
    predictions = result["predictions"]
    assert len(predictions) == 0
    assert tuple(predictions.mask.shape) == (0, *SIZE_HW)
    assert predictions.image_metadata[CLASS_NAMES_KEY] == PROMPT_NAMES
    record = row_record(predictions)
    assert (record.namespace, record.rows) == ("detector", 0)
    assert len(result["segmentation"]) == 0


def test_sam2_mixed_batch_embeds_only_images_with_prompt_rows() -> None:
    model = FakeSam2()
    images = [_image("a"), _image("b"), _image("c")]
    prompts = [
        _prompt(images[0], namespace="pa"),
        _prompt(images[1], [], [], namespace="pb"),
        _prompt(images[2], [[1, 1, 4, 4]], [2], namespace="pc"),
    ]

    results = _call(_sam2(model), image=Batch(images), boxes=Batch(prompts))

    assert model.embedded == [2]
    assert [len(result["predictions"]) for result in results] == [3, 0, 1]
    assert [row_record(r["predictions"]).namespace for r in results] == [
        "pa",
        "pb",
        "pc",
    ]
    assert results[2]["predictions"].xyxy.tolist() == [[1, 1, 5, 5]]


def test_sam2_reads_boxes_from_keypoint_predictions() -> None:
    image = _image("a")
    boxes = _prompt(image, PROMPT_XYXY[:1], [1])
    key_points = KeyPoints(
        xy=torch.zeros((1, 2, 2)),
        class_id=torch.tensor([1]),
        confidence=torch.ones((1, 2)),
    )

    (result,) = _call(_sam2(), image=Batch([image]), boxes=Batch([(key_points, boxes)]))

    assert result["predictions"].xyxy.tolist() == PROMPT_EXTENTS[:1]
    assert row_record(result["predictions"]).namespace == "detector"


def test_sam2_keypoints_without_boxes_are_an_error() -> None:
    key_points = KeyPoints(
        xy=torch.zeros((1, 2, 2)),
        class_id=torch.tensor([1]),
        confidence=torch.ones((1, 2)),
    )

    with pytest.raises(ValueError, match="keypoint prediction has no boxes"):
        _call(_sam2(), image=Batch([_image("a")]), boxes=Batch([(key_points, None)]))


def test_sam2_reads_boxes_from_a_segmentation_view() -> None:
    image = _image("a")
    boxes = _prompt(image)
    segmented = InstanceDetections(
        xyxy=boxes.xyxy,
        class_id=boxes.class_id,
        confidence=boxes.confidence,
        mask=torch.zeros((3, *SIZE_HW), dtype=torch.bool),
        image_metadata=boxes.image_metadata,
        bboxes_metadata=boxes.bboxes_metadata,
    )
    view = SegmentationView.from_dense(segmented)

    (result,) = _call(_sam2(), image=Batch([image]), boxes=Batch([view]))

    assert result["predictions"].xyxy.tolist() == PROMPT_EXTENTS
    assert result["predictions"].class_id.tolist() == PROMPT_CLASSES
    assert row_record(result["predictions"]).namespace == "detector"


@pytest.mark.parametrize(
    "prompt_image",
    [_image("other"), _image("a", size_hw=(10, 30))],
    ids=["other image", "other size"],
)
def test_sam2_boxes_of_another_frame_are_an_error(prompt_image: ImageData) -> None:
    model = FakeSam2()
    prompt = _prompt(prompt_image, [[1, 1, 4, 4]], [0])

    with pytest.raises(ValueError, match="wire boxes predicted on the image"):
        _call(_sam2(model), image=Batch([_image("a")]), boxes=Batch([prompt]))
    assert model.embedded == []


def test_sam2_boxes_and_images_must_have_the_same_length() -> None:
    image = _image("a")

    with pytest.raises(ValueError, match="boxes has 2 values for 1 images"):
        _call(
            _sam2(),
            image=Batch([image]),
            boxes=Batch([_prompt(image), _prompt(image)]),
        )


def test_sam2_does_not_change_its_prompt() -> None:
    image = _image("a")
    prompt = _prompt(image)
    xyxy, class_id = prompt.xyxy.clone(), prompt.class_id.clone()
    metadata = dict(prompt.image_metadata)
    rows = [dict(row) for row in prompt.bboxes_metadata]

    _call(_sam2(), image=Batch([image]), boxes=Batch([prompt]), confidence=0.6)

    assert torch.equal(prompt.xyxy, xyxy) and torch.equal(prompt.class_id, class_id)
    assert prompt.image_metadata == metadata
    assert prompt.bboxes_metadata == rows
    assert row_record(prompt).host_ids() == [0, 1, 2]


# --- SAM2: proposals, logits and options ----------------------------------------


def test_sam2_multimask_output_reaches_the_model_and_the_best_proposal_wins() -> None:
    image = _image("a")
    prompt = _prompt(image, PROMPT_XYXY[:1], [1])
    for multimask in (True, False):
        model = FakeSam2()

        (result,) = _call(
            _sam2(model),
            image=Batch([image]),
            boxes=Batch([prompt]),
            multimask_output=multimask,
        )

        assert model.multimask == [multimask]
        # With three proposals the best is the middle one (the exact box).
        assert result["predictions"].xyxy.tolist() == PROMPT_EXTENTS[:1]
        assert result["predictions"].confidence.tolist() == pytest.approx([0.9])


def test_most_confident_picks_the_best_proposal_per_prompt() -> None:
    masks = torch.arange(2 * 3 * 1 * 1, dtype=torch.float32).reshape(2, 3, 1, 1)
    scores = torch.tensor([[0.1, 0.2, 0.7], [0.6, 0.3, 0.1]])

    selected, best = _most_confident(masks, scores)

    assert selected.flatten().tolist() == [2.0, 3.0]
    assert best.tolist() == pytest.approx([0.7, 0.6])
    # One prompt: SAM2 drops the prompt axis.
    single, single_best = _most_confident(masks[1], scores[1])
    assert single.flatten().tolist() == [3.0]
    assert single_best.tolist() == pytest.approx([0.6])


class _LogitModel:
    """One prompt, one proposal: fixed logits on a 2x3 image."""

    def __init__(self, logits: List[List[float]], *, rows: int = 1):
        self.logits = torch.tensor(logits).expand(rows, 1, 2, 3)
        self.rows = rows

    def embed_images(self, images, **options):
        return [object() for _ in images]

    def segment_images(self, embeddings, boxes=None, **options):
        masks = self.logits if self.rows > 1 else self.logits[0]
        scores = torch.full((self.rows, 1), 0.9)
        scores = scores if self.rows > 1 else scores[0]
        return [SAM2Prediction(masks=masks, scores=scores, logits=masks)]


def test_sam2_logits_at_zero_are_foreground() -> None:
    model = _LogitModel([[-0.5, 0.0, 0.5], [-1.0, -1.0, -1.0]])

    (result,) = _call(_sam2(model), image=Batch([_image("a", size_hw=(2, 3))]))

    expected = [[[False, True, True], [False, False, False]]]
    assert result["predictions"].mask.tolist() == expected
    assert result["predictions"].xyxy.tolist() == [[1, 0, 3, 1]]


def test_sam2_empty_mask_has_a_zero_box() -> None:
    model = _LogitModel([[-1.0] * 3, [-1.0] * 3])

    (result,) = _call(_sam2(model), image=Batch([_image("a", size_hw=(2, 3))]))

    assert result["predictions"].xyxy.tolist() == [[0, 0, 0, 0]]
    assert result["predictions"].confidence.tolist() == pytest.approx([0.9])


def test_sam2_model_masks_that_do_not_match_the_prompts_are_an_error() -> None:
    image = _image("a", size_hw=(2, 3))
    prompt = _prompt(image, [[0, 0, 1, 1], [1, 0, 2, 1]], [0, 0])

    with pytest.raises(ValueError, match=r"masks of shape \(1, 2, 3\) for 2 prompts"):
        _call(
            _sam2(_LogitModel([[1.0] * 3] * 2)),
            image=Batch([image]),
            boxes=Batch([prompt]),
        )


@pytest.mark.parametrize(
    "options, message",
    [
        ({"confidence": 1.5}, "confidence must be a number 0..1"),
        ({"confidence": True}, "confidence must be a number 0..1"),
        ({"multimask_output": "yes"}, "multimask_output must be true or false"),
    ],
)
def test_sam2_invalid_direct_call_options_fail_before_the_model(
    options, message
) -> None:
    model = FakeSam2()

    with pytest.raises(ValueError, match=message):
        _call(_sam2(model), image=Batch([_image("a")]), **options)
    assert model.embedded == []


def test_sam2_needs_rgb_images() -> None:
    with pytest.raises(ValueError, match="needs a 3-channel RGB image"):
        _call(_sam2(), image=Batch([_image("a", channels=1)]))


# --- SAM2: dense and view outputs ---------------------------------------------


def test_sam2_view_and_dense_predictions_hold_the_same_rows() -> None:
    image = _image("a")

    (result,) = _call(_sam2(), image=Batch([image]), boxes=Batch([_prompt(image)]))

    view, dense = result["segmentation"], result["predictions"]
    assert view.full_res() is dense
    assert torch.equal(view.detections.xyxy, dense.xyxy)
    assert row_record(view.detections).namespace == "detector"


@pytest.mark.parametrize("wanted", [{"predictions"}, {"segmentation"}])
def test_sam2_leaves_out_an_unwanted_mask_output(wanted) -> None:
    image = _image("a")

    (result,) = _call(_sam2(), wanted=wanted, image=Batch([image]))

    assert set(result) == {*wanted, "model_id", "inference_id"}


# --- SAM2: embedding reuse ------------------------------------------------------


def test_sam2_reuses_the_embedding_of_the_same_pixels() -> None:
    model = FakeSam2()
    block = _sam2(model)
    image = _image("a")

    _call(block, image=Batch([image]))
    _call(block, image=Batch([image]), boxes=Batch([_prompt(image)]))

    assert model.embedded == [1]
    assert model.embed_options == [{"use_embeddings_cache": False}]
    first, second = model.segmented
    assert second is first


def test_sam2_new_pixels_under_the_same_image_id_are_embedded_again() -> None:
    model = FakeSam2()
    block = _sam2(model)
    image = _image("a")
    brighter = image.with_pixels(torch.full((3, *SIZE_HW), 200, dtype=torch.uint8))

    _call(block, image=Batch([image]))
    _call(block, image=Batch([brighter]))

    assert model.embedded == [1, 1]
    assert model.segmented[0] is not model.segmented[1]


def test_sam2_keeps_only_the_last_calls_embeddings() -> None:
    model = FakeSam2()
    block = _sam2(model)
    a, b = _image("a"), _image("b")

    _call(block, image=Batch([a]))
    _call(block, image=Batch([b]))
    _call(block, image=Batch([a, b]))

    # a was dropped by the second call; b is reused in the third.
    assert model.embedded == [1, 1, 1]
    assert model.segmented[3] is model.segmented[1]


def test_sam2_reset_state_drops_embeddings_but_keeps_the_model() -> None:
    model = FakeSam2()
    block = _sam2(model)
    image = _image("a")

    _call(block, image=Batch([image]))
    block.reset_state()
    _call(block, image=Batch([image]))

    assert model.embedded == [1, 1]


def test_sam2_embeddings_are_kept_per_model_request() -> None:
    small, large = FakeSam2(), FakeSam2()
    block = SegmentAnything2Block(
        models={"sam2/hiera_small": small, "sam2/hiera_large": large}
    )
    image = _image("a")

    _call(block, image=Batch([image]), model_id="sam2/hiera_small")
    _call(block, image=Batch([image]), model_id="sam2/hiera_large")
    _call(block, image=Batch([image]), model_id="sam2/hiera_large")

    assert (small.embedded, large.embedded) == ([1], [1])


class _ReleaseProbe(FakeSam2):
    """FakeSam2 that holds its embeddings only weakly.

    Each ``embed_images`` call first records which earlier embeddings are still
    reachable, in creation order. No ``gc.collect()``: a released embedding
    must be freed by reference counting alone.
    """

    def __init__(self):
        super().__init__()
        self.made: List[weakref.ref] = []
        self.alive_at_embed: List[List[bool]] = []
        self.fail = False

    def embed_images(self, images, **options):
        self.alive_at_embed.append([ref() is not None for ref in self.made])
        if self.fail:
            raise RuntimeError("embed failed")
        embeddings = super().embed_images(images, **options)
        self.made.extend(weakref.ref(embedding) for embedding in embeddings)

        return embeddings

    def segment_images(self, embeddings, **options):
        predictions = super().segment_images(embeddings, **options)
        self.segmented.clear()

        return predictions

    def alive(self) -> List[bool]:
        return [ref() is not None for ref in self.made]


def test_sam2_releases_unhit_embeddings_before_embedding_new_images() -> None:
    model = _ReleaseProbe()
    block = _sam2(model)
    a, b, c = _image("a"), _image("b"), _image("c")

    _call(block, image=Batch([a, b]))
    _call(block, image=Batch([b, c]))

    # At the second embed_images, a (unhit) is gone and b (hit) is live.
    assert model.alive_at_embed == [[], [False, True]]
    assert model.alive() == [False, True, True]
    assert model.embedded == [2, 1]


def test_sam2_releases_a_stale_embedding_of_the_same_image_id() -> None:
    model = _ReleaseProbe()
    block = _sam2(model)
    image = _image("a")
    brighter = image.with_pixels(torch.full((3, *SIZE_HW), 200, dtype=torch.uint8))

    _call(block, image=Batch([image]))
    _call(block, image=Batch([brighter]))

    # The miss leaves the stale entry in the hit loop's local; it is cleared too.
    assert model.alive_at_embed == [[], [False]]
    assert model.embedded == [1, 1]


def test_sam2_failed_embedding_keeps_only_the_calls_hits() -> None:
    model = _ReleaseProbe()
    block = _sam2(model)
    a, b, c = _image("a"), _image("b"), _image("c")
    _call(block, image=Batch([a, b]))

    model.fail = True
    with pytest.raises(RuntimeError, match="embed failed"):
        _call(block, image=Batch([b, c]))
    model.fail = False

    assert model.alive_at_embed[1] == [False, True]
    assert model.alive() == [False, True]
    _call(block, image=Batch([b]))
    assert model.embedded == [2]
    _call(block, image=Batch([a]))
    assert model.embedded == [2, 1]


def test_sam2_reset_state_releases_the_kept_embeddings() -> None:
    model = _ReleaseProbe()
    block = _sam2(model)
    image = _image("a")
    _call(block, image=Batch([image]))
    assert model.alive() == [True]

    block.reset_state()

    assert model.alive() == [False]


def test_sam2_reports_its_model_dependency() -> None:
    params = SegmentAnything2Block.Params(
        image="$inputs.image", model_id="sam2/hiera_large"
    )

    (resource,) = SegmentAnything2Block.discover_dependent_resources(params)

    assert (resource.resource_type, resource.identifier) == (
        "model",
        "sam2/hiera_large",
    )
    assert resource.details["task_type"] == "instance-segmentation"


# --- SAM3 -----------------------------------------------------------------------
# FakeSam3 table (20x30 image): car 0.9 (2,2,11,9), car 0.7 (2,2,11,7),
# car 0.4 (20,3,25,9); person 0.6 (2,2,11,9), person 0.95 empty mask;
# no name 0.8 (1,1,28,18). Mask IoU car 0.9 / car 0.7 = 60 / 80 = 0.75;
# person 0.6 equals car 0.9 (IoU 1).
CAR_9, CAR_7, CAR_4 = [2, 2, 12, 10], [2, 2, 12, 8], [20, 3, 26, 10]


def _rows(predictions: InstanceDetections):
    rows = list(
        zip(
            predictions.class_id.tolist(),
            [round(score, 3) for score in predictions.confidence.tolist()],
            predictions.xyxy.tolist(),
        )
    )

    return rows


def test_sam3_one_prompt_per_class_and_empty_masks_dropped() -> None:
    model = FakeSam3()

    (result,) = _call(
        _sam3(model),
        image=Batch([_image("a")]),
        class_names=["car", "person"],
        apply_nms=False,
    )

    (call,) = model.calls
    assert call["prompts"] == [{"text": "car"}, {"text": "person"}]
    assert (call["output_prob_thresh"], call["mask_format"]) == (0.5, "dense")
    predictions = result["predictions"]
    assert _rows(predictions) == [(0, 0.9, CAR_9), (0, 0.7, CAR_7), (1, 0.6, CAR_9)]
    assert torch.equal(
        predictions.mask, _masks_of([[2, 2, 11, 9], [2, 2, 11, 7], [2, 2, 11, 9]])
    )
    assert predictions.image_metadata[CLASS_NAMES_KEY] == {0: "car", 1: "person"}
    record = row_record(predictions)
    assert (record.namespace, record.host_ids()) == (result["inference_id"], [0, 1, 2])
    assert frame_of(predictions).image_id == "a"


@pytest.mark.parametrize(
    "threshold, expected",
    [
        (0.9, [(0, 0.9, CAR_9), (0, 0.7, CAR_7)]),  # person: IoU 1 with car 0.9
        (0.5, [(0, 0.9, CAR_9)]),  # car 0.7: IoU 0.75 with car 0.9
        (1.0, [(0, 0.9, CAR_9), (0, 0.7, CAR_7), (1, 0.6, CAR_9)]),
    ],
)
def test_sam3_nms_drops_masks_overlapping_a_better_mask_of_any_class(
    threshold: float, expected
) -> None:
    (result,) = _call(
        _sam3(),
        image=Batch([_image("a")]),
        class_names=["car", "person"],
        nms_iou_threshold=threshold,
    )

    assert _rows(result["predictions"]) == expected


def test_sam3_per_class_confidence_sets_the_floor_and_each_class_threshold() -> None:
    model = FakeSam3()

    (result,) = _call(
        _sam3(model),
        image=Batch([_image("a")]),
        class_names=["car", "person"],
        per_class_confidence=[0.8, 0.3],
        apply_nms=False,
    )

    assert model.calls[0]["output_prob_thresh"] == 0.3
    # car 0.7 and 0.4 are below 0.8; person 0.6 passes 0.3.
    assert _rows(result["predictions"]) == [(0, 0.9, CAR_9), (1, 0.6, CAR_9)]


def test_sam3_without_class_names_runs_one_foreground_prompt() -> None:
    model = FakeSam3()

    (result,) = _call(_sam3(model), image=Batch([_image("a")]))

    assert model.calls[0]["prompts"] == [{"text": None}]
    assert _rows(result["predictions"]) == [(0, 0.8, [1, 1, 29, 19])]
    assert result["predictions"].image_metadata[CLASS_NAMES_KEY] == {0: "foreground"}


@pytest.mark.parametrize(
    "class_names, prompts",
    [
        ("car, person", ["car", "person"]),
        (" car ", ["car"]),
        (("car",), ["car"]),
        ([], [None]),
    ],
)
def test_sam3_class_name_forms(class_names, prompts) -> None:
    model = FakeSam3()

    _call(_sam3(model), image=Batch([_image("a")]), class_names=class_names)

    assert [prompt["text"] for prompt in model.calls[0]["prompts"]] == prompts


@pytest.mark.parametrize("class_names", ["car,,person", ["car", ""], ["car", 3]])
def test_sam3_empty_or_non_text_class_names_are_errors(class_names) -> None:
    with pytest.raises(ValueError, match="class_names holds non-empty names"):
        _call(_sam3(), image=Batch([_image("a")]), class_names=class_names)


def test_sam3_class_mapping_renames_results_but_not_prompts() -> None:
    model = FakeSam3()

    (result,) = _call(
        _sam3(model),
        image=Batch([_image("a")]),
        class_names=["car", "person"],
        class_mapping={"car": "auto", "dog": "perro"},
    )

    assert model.calls[0]["prompts"] == [{"text": "car"}, {"text": "person"}]
    names = result["predictions"].image_metadata[CLASS_NAMES_KEY]
    assert names == {0: "auto", 1: "person"}


def test_sam3_class_mapping_renames_the_foreground_prompt() -> None:
    (result,) = _call(
        _sam3(), image=Batch([_image("a")]), class_mapping={"foreground": "thing"}
    )

    assert result["predictions"].image_metadata[CLASS_NAMES_KEY] == {0: "thing"}


@pytest.mark.parametrize(
    "options, message",
    [
        ({"per_class_confidence": [0.5]}, "has 1 values for 2 prompts"),
        ({"per_class_confidence": 0.5}, "must be a list of numbers"),
        ({"per_class_confidence": [0.5, 2.0]}, "per_class_confidence must be"),
        ({"confidence": -0.1}, "confidence must be a number 0..1"),
        ({"nms_iou_threshold": 1.1}, "nms_iou_threshold must be a number 0..1"),
        ({"apply_nms": 1}, "apply_nms must be true or false"),
        ({"class_mapping": ["car"]}, "class_mapping must map class names"),
    ],
)
def test_sam3_invalid_direct_call_options_fail_before_the_model(
    options, message
) -> None:
    model = FakeSam3()

    with pytest.raises(ValueError, match=message):
        _call(
            _sam3(model),
            image=Batch([_image("a")]),
            class_names=["car", "person"],
            **options,
        )
    assert model.calls == []


def test_sam3_params_check_literal_threshold_counts_but_not_selectors() -> None:
    with pytest.raises(pydantic.ValidationError, match="has 1 values for 2 prompts"):
        SegmentAnything3Block.Params(
            image="$inputs.image",
            class_names=["car", "person"],
            per_class_confidence=[0.5],
        )

    params = SegmentAnything3Block.Params(
        image="$inputs.image",
        class_names="$inputs.classes",
        per_class_confidence=[0.5],
    )
    assert params.class_names == "$inputs.classes"


def test_sam3_hosted_execution_is_deferred_and_rejected_before_any_request() -> None:
    # V1 chose the hosted proxy with SAM3_EXEC_MODE; V2 has no such mode yet.
    with pytest.raises(pydantic.ValidationError, match="execution"):
        SegmentAnything3Block.Params(image="$inputs.image", execution="remote")
    with pytest.raises(TypeError, match="platform_client"):
        SegmentAnything3Block(models={}, platform_client=object())


def test_sam3_model_results_of_the_wrong_shape_are_errors() -> None:
    class WrongSize(FakeSam3):
        def segment_with_text_prompts(self, images, **options):
            return super().segment_with_text_prompts(
                [torch.zeros((3, 4, 4), dtype=torch.uint8)], **options
            )

    class Missing(FakeSam3):
        def segment_with_text_prompts(self, images, **options):
            return []

    with pytest.raises(ValueError, match=r"returned \(4, 4\) masks for a \(20, 30\)"):
        _call(_sam3(WrongSize()), image=Batch([_image("a")]), class_names=["car"])
    with pytest.raises(ValueError, match="returned 0 results for 1 images"):
        _call(_sam3(Missing()), image=Batch([_image("a")]), class_names=["car"])


def test_sam3_batches_images_in_one_model_call() -> None:
    model = FakeSam3()
    images = [_image("a"), _image("b", size_hw=(10, 12))]

    results = _call(
        _sam3(model), image=Batch(images), class_names=["car"], apply_nms=False
    )

    assert [call["images"] for call in model.calls] == [2]
    assert [frame_of(r["predictions"]).image_id for r in results] == ["a", "b"]
    # On 10x12 the occupied pixels reach x=11; their exclusive end is 12.
    assert results[1]["predictions"].xyxy.tolist() == [[2, 2, 12, 10], [2, 2, 12, 8]]


def test_sam3_reports_its_model_dependency() -> None:
    params = SegmentAnything3Block.Params(
        image="$inputs.image", model_id="$inputs.sam3", model_package_id="pkg"
    )

    (resource,) = SegmentAnything3Block.discover_dependent_resources(params)

    assert resource.identifier == "$inputs.sam3"
    assert resource.details["model_package_id"] == "pkg"


# --- shared mask helpers --------------------------------------------------------


def test_mask_boxes_are_half_open_extents_and_zero_when_empty() -> None:
    masks = torch.zeros((3, 4, 5), dtype=torch.bool)
    masks[0, 1:3, 2:5] = True
    masks[1, 3, 0] = True

    boxes = _mask_boxes(masks)

    assert boxes.tolist() == [[2, 1, 5, 3], [0, 3, 1, 4], [0, 0, 0, 0]]
    assert boxes.dtype == torch.float32


def _row(prompt: int, box, score: float):
    return _Row(prompt_index=prompt, mask=box_mask(box, size_hw=(4, 4)), score=score)


def test_mask_nms_keeps_survivors_in_row_order() -> None:
    rows = [_row(0, (0, 0, 0, 0), 0.3), _row(1, (2, 2, 3, 3), 0.9)]

    kept = _mask_nms(rows, iou_threshold=0.5)

    assert [row.score for row in kept] == [0.3, 0.9]


def test_mask_nms_visits_equal_scores_in_row_order() -> None:
    # V1 visits ties in reverse row order and would keep the second row.
    rows = [_row(0, (0, 0, 1, 1), 0.5), _row(1, (0, 0, 1, 1), 0.5)]

    kept = _mask_nms(rows, iou_threshold=0.5)

    assert [row.prompt_index for row in kept] == [0]


def test_mask_nms_drops_only_above_the_threshold() -> None:
    # A 2x4 mask and the 2x2 mask inside it: IoU 4 / 8 = 0.5.
    rows = [_row(0, (0, 0, 1, 3), 0.9), _row(1, (0, 0, 1, 1), 0.8)]

    assert len(_mask_nms(rows, iou_threshold=0.5)) == 2
    assert len(_mask_nms(rows, iou_threshold=0.49)) == 1


@pytest.mark.parametrize(
    "empty_prompt", [True, False], ids=["zero rows", "filtered empty"]
)
def test_sam2_empty_results_use_selected_prompt_identity(empty_prompt: bool) -> None:
    image = _image("a")
    prompt = _prompt(image, [], []) if empty_prompt else _prompt(image)
    source_record = row_record(prompt)

    (result,) = _call(
        _sam2(), image=Batch([image]), boxes=Batch([prompt]), confidence=1.0
    )

    record = row_record(result["predictions"])
    assert (record.namespace, record.rows, record.host_ids()) == ("detector", 0, [])
    assert source_record.host_ids() == ([] if empty_prompt else [0, 1, 2])
    assert row_record(result["segmentation"].detections).namespace == "detector"


def test_sam2_changed_boxes_and_scores_drop_prompt_host_mirrors() -> None:
    image = _image("a")
    prompt = _prompt(image, [[0, 0, 30, 20]], [1])
    prompt.bboxes_metadata[0].update(
        _host_xyxy=[0, 0, 30, 20],
        _host_class_id=1,
        _host_confidence=0.8,
        custom="kept",
    )
    original = dict(prompt.bboxes_metadata[0])

    (result,) = _call(_sam2(), image=Batch([image]), boxes=Batch([prompt]))

    predictions = result["predictions"]
    assert predictions.xyxy.tolist() == [[0, 0, 30, 20]]
    assert predictions.confidence.tolist() == pytest.approx([0.9])
    assert predictions.bboxes_metadata == [{"detection_id": "d0", "custom": "kept"}]
    assert prompt.bboxes_metadata == [original]
    assert (
        result["segmentation"].detections.bboxes_metadata == predictions.bboxes_metadata
    )


def test_continuous_prompt_boxes_reach_sam2_unchanged() -> None:
    image = _image("continuous")
    box = [[1.25, 2.5, 8.75, 10.5]]
    prompt = _prompt(image, box, [1])
    model = FakeSam2()

    _call(_sam2(model), image=Batch([image]), boxes=Batch([prompt]))

    assert model.prompts[0].tolist() == box
    assert prompt.xyxy.tolist() == box


def test_sam3_backend_geometry_is_not_modified_when_reboxing_masks() -> None:
    class _BackendBoxes(FakeSam3):
        def segment_with_text_prompts(self, *args, **kwargs):
            self.result = super().segment_with_text_prompts(*args, **kwargs)
            self.boxes = torch.tensor([[1.25, 2.5, 8.75, 10.5]])
            self.result[0][0]["boxes"] = self.boxes
            return self.result

    model = _BackendBoxes(table={"part": [((4, 3, 4, 3), 0.9)]})
    (row,) = _call(_sam3(model), image=Batch([_image("a")]), class_names=["part"])

    assert row["predictions"].xyxy.tolist() == [[4, 3, 5, 4]]
    assert model.result[0][0]["boxes"] is model.boxes
    assert model.boxes.tolist() == [[1.25, 2.5, 8.75, 10.5]]


INVALID_FRAME_CASES = (
    "non-mapping",
    "typed-missing-key",
    "typed-image",
    "typed-size",
    "typed-source",
    "typed-frame",
    "typed-time",
    "typed-parent",
    "typed-root",
    "typed-version",
    "typed-mapping",
    "typed-video",
    "legacy-image",
    "legacy-root",
    "legacy-size",
    "legacy-offset",
    "legacy-scale",
    "conflict-size",
    "conflict-flag",
    "partial-root-scale",
)
VALID_FRAME_CASES = (
    "absent",
    "empty",
    "unrelated",
    "size-only",
    "root-id-only",
    "offset-only",
    "none-fields",
    "typed-only",
    "legacy-only",
    "mirrored",
)


def _guard_image(image):
    return dataclasses.replace(
        image,
        video_metadata=VideoMetadata(
            video_identifier="source-a",
            frame_number=4,
            frame_timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ),
    )


def _guard_metadata(image, case):
    metadata = copy.deepcopy(image.prediction_metadata())
    record = metadata["workflows_v2_frame"]
    if case == "non-mapping":
        return []
    if case.startswith("typed-"):
        metadata = {"workflows_v2_frame": record}
        if case == "typed-missing-key":
            record.pop("root")
        elif case == "typed-image":
            record["image_id"] = "other"
        elif case == "typed-size":
            record["size_hw"][0] += 1
        elif case == "typed-source":
            record["video_metadata"]["video_identifier"] = "source-b"
        elif case == "typed-frame":
            record["video_metadata"]["frame_number"] += 1
        elif case == "typed-time":
            record["video_metadata"]["frame_timestamp"] = "2026-01-02T00:00:00+00:00"
        elif case == "typed-parent":
            record["parent"]["offset_xy"][0] += 1
        elif case == "typed-root":
            record["root"]["frame_id"] = "other-root"
        elif case == "typed-version":
            record["version"] = True
        elif case == "typed-mapping":
            record["parent"]["scale_xy"] = [0, 1]
        elif case == "typed-video":
            record["video_metadata"]["frame_number"] = True
        return metadata
    if case == "legacy-image":
        return {"parent_id": "other"}
    if case == "legacy-root":
        return {"root_parent_id": "other-root"}
    if case == "legacy-size":
        return {"image_dimensions": [True, image.size_hw[1]]}
    if case == "legacy-offset":
        return {"parent_coordinates": ["bad", 0]}
    if case == "legacy-scale":
        return {"scaling_relative_to_root_parent": 0}
    if case == "conflict-size":
        metadata["image_dimensions"] = [image.size_hw[0] + 1, image.size_hw[1]]
        return metadata
    if case == "conflict-flag":
        metadata["is_composite"] = 1
        return metadata
    if case == "partial-root-scale":
        return {"scaling_relative_to_root_parent": 2}
    if case == "absent":
        return None
    if case == "empty":
        return {}
    if case == "unrelated":
        return {"class_names": {0: "part"}}
    if case == "size-only":
        return {"image_dimensions": metadata["image_dimensions"]}
    if case == "root-id-only":
        return {"root_parent_id": metadata["root_parent_id"]}
    if case == "offset-only":
        return {"parent_coordinates": metadata["parent_coordinates"]}
    if case == "none-fields":
        return {"parent_id": None, "image_dimensions": None, "workflows_v2_frame": None}
    if case == "legacy-only":
        metadata.pop("workflows_v2_frame")
    return metadata


@pytest.mark.parametrize("case", INVALID_FRAME_CASES)
@pytest.mark.parametrize("empty", [False, True])
def test_sam2_rejects_frame_metadata_before_model_work(case, empty):
    image = _guard_image(_image("a"))
    prompt = _prompt(image, [] if empty else [[1, 1, 4, 4]], [] if empty else [0])
    prompt.image_metadata = _guard_metadata(image, case)
    model = FakeSam2()
    with pytest.raises(ValueError, match="wire boxes predicted on the image"):
        _call(_sam2(model), image=Batch([image]), boxes=Batch([prompt]), wanted=set())
    assert model.embedded == []
    assert model.segmented == []


@pytest.mark.parametrize("case", VALID_FRAME_CASES)
@pytest.mark.parametrize("empty", [False, True])
def test_sam2_accepts_absent_and_valid_partial_frame_metadata(case, empty):
    image = _guard_image(_image("a"))
    prompt = _prompt(image, [] if empty else [[1, 1, 4, 4]], [] if empty else [0])
    prompt.image_metadata = _guard_metadata(image, case)
    before = copy.deepcopy(prompt.image_metadata)
    model = FakeSam2()
    (result,) = _call(_sam2(model), image=Batch([image]), boxes=Batch([prompt]))
    assert len(result["predictions"]) == (0 if empty else 1)
    assert model.embedded == ([] if empty else [1])
    assert prompt.image_metadata == before


def test_sam2_view_prompt_validates_legacy_identity():
    image = _image("a")
    predictions = InstanceDetections(
        xyxy=torch.tensor([[1.0, 1.0, 4.0, 4.0]]),
        class_id=torch.tensor([0]),
        confidence=torch.tensor([0.8]),
        mask=torch.ones((1, *SIZE_HW), dtype=torch.bool),
        image_metadata={"parent_id": "other"},
    )
    model = FakeSam2()
    with pytest.raises(ValueError, match="Frame mismatch"):
        _call(
            _sam2(model),
            image=Batch([image]),
            boxes=Batch([SegmentationView.from_dense(predictions)]),
        )
    assert model.embedded == []
