"""SAM3 interactive block on a fake model, alone and in compiled workflows.

``FakeSam3Interactive`` below copies the call shapes of ``inference_models``
``SAM3Torch.embed_images`` and ``segment_with_visual_prompts`` (sam3_torch.py):
one ``(P, H, W)`` logits tensor and ``(P,)`` scores per image, the best
proposal per prompt already chosen. Its masks are a fixed function of each
prompt, so every expected mask, box and score below is written by hand. This
is not model inference and nothing is downloaded::

    box i of P     mask = the box (inclusive pixels), score 0.9 - 0.2 * i
    point set      mask = 5x5 squares around positive points, minus the
                   squares around negative points; score 0.8
    logits         +4 inside a mask, -4 outside

Compiled cases use the shared fake detector (``fake_task_models``): an image
of mean brightness b has ``round(4 * b / 255)`` rows; row i is box
``[i, i, w - i, h - i]``, class ``i % 3`` of person/car/dog.
"""

import copy
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pydantic
import pytest
import torch
from roboflow_workflows.v2.blocks.image.dynamic_crop.v1 import DynamicCropBlock
from roboflow_workflows.v2.blocks.models.cache import (
    ModelCache,
    ModelRequest,
    ModelRequestConflict,
)
from roboflow_workflows.v2.blocks.models.roboflow_platform.object_detection_model.v3 import (
    ObjectDetectionBlock,
)
from roboflow_workflows.v2.blocks.models.sam3_interactive.v1 import SAM3_INTERACTIVE_DEFAULT_MODEL_ID, SegmentAnything3InteractiveBlock
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.frames import FrameRef, frame_of
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.masks.grid import (
    INSTANCE_SEGMENTATION_VIEW_KIND,
    SegmentationView,
)
from roboflow_workflows.v2.data.prediction_ops import CLASS_NAMES_KEY
from roboflow_workflows.v2.data.predictions import NATIVE_KINDS
from roboflow_workflows.v2.data.row_identity import row_record, with_row_identity
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.errors import (
    LineageError,
    ParamsValidationError,
    StepExecutionError,
)
from roboflow_workflows.v2.engine.introspection import discover_workload
from roboflow_workflows.v2.engine.plan import CompileOptions
from roboflow_workflows.v2.engine.reactions.testing import block_call

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections
from inference_models.models.sam3.entities import SAM3ImageEmbeddings, SAM3Prediction
from tests.v2.support.models.fake_sam_models import box_mask
from tests.v2.support.models.fake_task_models import NmsDetector

SIZE_HW = (20, 30)
PROMPT_NAMES = {0: "person", 1: "car", 2: "dog"}
# Prompt rows on a 20x30 image, scored 0.9, 0.7 and 0.5 by the fake.
PROMPT_XYXY = [[2, 3, 10, 8], [12, 4, 20, 15], [5, 5, 25, 18]]
PROMPT_CLASSES = [1, 0, 2]
POINT_SCORE = 0.8
# One positive point at (10, 8): its 5x5 square is x 8..12, y 6..10.
POINT = [{"x": 10, "y": 8}]
POINT_XYXY = [8, 6, 12, 10]
# Adding a negative point at (12, 8) carves x 10..14 out of that square.
POINT_AND_NEGATIVE = [{"x": 10, "y": 8}, {"x": 12, "y": 8, "positive": False}]
CARVED_XYXY = [8, 6, 9, 10]
# The *_XYXY values above are occupied pixels (inclusive). Output boxes are
# half-open extents: one past the last occupied column and row.
PROMPT_EXTENTS = [[2, 3, 11, 9], [12, 4, 21, 16], [5, 5, 26, 19]]
POINT_EXTENT = [8, 6, 13, 11]
CARVED_EXTENT = [8, 6, 10, 11]
NAMES_WITH_POINT = {**PROMPT_NAMES, 3: "foreground"}


def point_mask(
    coordinates: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    size_hw: Tuple[int, int],
) -> torch.Tensor:
    """The fake's mask for one point set: positive squares minus negative ones."""
    mask = torch.zeros(size_hw, dtype=torch.bool)
    for wanted_label in (1, 0):
        for (x, y), label in zip(coordinates, labels):
            if label != wanted_label:
                continue
            x, y = round(x), round(y)
            square = box_mask((x - 2, y - 2, x + 2, y + 2), size_hw=size_hw)
            mask = mask | square if label else mask & ~square

    return mask


class FakeSam3Interactive:
    """``embed_images`` and ``segment_with_visual_prompts`` with SAM3Torch's shapes."""

    def __init__(self, request: Optional[ModelRequest] = None):
        self.request = request
        self.embedded: List[int] = []
        self.embed_options: List[Dict[str, Any]] = []
        # Per segmented image and prompt group: its embeddings and prompt.
        self.calls: List[Dict[str, Any]] = []

    def embed_images(
        self, images: List[torch.Tensor], **options: Any
    ) -> List[SAM3ImageEmbeddings]:
        self.embedded.append(len(images))
        self.embed_options.append(options)
        embeddings = [
            SAM3ImageEmbeddings(
                image_hash=f"fake-{index}",
                image_size_hw=tuple(image.shape[1:]),
                embeddings={"pixels": image},
            )
            for index, image in enumerate(images)
        ]

        return embeddings

    def segment_with_visual_prompts(
        self,
        embeddings: List[SAM3ImageEmbeddings],
        point_coordinates: Optional[List[torch.Tensor]] = None,
        point_labels: Optional[List[torch.Tensor]] = None,
        boxes: Optional[List[torch.Tensor]] = None,
        multi_mask_output: bool = True,
        return_logits: bool = False,
    ) -> List[SAM3Prediction]:
        assert return_logits, "the block binarizes logits itself"
        # V1 sends box prompts and point prompts as separate requests.
        assert (boxes is None) != (point_coordinates is None)
        predictions = []
        for index, embedding in enumerate(embeddings):
            size_hw = embedding.image_size_hw
            if boxes is not None:
                masks, scores, prompt = self._box_masks(boxes[index], size_hw)
            else:
                masks, scores, prompt = self._point_masks(
                    point_coordinates[index], point_labels[index], size_hw
                )
            self.calls.append(
                {"embedding": embedding, "multimask": multi_mask_output, **prompt}
            )
            logits = torch.stack(masks).float() * 8.0 - 4.0
            predictions.append(
                SAM3Prediction(masks=logits, scores=torch.tensor(scores), logits=logits)
            )

        return predictions

    @staticmethod
    def _box_masks(xyxy: torch.Tensor, size_hw: Tuple[int, int]):
        assert xyxy.ndim == 2 and xyxy.shape[1] == 4
        rows = xyxy.tolist()
        masks = [box_mask(tuple(int(v) for v in row), size_hw=size_hw) for row in rows]
        scores = [0.9 - 0.2 * row for row in range(len(rows))]

        return masks, scores, {"boxes": rows}

    @staticmethod
    def _point_masks(
        coordinates: torch.Tensor, labels: torch.Tensor, size_hw: Tuple[int, int]
    ):
        # SAM3Torch.pad_points needs one label list per prompt: (1, N) labels.
        assert coordinates.ndim == 3 and coordinates.shape[2] == 2
        assert labels.shape == coordinates.shape[:2]
        mask = point_mask(coordinates[0].tolist(), labels[0].tolist(), size_hw=size_hw)
        prompt = {
            "points": coordinates.tolist(),
            "labels": labels.tolist(),
            "dtypes": (coordinates.dtype, labels.dtype),
        }

        return [mask], [POINT_SCORE], prompt


class _ScriptedSam3(FakeSam3Interactive):
    """Returns given masks and scores for every prompt group."""

    def __init__(self, masks: torch.Tensor, scores: torch.Tensor):
        super().__init__()
        self._masks, self._scores = masks, scores

    def segment_with_visual_prompts(self, embeddings, **prompts):
        super().segment_with_visual_prompts(embeddings, **prompts)
        prediction = SAM3Prediction(
            masks=self._masks, scores=self._scores, logits=self._masks
        )

        return [prediction]


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


def _block(model: Any = None) -> SegmentAnything3InteractiveBlock:
    model = model if model is not None else FakeSam3Interactive()
    block = SegmentAnything3InteractiveBlock(
        models={SAM3_INTERACTIVE_DEFAULT_MODEL_ID: model}
    )

    return block


def _call(block: Any, *, wanted=None, **params: Any) -> List[Dict[str, Any]]:
    # The block asks wants() for its mask outputs: run it as one engine call.
    with block_call(block, wanted=wanted):
        results = block.run(**params)

    return results


def _masks_of(xyxy: List[List[int]], size_hw=SIZE_HW) -> torch.Tensor:
    masks = torch.stack([box_mask(tuple(box), size_hw=size_hw) for box in xyxy])

    return masks


def _params(**values: Any) -> SegmentAnything3InteractiveBlock.Params:
    params = SegmentAnything3InteractiveBlock.Params(image="$inputs.image", **values)

    return params


# --- point prompts --------------------------------------------------------------


def test_points_give_one_foreground_row_with_a_fresh_identity() -> None:
    model = FakeSam3Interactive()

    (result,) = _call(_block(model), image=Batch([_image("a")]), points=POINT)

    predictions = result["predictions"]
    assert predictions.xyxy.tolist() == [POINT_EXTENT]
    assert torch.equal(predictions.mask, _masks_of([POINT_XYXY]))
    assert predictions.class_id.tolist() == [0]
    assert predictions.confidence.tolist() == pytest.approx([POINT_SCORE])
    assert predictions.image_metadata[CLASS_NAMES_KEY] == {0: "foreground"}
    assert predictions.image_metadata["prediction_type"] == "instance-segmentation"
    assert predictions.bboxes_metadata is None
    assert row_record(predictions).namespace == result["inference_id"]
    assert frame_of(predictions).image_id == "a"


def test_negative_points_exclude_their_region() -> None:
    model = FakeSam3Interactive()

    (result,) = _call(
        _block(model), image=Batch([_image("a")]), points=POINT_AND_NEGATIVE
    )

    (call,) = model.calls
    assert call["labels"] == [[1, 0]]
    assert result["predictions"].xyxy.tolist() == [CARVED_EXTENT]


@pytest.mark.parametrize(
    "points",
    [
        [{"x": 10, "y": 8, "positive": True}, {"x": 4.5, "y": 2, "positive": False}],
        [[10, 8], [4.5, 2, False]],
        [(10, 8, 1), (4.5, 2, 0)],
        [{"x": 10, "y": 8}, [4.5, 2, 0]],
    ],
)
def test_point_forms_reach_the_model_as_one_prompt(points) -> None:
    model = FakeSam3Interactive()

    _call(_block(model), image=Batch([_image("a")]), points=points)

    (call,) = model.calls
    assert call["points"] == [[[10.0, 8.0], [4.5, 2.0]]]
    assert call["labels"] == [[1, 0]]
    assert call["dtypes"] == (torch.float64, torch.int64)


INVALID_POINTS = [
    ([{"x": 1, "y": 2, "postive": False}], "keys x, y and optionally positive"),
    ([{"x": 1}], "keys x, y and optionally positive"),
    ([[1]], "a point is an"),
    ([[1, 2, True, 4]], "a point is an"),
    (["1, 2"], "a point is an"),
    ([[1, "2"]], "coordinates are numbers"),
    ([[True, 2]], "coordinates are numbers"),
    ([[math.nan, 2]], "finite numbers"),
    ([{"x": 1, "y": math.inf}], "finite numbers"),
    ([[1, 2, "false"]], "true, false, 1 or 0"),
    ([[1, 2, 2]], "true, false, 1 or 0"),
    ([[1, 2, 1.0]], "true, false, 1 or 0"),
    ([{"x": 1, "y": 2, "positive": None}], "true, false, 1 or 0"),
]


@pytest.mark.parametrize("points, message", INVALID_POINTS)
def test_malformed_literal_points_fail_validation(points, message) -> None:
    with pytest.raises(pydantic.ValidationError, match=message):
        _params(points=points)


@pytest.mark.parametrize(
    "points, message",
    [*INVALID_POINTS, ({"x": 1, "y": 2}, "points must be a list of points")],
)
def test_malformed_selected_points_fail_before_the_model(points, message) -> None:
    model = FakeSam3Interactive()

    with pytest.raises(ValueError, match=message):
        _call(_block(model), image=Batch([_image("a")]), points=points)
    assert (model.embedded, model.calls) == ([], [])


@pytest.mark.parametrize(
    "point, image_id",
    [([31, 5], "small"), ([5, 20.5], "small"), ([-0.5, 5], "big"), ([5, -1], "big")],
)
def test_points_outside_an_image_fail_before_the_model(point, image_id) -> None:
    model = FakeSam3Interactive()
    images = Batch([_image("big", size_hw=(40, 40)), _image("small")])

    with pytest.raises(ValueError, match=f"is outside image '{image_id}' of size"):
        _call(_block(model), image=images, points=[point])
    assert (model.embedded, model.calls) == ([], [])


def test_points_on_the_image_border_are_inside() -> None:
    model = FakeSam3Interactive()

    _call(_block(model), image=Batch([_image("a")]), points=[[0, 0], [30, 20]])

    assert model.calls[0]["points"] == [[[0.0, 0.0], [30.0, 20.0]]]


# --- no prompt ------------------------------------------------------------------


@pytest.mark.parametrize("points", [None, []])
def test_a_step_without_points_or_boxes_fails_validation(points) -> None:
    with pytest.raises(pydantic.ValidationError, match="needs points, boxes or both"):
        _params(points=points)


def test_selected_points_or_wired_boxes_satisfy_validation() -> None:
    _params(points="$inputs.points")
    _params(boxes="$steps.detector.predictions")
    _params(points=[], boxes="$steps.detector.predictions")


@pytest.mark.parametrize("points", [None, []])
def test_empty_selected_points_without_boxes_fail_before_the_model(points) -> None:
    model = FakeSam3Interactive()

    with pytest.raises(ValueError, match="needs points, boxes or both"):
        _call(_block(model), image=Batch([_image("a")]), points=points)
    assert model.embedded == []


# --- box prompts ----------------------------------------------------------------


def test_box_prompts_give_one_mask_per_kept_row_with_its_lineage() -> None:
    model = FakeSam3Interactive()
    image = _image("a")
    prompt = _prompt(image)

    (result,) = _call(
        _block(model), image=Batch([image]), boxes=Batch([prompt]), confidence=0.6
    )

    (call,) = model.calls
    # Boxes reach the model as xyxy, all rows in one call.
    assert call["boxes"] == PROMPT_XYXY
    predictions = result["predictions"]
    # Row 2 scored 0.5 < 0.6 and is dropped.
    assert predictions.xyxy.tolist() == PROMPT_EXTENTS[:2]
    assert torch.equal(predictions.mask, _masks_of(PROMPT_XYXY[:2]))
    assert predictions.class_id.tolist() == [1, 0]
    assert predictions.confidence.tolist() == pytest.approx([0.9, 0.7])
    assert predictions.image_metadata[CLASS_NAMES_KEY] == PROMPT_NAMES
    assert predictions.bboxes_metadata == [
        {"detection_id": "d0"},
        {"detection_id": "d1"},
    ]
    record = row_record(predictions)
    assert (record.namespace, record.host_ids()) == ("detector", [0, 1])


@pytest.mark.parametrize("points", [None, POINT], ids=["boxes", "boxes-and-points"])
@pytest.mark.parametrize("confidence", [0.6, 0.85, 1.0])
@pytest.mark.parametrize(
    "wanted", [None, {"predictions"}, {"segmentation"}], ids=["both", "dense", "view"]
)
def test_transformed_prompt_rows_discard_host_mirrors(
    points, confidence: float, wanted
) -> None:
    image = _image("a")
    boxes = [[0, 0, 30, 20], *PROMPT_XYXY[1:]]
    prompt = _prompt(image, xyxy=boxes)
    for index, row in enumerate(prompt.bboxes_metadata):
        row.update(
            _host_xyxy=boxes[index],
            _host_class_id=PROMPT_CLASSES[index],
            _host_confidence=0.8,
            custom={"source": "detector", "index": index},
        )
    before_rows = copy.deepcopy(prompt.bboxes_metadata)
    before_metadata = dict(prompt.image_metadata)
    before_tensors = [
        tensor.clone() for tensor in (prompt.xyxy, prompt.class_id, prompt.confidence)
    ]
    before_record = (row_record(prompt).namespace, row_record(prompt).host_ids())

    (result,) = _call(
        _block(),
        wanted=wanted,
        image=Batch([image]),
        boxes=Batch([prompt]),
        points=points,
        confidence=confidence,
    )

    box_count = 2 if confidence == 0.6 else 1 if confidence == 0.85 else 0
    point_kept = points is not None and confidence <= POINT_SCORE
    # The first prompt box covers the whole image; its mask is every pixel.
    expected_occupied = [[0, 0, 29, 19], PROMPT_XYXY[1]][:box_count]
    expected_boxes = [[0, 0, 30, 20], PROMPT_EXTENTS[1]][:box_count]
    expected_classes = PROMPT_CLASSES[:box_count]
    expected_scores = [0.9, 0.7][:box_count]
    expected_rows = [
        {"detection_id": f"d{index}", "custom": {"source": "detector", "index": index}}
        for index in range(box_count)
    ]
    if point_kept:
        expected_occupied.append(POINT_XYXY)
        expected_boxes.append(POINT_EXTENT)
        expected_classes.append(3)
        expected_scores.append(POINT_SCORE)
        expected_rows.append({})
    requested = {"predictions", "segmentation"} if wanted is None else wanted
    assert set(result) == requested | {"model_id", "inference_id"}
    outputs = []
    if "predictions" in result:
        outputs.append(result["predictions"])
    if "segmentation" in result:
        outputs.append(result["segmentation"].full_res())
    for output in outputs:
        assert output.xyxy.tolist() == expected_boxes
        assert output.class_id.tolist() == expected_classes
        assert output.confidence.tolist() == pytest.approx(expected_scores)
        assert output.bboxes_metadata == expected_rows
        expected_masks = (
            _masks_of(expected_occupied)
            if expected_occupied
            else torch.empty((0, *SIZE_HW), dtype=torch.bool)
        )
        assert torch.equal(output.mask, expected_masks)
        record = row_record(output)
        namespace = "detector" if points is None else result["inference_id"]
        assert (record.namespace, record.host_ids()) == (
            namespace,
            list(range(len(expected_rows))),
        )
    if len(outputs) == 2:
        assert torch.equal(outputs[0].mask, outputs[1].mask)
    if expected_rows:
        outputs[0].bboxes_metadata[0]["detection_id"] = "changed-output"
    assert prompt.bboxes_metadata == before_rows
    assert prompt.image_metadata == before_metadata
    for tensor, before in zip(
        (prompt.xyxy, prompt.class_id, prompt.confidence), before_tensors
    ):
        assert torch.equal(tensor, before)
    assert (
        row_record(prompt).namespace,
        row_record(prompt).host_ids(),
    ) == before_record


def test_empty_box_prompt_gives_no_masks_and_calls_no_model() -> None:
    model = FakeSam3Interactive()
    image = _image("a")
    prompt = _prompt(image, xyxy=[], classes=[])

    (result,) = _call(_block(model), image=Batch([image]), boxes=Batch([prompt]))

    assert (model.embedded, model.calls) == ([], [])
    predictions = result["predictions"]
    assert (len(predictions), tuple(predictions.mask.shape)) == (0, (0, *SIZE_HW))
    assert predictions.image_metadata[CLASS_NAMES_KEY] == PROMPT_NAMES
    assert row_record(predictions).namespace == "detector"


def test_boxes_and_points_give_box_rows_then_the_point_row() -> None:
    model = FakeSam3Interactive()
    image = _image("a")

    (result,) = _call(
        _block(model),
        image=Batch([image]),
        boxes=Batch([_prompt(image)]),
        points=POINT,
        confidence=0.6,
    )

    # Two model calls for one image share its one embedding.
    assert model.embedded == [1]
    box_call, point_call = model.calls
    assert point_call["embedding"] is box_call["embedding"]
    predictions = result["predictions"]
    assert predictions.xyxy.tolist() == [*PROMPT_EXTENTS[:2], POINT_EXTENT]
    assert predictions.class_id.tolist() == [1, 0, 3]
    assert predictions.confidence.tolist() == pytest.approx([0.9, 0.7, POINT_SCORE])
    assert predictions.image_metadata[CLASS_NAMES_KEY] == NAMES_WITH_POINT
    assert predictions.bboxes_metadata == [
        {"detection_id": "d0"},
        {"detection_id": "d1"},
        {},
    ]
    # The point row is not a detector row: the result gets a fresh identity.
    record = row_record(predictions)
    assert (record.namespace, record.host_ids()) == (result["inference_id"], [0, 1, 2])


def test_point_row_class_follows_the_prompt_even_without_its_rows() -> None:
    model = FakeSam3Interactive()
    image = _image("a")
    prompt = _prompt(image, xyxy=[], classes=[])

    (result,) = _call(
        _block(model), image=Batch([image]), boxes=Batch([prompt]), points=POINT
    )

    predictions = result["predictions"]
    assert predictions.class_id.tolist() == [3]
    assert predictions.image_metadata[CLASS_NAMES_KEY] == NAMES_WITH_POINT
    assert predictions.bboxes_metadata is None
    assert row_record(predictions).namespace == result["inference_id"]


def test_point_row_class_is_past_unnamed_prompt_classes_too() -> None:
    image = _image("a")
    prompt = _prompt(image, xyxy=[[2, 3, 10, 8]], classes=[7])

    (result,) = _call(
        _block(), image=Batch([image]), boxes=Batch([prompt]), points=POINT
    )

    assert result["predictions"].class_id.tolist() == [7, 8]
    assert result["predictions"].image_metadata[CLASS_NAMES_KEY][8] == "foreground"


def test_confidence_filters_the_point_row_too() -> None:
    (result,) = _call(
        _block(), image=Batch([_image("a")]), points=POINT, confidence=0.85
    )

    predictions = result["predictions"]
    assert (len(predictions), tuple(predictions.mask.shape)) == (0, (0, *SIZE_HW))
    assert row_record(predictions).namespace == result["inference_id"]


def test_a_score_equal_to_confidence_is_kept() -> None:
    # V1 dropped only scores below its threshold.
    model = _ScriptedSam3(torch.ones((2, *SIZE_HW)), torch.tensor([0.5, 0.25]))
    image = _image("a")
    prompt = _prompt(image, xyxy=PROMPT_XYXY[:2], classes=[0, 1])

    (result,) = _call(
        _block(model), image=Batch([image]), boxes=Batch([prompt]), confidence=0.5
    )

    assert result["predictions"].confidence.tolist() == [0.5]


def test_reads_boxes_from_keypoint_predictions() -> None:
    image = _image("a")
    boxes = _prompt(image)
    key_points = KeyPoints(
        xy=torch.zeros((3, 1, 2)),
        class_id=boxes.class_id,
        confidence=torch.ones((3, 1)),
    )

    (result,) = _call(
        _block(), image=Batch([image]), boxes=Batch([(key_points, boxes)])
    )

    assert result["predictions"].xyxy.tolist() == PROMPT_EXTENTS


def test_keypoints_without_boxes_are_an_error() -> None:
    key_points = KeyPoints(
        xy=torch.zeros((1, 1, 2)),
        class_id=torch.tensor([0]),
        confidence=torch.ones((1, 1)),
    )

    with pytest.raises(ValueError, match="no boxes; SAM3 needs boxes"):
        _call(_block(), image=Batch([_image("a")]), boxes=Batch([(key_points, None)]))


def test_reads_boxes_from_a_segmentation_view() -> None:
    image = _image("a")
    boxes = _prompt(image)
    segmented = InstanceDetections(
        xyxy=boxes.xyxy,
        class_id=boxes.class_id,
        confidence=boxes.confidence,
        mask=_masks_of(PROMPT_XYXY),
        image_metadata=boxes.image_metadata,
    )

    (result,) = _call(
        _block(),
        image=Batch([image]),
        boxes=Batch([SegmentationView.from_dense(segmented)]),
    )

    assert result["predictions"].xyxy.tolist() == PROMPT_EXTENTS


@pytest.fixture(params=["other-id", "other-size"])
def prompt_image(request) -> ImageData:
    if request.param == "other-id":
        return _image("b")

    return _image("a", size_hw=(10, 30))


def test_boxes_of_another_frame_are_an_error(prompt_image: ImageData) -> None:
    model = FakeSam3Interactive()

    with pytest.raises(ValueError, match="wire boxes predicted on the image"):
        _call(
            _block(model),
            image=Batch([_image("a")]),
            boxes=Batch([_prompt(prompt_image)]),
        )
    assert model.embedded == []


def test_boxes_and_images_must_have_the_same_length() -> None:
    image = _image("a")

    with pytest.raises(ValueError, match="boxes has 2 values for 1 images"):
        _call(
            _block(),
            image=Batch([image]),
            boxes=Batch([_prompt(image), _prompt(image)]),
        )


def test_does_not_change_its_prompt() -> None:
    image = _image("a")
    prompt = _prompt(image)
    before = (prompt.xyxy.clone(), [dict(row) for row in prompt.bboxes_metadata])

    (result,) = _call(
        _block(), image=Batch([image]), boxes=Batch([prompt]), points=POINT
    )
    result["predictions"].bboxes_metadata[0]["detection_id"] = "changed"

    assert torch.equal(prompt.xyxy, before[0])
    assert prompt.bboxes_metadata == before[1]
    assert len(prompt.image_metadata[CLASS_NAMES_KEY]) == 3


def test_multimask_output_reaches_the_model() -> None:
    model = FakeSam3Interactive()
    image = _image("a")

    _call(
        _block(model),
        image=Batch([image]),
        boxes=Batch([_prompt(image)]),
        points=POINT,
        multimask_output=False,
    )

    assert [call["multimask"] for call in model.calls] == [False, False]


# --- masks and scores -----------------------------------------------------------


def test_logits_at_zero_are_mask_pixels_and_an_empty_mask_has_a_zero_box() -> None:
    logits = torch.full((2, *SIZE_HW), -1.0)
    logits[0, 4, 5] = 0.0
    model = _ScriptedSam3(logits, torch.tensor([0.9, 0.8]))
    image = _image("a")
    prompt = _prompt(image, xyxy=PROMPT_XYXY[:2], classes=[0, 1])

    (result,) = _call(_block(model), image=Batch([image]), boxes=Batch([prompt]))

    assert result["predictions"].xyxy.tolist() == [[5, 4, 6, 5], [0, 0, 0, 0]]


@pytest.mark.parametrize(
    "masks, scores",
    [
        (torch.zeros((3, *SIZE_HW)), torch.zeros(2)),
        (torch.zeros((2, *SIZE_HW)), torch.zeros(3)),
        (torch.zeros((2, 10, 30)), torch.zeros(2)),
        (torch.zeros((2, 3, *SIZE_HW)), torch.zeros((2, 3))),
        (torch.zeros((2, *SIZE_HW)), torch.zeros((2, 1))),
    ],
)
def test_masks_and_scores_must_align_with_the_prompts(masks, scores) -> None:
    image = _image("a")
    prompt = _prompt(image, xyxy=PROMPT_XYXY[:2], classes=[0, 1])

    with pytest.raises(ValueError, match="for 2 prompts on a"):
        _call(
            _block(_ScriptedSam3(masks, scores)),
            image=Batch([image]),
            boxes=Batch([prompt]),
        )


# --- options, images and outputs ------------------------------------------------


@pytest.mark.parametrize(
    "options, message",
    [
        ({"confidence": 1.5}, "confidence must be a number 0..1"),
        ({"confidence": "0.5"}, "confidence must be a number 0..1"),
        ({"confidence": True}, "confidence must be a number 0..1"),
        ({"multimask_output": "yes"}, "multimask_output must be true or false"),
    ],
)
def test_invalid_selected_options_fail_before_the_model(options, message) -> None:
    model = FakeSam3Interactive()

    with pytest.raises(ValueError, match=message):
        _call(_block(model), image=Batch([_image("a")]), points=POINT, **options)
    assert model.embedded == []


def test_needs_rgb_images() -> None:
    with pytest.raises(ValueError, match="needs a 3-channel RGB image"):
        _call(_block(), image=Batch([_image("a", channels=1)]), points=POINT)


def test_view_and_dense_predictions_hold_the_same_rows() -> None:
    image = _image("a")

    (result,) = _call(
        _block(), image=Batch([image]), boxes=Batch([_prompt(image)]), points=POINT
    )

    view, dense = result["segmentation"], result["predictions"]
    assert torch.equal(view.full_res().mask, dense.mask)
    assert torch.equal(view.detections.xyxy, dense.xyxy)
    assert torch.equal(view.detections.class_id, dense.class_id)


@pytest.mark.parametrize("wanted", [{"predictions"}, {"segmentation"}])
def test_leaves_out_an_unwanted_mask_output(wanted) -> None:
    (result,) = _call(_block(), wanted=wanted, image=Batch([_image("a")]), points=POINT)

    assert set(result) == {*wanted, "model_id", "inference_id"}


# --- embeddings -----------------------------------------------------------------


def test_reuses_the_embedding_of_the_same_pixels() -> None:
    model = FakeSam3Interactive()
    block = _block(model)
    image = _image("a")

    _call(block, image=Batch([image]), points=POINT)
    _call(block, image=Batch([image]), boxes=Batch([_prompt(image)]))

    assert model.embedded == [1]
    assert model.embed_options == [{"use_embeddings_cache": False}]
    first, second = model.calls
    assert second["embedding"] is first["embedding"]


def test_new_pixels_under_the_same_image_id_are_embedded_again() -> None:
    model = FakeSam3Interactive()
    block = _block(model)
    image = _image("a")
    brighter = image.with_pixels(torch.full((3, *SIZE_HW), 200, dtype=torch.uint8))

    _call(block, image=Batch([image]), points=POINT)
    _call(block, image=Batch([brighter]), points=POINT)

    assert model.embedded == [1, 1]
    assert model.calls[0]["embedding"] is not model.calls[1]["embedding"]


def test_keeps_only_the_last_calls_embeddings() -> None:
    model = FakeSam3Interactive()
    block = _block(model)
    a, b = _image("a"), _image("b")

    _call(block, image=Batch([a]), points=POINT)
    _call(block, image=Batch([b]), points=POINT)
    _call(block, image=Batch([a, b]), points=POINT)

    # a was dropped by the second call; b is reused in the third.
    assert model.embedded == [1, 1, 1]
    assert model.calls[3]["embedding"] is model.calls[1]["embedding"]


def test_reset_state_drops_embeddings_but_keeps_the_model() -> None:
    model = FakeSam3Interactive()
    block = _block(model)
    image = _image("a")

    _call(block, image=Batch([image]), points=POINT)
    block.reset_state()
    _call(block, image=Batch([image]), points=POINT)

    assert model.embedded == [1, 1]


def test_mixed_batch_embeds_only_images_with_prompts_in_one_call() -> None:
    model = FakeSam3Interactive()
    a, b, c = _image("a"), _image("b"), _image("c")
    prompts = [_prompt(a), _prompt(b, xyxy=[], classes=[]), _prompt(c)]

    results = _call(_block(model), image=Batch([a, b, c]), boxes=Batch(prompts))

    assert model.embedded == [2]
    assert [len(result["predictions"]) for result in results] == [3, 0, 3]


# --- local model selection ------------------------------------------------------


def test_model_and_package_ids_reach_the_loader_as_one_request() -> None:
    requests: List[ModelRequest] = []

    def load(request: ModelRequest) -> FakeSam3Interactive:
        requests.append(request)
        return FakeSam3Interactive(request)

    block = SegmentAnything3InteractiveBlock(models=ModelCache(load))
    image = _image("a")
    for _ in range(2):
        _call(
            block,
            image=Batch([image]),
            points=POINT,
            model_id="/models/sam3-pvs",
            model_package_id="pkg-1",
        )

    assert requests == [ModelRequest("/models/sam3-pvs", "pkg-1")]


def test_embeddings_are_kept_per_model_request() -> None:
    small, large = FakeSam3Interactive(), FakeSam3Interactive()
    block = SegmentAnything3InteractiveBlock(models={"sam3/a": small, "sam3/b": large})
    image = _image("a")

    _call(block, image=Batch([image]), points=POINT, model_id="sam3/a")
    _call(block, image=Batch([image]), points=POINT, model_id="sam3/b")
    _call(block, image=Batch([image]), points=POINT, model_id="sam3/b")

    assert (small.embedded, large.embedded) == ([1], [1])


def test_a_model_supplied_without_a_package_rejects_a_package() -> None:
    model = FakeSam3Interactive()

    with pytest.raises(ModelRequestConflict):
        _call(
            _block(model),
            image=Batch([_image("a")]),
            points=POINT,
            model_package_id="pkg-1",
        )
    assert model.embedded == []


def test_reports_its_model_dependency() -> None:
    params = _params(points=POINT, model_id="/models/sam3-pvs", model_package_id="p")

    (resource,) = SegmentAnything3InteractiveBlock.discover_dependent_resources(params)

    assert (resource.resource_type, resource.identifier) == (
        "model",
        "/models/sam3-pvs",
    )
    assert resource.details["model_package_id"] == "p"
    assert resource.details["task_type"] == "instance-segmentation"


def test_hosted_execution_is_deferred_and_rejected_before_any_request() -> None:
    with pytest.raises(pydantic.ValidationError, match="execution"):
        _params(points=POINT, execution="remote")


# --- compiled workflows ---------------------------------------------------------

MODES = ("run", "phases")
BRIGHT, DARK, MID = 200, 20, 100
DETECTOR_NAMES = {0: "person", 1: "car", 2: "dog"}


class _Loader:
    """``ModelCache`` loader: one fake per model id, recorded in load order."""

    def __init__(self):
        self.models: Dict[str, Any] = {}

    def __call__(self, request: ModelRequest) -> Any:
        factory = NmsDetector if request.model_id == "od" else FakeSam3Interactive
        model = self.models[request.model_id] = factory(request)

        return model


def _compile(
    steps: List[Dict[str, Any]],
    outputs: Dict[str, str],
    *,
    mode: str = "run",
    inputs: Sequence[Dict[str, Any]] = (
        {"type": "WorkflowParameter", "name": "points"},
    ),
):
    definition = {
        "version": "2.0",
        "inputs": [
            {"type": "WorkflowBatchInput", "name": "image", "kind": ["image"]},
            *inputs,
        ],
        "steps": steps,
        "outputs": [
            {"type": "JsonField", "name": name, "selector": selector}
            for name, selector in outputs.items()
        ],
    }
    catalogue = Catalogue(
        [ObjectDetectionBlock, SegmentAnything3InteractiveBlock, DynamicCropBlock],
        kinds=[*NATIVE_KINDS, INSTANCE_SEGMENTATION_VIEW_KIND],
    )
    plan = compile_workflow(
        definition, catalogue=catalogue, options=CompileOptions(block_execution=mode)
    )

    return plan


def _flat(value: int, image_id: str, *, height: int = 20, width: int = 30):
    pixels = torch.full((3, height, width), value, dtype=torch.uint8)
    image = ImageData.from_tensor(pixels, image_id=image_id)

    return image


def _images() -> List[ImageData]:
    images = [_flat(BRIGHT, "x"), _flat(DARK, "y", height=12), _flat(MID, "z")]

    return images


DETECTOR_STEP = {
    "type": "roboflow_core/roboflow_object_detection_model@v3",
    "name": "detector",
    "image": "$inputs.image",
    "model_id": "od",
    "confidence": 0.0,
}
SAM_STEP = {
    "type": "roboflow_core/sam3_interactive@v1",
    "name": "sam",
    "image": "$inputs.image",
    "boxes": "$steps.detector.predictions",
    "confidence": 0.6,
}
OUTPUTS = {
    "boxes": "$steps.detector.predictions",
    "masks": "$steps.sam.predictions",
    "view": "$steps.sam.segmentation",
}


@pytest.mark.parametrize("mode", MODES)
def test_compiled_detector_boxes_prompt_sam3_and_rows_stay_detector_rows(
    mode: str,
) -> None:
    loader = _Loader()
    plan = _compile([DETECTOR_STEP, SAM_STEP], OUTPUTS, mode=mode)
    session = plan.create_session({"models": ModelCache(loader)})

    rows = session.run({"image": _images()}).rows()

    # x: 3 rows scored 0.9/0.7/0.5 -> 2 kept; y: no rows; z: 2 rows, both kept.
    x, y, z = (row["masks"] for row in rows)
    assert x.xyxy.tolist() == [[0, 0, 30, 20], [1, 1, 30, 20]]
    assert x.class_id.tolist() == [0, 1]
    assert x.image_metadata[CLASS_NAMES_KEY] == DETECTOR_NAMES
    assert (len(y), tuple(y.mask.shape)) == (0, (0, 12, 30))
    assert z.xyxy.tolist() == [[0, 0, 30, 20], [1, 1, 30, 20]]
    for row in rows:
        assert row_record(row["masks"]).namespace == row_record(row["boxes"]).namespace
        assert torch.equal(row["view"].full_res().mask, row["masks"].mask)
    assert [frame_of(row["masks"]).image_id for row in rows] == ["x", "y", "z"]
    # Only the images with prompt rows were embedded, in one call.
    assert loader.models[SAM3_INTERACTIVE_DEFAULT_MODEL_ID].embedded == [2]


@pytest.mark.parametrize("mode", MODES)
def test_compiled_bound_points_change_per_run_and_embeddings_are_kept(
    mode: str,
) -> None:
    loader = _Loader()
    step = {**SAM_STEP, "points": "$inputs.points", "confidence": 0.0}
    plan = _compile([DETECTOR_STEP, step], OUTPUTS, mode=mode)
    session = plan.create_session({"models": ModelCache(loader)})
    images = _images()

    first = session.run({"image": images, "points": POINT}).rows()
    second = session.run({"image": images, "points": POINT_AND_NEGATIVE}).rows()

    model = loader.models[SAM3_INTERACTIVE_DEFAULT_MODEL_ID]
    # Every image has a point prompt; the same tensors are embedded once.
    assert model.embedded == [3]
    for rows, point_extent in ((first, POINT_EXTENT), (second, CARVED_EXTENT)):
        y = rows[1]["masks"]
        assert y.xyxy.tolist() == [point_extent]
        assert y.class_id.tolist() == [3]
        x = rows[0]["masks"]
        assert x.class_id.tolist() == [0, 1, 2, 3]
        assert x.image_metadata[CLASS_NAMES_KEY] == {**DETECTOR_NAMES, 3: "foreground"}


def test_compiled_literal_points_without_boxes() -> None:
    step = {key: value for key, value in SAM_STEP.items() if key != "boxes"}
    plan = _compile(
        [{**step, "points": [[10, 8]]}], {"masks": "$steps.sam.predictions"}
    )
    session = plan.create_session({"models": ModelCache(_Loader())})

    (row,) = session.run({"image": [_flat(BRIGHT, "x")]}).rows()

    assert row["masks"].xyxy.tolist() == [POINT_EXTENT]
    assert row["masks"].image_metadata[CLASS_NAMES_KEY] == {0: "foreground"}


# Occupied raster pixels, not prompt boxes, and the crop size they give.
RASTER_CASES = [
    pytest.param((4, 3, 4, 3), (1, 1), id="pixel"),
    pytest.param((2, 3, 8, 3), (1, 7), id="row"),
    pytest.param((4, 2, 4, 8), (7, 1), id="column"),
    pytest.param((7, 5, 21, 14), (10, 15), id="rectangle"),
    pytest.param((0, 0, 29, 19), (20, 30), id="border"),
    pytest.param(None, None, id="empty"),
]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("output", ["predictions", "segmentation"])
@pytest.mark.parametrize("occupied, expected_hw", RASTER_CASES)
def test_compiled_mask_boxes_keep_last_pixels_through_dynamic_crop(
    mode: str, output: str, occupied, expected_hw
) -> None:
    mask = box_mask(occupied, size_hw=SIZE_HW)
    model = _ScriptedSam3(mask[None].float() * 8.0 - 4.0, torch.tensor([0.9]))
    step = {key: value for key, value in SAM_STEP.items() if key != "boxes"}
    crop = {
        "type": DynamicCropBlock.type,
        "name": "crop",
        "image": "$inputs.image",
        "predictions": f"$steps.sam.{output}",
        "mask_opacity": 0.0,
    }
    plan = _compile(
        [{**step, "points": POINT}, crop],
        {"masks": "$steps.sam.predictions", "crops": "$steps.crop.crops"},
        mode=mode,
    )
    # Distinct pixels make losing or shifting the last row or column visible.
    pixels = torch.arange(3 * 20 * 30).reshape(3, *SIZE_HW).to(torch.uint8)
    image = ImageData.from_tensor(pixels.clone(), image_id="raster-crop")
    with plan.create_session({"models": ModelCache(lambda request: model)}) as session:
        (row,) = session.run({"image": [image]}).rows()

    predictions, crops = row["masks"], row["crops"]
    assert torch.equal(predictions.mask, mask[None])
    assert torch.equal(image.tensor_image, pixels)
    if occupied is None:
        assert predictions.xyxy.tolist() == [[0, 0, 0, 0]]
        assert crops == [None]
        return

    x1, y1, x2, y2 = occupied
    assert predictions.xyxy.tolist() == [[x1, y1, x2 + 1, y2 + 1]]
    (cropped,) = crops
    assert cropped.size_hw == expected_hw
    assert torch.equal(cropped.tensor_image, pixels[:, y1 : y2 + 1, x1 : x2 + 1])
    record = row_record(predictions)
    assert (cropped.parent_row.namespace, cropped.parent_row.id) == (
        record.namespace,
        record.host_ids()[0],
    )


def test_compiled_new_session_embeds_again() -> None:
    loader = _Loader()
    step = {key: value for key, value in SAM_STEP.items() if key != "boxes"}
    plan = _compile([{**step, "points": POINT}], {"masks": "$steps.sam.predictions"})
    cache = ModelCache(loader)
    images = [_flat(BRIGHT, "x")]

    for _ in range(2):
        session = plan.create_session({"models": cache})
        session.run({"image": images})
        session.run({"image": images})
        session.close()

    # One shared model; each session's step instance embeds once.
    assert loader.models[SAM3_INTERACTIVE_DEFAULT_MODEL_ID].embedded == [1, 1]


@pytest.mark.parametrize(
    "change",
    [
        {"boxes": None},
        {"points": [{"x": 1, "y": 2, "positive": "no"}]},
        {"execution": "remote"},
    ],
)
def test_compiled_invalid_literal_steps_fail_to_compile(change) -> None:
    step = {**SAM_STEP, **change}
    step = {key: value for key, value in step.items() if value is not None}

    with pytest.raises(ParamsValidationError):
        _compile([DETECTOR_STEP, step], {"masks": "$steps.sam.predictions"})


def test_compiled_empty_bound_points_without_boxes_fail_the_run() -> None:
    step = {key: value for key, value in SAM_STEP.items() if key != "boxes"}
    plan = _compile(
        [{**step, "points": "$inputs.points"}], {"masks": "$steps.sam.predictions"}
    )
    session = plan.create_session({"models": ModelCache(_Loader())})

    with pytest.raises(StepExecutionError, match="needs points, boxes or both"):
        session.run({"image": [_flat(BRIGHT, "x")], "points": []})


def test_compiled_per_image_points_are_rejected() -> None:
    step = {**SAM_STEP, "points": "$inputs.points"}
    inputs = (
        {"type": "WorkflowBatchInput", "name": "points", "kind": ["list_of_values"]},
    )

    with pytest.raises(LineageError, match="must be constant"):
        _compile([DETECTOR_STEP, step], OUTPUTS, inputs=inputs)


def test_compiled_reports_its_model_and_loads_it_only_at_run_time() -> None:
    loader = _Loader()
    step = {**SAM_STEP, "model_package_id": "pkg-1"}
    plan = _compile([DETECTOR_STEP, step], OUTPUTS)

    usages = {
        usage.identifier: usage for usage in discover_workload(plan).resources.items
    }
    session = plan.create_session({"models": ModelCache(loader)})

    assert set(usages) == {"od", SAM3_INTERACTIVE_DEFAULT_MODEL_ID}
    assert usages[SAM3_INTERACTIVE_DEFAULT_MODEL_ID].used_by_steps == ["$steps.sam"]
    assert loader.models == {}
    session.run({"image": [_flat(BRIGHT, "x")]})
    assert loader.models[SAM3_INTERACTIVE_DEFAULT_MODEL_ID].request == ModelRequest(
        SAM3_INTERACTIVE_DEFAULT_MODEL_ID, "pkg-1"
    )
