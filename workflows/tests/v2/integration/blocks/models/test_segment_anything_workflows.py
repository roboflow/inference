"""SAM2 and SAM3 in compiled V2 workflows with a ``ModelCache``.

Fake models (``tests/v2/support/models``) stand in for loaded weights; nothing
is downloaded and this is not real inference. The tests check the wiring a
workflow builder relies on: detector boxes and segmentation views prompt SAM2,
mask rows stay the detector's rows, SAM3 reads selected class names and
thresholds, run and phase execution agree, and the block keeps SAM2 image
embeddings across runs of one session.

Fake detector rows (``fake_task_models``): an image of mean brightness b has
``round(4 * b / 255)`` rows; row i is box ``[i, i, w - i, h - i]``, class
``i % 3`` of person/car/dog. FakeSam2 makes each box's best mask the box
itself, clipped to the image, scored ``0.9 - 0.2 * i``.
"""

from typing import Any, Callable, Dict, List

import pytest
import torch
from roboflow_workflows.v2.blocks.models.cache import ModelCache, ModelRequest
from roboflow_workflows.v2.blocks.models.detection import ObjectDetectionBlock
from roboflow_workflows.v2.blocks.models.instance_segmentation import (
    InstanceSegmentationBlock,
)
from roboflow_workflows.v2.blocks.models.segment_anything import (
    SAM2_DEFAULT_MODEL_ID,
    SAM3_DEFAULT_MODEL_ID,
    SegmentAnything2Block,
    SegmentAnything3Block,
)
from roboflow_workflows.v2.data.frames import frame_of
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.masks.grid import (
    INSTANCE_SEGMENTATION_VIEW_KIND,
    SegmentationView,
)
from roboflow_workflows.v2.data.prediction_ops import CLASS_NAMES_KEY
from roboflow_workflows.v2.data.predictions import NATIVE_KINDS
from roboflow_workflows.v2.data.row_identity import row_record
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.errors import (
    ParamsValidationError,
    ResolvedParameterError,
    StepExecutionError,
)
from roboflow_workflows.v2.engine.introspection import discover_workload
from roboflow_workflows.v2.engine.plan import CompileOptions

from tests.v2.support.models.fake_sam_models import FakeSam2, FakeSam3, box_mask
from tests.v2.support.models.fake_task_models import InstanceModel, NmsDetector

MODES = ("run", "phases")
DARK, MID, BRIGHT = 20, 100, 200
DETECTOR_NAMES = {0: "person", 1: "car", 2: "dog"}


class _Loader:
    """``ModelCache`` loader: one fake per model id, recorded in load order."""

    FACTORIES: Dict[str, Callable[[ModelRequest], Any]] = {
        "od": NmsDetector,
        "seg": InstanceModel,
        SAM2_DEFAULT_MODEL_ID: FakeSam2,
        SAM3_DEFAULT_MODEL_ID: FakeSam3,
    }

    def __init__(self):
        self.models: Dict[str, Any] = {}

    def __call__(self, request: ModelRequest) -> Any:
        model = self.FACTORIES[request.model_id](request)
        self.models[request.model_id] = model

        return model


def _catalogue() -> Catalogue:
    catalogue = Catalogue(
        [
            ObjectDetectionBlock,
            InstanceSegmentationBlock,
            SegmentAnything2Block,
            SegmentAnything3Block,
        ],
        kinds=[*NATIVE_KINDS, INSTANCE_SEGMENTATION_VIEW_KIND],
    )

    return catalogue


def _compile(
    steps: List[Dict[str, Any]],
    outputs: Dict[str, str],
    *,
    mode: str = "run",
    parameters: tuple = (),
):
    definition = {
        "version": "2.0",
        "inputs": [
            {"type": "WorkflowBatchInput", "name": "image", "kind": ["image"]},
            *({"type": "WorkflowParameter", "name": name} for name in parameters),
        ],
        "steps": steps,
        "outputs": [
            {"type": "JsonField", "name": name, "selector": selector}
            for name, selector in outputs.items()
        ],
    }
    plan = compile_workflow(
        definition, catalogue=_catalogue(), options=CompileOptions(block_execution=mode)
    )

    return plan


def _image(value: int, image_id: str, *, height: int = 20, width: int = 30):
    pixels = torch.full((3, height, width), value, dtype=torch.uint8)
    image = ImageData.from_tensor(pixels, image_id=image_id)

    return image


def _images() -> List[ImageData]:
    images = [
        _image(BRIGHT, "x"),
        _image(DARK, "y", height=12),
        _image(MID, "z", width=16),
    ]

    return images


def _masks_of(xyxy: List[List[int]], size_hw) -> torch.Tensor:
    masks = torch.stack([box_mask(tuple(box), size_hw=size_hw) for box in xyxy])

    return masks


DETECTOR_STEP = {
    "type": "v2/object_detection@v1",
    "name": "detector",
    "image": "$inputs.image",
    "model_id": "od",
    "confidence": 0.0,
}
SAM2_STEP = {
    "type": "v2/segment_anything@v1",
    "name": "sam",
    "image": "$inputs.image",
    "boxes": "$steps.detector.predictions",
    "confidence": 0.6,
}
SAM2_OUTPUTS = {
    "boxes": "$steps.detector.predictions",
    "masks": "$steps.sam.predictions",
    "view": "$steps.sam.segmentation",
}


# --- detector -> SAM2 ---------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_detector_boxes_prompt_sam2_and_mask_rows_stay_detector_rows(
    mode: str,
) -> None:
    loader = _Loader()
    plan = _compile([DETECTOR_STEP, SAM2_STEP], SAM2_OUTPUTS, mode=mode)
    session = plan.create_session({"models": ModelCache(loader)})

    rows = session.run({"image": _images()}).rows()

    # x: 3 rows scored 0.9/0.7/0.5 -> 2 kept; y: no rows; z: 2 rows, both kept.
    x, y, z = (row["masks"] for row in rows)
    assert x.xyxy.tolist() == [[0, 0, 29, 19], [1, 1, 29, 19]]
    assert torch.equal(x.mask, _masks_of([[0, 0, 29, 19], [1, 1, 29, 19]], (20, 30)))
    assert x.confidence.tolist() == pytest.approx([0.9, 0.7])
    assert x.class_id.tolist() == [0, 1]
    assert x.image_metadata[CLASS_NAMES_KEY] == DETECTOR_NAMES
    assert (len(y), tuple(y.mask.shape)) == (0, (0, 12, 30))
    assert z.xyxy.tolist() == [[0, 0, 15, 19], [1, 1, 15, 19]]
    for row in rows:
        detector, masks = row_record(row["boxes"]), row_record(row["masks"])
        assert masks.namespace == detector.namespace
    assert row_record(x).host_ids() == [0, 1]
    assert row_record(z).host_ids() == [0, 1]
    assert [frame_of(row["masks"]).image_id for row in rows] == ["x", "y", "z"]
    # Only the images with prompt rows were embedded, in one call.
    assert loader.models[SAM2_DEFAULT_MODEL_ID].embedded == [2]


@pytest.mark.parametrize("mode", MODES)
def test_sam2_view_output_matches_its_dense_predictions(mode: str) -> None:
    plan = _compile([DETECTOR_STEP, SAM2_STEP], SAM2_OUTPUTS, mode=mode)
    session = plan.create_session({"models": ModelCache(_Loader())})

    rows = session.run({"image": _images()}).rows()

    for row in rows:
        assert isinstance(row["view"], SegmentationView)
        assert torch.equal(row["view"].full_res().mask, row["masks"].mask)
        assert torch.equal(row["view"].detections.xyxy, row["masks"].xyxy)


def test_run_and_phase_execution_give_identical_masks() -> None:
    by_mode = {}
    for mode in MODES:
        plan = _compile([DETECTOR_STEP, SAM2_STEP], SAM2_OUTPUTS, mode=mode)
        session = plan.create_session({"models": ModelCache(_Loader())})
        by_mode[mode] = session.run({"image": _images()}).rows()

    for run_row, phase_row in zip(by_mode["run"], by_mode["phases"]):
        run_masks, phase_masks = run_row["masks"], phase_row["masks"]
        assert torch.equal(run_masks.mask, phase_masks.mask)
        assert torch.equal(run_masks.xyxy, phase_masks.xyxy)
        assert torch.equal(run_masks.class_id, phase_masks.class_id)


def test_sam2_keeps_image_embeddings_across_runs_of_one_session() -> None:
    loader = _Loader()
    plan = _compile([DETECTOR_STEP, SAM2_STEP], SAM2_OUTPUTS)
    session = plan.create_session({"models": ModelCache(loader)})
    images = _images()

    session.run({"image": images})
    session.run({"image": images})
    session.run({"image": _images()})

    # Same pixel tensors reuse their embeddings; new tensors are embedded.
    assert loader.models[SAM2_DEFAULT_MODEL_ID].embedded == [2, 2]


@pytest.mark.parametrize("mode", MODES)
def test_an_instance_segmentation_view_prompts_sam2(mode: str) -> None:
    steps = [
        {
            "type": "v2/instance_segmentation@v1",
            "name": "segmenter",
            "image": "$inputs.image",
            "model_id": "seg",
            "confidence": 0.0,
        },
        {**SAM2_STEP, "boxes": "$steps.segmenter.segmentation", "confidence": 0.0},
    ]
    outputs = {
        "segmenter": "$steps.segmenter.predictions",
        "masks": "$steps.sam.predictions",
    }
    plan = _compile(steps, outputs, mode=mode)
    session = plan.create_session({"models": ModelCache(_Loader())})

    (row,) = session.run({"image": [_image(BRIGHT, "x")]}).rows()

    masks, segmenter = row["masks"], row["segmenter"]
    assert masks.xyxy.tolist() == [[0, 0, 29, 19], [1, 1, 29, 19], [2, 2, 28, 18]]
    assert masks.class_id.tolist() == segmenter.class_id.tolist() == [0, 1, 2]
    assert row_record(masks).namespace == row_record(segmenter).namespace


def test_whole_image_sam2_without_boxes() -> None:
    step = {key: value for key, value in SAM2_STEP.items() if key != "boxes"}
    plan = _compile([step], {"masks": "$steps.sam.predictions"})
    session = plan.create_session({"models": ModelCache(_Loader())})

    (row,) = session.run({"image": [_image(BRIGHT, "x")]}).rows()

    assert row["masks"].xyxy.tolist() == [[7, 5, 21, 14]]
    assert row["masks"].image_metadata[CLASS_NAMES_KEY] == {0: "foreground"}


def test_sam2_reports_its_model_and_loads_it_only_at_run_time() -> None:
    loader = _Loader()
    plan = _compile([DETECTOR_STEP, SAM2_STEP], SAM2_OUTPUTS)

    usages = {
        usage.identifier: usage for usage in discover_workload(plan).resources.items
    }
    session = plan.create_session({"models": ModelCache(loader)})

    assert set(usages) == {"od", SAM2_DEFAULT_MODEL_ID}
    assert usages[SAM2_DEFAULT_MODEL_ID].used_by_steps == ["$steps.sam"]
    assert loader.models == {}
    session.run({"image": [_image(BRIGHT, "x")]})
    assert set(loader.models) == {"od", SAM2_DEFAULT_MODEL_ID}


# --- SAM3 ------------------------------------------------------------------------

SAM3_STEP = {
    "type": "v2/sam3@v3",
    "name": "sam3",
    "image": "$inputs.image",
    "class_names": "$inputs.classes",
    "per_class_confidence": "$inputs.thresholds",
    "class_mapping": {"car": "auto"},
    "apply_nms": False,
}


@pytest.mark.parametrize("mode", MODES)
def test_sam3_segments_selected_classes_with_their_thresholds(mode: str) -> None:
    loader = _Loader()
    plan = _compile(
        [SAM3_STEP],
        {"masks": "$steps.sam3.predictions", "view": "$steps.sam3.segmentation"},
        mode=mode,
        parameters=("classes", "thresholds"),
    )
    session = plan.create_session({"models": ModelCache(loader)})

    rows = session.run(
        {
            "image": [_image(BRIGHT, "x"), _image(MID, "z", width=16)],
            "classes": ["car", "person"],
            "thresholds": [0.8, 0.3],
        }
    ).rows()

    model = loader.models[SAM3_DEFAULT_MODEL_ID]
    assert [call["images"] for call in model.calls] == [2]
    assert model.calls[0]["prompts"] == [{"text": "car"}, {"text": "person"}]
    assert model.calls[0]["output_prob_thresh"] == 0.3
    for row in rows:
        masks = row["masks"]
        # car 0.9 passes 0.8; person 0.6 passes 0.3; empty masks are dropped.
        assert masks.class_id.tolist() == [0, 1]
        assert masks.confidence.tolist() == pytest.approx([0.9, 0.6])
        assert masks.image_metadata[CLASS_NAMES_KEY] == {0: "auto", 1: "person"}
        assert torch.equal(row["view"].full_res().mask, masks.mask)
    assert rows[0]["masks"].xyxy.tolist() == [[2, 2, 11, 9], [2, 2, 11, 9]]
    # On a 16-pixel-wide image the boxes still fit.
    assert rows[1]["masks"].xyxy.tolist() == [[2, 2, 11, 9], [2, 2, 11, 9]]
    assert [frame_of(row["masks"]).image_id for row in rows] == ["x", "z"]


def test_sam3_applies_nms_in_a_compiled_workflow() -> None:
    step = {
        **SAM3_STEP,
        "class_names": "car, person",
        "per_class_confidence": None,
        "apply_nms": True,
        "nms_iou_threshold": 0.5,
    }
    plan = _compile([step], {"masks": "$steps.sam3.predictions"})
    session = plan.create_session({"models": ModelCache(_Loader())})

    (row,) = session.run({"image": [_image(BRIGHT, "x")]}).rows()

    # car 0.7 (IoU 0.75) and person 0.6 (IoU 1) overlap car 0.9.
    assert row["masks"].class_id.tolist() == [0]
    assert row["masks"].confidence.tolist() == pytest.approx([0.9])


def test_sam3_has_no_execution_parameter() -> None:
    loader = _Loader()
    step = {**SAM3_STEP, "class_names": ["car"], "per_class_confidence": None}

    with pytest.raises(ParamsValidationError):
        _compile(
            [{**step, "execution": "remote"}], {"masks": "$steps.sam3.predictions"}
        )
    assert loader.models == {}


def test_sam3_literal_threshold_count_is_checked_at_compile_time() -> None:
    step = {
        **SAM3_STEP,
        "class_names": ["car", "person"],
        "per_class_confidence": [0.5],
    }

    with pytest.raises(ParamsValidationError):
        _compile([step], {"masks": "$steps.sam3.predictions"})


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("selected", [False, True], ids=["literal", "selector"])
def test_sam3_class_names_and_literal_thresholds_agree(
    mode: str, selected: bool
) -> None:
    step = {
        **SAM3_STEP,
        "class_names": "$inputs.classes" if selected else "car, person",
        "per_class_confidence": [0.8, 0.3],
    }
    plan = _compile(
        [step],
        {"masks": "$steps.sam3.predictions"},
        mode=mode,
        parameters=("classes",) if selected else (),
    )
    inputs = {"image": [_image(BRIGHT, "x")]}
    if selected:
        inputs["classes"] = "car, person"

    with plan.create_session({"models": ModelCache(_Loader())}) as session:
        (row,) = session.run(inputs).rows()

    assert row["masks"].class_id.tolist() == [0, 1]
    assert row["masks"].confidence.tolist() == pytest.approx([0.9, 0.6])


@pytest.mark.parametrize("mode", MODES)
def test_sam2_filtered_to_empty_rows_keep_detector_identity(mode: str) -> None:
    plan = _compile(
        [DETECTOR_STEP, {**SAM2_STEP, "confidence": 1.0}], SAM2_OUTPUTS, mode=mode
    )
    with plan.create_session({"models": ModelCache(_Loader())}) as session:
        rows = session.run({"image": _images()}).rows()

    for row in rows:
        record = row_record(row["masks"])
        assert record.namespace == row_record(row["boxes"]).namespace
        assert (record.rows, record.host_ids()) == (0, [])


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "block_type, option",
    [
        (SegmentAnything2Block.type, "confidence"),
        (SegmentAnything3Block.type, "confidence"),
        (SegmentAnything3Block.type, "nms_iou_threshold"),
    ],
)
def test_selected_scalar_threshold_bounds_are_checked_by_engine(
    mode: str, block_type: str, option: str
) -> None:
    step = {
        "type": block_type,
        "name": "sam",
        "image": "$inputs.image",
        option: "$inputs.threshold",
    }
    loader = _Loader()
    plan = _compile(
        [step],
        {"masks": "$steps.sam.predictions"},
        mode=mode,
        parameters=("threshold",),
    )

    with plan.create_session({"models": ModelCache(loader)}) as session:
        with pytest.raises(StepExecutionError, match=option) as failure:
            session.run({"image": [_image(BRIGHT, "x")], "threshold": 1.5})

    assert isinstance(failure.value.__cause__, ResolvedParameterError)
    assert loader.models == {}


@pytest.mark.parametrize("mode", MODES)
def test_selected_class_names_recheck_literal_threshold_count(mode: str) -> None:
    step = {**SAM3_STEP, "per_class_confidence": [0.5]}
    loader = _Loader()
    plan = _compile(
        [step],
        {"masks": "$steps.sam3.predictions"},
        mode=mode,
        parameters=("classes",),
    )

    with plan.create_session({"models": ModelCache(loader)}) as session:
        with pytest.raises(StepExecutionError, match="values for 2 prompts") as failure:
            session.run({"image": [_image(BRIGHT, "x")], "classes": ["car", "person"]})

    assert isinstance(failure.value.__cause__, ResolvedParameterError)
    assert loader.models == {}
