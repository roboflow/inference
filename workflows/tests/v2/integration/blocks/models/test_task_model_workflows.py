"""The six task-model blocks in compiled V2 workflows with a ``ModelCache``.

Fake models (``tests/v2/support/models``) stand in for loaded weights; nothing
is downloaded and this is not real inference. The tests check what a workflow
builder relies on: models load at run time once per request, one model
serves every image of a call in row order, the model may change between
runs, a per-image model selector fails compilation, run and phase execution
agree, selected options reach the model, gates skip images, and families
compose in one workflow over one cache.
"""

import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytest
import torch
from roboflow_workflows.execution_engine.entities.base import VideoMetadata
from roboflow_workflows.v2.blocks.models import BLOCKS
from roboflow_workflows.v2.blocks.models.cache import ModelCache, ModelRequest
from roboflow_workflows.v2.data.frames import frame_of
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.masks import mask_provider_of
from roboflow_workflows.v2.data.masks.grid import INSTANCE_SEGMENTATION_VIEW_KIND
from roboflow_workflows.v2.data.predictions import NATIVE_KINDS
from roboflow_workflows.v2.data.row_identity import row_record
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.errors import (
    LineageError,
    ParamsValidationError,
    StepExecutionError,
)
from roboflow_workflows.v2.engine.introspection import discover_workload
from roboflow_workflows.v2.engine.plan import CompileOptions

from tests.v2.support.compilation.fixture_blocks import Gate
from tests.v2.support.models.fake_task_models import (
    Classifier,
    FakeLoader,
    InstanceModel,
    KeypointModel,
    MultiLabelClassifier,
    NmsDetector,
    NmsFreeDetector,
    SemanticModel,
)

MODES = ("run", "phases")
DARK, MID, BRIGHT = 20, 100, 200


class Family:
    """One block type, its fake model and how to read its predictions."""

    def __init__(
        self,
        block_type: str,
        model_type: type,
        tensors: Callable[[Any], List[torch.Tensor]],
        *,
        options: Optional[Dict[str, Any]] = None,
        extra_outputs: Tuple[str, ...] = (),
    ):
        self.block_type = block_type
        self.model_type = model_type
        self.tensors = tensors
        self.options = options or {}
        self.outputs = ("predictions", "model_id", "inference_id", *extra_outputs)


def _detections(p: Any) -> List[torch.Tensor]:
    return [p.xyxy, p.class_id, p.confidence]


FAMILIES = {
    "object_detection": Family(
        "roboflow_core/roboflow_object_detection_model@v3", NmsDetector, _detections, options={"confidence": 0.0}
    ),
    "classification": Family(
        "roboflow_core/roboflow_classification_model@v3", Classifier, lambda p: [p.class_id, p.confidence]
    ),
    "multi_label": Family(
        "roboflow_core/roboflow_multi_label_classification_model@v3",
        MultiLabelClassifier,
        lambda p: [p.class_ids, p.confidence],
        options={"confidence": 0.3},
    ),
    "keypoints": Family(
        "roboflow_core/roboflow_keypoint_detection_model@v3",
        KeypointModel,
        lambda p: [p[0].xy, p[0].class_id, p[0].confidence, *_detections(p[1])],
        options={"confidence": 0.0},
    ),
    "instance_segmentation": Family(
        "roboflow_core/roboflow_instance_segmentation_model@v4",
        InstanceModel,
        lambda p: [*_detections(p), p.mask],
        options={"confidence": 0.0},
        extra_outputs=("segmentation",),
    ),
    "semantic_segmentation": Family(
        "roboflow_core/roboflow_semantic_segmentation_model@v2",
        SemanticModel,
        lambda p: [p.segmentation_map, p.confidence],
        options={"confidence": "default"},
    ),
}


def _catalogue() -> Catalogue:
    catalogue = Catalogue(
        [*BLOCKS, Gate], kinds=[*NATIVE_KINDS, INSTANCE_SEGMENTATION_VIEW_KIND]
    )

    return catalogue


def _step(family: Family, name: str, **params: Any) -> Dict[str, Any]:
    step = {
        "type": family.block_type,
        "name": name,
        "image": "$inputs.image",
        "model_id": "$inputs.model_id",
        **family.options,
        **params,
    }

    return step


def _outputs(family: Family, name: str) -> List[Dict[str, str]]:
    outputs = [
        {"type": "JsonField", "name": output, "selector": f"$steps.{name}.{output}"}
        for output in family.outputs
    ]

    return outputs


def _plan(
    family: Family,
    *,
    mode: str = "run",
    per_image_model: bool = False,
    inputs: Tuple[Dict[str, Any], ...] = (),
    gated: bool = False,
    **params: Any,
):
    definition = {
        "version": "2.0",
        "inputs": [
            {"type": "WorkflowBatchInput", "name": "image", "kind": ["image"]},
            {
                "type": (
                    "WorkflowBatchInput" if per_image_model else "WorkflowParameter"
                ),
                "name": "model_id",
                "kind": ["string"],
            },
            *inputs,
        ],
        "steps": [_step(family, "model", **params)],
        "outputs": _outputs(family, "model"),
    }
    if gated:
        definition["inputs"].append({"type": "WorkflowBatchInput", "name": "keep"})
        definition["steps"].insert(
            0,
            {
                "type": Gate.type,
                "name": "keep",
                "value": "$inputs.keep",
                "next_steps": ["$steps.model"],
            },
        )
    plan = compile_workflow(
        definition, catalogue=_catalogue(), options=CompileOptions(block_execution=mode)
    )

    return plan


def _image(value: int, image_id: str, *, height: int = 20, width: int = 30, **kw):
    pixels = torch.full((3, height, width), value, dtype=torch.uint8)
    image = ImageData.from_tensor(pixels, image_id=image_id, **kw)

    return image


def _images() -> List[ImageData]:
    images = [
        _image(BRIGHT, "x"),
        _image(DARK, "y", height=12),
        _image(MID, "z", width=16),
    ]

    return images


def _frame_ids(rows: List[Dict[str, Any]]) -> List[Optional[str]]:
    frame_ids = [
        None if row["predictions"] is None else frame_of(row["predictions"]).image_id
        for row in rows
    ]

    return frame_ids


# --- loading, grouping and order ---------------------------------------------


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", FAMILIES)
def test_selected_models_load_once_and_results_keep_row_order(
    name: str, mode: str
) -> None:
    family = FAMILIES[name]
    loader = FakeLoader(family.model_type)
    plan = _plan(family, mode=mode)

    session = plan.create_session({"models": ModelCache(loader)})
    assert loader.loaded == []  # compiling and constructing load nothing
    rows = session.run({"image": _images(), "model_id": "a"}).rows()
    other = session.run({"image": _images(), "model_id": "b"}).rows()
    again = session.run({"image": _images(), "model_id": "a"}).rows()

    assert loader.loaded == [ModelRequest("a"), ModelRequest("b")]
    first, second = loader.models
    assert (first.batch_sizes, second.batch_sizes) == ([3, 3], [3])
    assert [row["model_id"] for row in rows + other + again] == ["a"] * 3 + [
        "b"
    ] * 3 + ["a"] * 3
    assert _frame_ids(rows) == _frame_ids(other) == _frame_ids(again) == [
        "x",
        "y",
        "z",
    ]
    inference_ids = [row["inference_id"] for row in rows + other + again]
    assert len(set(inference_ids)) == 9


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", FAMILIES)
def test_a_per_image_model_selector_fails_compilation(name: str, mode: str) -> None:
    with pytest.raises(LineageError, match="model model_id is passed once per call"):
        _plan(FAMILIES[name], mode=mode, per_image_model=True)


@pytest.mark.parametrize("name", FAMILIES)
def test_run_and_phase_execution_give_identical_predictions(name: str) -> None:
    family = FAMILIES[name]
    by_mode = {}
    for mode in MODES:
        session = _plan(family, mode=mode).create_session(
            {"models": ModelCache(FakeLoader(family.model_type))}
        )
        by_mode[mode] = [
            row
            for model_id in ("a", "b")
            for row in session.run({"image": _images(), "model_id": model_id}).rows()
        ]

    for run_row, phase_row in zip(by_mode["run"], by_mode["phases"]):
        run_tensors = family.tensors(run_row["predictions"])
        phase_tensors = family.tensors(phase_row["predictions"])
        assert len(run_tensors) == len(phase_tensors)
        for run_tensor, phase_tensor in zip(run_tensors, phase_tensors):
            assert torch.equal(run_tensor, phase_tensor)
        assert run_row["model_id"] == phase_row["model_id"]
        assert run_row["inference_id"] != phase_row["inference_id"]


@pytest.mark.parametrize("mode", MODES)
def test_a_gated_image_is_skipped_and_the_others_keep_their_rows(mode: str) -> None:
    family = FAMILIES["instance_segmentation"]
    loader = FakeLoader(family.model_type)
    session = _plan(family, mode=mode, gated=True).create_session(
        {"models": ModelCache(loader)}
    )

    rows = session.run(
        {"image": _images(), "model_id": "seg", "keep": [True, False, True]}
    ).rows()

    assert loader.models[0].batch_sizes == [2]
    assert _frame_ids(rows) == ["x", None, "z"]
    lengths = [
        None if row["predictions"] is None else len(row["predictions"]) for row in rows
    ]
    assert lengths == [3, None, 2]
    assert rows[1]["segmentation"] is None


# --- selected options ---------------------------------------------------------


def test_selected_confidence_mode_and_class_filter_reach_the_model() -> None:
    family = FAMILIES["object_detection"]
    loader = FakeLoader(NmsDetector)
    plan = _plan(
        family,
        inputs=(
            {"type": "WorkflowParameter", "name": "confidence"},
            {"type": "WorkflowParameter", "name": "classes"},
        ),
        confidence="$inputs.confidence",
        class_filter="$inputs.classes",
    )
    session = plan.create_session({"models": ModelCache(loader)})

    (best,) = session.run(
        {
            "image": [_image(BRIGHT, "x")],
            "model_id": "od",
            "confidence": "best",
            "classes": ["person", "car"],
        }
    ).rows()
    (low,) = session.run(
        {
            "image": [_image(BRIGHT, "x")],
            "model_id": "od",
            "confidence": 0.0,
            "classes": ["dog"],
        }
    ).rows()

    model = loader.models[0]
    assert [r["confidence"] for r in model.received] == ["best", 0.0]
    # "best" resolves to 0.5 in the fake: rows 0..1 survive (0.9, 0.8), both kept.
    assert best["predictions"].class_id.tolist() == [0, 1]
    assert low["predictions"].class_id.tolist() == [2]


@pytest.mark.parametrize(
    "value, message",
    [("fast", "confidence must be a number 0..1 or one of"), (1.5, "confidence")],
)
def test_an_invalid_selected_confidence_fails_without_a_model_call(
    value: Any, message: str
) -> None:
    family = FAMILIES["object_detection"]
    loader = FakeLoader(NmsDetector)
    plan = _plan(
        family,
        inputs=({"type": "WorkflowParameter", "name": "confidence"},),
        confidence="$inputs.confidence",
    )
    session = plan.create_session({"models": ModelCache(loader)})

    with pytest.raises(StepExecutionError, match=message):
        session.run(
            {"image": [_image(BRIGHT, "x")], "model_id": "od", "confidence": value}
        )

    assert all(model.calls == [] for model in loader.models)


def test_an_nms_option_for_an_nms_free_model_fails_the_step() -> None:
    family = FAMILIES["object_detection"]
    loader = FakeLoader(NmsFreeDetector)
    session = _plan(family, iou_threshold=0.5).create_session(
        {"models": ModelCache(loader)}
    )

    with pytest.raises(StepExecutionError) as caught:
        session.run({"image": [_image(BRIGHT, "x")], "model_id": "rfdetr"})

    assert "does not read 'iou_threshold'" in repr(caught.value.__cause__)
    assert loader.models[0].calls == []


@pytest.mark.parametrize(
    "params",
    [
        {"confidence": "fast"},
        {"class_filter": []},
        {"max_detections": 0},
        {"iou_threshold": 1.5},
        {"max_candidates": 3000},
        {"disable_active_learning": True},
    ],
    ids=lambda params: next(iter(params)),
)
def test_invalid_or_unported_literal_options_fail_at_compile_time(
    params: Dict[str, Any],
) -> None:
    with pytest.raises(ParamsValidationError):
        _plan(FAMILIES["object_detection"], **params)


def test_unported_instance_mask_options_fail_at_compile_time() -> None:
    with pytest.raises(ParamsValidationError):
        _plan(FAMILIES["instance_segmentation"], mask_decode_mode="fast")


# --- metadata and composition -------------------------------------------------


def test_classification_keeps_the_video_frame_of_its_image() -> None:
    video = VideoMetadata(
        video_identifier="camera-1",
        frame_number=7,
        frame_timestamp=datetime.datetime(2026, 10, 13, tzinfo=datetime.timezone.utc),
        fps=10.0,
    )
    frame = _image(MID, "frame", video_metadata=video)
    session = _plan(FAMILIES["classification"]).create_session(
        {"models": {"cls": Classifier()}}
    )

    (row,) = session.run({"image": [frame], "model_id": "cls"}).rows()

    predicted = frame_of(row["predictions"])
    assert predicted.video_metadata == video
    assert predicted.image_id == "frame"


def test_families_compose_in_one_workflow_over_one_cache() -> None:
    model_types = {
        "od": NmsDetector,
        "cls": Classifier,
        "seg": InstanceModel,
        "sem": SemanticModel,
    }
    loaded: List[ModelRequest] = []

    def load(request: ModelRequest) -> Any:
        loaded.append(request)
        model = model_types[request.model_id](request)

        return model

    steps = [
        _step(FAMILIES["object_detection"], "detector", model_id="od"),
        _step(FAMILIES["classification"], "classifier", model_id="cls"),
        _step(FAMILIES["instance_segmentation"], "segmenter", model_id="seg"),
        _step(FAMILIES["semantic_segmentation"], "semantic", model_id="sem"),
    ]
    selectors = {
        "boxes": "$steps.detector.predictions",
        "label": "$steps.classifier.predictions",
        "instances": "$steps.segmenter.predictions",
        "masks": "$steps.segmenter.segmentation",
        "classes": "$steps.semantic.predictions",
    }
    definition = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowBatchInput", "name": "image", "kind": ["image"]}],
        "steps": steps,
        "outputs": [
            {"type": "JsonField", "name": name, "selector": selector}
            for name, selector in selectors.items()
        ],
    }
    session = compile_workflow(definition, catalogue=_catalogue()).create_session(
        {"models": ModelCache(load)}
    )

    rows = session.run({"image": _images()}).rows()

    assert sorted(request.model_id for request in loaded) == ["cls", "od", "seg", "sem"]
    assert [len(row["boxes"]) for row in rows] == [3, 0, 2]
    assert [row["label"].class_id.tolist() for row in rows] == [[2], [0], [1]]
    for row in rows:
        # Two independent results: two row namespaces, never shared.
        boxes, instances = row_record(row["boxes"]), row_record(row["instances"])
        assert boxes.namespace != instances.namespace
        # The view output is the instance output's rows behind the mask interface.
        masks = mask_provider_of(row["masks"]).binary(fidelity="reference")
        assert torch.equal(masks.predictions.mask, row["instances"].mask)
        assert row_record(row["masks"].detections).namespace == instances.namespace
        assert row["classes"].segmentation_map.dtype == torch.long


def test_discovery_reports_each_literal_model_with_its_task_type() -> None:
    expected = {
        "object_detection": "object-detection",
        "classification": "classification",
        "multi_label": "classification",
        "keypoints": "keypoint-detection",
        "instance_segmentation": "instance-segmentation",
        "semantic_segmentation": "semantic-segmentation",
    }
    blocks = {block.type: block for block in BLOCKS}
    for name, task_type in expected.items():
        family = FAMILIES[name]
        plan = _plan(family, model_id="literal-model", model_package_id="pkg")
        block = blocks[family.block_type]

        (usage,) = discover_workload(plan).resources.items
        (declared,) = block.discover_dependent_resources(
            block.Params(
                image="$inputs.image", model_id="literal-model", model_package_id="pkg"
            )
        )

        assert (usage.resource_type, usage.identifier) == ("model", "literal-model")
        assert usage.used_by_steps == ["$steps.model"]
        assert declared.details["task_type"] == task_type
        assert declared.details["model_package_id"] == "pkg"
