"""The OCR blocks in compiled V2 workflows over one ``ModelCache``.

Fake models (``tests/v2/support/models/fake_ocr_models.py``) stand in for
loaded weights; nothing is downloaded and this is not real OCR. The tests
check what a workflow builder relies on: the three blocks compile and run in
one workflow, load each model once at run time, agree between run and phase
execution (PP-OCR has no phases and always runs ``run``), read crops of a detection step with the crops' own geometry, and
reject invalid step definitions at compile time.
"""

from typing import Any, Dict, List

import pytest
import torch
from roboflow_workflows.v2.blocks.image.dynamic_crop import DynamicCropBlock
from roboflow_workflows.v2.blocks.models.cache import ModelCache, ModelRequest
from roboflow_workflows.v2.blocks.models.detection import ObjectDetectionBlock
from roboflow_workflows.v2.blocks.models.ocr import (
    EasyOCRBlock,
    OCRModelBlock,
    PPOCRBlock,
)
from roboflow_workflows.v2.data.frames import frame_of
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.masks.grid import INSTANCE_SEGMENTATION_VIEW_KIND
from roboflow_workflows.v2.data.predictions import NATIVE_KINDS
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.errors import ParamsValidationError
from roboflow_workflows.v2.engine.introspection import discover_workload
from roboflow_workflows.v2.engine.plan import CompileOptions
from tests.v2.support.models.fake_ocr_models import FakeOCRLoader
from tests.v2.support.models.fake_task_models import NmsDetector

MODES = ("run", "phases")
BRIGHT, DARK = 200, 20
OCR_OUTPUTS = ("result", "predictions", "model_id", "inference_id")


class Loader(FakeOCRLoader):
    """OCR fakes, plus the fake detector for model id ``detector``."""

    def __call__(self, request: ModelRequest) -> Any:
        if request.model_id != "detector":
            return super().__call__(request)

        self.loaded.append(request)
        model = NmsDetector(request)

        return model


def _catalogue() -> Catalogue:
    catalogue = Catalogue(
        [
            OCRModelBlock,
            EasyOCRBlock,
            PPOCRBlock,
            ObjectDetectionBlock,
            DynamicCropBlock,
        ],
        kinds=[*NATIVE_KINDS, INSTANCE_SEGMENTATION_VIEW_KIND],
    )

    return catalogue


def _compile(
    steps: List[Dict[str, Any]],
    outputs: Dict[str, str],
    *,
    mode: str,
    coordinates_system: str = "parent",
):
    definition = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowBatchInput", "name": "image", "kind": ["image"]}],
        "steps": steps,
        "outputs": [
            {
                "type": "JsonField",
                "name": name,
                "selector": selector,
                "coordinates_system": coordinates_system,
            }
            for name, selector in outputs.items()
        ],
    }
    plan = compile_workflow(
        definition, catalogue=_catalogue(), options=CompileOptions(block_execution=mode)
    )

    return plan


def _image(value: int, image_id: str, *, height: int = 40, width: int = 60):
    pixels = torch.full((3, height, width), value, dtype=torch.uint8)
    image = ImageData.from_tensor(pixels, image_id=image_id)

    return image


OCR_STEPS = [
    {"type": "v2/ocr_model@v1", "name": "doctr", "image": "$inputs.image"},
    {
        "type": "v2/easy_ocr@v1",
        "name": "easy",
        "image": "$inputs.image",
        "language": "Japanese",
    },
    {"type": "v2/pp_ocr@v1", "name": "paddle", "image": "$inputs.image"},
]
OCR_SELECTORS = {
    f"{step['name']}_{output}": f"$steps.{step['name']}.{output}"
    for step in OCR_STEPS
    for output in OCR_OUTPUTS
}


@pytest.mark.parametrize("mode", MODES)
def test_three_ocr_blocks_share_one_cache_and_load_each_model_once(mode: str) -> None:
    plan = _compile(OCR_STEPS, OCR_SELECTORS, mode=mode)
    loader = Loader()

    session = plan.create_session({"models": ModelCache(loader)})
    try:
        assert loader.loaded == []  # compiling and constructing load nothing
        images = [_image(BRIGHT, "a"), _image(DARK, "b")]
        rows = session.run({"image": images}).rows()
        again = session.run({"image": images}).rows()
    finally:
        session.close()

    assert sorted(request.model_id for request in loader.loaded) == [
        "doctr/default",
        "easy_ocr/japanese_g2",
        "pp-ocrv6-det/small",
        "pp-ocrv6-rec/small",
    ]
    bright, dark = rows
    assert (bright["doctr_result"], bright["easy_result"]) == ("w0 w1", "r0 r1 r2")
    assert bright["paddle_result"] == "line20\nline10"
    assert (bright["easy_model_id"], bright["paddle_model_id"]) == (
        "easy_ocr/japanese_g2",
        "pp-ocrv6-det/small",
    )
    assert [dark[f"{name}_result"] for name in ("doctr", "easy", "paddle")] == [""] * 3
    for row, image_id in zip(rows + again, "abab"):
        for name in ("doctr", "easy", "paddle"):
            assert frame_of(row[f"{name}_predictions"]).image_id == image_id
    inference_ids = {row["doctr_inference_id"] for row in rows + again}
    assert len(inference_ids) == 4


def test_run_and_phase_execution_read_the_same_text_and_boxes() -> None:
    by_mode, executions = {}, {}
    for mode in MODES:
        plan = _compile(OCR_STEPS, OCR_SELECTORS, mode=mode)
        executions[mode] = {step.path[-1]: step.execution for step in plan.steps}
        session = plan.create_session({"models": ModelCache(Loader())})
        try:
            by_mode[mode] = session.run({"image": [_image(BRIGHT, "a")]}).rows()
        finally:
            session.close()

    # PP-OCR has no phases, so phase execution calls its run: nothing to compare.
    assert executions["phases"] == {
        "doctr": "phases",
        "easy": "phases",
        "paddle": "run",
    }
    (run_row,), (phase_row,) = by_mode["run"], by_mode["phases"]
    for name in ("doctr", "easy"):
        assert run_row[f"{name}_result"] == phase_row[f"{name}_result"]
        run_predictions = run_row[f"{name}_predictions"]
        phase_predictions = phase_row[f"{name}_predictions"]
        assert torch.equal(run_predictions.xyxy, phase_predictions.xyxy)
        assert [
            (row["text"], row.get("class")) for row in run_predictions.bboxes_metadata
        ] == [
            (row["text"], row.get("class")) for row in phase_predictions.bboxes_metadata
        ]


@pytest.mark.parametrize("mode", MODES)
def test_ocr_reads_each_crop_of_a_detection_step(mode: str) -> None:
    steps = [
        {
            "type": "v2/object_detection@v1",
            "name": "detect",
            "image": "$inputs.image",
            "model_id": "detector",
            "confidence": 0.0,
        },
        {
            "type": "v2/dynamic_crop@v1",
            "name": "crop",
            "image": "$inputs.image",
            "predictions": "$steps.detect.predictions",
        },
        {
            "type": "v2/pp_ocr@v1",
            "name": "read",
            "image": "$steps.crop.crops",
            "text_detection": "none",
        },
    ]
    plan = _compile(
        steps,
        {"text": "$steps.read.result", "predictions": "$steps.read.predictions"},
        mode=mode,
        coordinates_system="own",
    )
    loader = Loader()

    session = plan.create_session({"models": ModelCache(loader)})
    try:
        (row,) = session.run({"image": [_image(BRIGHT, "a")]}).rows()
    finally:
        session.close()

    # Detector boxes [i, i, 60 - i, 40 - i]: crops 60, 58 and 56 pixels wide.
    assert row["text"] == ["image60", "image58", "image56"]
    # "own" output coordinates keep each crop's frame; "parent" maps to the root.
    frames = [frame_of(predictions) for predictions in row["predictions"]]
    assert [frame.size_hw for frame in frames] == [(40, 60), (38, 58), (36, 56)]
    assert [frame.parent.offset_xy for frame in frames] == [(0, 0), (1, 1), (2, 2)]
    (recognizer,) = [
        model
        for request, model in loader.models.items()
        if request.model_id.startswith("pp-ocrv6-rec/")
    ]
    assert recognizer.batch_sizes == [3]  # every crop of the image in one call


def test_workload_lists_every_model_the_steps_need() -> None:
    plan = _compile(OCR_STEPS, OCR_SELECTORS, mode="run")

    resources = discover_workload(plan).resources.items

    assert sorted(resource.identifier for resource in resources) == [
        "doctr/default",
        "easy_ocr/japanese_g2",
        "pp-ocrv6-det/small",
        "pp-ocrv6-rec/small",
    ]


@pytest.mark.parametrize(
    "step",
    [
        {"type": "v2/pp_ocr@v1", "text_detection": "none", "text_recognition": "none"},
        {"type": "v2/pp_ocr@v1", "text_detection": "large"},
        {"type": "v2/easy_ocr@v1", "language": "Klingon"},
        {"type": "v2/easy_ocr@v1", "quantize": True},
        {"type": "v2/ocr_model@v1", "model_id": ""},
    ],
)
def test_invalid_steps_fail_at_compile_time(step: Dict[str, Any]) -> None:
    steps = [{"name": "ocr", "image": "$inputs.image", **step}]

    with pytest.raises(ParamsValidationError):
        _compile(steps, {"text": "$steps.ocr.result"}, mode="run")


def _package_workflow(
    block_type: str, packages: Dict[str, Any], *, declared: bool = True
):
    return {
        "version": "2.0",
        "inputs": [
            {"type": "WorkflowBatchInput", "name": "image", "kind": ["image"]},
            *(
                [{"type": "WorkflowParameter", "name": "package", "kind": ["string"]}]
                if declared
                else []
            ),
        ],
        "steps": [
            {"type": block_type, "name": "ocr", "image": "$inputs.image", **packages}
        ],
        "outputs": [
            {"type": "JsonField", "name": "text", "selector": "$steps.ocr.result"}
        ],
    }


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "field", ["text_detection_package_id", "text_recognition_package_id"]
)
@pytest.mark.parametrize("declared", [True, False])
@pytest.mark.parametrize("value", ["$inputs.package", "$steps.read.package"])
def test_pp_ocr_selector_looking_packages_fail_compilation_before_loading(
    mode: str, field: str, declared: bool, value: str
) -> None:
    loader = Loader()
    # Compilation must fail before session creation or inference, whether or
    # not a referenced workflow input/step exists.
    with pytest.raises(ParamsValidationError) as caught:
        plan = compile_workflow(
            _package_workflow(PPOCRBlock.type, {field: value}, declared=declared),
            catalogue=_catalogue(),
            options=CompileOptions(block_execution=mode),
        )
        session = plan.create_session({"models": ModelCache(loader)})
        try:
            session.run({"image": [_image(BRIGHT, "a")], "package": "actual"})
        finally:
            session.close()
    message = str(caught.value)
    assert field in message
    assert "literal-only package IDs" in message
    assert "Supply the actual package ID as a literal string" in message
    assert value in message
    assert loader.loaded == []
    assert loader.models == {}


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("package", [None, "actual", "onnx/pkg-1", "$custom"])
def test_pp_ocr_literal_packages_compile_and_keep_exact_request_identity(
    mode: str, package: str | None
) -> None:
    loader = Loader()
    plan = compile_workflow(
        _package_workflow(
            PPOCRBlock.type,
            {
                "text_detection_package_id": package,
                "text_recognition_package_id": package,
            },
        ),
        catalogue=_catalogue(),
        options=CompileOptions(block_execution=mode),
    )
    assert loader.loaded == []
    session = plan.create_session({"models": ModelCache(loader)})
    try:
        (row,) = session.run(
            {"image": [_image(BRIGHT, "a")], "package": "unused"}
        ).rows()
    finally:
        session.close()
    assert row["text"] == "line20\nline10"
    assert loader.loaded == [
        ModelRequest("pp-ocrv6-det/small", package),
        ModelRequest("pp-ocrv6-rec/small", package),
    ]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "block_type, model_id",
    [(OCRModelBlock.type, "doctr/default"), (EasyOCRBlock.type, "easy_ocr/english_g2")],
)
def test_doctr_and_easyocr_still_resolve_package_selectors(
    mode: str, block_type: str, model_id: str
) -> None:
    loader = Loader()
    plan = compile_workflow(
        _package_workflow(block_type, {"model_package_id": "$inputs.package"}),
        catalogue=_catalogue(),
        options=CompileOptions(block_execution=mode),
    )
    session = plan.create_session({"models": ModelCache(loader)})
    try:
        session.run({"image": [_image(BRIGHT, "a")], "package": "actual"})
    finally:
        session.close()
    assert loader.loaded == [ModelRequest(model_id, "actual")]
