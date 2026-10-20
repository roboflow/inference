"""The DocTR, EasyOCR and PP-OCR blocks called directly on fake models.

The fakes (``tests/v2/support/models/fake_ocr_models.py``) copy each backend's
signatures and output dtypes; text and boxes are a fixed function of image
size and brightness, so every expectation is exact. The PP-OCR fakes run in
the real ``PPOCRv6Pipeline``; a test-local recognizer returns a wrong number
of line texts to check the block's count validation. This is not real OCR. Compiled workflows are in
``integration/blocks/models/test_ocr_workflows.py``.
"""

from typing import Any, Dict, List, get_args

import pytest
import torch
from inference_models.models.pp_ocrv6.pp_ocrv6_pipeline import PPOCRv6Pipeline
from pydantic import ValidationError
from roboflow_workflows.v2.blocks.models.cache import ModelCache, ModelRequest
from roboflow_workflows.v2.blocks.models.easy_ocr.v1 import (
    EASY_OCR_MODELS,
    EasyOCRBlock,
)
from roboflow_workflows.v2.blocks.models.ocr_model.v1 import OCRModelBlock
from roboflow_workflows.v2.blocks.models.pp_ocr.v1 import (
    PP_OCR_STAGE_REQUIRED,
    PPOCRBlock,
)
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.frames import frame_of
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.kinds import STRING_KIND
from roboflow_workflows.v2.data.predictions import OBJECT_DETECTION_PREDICTION_KIND
from roboflow_workflows.v2.data.row_identity import row_record
from roboflow_workflows.v2.engine.declaration import Output
from roboflow_workflows.v2.engine.phases import PhaseFailure
from tests.v2.support.models.fake_ocr_models import (
    FakeDocTR,
    FakeEasyOCR,
    FakeOCRLoader,
    FakePPDetector,
    FakePPRecognizer,
)

# Brightness -> DocTR words / EasyOCR regions / PP-OCR lines (fake_ocr_models):
# 255 -> 3 / 4 / 3, 200 -> 2 / 3 / 2, 20 -> 0 / 0 / 0.
FULL, BRIGHT, DARK = 255, 200, 20


def _image(value: int, image_id: str, *, height: int = 40, width: int = 60):
    pixels = torch.full((3, height, width), value, dtype=torch.uint8)
    image = ImageData.from_tensor(pixels, image_id=image_id)

    return image


def _batch(*images: ImageData) -> Batch:
    batch = Batch(list(images))

    return batch


def _block(block_type: type) -> tuple:
    loader = FakeOCRLoader()
    block = block_type(models=ModelCache(loader))

    return block, loader


def _texts(result: Dict[str, Any]) -> List[str]:
    texts = [row["text"] for row in result["predictions"].bboxes_metadata]

    return texts


# --- DocTR ----------------------------------------------------------------------


def test_doctr_reads_words_lines_and_blocks_per_image() -> None:
    block, loader = _block(OCRModelBlock)

    results = block.run(image=_batch(_image(BRIGHT, "a"), _image(DARK, "b")))

    assert loader.loaded == [ModelRequest("doctr/default")]
    bright, dark = results
    assert bright["result"] == "w0 w1"
    assert _texts(bright) == ["w0", "w1", "w0 w1", "w0 w1"]
    assert bright["predictions"].class_id.tolist() == [2, 2, 1, 0]
    metadata = bright["predictions"].image_metadata
    assert metadata["class_names"] == {0: "block", 1: "line", 2: "word"}
    assert metadata["prediction_type"] == "ocr"
    assert dark["result"] == "" and len(dark["predictions"]) == 0
    for result, image_id in zip(results, "ab"):
        OBJECT_DETECTION_PREDICTION_KIND.validate(result["predictions"])
        assert frame_of(result["predictions"]).image_id == image_id
        assert result["model_id"] == "doctr/default"
    # DocTR rows keep the model's class names; no row names itself.
    assert all("class" not in row for row in bright["predictions"].bboxes_metadata)


def test_doctr_runs_one_model_per_call_and_passes_no_options() -> None:
    block, loader = _block(OCRModelBlock)
    images = _batch(_image(BRIGHT, "a"), _image(FULL, "b"), _image(BRIGHT, "c"))

    first_results = block.run(image=images, model_id="doctr/default")
    second_results = block.run(image=images, model_id="doctr/other")

    first, second = loader.models.values()
    assert (first.batch_sizes, second.batch_sizes) == ([3], [3])
    assert first.calls == ["pre_process", "forward", "post_process"]
    assert first.received == [{}]
    assert [result["model_id"] for result in first_results + second_results] == [
        "doctr/default"
    ] * 3 + ["doctr/other"] * 3
    assert [result["result"] for result in first_results] == [
        "w0 w1",
        "w0 w1 w2",
        "w0 w1",
    ]


def test_doctr_rejects_per_image_model_ids_before_any_load() -> None:
    block, loader = _block(OCRModelBlock)

    with pytest.raises(PhaseFailure, match="model_id is one value for every image"):
        block.run(
            image=_batch(_image(BRIGHT, "a"), _image(FULL, "b")),
            model_id=Batch(["doctr/default", "doctr/other"]),
        )

    assert not loader.models


def test_every_image_gets_fresh_row_identity() -> None:
    block, _ = _block(OCRModelBlock)

    results = block.run(image=_batch(_image(BRIGHT, "a"), _image(BRIGHT, "b")))

    inference_ids = [result["inference_id"] for result in results]
    assert len(set(inference_ids)) == 2
    for result in results:
        record = row_record(result["predictions"])
        assert record.namespace == result["inference_id"]
        assert record.rows == len(result["predictions"]) == 4


@pytest.mark.parametrize(
    "block_type, model_param",
    [(OCRModelBlock, "doctr/default"), (EasyOCRBlock, "English")],
)
def test_phases_compose_like_run(block_type: type, model_param: str) -> None:
    # EasyOCR's row naming and integer class ids run in its result phase.
    block, _ = _block(block_type)
    images = _batch(_image(FULL, "a"), _image(DARK, "b"))

    network_input = block.network_input(images, model_param, None)
    raw_output = block.raw_output(network_input)
    phased = block.result(raw_output, network_input, images)
    ran = block.run(image=images)

    for phase_result, run_result in zip(phased, ran):
        phase_predictions = phase_result["predictions"]
        run_predictions = run_result["predictions"]
        assert phase_result["result"] == run_result["result"]
        assert torch.equal(phase_predictions.xyxy, run_predictions.xyxy)
        assert torch.equal(phase_predictions.class_id, run_predictions.class_id)
        assert [
            (row["text"], row.get("class")) for row in phase_predictions.bboxes_metadata
        ] == [
            (row["text"], row.get("class")) for row in run_predictions.bboxes_metadata
        ]


def test_a_non_rgb_image_fails_before_any_model_runs() -> None:
    block, loader = _block(OCRModelBlock)
    gray = ImageData.from_tensor(torch.zeros((1, 8, 8), dtype=torch.uint8))

    with pytest.raises(PhaseFailure) as failure:
        block.run(image=_batch(_image(BRIGHT, "a"), gray))

    assert "3-channel RGB" in str(failure.value.__cause__)
    assert loader.loaded == []


def test_a_mapping_resource_serves_only_what_it_holds() -> None:
    block = OCRModelBlock(models={"doctr/default": FakeDocTR()})

    with pytest.raises(PhaseFailure) as failure:
        block.run(image=_batch(_image(BRIGHT, "a")), model_id="doctr/other")

    assert isinstance(failure.value.__cause__, LookupError)


def test_a_model_returning_too_few_results_fails() -> None:
    class Short(FakeDocTR):
        def post_process(self, *args: Any, **kwargs: Any):
            texts, batch = super().post_process(*args, **kwargs)
            return texts[:1], batch[:1]

    block = OCRModelBlock(models={"doctr/default": Short()})

    with pytest.raises(PhaseFailure, match="result"):
        block.run(image=_batch(_image(BRIGHT, "a"), _image(BRIGHT, "b")))


def test_doctr_dependency_names_the_model_and_package() -> None:
    params = OCRModelBlock.Params(image="$inputs.image")
    packaged = OCRModelBlock.Params(
        image="$inputs.image",
        model_package_id="onnx",
    )

    (resource,) = OCRModelBlock.discover_dependent_resources(params)
    (with_package,) = OCRModelBlock.discover_dependent_resources(packaged)

    assert resource.identifier == "doctr/default"
    assert resource.details["model_package_id"] is None
    assert with_package.details["model_package_id"] == "onnx"


# --- EasyOCR --------------------------------------------------------------------


def test_easy_ocr_keeps_every_region_and_names_rows_by_text() -> None:
    block, loader = _block(EasyOCRBlock)

    (result,) = block.run(image=_batch(_image(FULL, "a")))

    # The fake's own default (0.25) would drop "r3" (0.1); the block passes 0.0.
    (model,) = loader.models.values()
    assert model.received == [{"confidence": 0.0}]
    assert result["result"] == "r0 r1 r2 r3"
    rows = result["predictions"].bboxes_metadata
    assert [row["class"] for row in rows] == ["r0", "r1", "r2", "r3"]
    assert result["predictions"].image_metadata["class_names"] == {0: "text-region"}
    OBJECT_DETECTION_PREDICTION_KIND.validate(result["predictions"])


def test_easy_ocr_image_without_text_is_a_valid_prediction() -> None:
    # EasyOCRTorch gives float32 class ids when no region is kept.
    model = FakeEasyOCR()
    arrays, dimensions = model.pre_process(
        [torch.full((3, 8, 8), DARK, dtype=torch.uint8)], input_color_format="rgb"
    )
    (raw,) = model.post_process(model.forward(arrays), dimensions)[1]
    assert raw.class_id.dtype == torch.float32
    block, _ = _block(EasyOCRBlock)

    (result,) = block.run(image=_batch(_image(DARK, "a")))

    assert result["result"] == ""
    assert result["predictions"].class_id.dtype == torch.int64
    OBJECT_DETECTION_PREDICTION_KIND.validate(result["predictions"])


@pytest.mark.parametrize("language", list(EASY_OCR_MODELS))
def test_easy_ocr_language_selects_its_model(language: str) -> None:
    block, loader = _block(EasyOCRBlock)
    params = EasyOCRBlock.Params(image="$inputs.image", language=language)

    (result,) = block.run(image=_batch(_image(BRIGHT, "a")), language=language)
    (resource,) = EasyOCRBlock.discover_dependent_resources(params)

    assert loader.loaded == [ModelRequest(EASY_OCR_MODELS[language])]
    assert result["model_id"] == resource.identifier == EASY_OCR_MODELS[language]


def test_easy_ocr_languages_are_exactly_the_mapped_models() -> None:
    languages = EasyOCRBlock.Params.model_fields["language"].annotation

    assert list(get_args(languages)) == list(EASY_OCR_MODELS)


def test_easy_ocr_rejects_unknown_languages_and_v1_quantize() -> None:
    base = {"image": "$inputs.image"}

    with pytest.raises(ValidationError):
        EasyOCRBlock.Params(**base, language="Klingon")
    with pytest.raises(ValidationError):
        EasyOCRBlock.Params(**base, model_package_id="")
    with pytest.raises(ValidationError):
        EasyOCRBlock.Params(**base, quantize=True)


# --- PP-OCR ---------------------------------------------------------------------


def test_pp_ocr_reads_lines_in_reading_order() -> None:
    block, loader = _block(PPOCRBlock)

    bright, dark = block.run(image=_batch(_image(BRIGHT, "a"), _image(DARK, "b")))

    assert loader.loaded == [
        ModelRequest("pp-ocrv6-det/small"),
        ModelRequest("pp-ocrv6-rec/small"),
    ]
    # The fake detector emits the bottom line first; the pipeline reorders.
    assert bright["result"] == "line20\nline10"
    assert _texts(bright) == ["line20", "line10"]
    rows = bright["predictions"].bboxes_metadata
    assert [row["class"] for row in rows] == ["line20", "line10"]
    assert rows[0]["polygon"] == [[2.0, 2.0], [22.0, 2.0], [22.0, 8.0], [2.0, 8.0]]
    assert bright["predictions"].xyxy.tolist() == [
        [2.0, 2.0, 22.0, 8.0],
        [2.0, 10.0, 12.0, 16.0],
    ]
    assert bright["model_id"] == "pp-ocrv6-det/small"
    assert dark["result"] == "" and len(dark["predictions"]) == 0
    for result in (bright, dark):
        OBJECT_DETECTION_PREDICTION_KIND.validate(result["predictions"])
        metadata = result["predictions"].image_metadata
        assert metadata["class_names"] == {0: "text-region"}
        assert metadata["prediction_type"] == "ocr"


def test_pp_ocr_detects_per_batch_and_recognizes_per_image() -> None:
    block, loader = _block(PPOCRBlock)
    images = _batch(_image(BRIGHT, "a"), _image(DARK, "b"), _image(FULL, "c"))

    block.run(image=images)

    detector, recognizer = loader.models.values()
    assert detector.batch_sizes == [3]  # one call for the whole batch
    # One call per image with lines, over its crops: "a" 2 lines, "c" 3 lines;
    # "b" has no lines and no call.
    assert recognizer.batch_sizes == [2, 3]


def test_pp_ocr_model_id_names_the_detector_else_the_recognizer() -> None:
    block, _ = _block(PPOCRBlock)
    images = _batch(_image(BRIGHT, "a"))

    (both,) = block.run(image=images, text_detection="tiny", text_recognition="medium")
    (detection_only,) = block.run(image=images, text_recognition="none")
    (recognition_only,) = block.run(image=images, text_detection="none")

    assert both["model_id"] == "pp-ocrv6-det/tiny"
    assert detection_only["model_id"] == "pp-ocrv6-det/small"
    assert recognition_only["model_id"] == "pp-ocrv6-rec/small"
    description = PPOCRBlock.outputs["model_id"].description
    assert "detection model" in description
    assert "recognition model's id when text detection is 'none'" in description
    model_id_output = Output(
        STRING_KIND, source="image", description="Model id that served this image."
    )
    for block_type in (OCRModelBlock, EasyOCRBlock):
        assert block_type.outputs["model_id"] == model_id_output


def test_pp_ocr_recognition_only_reads_each_image_as_one_line() -> None:
    block, loader = _block(PPOCRBlock)
    images = _batch(_image(BRIGHT, "a"), _image(BRIGHT, "b", width=30))

    results = block.run(image=images, text_detection="none", text_recognition="tiny")

    assert loader.loaded == [ModelRequest("pp-ocrv6-rec/tiny")]
    (recognizer,) = loader.models.values()
    assert recognizer.received == [{"input_color_format": "rgb"}]
    assert [result["result"] for result in results] == ["image60", "image30"]
    assert [len(result["predictions"]) for result in results] == [0, 0]
    assert results[0]["model_id"] == "pp-ocrv6-rec/tiny"
    OBJECT_DETECTION_PREDICTION_KIND.validate(results[0]["predictions"])


def test_pp_ocr_detection_only_returns_unnamed_boxes() -> None:
    block, loader = _block(PPOCRBlock)

    with_lines, without_lines = block.run(
        image=_batch(_image(BRIGHT, "a"), _image(DARK, "b")),
        text_detection="medium",
        text_recognition="none",
    )

    assert loader.loaded == [ModelRequest("pp-ocrv6-det/medium")]
    assert with_lines["result"] == ""
    rows = with_lines["predictions"].bboxes_metadata
    assert [(row["text"], row["class"]) for row in rows] == [("", ""), ("", "")]
    assert all("polygon" in row for row in rows)
    assert without_lines["result"] == "" and len(without_lines["predictions"]) == 0
    for result in (with_lines, without_lines):
        OBJECT_DETECTION_PREDICTION_KIND.validate(result["predictions"])


class _MiscountingRecognizer(FakePPRecognizer):
    """Recognizer that returns ``extra`` more (or fewer) texts than crops."""

    def __init__(self, extra: int):
        super().__init__()
        self._extra = extra

    def __call__(self, images: Any, **kwargs: Any) -> List[str]:
        texts = super().__call__(images, **kwargs)
        if self._extra < 0:
            return texts[: self._extra]

        return texts + ["extra"] * self._extra


@pytest.mark.parametrize(
    "extra, returned", [(-1, 1), (-2, 0), (1, 3)], ids=["short", "none", "long"]
)
def test_pp_ocr_fails_when_recognized_texts_do_not_match_lines(
    extra: int, returned: int
) -> None:
    # Recognition ran, so even an empty text list is a mismatch, not "disabled".
    block = PPOCRBlock(
        models={
            "pp-ocrv6-det/small": FakePPDetector(),
            "pp-ocrv6-rec/small": _MiscountingRecognizer(extra),
        }
    )

    with pytest.raises(
        ValueError, match=f"returned {returned} line texts for 2 detected lines"
    ):
        block.run(image=_batch(_image(BRIGHT, "a")))


def test_pp_ocr_package_ids_reach_each_request() -> None:
    block, loader = _block(PPOCRBlock)

    block.run(
        image=_batch(_image(BRIGHT, "a")),
        text_detection_package_id="det-onnx",
        text_recognition_package_id="rec-onnx",
    )

    assert loader.loaded == [
        ModelRequest("pp-ocrv6-det/small", "det-onnx"),
        ModelRequest("pp-ocrv6-rec/small", "rec-onnx"),
    ]


def test_pp_ocr_supplied_models_compose_without_a_loader() -> None:
    detector, recognizer = FakePPDetector(), FakePPRecognizer()
    block = PPOCRBlock(
        models={"pp-ocrv6-det/small": detector, "pp-ocrv6-rec/small": recognizer}
    )

    (result,) = block.run(image=_batch(_image(BRIGHT, "a")))

    assert result["result"] == "line20\nline10"
    assert recognizer.batch_sizes == [2]  # one call over the image's 2 line crops


@pytest.mark.parametrize(
    "stages, expected",
    [
        ({}, ["pp-ocrv6-det/small", "pp-ocrv6-rec/small"]),
        ({"text_detection": "none"}, ["pp-ocrv6-rec/small"]),
        ({"text_recognition": "none", "text_detection": "tiny"}, ["pp-ocrv6-det/tiny"]),
    ],
)
def test_pp_ocr_dependencies_are_the_enabled_stages(
    stages: Dict[str, str], expected: List[str]
) -> None:
    params = PPOCRBlock.Params(image="$inputs.image", **stages)

    resources = PPOCRBlock.discover_dependent_resources(params)

    assert [resource.identifier for resource in resources] == expected


def test_pp_ocr_needs_at_least_one_stage() -> None:
    block, loader = _block(PPOCRBlock)

    with pytest.raises(ValidationError, match=PP_OCR_STAGE_REQUIRED):
        PPOCRBlock.Params(
            image="$inputs.image",
            text_detection="none",
            text_recognition="none",
        )
    with pytest.raises(ValueError, match=PP_OCR_STAGE_REQUIRED):
        block.run(
            image=_batch(_image(BRIGHT, "a")),
            text_detection="none",
            text_recognition="none",
        )
    assert loader.loaded == []


@pytest.mark.parametrize(
    "field", ["text_detection_package_id", "text_recognition_package_id"]
)
def test_pp_ocr_package_ids_are_non_empty_literals(field: str) -> None:
    params = PPOCRBlock.Params(image="$inputs.image", **{field: "onnx"})

    assert getattr(params, field) == "onnx"
    with pytest.raises(ValidationError, match="must be a non-empty string"):
        PPOCRBlock.Params(image="$inputs.image", **{field: ""})
    with pytest.raises(ValidationError):
        PPOCRBlock.Params(image="$inputs.image", **{field: 7})


def test_pp_ocr_rejects_a_non_rgb_image_before_loading() -> None:
    block, loader = _block(PPOCRBlock)
    gray = ImageData.from_tensor(torch.zeros((1, 8, 8), dtype=torch.uint8))

    with pytest.raises(ValueError, match="3-channel RGB"):
        block.run(image=_batch(gray))

    assert loader.loaded == []


class _RefusingLoader:
    """Records and refuses every request: no PP-OCR model may load."""

    def __init__(self) -> None:
        self.requests: List[ModelRequest] = []

    def __call__(self, request: ModelRequest) -> Any:
        self.requests.append(request)
        raise AssertionError(f"model {request} was resolved")


@pytest.fixture
def pipelines(monkeypatch) -> List[Any]:
    """Every PP-OCR pipeline the block builds; building one is refused."""
    built: List[Any] = []

    def refuse(models: Any) -> Any:
        built.append(models)
        raise AssertionError("a PP-OCR pipeline was built")

    monkeypatch.setattr(PPOCRv6Pipeline, "with_models", refuse)

    return built


@pytest.mark.parametrize(
    "field, values",
    [
        ("text_detection", ["small", "small"]),
        ("text_recognition", ["small", "small"]),
        ("text_detection_package_id", ["onnx", "onnx"]),
        ("text_recognition_package_id", ["onnx", "onnx"]),
    ],
)
@pytest.mark.parametrize("images", [2, 0], ids=["images", "empty-batch"])
def test_pp_ocr_rejects_a_batch_setting_before_any_load(
    field: str, values: List[str], images: int, pipelines: List[Any]
) -> None:
    loader = _RefusingLoader()
    block = PPOCRBlock(models=ModelCache(loader))
    image = _batch(_image(BRIGHT, "a"), _image(FULL, "b"))
    if not images:
        image = _batch()

    with pytest.raises(ValueError, match=f"{field} is one value for every image"):
        block.run(image=image, **{field: Batch(values)})

    assert loader.requests == [] and pipelines == []


@pytest.mark.parametrize(
    "settings",
    [
        {},
        {"text_detection": "none"},
        {"text_recognition": "none"},
        {"text_detection_package_id": "onnx", "text_recognition_package_id": "trt"},
    ],
    ids=["both", "recognition-only", "detection-only", "packages"],
)
def test_pp_ocr_an_empty_batch_loads_no_model_and_builds_no_pipeline(
    settings: Dict[str, str], pipelines: List[Any]
) -> None:
    loader = _RefusingLoader()
    block = PPOCRBlock(models=ModelCache(loader))

    assert block.run(image=_batch(), **settings) == []
    assert loader.requests == [] and pipelines == []


def test_pp_ocr_an_empty_batch_still_needs_a_stage() -> None:
    loader = _RefusingLoader()
    block = PPOCRBlock(models=ModelCache(loader))

    with pytest.raises(ValueError, match=PP_OCR_STAGE_REQUIRED):
        block.run(image=_batch(), text_detection="none", text_recognition="none")

    assert loader.requests == []


@pytest.mark.parametrize(
    "field", ["text_detection_package_id", "text_recognition_package_id"]
)
@pytest.mark.parametrize(
    "value",
    [
        "$inputs.package",
        "$inputs.",
        "$steps.read.package",
        "$steps.read",
        "$sources.camera.package",
        "$operators.choose.package",
    ],
)
def test_pp_ocr_package_fields_reject_workflow_selector_prefixes(
    field: str, value: str
) -> None:
    with pytest.raises(ValidationError) as caught:
        PPOCRBlock.Params(image="$inputs.image", **{field: value})
    message = str(caught.value)
    assert field in message
    assert "literal-only package IDs" in message
    assert "Supply the actual package ID as a literal string" in message
    assert value in message


@pytest.mark.parametrize(
    "field", ["text_detection_package_id", "text_recognition_package_id"]
)
@pytest.mark.parametrize(
    "value",
    [
        None,
        "actual",
        "onnx/pkg-1",
        "$custom",
        "price$5",
        "prefix-$inputs.package",
        "$inputs-package",
    ],
)
def test_pp_ocr_preserves_ordinary_literal_package_ids(
    field: str, value: str | None
) -> None:
    params = PPOCRBlock.Params(image="$inputs.image", **{field: value})
    assert getattr(params, field) == value
    resources = PPOCRBlock.discover_dependent_resources(params)
    index = 0 if field == "text_detection_package_id" else 1
    assert resources[index].details["model_package_id"] == value
