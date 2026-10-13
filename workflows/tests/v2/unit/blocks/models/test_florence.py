"""Florence task parity and real compiled workflows; injected local models only."""

import ast
import inspect
import json
import os
from pathlib import Path

import pytest
import torch
from pydantic import Field
from roboflow_workflows.v2.blocks.formatters.json import JsonParserBlock
from roboflow_workflows.v2.blocks.models.cache import (
    ModelCache,
    ModelRequest,
    ModelRequestConflict,
)
from roboflow_workflows.v2.blocks.models.florence import (
    BLOCKS,
    TASK_TOKENS,
    Florence2Block,
)
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.kinds import STRING_KIND
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref
from roboflow_workflows.v2.engine.errors import (
    ParamsValidationError,
    WorkflowExecutionError,
)
from roboflow_workflows.v2.engine.plan import CompileOptions

from inference_models.configuration import (
    INFERENCE_MODELS_FLORENCE2_DEFAULT_DO_SAMPLE,
    INFERENCE_MODELS_FLORENCE2_DEFAULT_MAX_NEW_TOKENS,
    INFERENCE_MODELS_FLORENCE2_DEFAULT_NUM_BEAMS,
)
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections
from tests.v2.support.paths import CHECKOUT_ROOT, WORKFLOWS_ROOT


class FakeFlorence:
    """Expose the pinned Florence2HF.prompt keyword surface without weights."""

    def __init__(self, answer=None):
        self.calls = []
        self.answer = answer

    def prompt(
        self,
        images,
        prompt,
        max_new_tokens=INFERENCE_MODELS_FLORENCE2_DEFAULT_MAX_NEW_TOKENS,
        num_beams=INFERENCE_MODELS_FLORENCE2_DEFAULT_NUM_BEAMS,
        do_sample=INFERENCE_MODELS_FLORENCE2_DEFAULT_DO_SAMPLE,
        skip_special_tokens=False,
        task=None,
        input_color_format=None,
        pre_processing_overrides=None,
        **kwargs,
    ):
        self.calls.append(
            dict(
                images=images,
                prompt=prompt,
                task=task,
                input_color_format=input_color_format,
                kwargs=kwargs,
            )
        )
        assert len(images) == 1
        if self.answer is not None:
            return self.answer
        if task in {
            "<OD>",
            "<DENSE_REGION_CAPTION>",
            "<OPEN_VOCABULARY_DETECTION>",
            "<CAPTION_TO_PHRASE_GROUNDING>",
            "<REGION_PROPOSAL>",
        }:
            decoded = {"bboxes": [[1, 2, 5, 6]], "labels": ["cat"]}
        elif task == "<OCR_WITH_REGION>":
            decoded = {"quad_boxes": [[1, 2, 5, 2, 5, 6, 1, 6]], "labels": ["hello"]}
        elif task in {
            "<REGION_TO_SEGMENTATION>",
            "<REFERRING_EXPRESSION_SEGMENTATION>",
        }:
            decoded = {"polygons": [[[1, 2, 5, 2, 5, 6]]], "labels": ["cat"]}
        else:
            decoded = f"answer-{int(images[0][0, 0, 0])}"
        return [{task: decoded}]


class Loader:
    def __init__(self):
        self.loaded = []
        self.models = {}

    def __call__(self, request):
        self.loaded.append(request)
        self.models[request] = FakeFlorence()
        return self.models[request]


def _images():
    return [
        ImageData.from_tensor(
            torch.full((3, 20, 40), pixel, dtype=torch.uint8), image_id=f"img-{pixel}"
        )
        for pixel in (10, 30, 90)
    ]


def _detections(boxes=((0, 0, 10, 10), (10, 2, 30, 12)), *, frame=None):
    return Detections(
        xyxy=torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        class_id=torch.zeros(len(boxes), dtype=torch.int64),
        confidence=torch.tensor([0.9, 0.2][: len(boxes)]),
        image_metadata=None if frame is None else frame.prediction_metadata(),
    )


class _Shout(Block):
    """A string consumer, standing in for any step input of kind string."""

    type = "test/shout@v1"
    outputs = {"value": Output(STRING_KIND, source="text")}

    class Params(BlockParams):
        text: Ref(STRING_KIND) = Field(description="Plain text to upper-case.")

    def run(self, *, text):
        return {"value": text.upper()}


def _plan(*, mode="run", inputs=(), steps=(), outputs=(), **params):
    definition = {
        "version": "2.0",
        "inputs": [
            {"type": "WorkflowBatchInput", "name": "image", "kind": ["image"]},
            *inputs,
        ],
        "steps": [
            {
                "type": Florence2Block.type,
                "name": "florence",
                "image": "$inputs.image",
                **params,
            },
            *steps,
        ],
        "outputs": [
            {"type": "JsonField", "name": name, "selector": f"$steps.florence.{name}"}
            for name in Florence2Block.outputs
        ]
        + list(outputs),
    }
    return compile_workflow(
        definition,
        catalogue=Catalogue([*BLOCKS, JsonParserBlock, _Shout]),
        options=CompileOptions(block_execution=mode),
    )


def _task_params(task):
    if task.startswith("detection-grounded-"):
        return {"grounding_detection": [0, 0, 20, 10]}
    if task.startswith("phrase-grounded-") or task == "custom":
        return {"prompt": "the cat" if task != "custom" else "<CUSTOM>question"}
    if task == "open-vocabulary-object-detection":
        return {"classes": ["cat", "dog"]}
    return {}


@pytest.mark.parametrize("mode", ["run", "phases"])
@pytest.mark.parametrize("task", TASK_TOKENS)
def test_every_v1_task_compiles_and_sends_the_exact_prompt(task, mode):
    fake = FakeFlorence()
    rows = (
        _plan(mode=mode, model_id="m", task_type=task, **_task_params(task))
        .create_session({"models": {"m": fake}})
        .run({"image": _images()[:1]})
        .rows()
    )
    call = fake.calls[0]
    suffix = (
        "<loc_0><loc_0><loc_500><loc_500>"
        if task.startswith("detection-grounded-")
        else (
            "<and>".join(["cat", "dog"])
            if task == "open-vocabulary-object-detection"
            else _task_params(task).get("prompt", "")
        )
    )
    assert call["prompt"] == (TASK_TOKENS[task] or "") + suffix
    assert call["task"] == (TASK_TOKENS[task] or "<CUSTOM>")
    assert call["input_color_format"] == "rgb"
    assert call["kwargs"] == {}
    assert len(rows) == 1 and rows[0]["model_id"] == "m"
    decoded = json.loads(rows[0]["raw_output"])
    assert rows[0]["parsed_output"] == (decoded if isinstance(decoded, dict) else None)
    if TASK_TOKENS[task] in {
        "<OD>",
        "<DENSE_REGION_CAPTION>",
        "<CAPTION_TO_PHRASE_GROUNDING>",
    }:
        assert rows[0]["classes"] == ["cat"]
    elif task == "ocr-with-text-detection":
        assert rows[0]["classes"] == ["hello"]
    else:
        assert rows[0]["classes"] == _task_params(task).get("classes", [])


@pytest.mark.parametrize("mode", ["run", "phases"])
def test_selected_tasks_prompts_classes_boxes_and_packages_keep_order_and_cache(mode):
    loader = Loader()
    names = ("task", "prompt", "classes", "box", "model", "package", "selection")
    kinds = (
        "string",
        "string",
        "list_of_values",
        "list_of_values",
        "string",
        "string",
        "string",
    )
    inputs = tuple(
        {"type": "WorkflowBatchInput", "name": name, "kind": [kind]}
        for name, kind in zip(names, kinds)
    )
    session = _plan(
        mode=mode,
        inputs=inputs,
        task_type="$inputs.task",
        prompt="$inputs.prompt",
        classes="$inputs.classes",
        grounding_detection="$inputs.box",
        model_id="$inputs.model",
        model_package_id="$inputs.package",
        grounding_selection_mode="$inputs.selection",
    ).create_session({"models": ModelCache(loader)})
    images = _images()
    values = dict(
        image=images,
        task=["caption", "open-vocabulary-object-detection", "detection-grounded-ocr"],
        prompt=[None, None, None],
        classes=[None, ["cat"], None],
        box=[None, None, [0.0, 0.0, 0.5, 0.5]],
        model=["m", "other", "m"],
        package=["p", None, "p"],
        selection=["first"] * 3,
    )
    rows = session.run(values).rows()
    again = session.run(values).rows()
    assert loader.loaded == [ModelRequest("m", "p"), ModelRequest("other")]
    assert [row["model_id"] for row in rows] == ["m", "other", "m"]
    assert json.loads(rows[0]["raw_output"]) == "answer-10"
    assert json.loads(rows[2]["raw_output"]) == "answer-90"
    calls = loader.models[ModelRequest("m", "p")].calls
    assert calls[0]["images"][0] is images[0].tensor_image
    assert calls[1]["images"][0] is images[2].tensor_image
    assert calls[1]["prompt"] == "<REGION_TO_OCR><loc_0><loc_0><loc_500><loc_500>"
    assert len({row["inference_id"] for row in rows + again}) == 6


@pytest.mark.parametrize("mode", ["run", "phases"])
def test_source_to_florence_to_json_parser_and_frame_context(mode):
    images = _images()
    session = _plan(
        mode=mode,
        model_id="m",
        task_type="object-detection",
        steps=(
            {
                "type": JsonParserBlock.type,
                "name": "parse",
                "raw_json": "$steps.florence.raw_output",
                "expected_fields": ["labels"],
            },
        ),
        outputs=(
            {
                "type": "JsonField",
                "name": "decoded_labels",
                "selector": "$steps.parse.labels",
            },
        ),
    ).create_session({"models": {"m": FakeFlorence()}})
    rows = session.run({"image": images}).rows()
    assert [r["decoded_labels"] for r in rows] == [["cat"]] * 3
    assert [r["image_metadata"]["workflows_v2_frame"]["image_id"] for r in rows] == [
        i.image_id for i in images
    ]


@pytest.mark.parametrize(
    "task,params",
    [
        ("custom", {}),
        ("custom", {"prompt": ""}),
        ("phrase-grounded-object-detection", {}),
        ("phrase-grounded-instance-segmentation", {}),
        ("open-vocabulary-object-detection", {}),
        ("open-vocabulary-object-detection", {"classes": []}),
        ("detection-grounded-caption", {}),
        ("bogus", {}),
        ("caption", {"grounding_selection_mode": "bad"}),
        ("caption", {"model_id": ""}),
        ("caption", {"prompt": 4}),
        ("caption", {"grounding_detection": [1, 2, 3]}),
        ("caption", {"grounding_detection": [1, 2, 3, 4, 5]}),
        ("caption", {"grounding_detection": [0.0, 0.0, 1.1, 1.0]}),
        ("caption", {"grounding_detection": [3, 0, 2, 4]}),
    ],
)
def test_invalid_literals_fail_to_compile(task, params):
    with pytest.raises(ParamsValidationError):
        _plan(task_type=task, **params)


@pytest.mark.parametrize(
    "task,field,value",
    [
        ("custom", "prompt", ""),
        ("custom", "prompt", 42),
        ("phrase-grounded-object-detection", "prompt", None),
        ("open-vocabulary-object-detection", "classes", []),
        ("open-vocabulary-object-detection", "classes", [12]),
        ("detection-grounded-caption", "grounding_detection", None),
        ("detection-grounded-caption", "grounding_detection", [1, 2, 3]),
        (
            "detection-grounded-caption",
            "grounding_detection",
            [0.0, 0.0, float("nan"), 1.0],
        ),
        ("caption", "grounding_selection_mode", "nope"),
        ("caption", "task_type", "nope"),
        ("caption", "model_id", ""),
    ],
)
def test_bad_selected_values_fail_before_any_model_load(task, field, value):
    loader = Loader()
    params = dict(model_id="m", task_type=task, **_task_params(task))
    params[field] = "$inputs.selected"
    session = _plan(
        inputs=({"type": "WorkflowParameter", "name": "selected"},), **params
    ).create_session({"models": ModelCache(loader)})
    with pytest.raises(WorkflowExecutionError):
        session.run({"image": _images(), "selected": value})
    assert loader.loaded == []


@pytest.mark.parametrize(
    "mode,index",
    [
        ("first", 0),
        ("last", 1),
        ("biggest", 1),
        ("smallest", 0),
        ("most-confident", 0),
        ("least-confident", 1),
    ],
)
@pytest.mark.parametrize("family", ["detection", "instance", "keypoint"])
def test_all_selection_modes_and_native_families(mode, index, family):
    image = _images()[0]
    detection = _detections()
    if family == "instance":
        detection = InstanceDetections(
            xyxy=detection.xyxy,
            class_id=detection.class_id,
            confidence=detection.confidence,
            mask=torch.zeros((2, 20, 40), dtype=torch.bool),
        )
    if family == "keypoint":
        detection = (
            KeyPoints(
                xy=torch.zeros((2, 1, 2)),
                confidence=torch.ones((2, 1)),
                class_id=torch.zeros((2,), dtype=torch.int64),
            ),
            detection,
        )
    fake = FakeFlorence()
    Florence2Block(models={"m": fake}).run(
        image=Batch([image]),
        model_id="m",
        task_type="detection-grounded-caption",
        grounding_detection=detection,
        grounding_selection_mode=mode,
    )
    loc = (
        "<loc_0><loc_0><loc_250><loc_500>"
        if index == 0
        else "<loc_250><loc_100><loc_750><loc_600>"
    )
    assert fake.calls[0]["prompt"] == "<REGION_TO_DESCRIPTION>" + loc


def test_empty_grounding_retains_explicit_null_row_and_loads_nothing_for_it():
    loader = Loader()
    block = Florence2Block(models=ModelCache(loader))
    images = _images()[:2]
    rows = block.run(
        image=Batch(images),
        model_id=Batch(["unused", "m"]),
        task_type="detection-grounded-ocr",
        grounding_detection=Batch([_detections(()), _detections()]),
    )
    assert loader.loaded == [ModelRequest("m")]
    assert [rows[0][name] for name in ("raw_output", "parsed_output", "classes")] == [
        None
    ] * 3
    assert json.loads(rows[1]["raw_output"]) == "answer-30"
    assert rows[0]["image_metadata"] == images[0].prediction_metadata()
    assert block.run(image=Batch([]), task_type="caption") == []


def test_empty_and_nonempty_decoded_geometry_preserves_exact_payload_and_origins():
    root = ImageData.from_tensor(
        torch.zeros((3, 80, 100), dtype=torch.uint8), image_id="root"
    )
    cropped = root.crop((20, 10, 60, 50), image_id="crop").resize(
        (20, 20), image_id="small"
    )
    for decoded in (
        {"bboxes": [], "labels": []},
        {"quad_boxes": [[1, 2, 3, 4, 5, 6, 7, 8]], "labels": ["a"]},
        {"polygons": [[[0.25, 1.5, 4.0, 2.0, 5.0, 6.0]]], "labels": ["a"]},
    ):
        fake = FakeFlorence(answer=[{"<CUSTOM>": decoded}])
        row = Florence2Block(models={"m": fake}).run(
            image=Batch([cropped]), model_id="m", task_type="custom", prompt="<CUSTOM>"
        )[0]
        assert row["parsed_output"] is decoded
        assert json.loads(row["raw_output"]) == decoded
        assert row["image_metadata"] == cropped.prediction_metadata()
        assert row["image_metadata"]["workflows_v2_frame"]["root"]["offset_xy"] == [
            20.0,
            10.0,
        ]
        assert row["image_metadata"]["workflows_v2_frame"]["root"]["scale_xy"] == [
            2.0,
            2.0,
        ]
        assert fake.calls[0]["images"][0] is cropped.tensor_image


def test_parent_origin_grounding_maps_through_crop_resize_without_mutating_boxes():
    root = ImageData.from_tensor(
        torch.zeros((3, 80, 100), dtype=torch.uint8), image_id="root"
    )
    crop = root.crop((20, 10, 60, 50), image_id="crop")
    small = crop.resize((20, 20), image_id="small")
    for detection in (
        _detections(((20, 10, 40, 30),), frame=root),
        _detections(((0, 0, 20, 20),), frame=crop),
        _detections(((0, 0, 10, 10),), frame=small),
    ):
        original = detection.xyxy.clone()
        fake = FakeFlorence()
        Florence2Block(models={"m": fake}).run(
            image=Batch([small]),
            model_id="m",
            task_type="detection-grounded-caption",
            grounding_detection=detection,
        )
        assert (
            fake.calls[0]["prompt"]
            == "<REGION_TO_DESCRIPTION><loc_0><loc_0><loc_500><loc_500>"
        )
        assert torch.equal(original, detection.xyxy)


def test_unrelated_grounding_frame_fails_before_load():
    loader = Loader()
    unrelated = ImageData.from_tensor(
        torch.zeros((3, 20, 40), dtype=torch.uint8), image_id="other-root"
    )
    with pytest.raises(ValueError, match="share a known workflow root"):
        Florence2Block(models=ModelCache(loader)).run(
            image=Batch(_images()[:1]),
            model_id="m",
            task_type="detection-grounded-caption",
            grounding_detection=_detections(frame=unrelated),
        )
    assert loader.loaded == []


@pytest.mark.parametrize(
    "box,loc",
    [
        ([0, 0, 40, 20], "<loc_0><loc_0><loc_999><loc_999>"),
        ([0.0, 0.0, 1.0, 1.0], "<loc_0><loc_0><loc_999><loc_999>"),
        ([-10, -5, 100, 50], "<loc_0><loc_0><loc_999><loc_999>"),
    ],
)
def test_integer_pixels_relative_floats_and_clipping(box, loc):
    fake = FakeFlorence()
    Florence2Block(models={"m": fake}).run(
        image=Batch(_images()[:1]),
        model_id="m",
        task_type="detection-grounded-caption",
        grounding_detection=box,
    )
    assert fake.calls[0]["prompt"] == "<REGION_TO_DESCRIPTION>" + loc


@pytest.mark.parametrize(
    "field,value",
    [
        ("prompt", 4),
        ("classes", [""]),
        ("grounding_detection", [0, 0.0, 1, 1.0]),
        ("grounding_detection", [0, 0, 1, 1, 2]),
        ("grounding_selection_mode", "bad"),
        ("model_id", 7),
        ("model_package_id", ""),
    ],
)
def test_direct_calls_validate_before_load(field, value):
    loader = Loader()
    params = dict(model_id="m", task_type="caption")
    params[field] = value
    with pytest.raises(ValueError):
        Florence2Block(models=ModelCache(loader)).run(image=Batch(_images()), **params)
    assert loader.loaded == []


def test_a_bad_later_row_does_not_load_or_invoke_earlier_rows():
    loader = Loader()
    with pytest.raises(ValueError, match="prompt must be non-empty"):
        Florence2Block(models=ModelCache(loader)).run(
            image=Batch(_images()[:2]),
            model_id="m",
            task_type=Batch(["caption", "custom"]),
            prompt=Batch([None, ""]),
        )
    assert loader.loaded == []
    with pytest.raises(ValueError, match="has 1 values for 3 images"):
        Florence2Block(models=ModelCache(loader)).run(
            image=Batch(_images()), task_type="caption", prompt=Batch(["one"])
        )
    assert loader.loaded == []


def _rows(values, rows):
    return Batch(values, indices=[(row,) for row in rows])


@pytest.mark.parametrize(
    "field,values",
    [
        ("model_id", ["m", "m"]),
        ("model_package_id", [None, None]),
        ("task_type", ["caption", "caption"]),
        ("prompt", ["one", "two"]),
        ("classes", [["cat"], ["dog"]]),
        ("grounding_detection", [[0, 0, 20, 10], [0, 0, 10, 10]]),
        ("grounding_selection_mode", ["first", "last"]),
    ],
)
@pytest.mark.parametrize(
    "image_rows,value_rows",
    [((0, 1), (1, 0)), ((1, 3), (2, 4))],
    ids=["reordered", "other"],
)
def test_misaligned_per_image_values_fail_before_loading_or_invoking(
    field, values, image_rows, value_rows
):
    fake, loads = FakeFlorence(), []

    def load(request):
        loads.append(request)
        return fake

    params = dict(model_id="m", task_type="caption")
    params[field] = _rows(values, value_rows)
    with pytest.raises(ValueError, match=f"{field} rows .* do not match image rows"):
        Florence2Block(models=ModelCache(load)).run(
            image=_rows(_images()[:2], image_rows), **params
        )
    assert loads == [] and fake.calls == []


def test_aligned_sparse_per_image_values_follow_their_images_and_scalars_broadcast():
    models = {"m": FakeFlorence(), "other": FakeFlorence()}
    images = _images()
    rows = Florence2Block(models=models).run(
        image=_rows(images, (2, 5, 7)),
        model_id=_rows(["m", "other", "m"], (2, 5, 7)),
        task_type=_rows(
            ["caption", "detection-grounded-ocr", "open-vocabulary-object-detection"],
            (2, 5, 7),
        ),
        classes=_rows([None, None, ["cat"]], (2, 5, 7)),
        grounding_detection=_rows([None, [0.0, 0.0, 0.5, 0.5], None], (2, 5, 7)),
        grounding_selection_mode="first",
    )
    assert [row["model_id"] for row in rows] == ["m", "other", "m"]
    assert [c["prompt"] for c in models["m"].calls] == [
        "<CAPTION>",
        "<OPEN_VOCABULARY_DETECTION>cat",
    ]
    assert [c["prompt"] for c in models["other"].calls] == [
        "<REGION_TO_OCR><loc_0><loc_0><loc_500><loc_500>"
    ]
    sent = [c["images"][0] for c in models["m"].calls + models["other"].calls]
    expected = [images[0], images[2], images[1]]
    assert all(tensor is image.tensor_image for tensor, image in zip(sent, expected))
    assert json.loads(rows[0]["raw_output"]) == "answer-10"
    assert json.loads(rows[1]["raw_output"]) == "answer-30"
    assert rows[2]["classes"] == ["cat"]


@pytest.mark.parametrize(
    "answer",
    [
        [],
        [{"<CAPTION>": "a"}, {"<CAPTION>": "b"}],
        [{"wrong": "x"}],
        ["raw text"],
        [{"<CAPTION>": 3}],
        [{"<CAPTION>": []}],
    ],
)
def test_malformed_backend_answers_are_rejected_without_fabricated_rows(answer):
    with pytest.raises(ValueError, match="Florence"):
        Florence2Block(models={"m": FakeFlorence(answer=answer)}).run(
            image=Batch(_images()[:1]), model_id="m", task_type="caption"
        )


def test_custom_plain_text_uses_the_v1_adapter_decoder_key():
    fake = FakeFlorence()
    row = Florence2Block(models={"m": fake}).run(
        image=Batch(_images()[:1]),
        model_id="m",
        task_type="custom",
        prompt="Describe this.",
    )[0]
    assert fake.calls[0]["task"] == "Describe this.>"
    assert json.loads(row["raw_output"]) == "answer-10"


def test_supplied_model_rejects_unverified_package_and_no_calls_happen():
    fake = FakeFlorence()
    with pytest.raises(ModelRequestConflict):
        Florence2Block(models={"m": fake}).run(
            image=Batch(_images()[:1]),
            model_id="m",
            model_package_id="p",
            task_type="caption",
        )
    assert fake.calls == []


def test_grounding_overrides_classes_and_classes_override_prompt_like_v1():
    fake = FakeFlorence()
    block = Florence2Block(models={"m": fake})
    block.run(
        image=Batch(_images()[:1]),
        model_id="m",
        task_type="caption",
        prompt="ignored",
        classes=["cat", "dog"],
    )
    block.run(
        image=Batch(_images()[:1]),
        model_id="m",
        task_type="caption",
        prompt="ignored",
        classes=["cat"],
        grounding_detection=[0, 0, 20, 10],
    )
    assert [c["prompt"] for c in fake.calls] == [
        "<CAPTION>cat<and>dog",
        "<CAPTION><loc_0><loc_0><loc_500><loc_500>",
    ]


def test_resource_discovery_includes_package_and_vlm_task():
    params = Florence2Block.Params(
        image="$inputs.image", model_id="m", model_package_id="p", task_type="caption"
    )
    (resource,) = Florence2Block.discover_dependent_resources(params)
    assert resource.identifier == "m"
    assert resource.details == {
        "resource": "models",
        "task_type": "vlm",
        "model_package_id": "p",
    }


def test_fake_surface_matches_source_named_parameters_and_all_v1_task_tokens():
    # Locate the same package tree as the imported block, never the live checkout.
    workflows = WORKFLOWS_ROOT
    root = CHECKOUT_ROOT
    backend = ast.parse(
        (
            root / "inference_models/inference_models/models/florence2/florence2_hf.py"
        ).read_text()
    )
    cls = next(
        n
        for n in backend.body
        if isinstance(n, ast.ClassDef) and n.name == "Florence2HF"
    )
    method = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "prompt"
    )
    assert [a.arg for a in method.args.args] + [method.args.kwarg.arg] == list(
        inspect.signature(FakeFlorence.prompt).parameters
    )
    source = ast.parse(
        (
            workflows
            / "roboflow_workflows/core_steps/models/foundation/florence2/v1.py"
        ).read_text()
    )
    tasks = next(
        n.value
        for n in source.body
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "SUPPORTED_TASK_TYPES_LIST"
            for t in n.targets
        )
    )
    assert {
        e["task_type"]: e["florence_task"] for e in ast.literal_eval(tasks)
    } == TASK_TOKENS


@pytest.mark.skipif(
    not os.environ.get("FLORENCE_LOCAL_PACKAGE"),
    reason="UNVERIFIED: no FLORENCE_LOCAL_PACKAGE; real model and devices were not exercised",
)
def test_opt_in_local_package_without_downloads(monkeypatch):
    package = Path(os.environ["FLORENCE_LOCAL_PACKAGE"]).resolve()
    assert (package / "model_config.json").is_file()
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    from inference_models import AutoModel

    model = AutoModel.from_pretrained(
        str(package),
        allow_direct_local_storage_loading=True,
        allow_loading_dependency_models=False,
        local_files_only=True,
        device="cpu",
    )
    rows = Florence2Block(models={"local": model}).run(
        image=Batch(_images()[:1]), model_id="local", task_type="caption"
    )
    assert isinstance(json.loads(rows[0]["raw_output"]), str)


@pytest.mark.parametrize("mode", ["run", "phases"])
@pytest.mark.parametrize(
    "family,kind",
    [
        ("detection", "object_detection_prediction"),
        ("instance", "instance_segmentation_prediction"),
        ("keypoint", "keypoint_detection_prediction"),
    ],
)
def test_native_grounding_selectors_compile_and_keep_empty_rows(mode, family, kind):
    fake = FakeFlorence()
    images = _images()[:2]

    def native(boxes):
        detection = _detections(boxes, frame=images[0])
        if family == "instance":
            detection = InstanceDetections(
                xyxy=detection.xyxy,
                class_id=detection.class_id,
                confidence=detection.confidence,
                mask=torch.zeros((len(boxes), 20, 40), dtype=torch.bool),
                image_metadata=detection.image_metadata,
            )
        if family == "keypoint":
            detection = (
                KeyPoints(
                    xy=torch.zeros((len(boxes), 1, 2)),
                    confidence=torch.ones((len(boxes), 1)),
                    class_id=torch.zeros((len(boxes),), dtype=torch.int64),
                    image_metadata=detection.image_metadata,
                ),
                detection,
            )
        return detection

    session = _plan(
        mode=mode,
        model_id="m",
        task_type="detection-grounded-caption",
        inputs=({"type": "WorkflowBatchInput", "name": "grounding", "kind": [kind]},),
        grounding_detection="$inputs.grounding",
    ).create_session({"models": {"m": fake}})
    # Both values describe the same image because rows may carry duplicate inputs.
    rows = session.run(
        {
            "image": [images[0], images[0]],
            "grounding": [native(()), native(((0, 0, 20, 10),))],
        }
    ).rows()
    assert len(fake.calls) == 1
    assert rows[0]["raw_output"] is None
    assert json.loads(rows[1]["raw_output"]) == "answer-10"
    assert (
        fake.calls[0]["prompt"]
        == "<REGION_TO_DESCRIPTION><loc_0><loc_0><loc_500><loc_500>"
    )


def test_compiled_empty_input_and_bad_later_selected_row_load_nothing():
    loader = Loader()
    session = _plan(model_id="m", task_type="caption").create_session(
        {"models": ModelCache(loader)}
    )
    assert session.run({"image": []}).rows() == []
    session = _plan(
        model_id="m",
        task_type="$inputs.task",
        prompt="$inputs.prompt",
        inputs=(
            {"type": "WorkflowBatchInput", "name": "task", "kind": ["string"]},
            {"type": "WorkflowBatchInput", "name": "prompt", "kind": ["string"]},
        ),
    ).create_session({"models": ModelCache(loader)})
    with pytest.raises(WorkflowExecutionError):
        session.run(
            {
                "image": _images()[:2],
                "task": ["caption", "custom"],
                "prompt": [None, ""],
            }
        )
    assert loader.loaded == []


def test_grayscale_and_missing_prompt_model_fail_without_inference():
    loader = Loader()
    gray = ImageData.from_tensor(torch.zeros((1, 20, 40), dtype=torch.uint8))
    with pytest.raises(ValueError, match="3-channel"):
        Florence2Block(models=ModelCache(loader)).run(
            image=Batch([gray]), task_type="caption"
        )
    assert loader.loaded == []
    fake = FakeFlorence()
    with pytest.raises(ValueError, match="provide prompt"):
        Florence2Block(models={"good": fake, "bad": object()}).run(
            image=Batch(_images()[:2]),
            model_id=Batch(["good", "bad"]),
            task_type="caption",
        )
    assert fake.calls == []


@pytest.mark.parametrize("box", [[0, 0.0, 1, 1.0], [False, 0, 2, 3]])
def test_mixed_coordinate_literals_fail_before_pydantic_numeric_conversion(box):
    with pytest.raises(ParamsValidationError):
        _plan(task_type="detection-grounded-caption", grounding_detection=box)


def test_named_backend_defaults_match_the_fake():
    root = CHECKOUT_ROOT
    backend = ast.parse(
        (
            root / "inference_models/inference_models/models/florence2/florence2_hf.py"
        ).read_text()
    )
    cls = next(
        n
        for n in backend.body
        if isinstance(n, ast.ClassDef) and n.name == "Florence2HF"
    )
    method = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "prompt"
    )
    defaults = dict(
        zip(
            [a.arg for a in method.args.args][-len(method.args.defaults) :],
            method.args.defaults,
        )
    )
    import inference_models.configuration as configuration

    signature = inspect.signature(FakeFlorence.prompt)
    for name in ("max_new_tokens", "num_beams", "do_sample"):
        assert signature.parameters[name].default == getattr(
            configuration, defaults[name].id
        )


@pytest.mark.parametrize(
    "boxes,confidence,mode,error",
    [
        (torch.ones((2, 3)), torch.ones(2), "first", "shape"),
        (
            torch.tensor([[0.0, 0.0, float("nan"), 2.0]]),
            torch.ones(1),
            "first",
            "finite",
        ),
        (torch.tensor([[2.0, 0.0, 1.0, 2.0]]), torch.ones(1), "first", "positive-area"),
        (torch.tensor([[0.0, 0.0, 1.0, 0.0]]), torch.ones(1), "first", "positive-area"),
        (
            torch.tensor([[0.0, 0.0, 1.0, 2.0]]),
            torch.ones(2),
            "most-confident",
            "confidence",
        ),
        (
            torch.tensor([[0.0, 0.0, 1.0, 2.0]]),
            torch.tensor([float("inf")]),
            "least-confident",
            "confidence",
        ),
    ],
)
def test_malformed_native_grounding_fails_before_loading_any_model(
    boxes, confidence, mode, error
):
    loader = Loader()
    prediction = Detections(
        xyxy=boxes,
        class_id=torch.zeros(len(boxes), dtype=torch.int64),
        confidence=confidence,
    )
    with pytest.raises(ValueError, match=error):
        Florence2Block(models=ModelCache(loader)).run(
            image=Batch(_images()[:1]),
            model_id="m",
            task_type="detection-grounded-caption",
            grounding_detection=prediction,
            grounding_selection_mode=mode,
        )
    assert loader.loaded == []


@pytest.mark.parametrize(
    "mode",
    ["first", "last", "biggest", "smallest", "most-confident", "least-confident"],
)
def test_tied_grounding_selection_is_stable_and_does_not_mutate_native_boxes(mode):
    # Equal areas and confidences distinguish stable argmin/argmax from row order.
    prediction = _detections(((0, 0, 10, 10), (20, 0, 30, 10)))
    prediction.confidence = torch.tensor([0.5, 0.5])
    original = prediction.xyxy.clone()
    fake = FakeFlorence()
    Florence2Block(models={"m": fake}).run(
        image=Batch(_images()[:1]),
        model_id="m",
        task_type="detection-grounded-caption",
        grounding_detection=prediction,
        grounding_selection_mode=mode,
    )
    expected = (
        "<loc_500><loc_0><loc_750><loc_500>"
        if mode == "last"
        else "<loc_0><loc_0><loc_250><loc_500>"
    )
    assert fake.calls[0]["prompt"] == "<REGION_TO_DESCRIPTION>" + expected
    assert torch.equal(prediction.xyxy, original)


@pytest.mark.parametrize("mode", ["run", "phases"])
@pytest.mark.parametrize("task", ["ocr", "caption"])
def test_text_answers_reach_a_string_consumer_unquoted(task, mode):
    session = _plan(
        mode=mode,
        model_id="m",
        task_type=task,
        steps=({"type": _Shout.type, "name": "shout", "text": "$steps.florence.text"},),
        outputs=(
            {"type": "JsonField", "name": "shouted", "selector": "$steps.shout.value"},
        ),
    ).create_session({"models": {"m": FakeFlorence()}})
    rows = session.run({"image": _images()[:2]}).rows()
    assert [row["text"] for row in rows] == ["answer-10", "answer-30"]
    assert [row["shouted"] for row in rows] == ["ANSWER-10", "ANSWER-30"]
    # The legacy outputs keep their V1 meaning.
    assert [row["raw_output"] for row in rows] == ['"answer-10"', '"answer-30"']
    assert [row["parsed_output"] for row in rows] == [None, None]


def test_text_is_null_for_dictionary_answers_and_empty_grounding():
    block = Florence2Block(models={"m": FakeFlorence()})
    detected = block.run(
        image=Batch(_images()[:1]), model_id="m", task_type="object-detection"
    )[0]
    assert detected["text"] is None
    assert detected["parsed_output"] == {"bboxes": [[1, 2, 5, 6]], "labels": ["cat"]}
    skipped = block.run(
        image=Batch(_images()[:1]),
        model_id="m",
        task_type="detection-grounded-ocr",
        grounding_detection=_detections(()),
    )[0]
    assert skipped["text"] is None and skipped["raw_output"] is None


# The 40x20 test images: [0, 0, 1, 1] in pixels is <loc_0><loc_0><loc_25><loc_50>.
@pytest.mark.parametrize("mode", ["run", "phases"])
@pytest.mark.parametrize(
    "coordinates,box,loc",
    [
        # JavaScript writes [0.0, 0.0, 1.0, 1.0] as [0, 0, 1, 1].
        ("relative", [0, 0, 1, 1], "<loc_0><loc_0><loc_999><loc_999>"),
        ("relative", [0, 0, 0.5, 1], "<loc_0><loc_0><loc_500><loc_999>"),
        ("relative", [0.0, 0.0, 0.5, 0.5], "<loc_0><loc_0><loc_500><loc_500>"),
        ("pixels", [0, 0, 20.0, 10], "<loc_0><loc_0><loc_500><loc_500>"),
        ("pixels", [0, 0, 1, 1], "<loc_0><loc_0><loc_25><loc_50>"),
        ("auto", [0, 0, 1, 1], "<loc_0><loc_0><loc_25><loc_50>"),
        ("auto", [0.0, 0.0, 1.0, 1.0], "<loc_0><loc_0><loc_999><loc_999>"),
    ],
)
@pytest.mark.parametrize("selected", [False, True], ids=["literal", "selected"])
def test_coordinate_system_reads_literal_and_selected_boxes(
    coordinates, box, loc, selected, mode
):
    fake = FakeFlorence()
    params = dict(
        mode=mode,
        model_id="m",
        task_type="detection-grounded-caption",
        grounding_coordinate_system=coordinates,
        grounding_detection="$inputs.box" if selected else box,
    )
    if selected:
        params["inputs"] = (
            {"type": "WorkflowParameter", "name": "box", "kind": ["list_of_values"]},
        )
    session = _plan(**params).create_session({"models": {"m": fake}})
    arguments = {"image": _images()[:1], **({"box": box} if selected else {})}
    rows = session.run(arguments).rows()
    assert fake.calls[0]["prompt"] == "<REGION_TO_DESCRIPTION>" + loc
    assert rows[0]["text"] == "answer-10"


def test_omitted_coordinate_system_keeps_v1_auto_reading():
    fake = FakeFlorence()
    _plan(
        model_id="m",
        task_type="detection-grounded-caption",
        grounding_detection=[0, 0, 1, 1],
    ).create_session({"models": {"m": fake}}).run({"image": _images()[:1]})
    assert fake.calls[0]["prompt"].endswith("<loc_0><loc_0><loc_25><loc_50>")


@pytest.mark.parametrize(
    "params",
    [
        {"grounding_coordinate_system": "bogus"},
        {"grounding_coordinate_system": "Relative"},
        {"grounding_coordinate_system": None},
        {
            "grounding_coordinate_system": "relative",
            "grounding_detection": [0, 0, 2, 1],
        },
        {
            "grounding_coordinate_system": "relative",
            "grounding_detection": [-0.1, 0, 1, 1],
        },
        {
            "grounding_coordinate_system": "relative",
            "grounding_detection": [0, 0, 0, 1],
        },
        {
            "grounding_coordinate_system": "relative",
            "grounding_detection": [True, 0, 1, 1],
        },
        {
            "grounding_coordinate_system": "pixels",
            "grounding_detection": [0, 0, "9", 9],
        },
        {"grounding_coordinate_system": "pixels", "grounding_detection": [5, 0, 5, 9]},
        {"grounding_coordinate_system": "pixels", "grounding_detection": [0, 0, 9]},
        {"grounding_detection": [0, 0, 0.5, 1]},
    ],
)
def test_invalid_coordinate_system_or_box_fails_to_compile(params):
    with pytest.raises(ParamsValidationError):
        _plan(task_type="detection-grounded-caption", **params)


@pytest.mark.parametrize(
    "coordinates,box,error",
    [
        ("relative", [0, 0, 2, 1], "relative grounding coordinates must lie in"),
        ("relative", [0, 0, 1, float("inf")], "four finite numbers"),
        ("pixels", [9, 0, 1, 9], "positive width and height"),
        ("auto", [0, 0, 0.5, 1], "set grounding_coordinate_system"),
    ],
)
def test_bad_selected_box_names_its_row_and_fails_before_any_model_load(
    coordinates, box, error
):
    loader = Loader()
    session = _plan(
        model_id="m",
        task_type="detection-grounded-caption",
        grounding_coordinate_system=coordinates,
        grounding_detection="$inputs.box",
        inputs=(
            {"type": "WorkflowBatchInput", "name": "box", "kind": ["list_of_values"]},
        ),
    ).create_session({"models": ModelCache(loader)})
    with pytest.raises(WorkflowExecutionError) as caught:
        session.run({"image": _images()[:2], "box": [[0, 0, 1, 1], box]})
    assert "at index [1]" in str(caught.value)
    assert error in str(caught.value)
    assert loader.loaded == []


@pytest.mark.parametrize("coordinates", ["pixels", "relative"])
def test_explicit_coordinate_system_leaves_native_boxes_in_pixels(coordinates):
    fake = FakeFlorence()
    Florence2Block(models={"m": fake}).run(
        image=Batch(_images()[:1]),
        model_id="m",
        task_type="detection-grounded-caption",
        grounding_detection=_detections(),
        grounding_coordinate_system=coordinates,
    )
    assert fake.calls[0]["prompt"].endswith("<loc_0><loc_0><loc_250><loc_500>")


def test_direct_call_rejects_an_unknown_coordinate_system_before_load():
    loader = Loader()
    with pytest.raises(ValueError, match="grounding_coordinate_system must be one of"):
        Florence2Block(models=ModelCache(loader)).run(
            image=Batch(_images()[:1]),
            model_id="m",
            task_type="caption",
            grounding_coordinate_system="Relative",
        )
    assert loader.loaded == []


@pytest.mark.parametrize(
    "field,values,error",
    [
        ("prompt", [None, 4], "prompt must be text or None"),
        ("task_type", ["caption", "Caption"], "Unsupported Florence task_type"),
        ("model_id", ["m", ""], "model_id must be a non-empty string"),
        ("grounding_detection", [None, [0, 0, 0.5, 1]], "set grounding_coordinate"),
    ],
)
def test_direct_call_errors_name_the_logical_index_of_the_failing_image(
    field, values, error
):
    loader = Loader()
    params = dict(model_id="m", task_type="caption")
    params[field] = _rows(values, (3, 8))
    with pytest.raises(ValueError) as caught:
        Florence2Block(models=ModelCache(loader)).run(
            image=_rows(_images()[:2], (3, 8)), **params
        )
    assert str(caught.value).startswith("v2/florence_2@v2: image at index [8]: ")
    assert error in str(caught.value)
    assert loader.loaded == []


def test_grayscale_later_image_names_its_index():
    gray = ImageData.from_tensor(torch.zeros((1, 20, 40), dtype=torch.uint8))
    with pytest.raises(ValueError, match=r"image at index \[1\]: Florence-2 needs a 3"):
        Florence2Block(models={"m": FakeFlorence()}).run(
            image=Batch([_images()[0], gray]), model_id="m", task_type="caption"
        )


def test_default_task_without_classes_says_it_is_the_default():
    with pytest.raises(ParamsValidationError) as caught:
        _plan(model_id="m")
    assert "(the default task_type); set classes or choose another task_type" in str(
        caught.value
    )
    with pytest.raises(ParamsValidationError) as caught:
        _plan(model_id="m", task_type="open-vocabulary-object-detection")
    assert "the default task_type" not in str(caught.value)
