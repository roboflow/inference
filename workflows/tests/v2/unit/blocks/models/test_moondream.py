"""Offline local-model and real compiled workflow contracts; fakes are not inference."""

import ast
import inspect
import os
from pathlib import Path
from threading import Lock
from types import MethodType, SimpleNamespace

import pytest
import torch
from pydantic import Field
from roboflow_workflows.v2.blocks.models.cache import (
    ModelCache,
    ModelRequest,
    ModelRequestConflict,
)
from roboflow_workflows.v2.blocks.models.moondream2.v1 import DEFAULT_MODEL_ID, Moondream2Block
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.frames import FrameRef, frame_of
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import INTEGER_KIND, STRING_KIND
from roboflow_workflows.v2.data.predictions import OBJECT_DETECTION_PREDICTION_KIND
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref
from roboflow_workflows.v2.engine.plan import CompileOptions

import inference_models
from inference_models.models.base.object_detection import Detections


def _images():
    return [
        ImageData.from_tensor(
            torch.full((3, 6, width), value, dtype=torch.uint8),
            image_id=f"frame-{value}",
        )
        for value, width in ((20, 8), (100, 4), (200, 10))
    ]


class _Fake:
    def __init__(self):
        self.calls = []
        self.answer = Detections(
            torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
            torch.tensor([0]),
            torch.tensor([0.9]),
            image_metadata={"backend": "kept"},
        )

    def detect(self, images, classes, max_new_tokens=700, input_color_format=None):
        assert len(images) == 1 and len(classes) == 1
        self.calls.append(
            {
                "image": images[0],
                "text": classes[0],
                "max_new_tokens": max_new_tokens,
                "input_color_format": input_color_format,
            }
        )
        return [self.answer]


class _Source(Block):
    type = "test/image_source@v1"
    outputs = {"image": Output(IMAGE_KIND, source="image")}

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(description="Image passed through the source.")

    def run(self, *, image):
        return {"image": image}


class _Consume(Block):
    type = "test/consume@v1"
    outputs = {"value": Output(STRING_KIND, source="prediction")}

    class Params(BlockParams):
        prediction: Ref(OBJECT_DETECTION_PREDICTION_KIND) = Field(
            description="Moondream detections."
        )

    def run(self, *, prediction):
        return {"value": prediction.image_metadata["class_names"][0]}


def _plan(*, mode="run", inputs=(), **params):
    definition = {
        "version": "2.0",
        "inputs": [
            {"type": "WorkflowBatchInput", "name": "image", "kind": ["image"]},
            *inputs,
        ],
        "steps": [
            {"type": _Source.type, "name": "source", "image": "$inputs.image"},
            {
                "type": Moondream2Block.type,
                "name": "model",
                "image": "$steps.source.image",
                **params,
            },
            {
                "type": _Consume.type,
                "name": "consume",
                "prediction": "$steps.model.predictions",
            },
        ],
        "outputs": [
            {"type": "JsonField", "name": name, "selector": selector}
            for name, selector in (
                ("predictions", "$steps.model.predictions"),
                ("model_id", "$steps.model.model_id"),
                ("inference_id", "$steps.model.inference_id"),
                ("consumed", "$steps.consume.value"),
            )
        ],
    }
    return compile_workflow(
        definition,
        catalogue=Catalogue([_Source, Moondream2Block, _Consume]),
        options=CompileOptions(block_execution=mode),
    )


def _signature(node):
    args = node.args
    defaults = [None] * (len(args.args) - len(args.defaults)) + list(args.defaults)
    return [
        (arg.arg, None if default is None else ast.unparse(default))
        for arg, default in zip(args.args, defaults)
    ]


def test_fake_method_matches_pinned_backend_signature():
    package = Path(inference_models.__file__).parent
    tree = ast.parse((package / "models/moondream2/moondream2_hf.py").read_text())
    cls = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "MoonDream2HF"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "detect"
    )
    signature = _signature(method)
    assert [name for name, _ in signature] == list(
        inspect.signature(_Fake.detect).parameters
    )
    assert (
        dict(signature)["max_new_tokens"]
        == "INFERENCE_MODELS_MOONDREAM2_DEFAULT_MAX_NEW_TOKENS"
    )
    # Real backend defaults come from configuration, not block-invented values.
    config = ast.parse((package / "configuration.py").read_text())
    assignment = next(
        node
        for node in ast.walk(config)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "INFERENCE_MODELS_MOONDREAM2_DEFAULT_MAX_NEW_TOKENS"
    )
    value = next(
        keyword.value
        for keyword in assignment.value.keywords
        if keyword.arg == "default"
    )
    assert ast.literal_eval(value) == 700


@pytest.mark.parametrize("params", [{}, {"prompt": None}, {"prompt": ""}])
def test_omitted_none_and_empty_prompts_preserve_v1(params):
    model = _Fake()
    rows = Moondream2Block(models={DEFAULT_MODEL_ID: model}).run(
        image=Batch(_images()[:1]), **params
    )
    assert model.calls[0]["text"] == ""
    assert rows[0]["model_id"] == DEFAULT_MODEL_ID


@pytest.mark.parametrize("tokens", [None, 9])
def test_generation_limit_omits_none_and_preserves_explicit_value(tokens):
    model = _Fake()
    method = model.detect
    sent = []

    def record(images, **keywords):
        sent.append(keywords)
        return method(images, **keywords)

    model.detect = record
    Moondream2Block(models={"m": model}).run(
        image=Batch(_images()[:1]), model_id="m", max_new_tokens=tokens
    )
    assert ("max_new_tokens" in sent[0]) is (tokens is not None)
    assert model.calls[0]["max_new_tokens"] == (700 if tokens is None else tokens)


def test_empty_batch_does_no_model_work():
    loads = []
    block = Moondream2Block(models=ModelCache(lambda request: loads.append(request)))
    assert block.run(image=Batch([]), model_id="m") == []
    assert loads == []


@pytest.mark.parametrize(
    "invalid",
    [
        {"prompt": 7},
        {"prompt": Batch(["ok", 9])},
        {"prompt": Batch(["too-short"])},
        {"model_id": ""},
        {"model_id": Batch(["ok", "ok"])},
        {"model_package_id": 4},
        {"model_package_id": Batch([None, None])},
        {"max_new_tokens": 0},
        {"max_new_tokens": True},
        {"max_new_tokens": 1.2},
    ],
)
def test_bad_resolved_parameters_fail_before_loading(invalid):
    loads = []
    block = Moondream2Block(models=ModelCache(lambda request: loads.append(request)))
    arguments = {"image": Batch(_images()[:2]), "model_id": "m", **invalid}
    with pytest.raises((ValueError, TypeError)):
        block.run(**arguments)
    assert loads == []


def _rows(values, rows):
    return Batch(values, indices=[(row,) for row in rows])


@pytest.mark.parametrize("field", ["prompt"])
@pytest.mark.parametrize(
    "image_rows,value_rows",
    [((0, 1), (1, 0)), ((1, 3), (2, 4))],
    ids=["reordered", "other"],
)
def test_misaligned_per_image_values_fail_before_loading_or_invoking(
    field, image_rows, value_rows
):
    model, loads = _Fake(), []

    def load(request):
        loads.append(request)
        return model

    values = {"prompt": ["cat", "dog"]}
    with pytest.raises(ValueError, match=f"{field} rows .* do not match image rows"):
        Moondream2Block(models=ModelCache(load)).run(
            image=_rows(_images()[:2], image_rows),
            **{"model_id": "m", field: _rows(values[field], value_rows)},
        )
    assert loads == [] and model.calls == []


def test_aligned_sparse_per_image_values_follow_their_images_and_scalars_broadcast():
    a, b = _Fake(), _Fake()
    images = _images()
    block = Moondream2Block(models={"a": a, "b": b})
    rows = block.run(
        image=_rows(images, (1, 3, 4)),
        model_id="a",
        prompt=_rows(["cat", "dog", "bird"], (1, 3, 4)),
        max_new_tokens=5,
    )
    block.run(image=_rows(images[:2], (2, 5)), model_id="b", prompt="car")
    assert [row["model_id"] for row in rows] == ["a", "a", "a"]
    assert [row["predictions"].image_metadata["class_names"][0] for row in rows] == [
        "cat",
        "dog",
        "bird",
    ]
    assert [call["text"] for call in a.calls] == ["cat", "dog", "bird"]
    assert [call["text"] for call in b.calls] == ["car", "car"]
    sent = [call["image"] for call in a.calls + b.calls]
    expected = [images[0], images[1], images[2], images[0], images[1]]
    assert all(tensor is image.tensor_image for tensor, image in zip(sent, expected))
    assert [call["max_new_tokens"] for call in a.calls] == [5, 5, 5]
    # The fakes return one shared carrier; each row must carry its own frame.
    assert [frame_of(row["predictions"]) for row in rows] == [
        FrameRef.from_image(image) for image in images
    ]


@pytest.mark.parametrize("field", ["model_id", "model_package_id"])
def test_per_image_model_selection_fails_before_loading_or_invoking(field):
    model, loads = _Fake(), []

    def load(request):
        loads.append(request)
        return model

    with pytest.raises(ValueError, match=f"{field} is one value for every image"):
        Moondream2Block(models=ModelCache(load)).run(
            image=Batch(_images()[:2]),
            **{"model_id": "m", field: Batch(["m", "m"])},
        )
    assert loads == [] and model.calls == []


def test_a_missing_model_fails_before_any_call():
    model = _Fake()
    block = Moondream2Block(models={"a": model})
    with pytest.raises(LookupError):
        block.run(image=Batch(_images()[:2]), model_id="missing")
    assert model.calls == []


def test_exact_package_request_is_required():
    model = _Fake()
    with pytest.raises(ModelRequestConflict):
        Moondream2Block(models={"m": model}).run(
            image=Batch(_images()[:1]), model_id="m", model_package_id="p"
        )
    assert model.calls == []


@pytest.mark.parametrize("bad", [None, "text", [], ["a", "b"], [{}], [7]])
def test_bad_model_output_type_or_cardinality_fails(bad):
    model = _Fake()

    def wrong(images, **keywords):
        return bad

    model.detect = wrong
    with pytest.raises((ValueError, TypeError)):
        Moondream2Block(models={"m": model}).run(
            image=Batch(_images()[:1]), model_id="m"
        )


@pytest.mark.parametrize("mode", ["run", "phases"])
def test_compiled_source_model_downstream_selectors_and_reuse(mode):
    loaded, models = [], {}

    def load(request):
        loaded.append(request)
        models[request] = _Fake()
        return models[request]

    plan = _plan(
        mode=mode,
        inputs=(
            {"type": "WorkflowParameter", "name": "model", "kind": ["string"]},
            {"type": "WorkflowParameter", "name": "package", "kind": ["string"]},
            {"type": "WorkflowBatchInput", "name": "question", "kind": ["string"]},
            {"type": "WorkflowParameter", "name": "tokens", "kind": ["integer"]},
        ),
        model_id="$inputs.model",
        model_package_id="$inputs.package",
        prompt="$inputs.question",
        max_new_tokens="$inputs.tokens",
    )
    session = plan.create_session({"models": ModelCache(load)})
    assert loaded == []
    images = _images()
    arguments = {
        "image": images,
        "model": "a",
        "package": "p",
        "question": ["first", "second", "third"],
        "tokens": 12,
    }
    rows = session.run(arguments).rows()
    other = session.run({**arguments, "model": "b", "package": None}).rows()
    again = session.run(arguments).rows()
    assert loaded == [ModelRequest("a", "p"), ModelRequest("b")]
    assert [row["model_id"] for row in rows + other] == ["a"] * 3 + ["b"] * 3
    assert [row["consumed"] for row in rows] == ["first", "second", "third"]
    assert len({row["inference_id"] for row in rows + other + again}) == 9
    for result in (rows, other, again):
        frames = [frame_of(row["predictions"]) for row in result]
        assert frames == [FrameRef.from_image(image) for image in images]
        assert len({frame.image_id for frame in frames}) == 3
    calls = models[ModelRequest("a", "p")].calls
    assert [call["text"] for call in calls] == ["first", "second", "third"] * 2
    assert [call["image"] for call in calls[:3]] == [
        image.tensor_image for image in images
    ]
    assert all(
        call["input_color_format"] == "rgb" and call["max_new_tokens"] == 12
        for call in calls
    )


def test_a_per_image_package_selector_fails_compilation():
    with pytest.raises(Exception, match="model_package_id is passed once per call"):
        _plan(
            inputs=(
                {"type": "WorkflowBatchInput", "name": "package", "kind": ["string"]},
            ),
            model_id="m",
            model_package_id="$inputs.package",
        )


@pytest.mark.parametrize(
    "invalid", [{"max_new_tokens": 0}, {"max_new_tokens": True}, {"prompt": 4}]
)
def test_literal_invalid_parameters_fail_compilation(invalid):
    with pytest.raises(Exception, match="invalid parameters"):
        _plan(**invalid)


def test_dynamic_invalid_tokens_fail_before_model_load():
    loaded = []
    plan = _plan(
        inputs=({"type": "WorkflowParameter", "name": "tokens"},),
        model_id="m",
        max_new_tokens="$inputs.tokens",
    )
    session = plan.create_session(
        {"models": ModelCache(lambda request: loaded.append(request))}
    )
    with pytest.raises(Exception, match="invalid|integer"):
        session.run({"image": _images()[:1], "tokens": "bad"})
    assert loaded == []


def test_default_model_in_compiled_workflow_and_explicit_nulls():
    model = _Fake()
    session = _plan(prompt=None, max_new_tokens=None).create_session(
        {"models": {DEFAULT_MODEL_ID: model}}
    )
    rows = session.run({"image": _images()[:1]}).rows()
    assert rows[0]["model_id"] == DEFAULT_MODEL_ID
    assert model.calls[0]["text"] == ""


@pytest.mark.skipif(
    not os.environ.get("MOONDREAM_LOCAL_PACKAGE"),
    reason="Opt-in existing local package only; no downloads",
)
def test_existing_real_package_offline():
    package = Path(os.environ["MOONDREAM_LOCAL_PACKAGE"]).resolve()
    assert package.is_dir(), "Supply an existing local package directory"
    assert (
        inference_models.OFFLINE_MODE
    ), "Real check requires OFFLINE_MODE=true at process startup"
    from inference_models import AutoModel

    model = AutoModel.from_pretrained(str(package))
    rows = Moondream2Block(models={"cached": model}).run(
        image=Batch(_images()[:1]), model_id="cached", prompt="person"
    )
    assert len(rows) == 1


def test_reused_detection_carrier_keeps_prior_frames_and_tensor_storage():
    model = _Fake()
    images = _images()[:2]
    block = Moondream2Block(models={"m": model})
    first = block.run(image=Batch(images[:1]), model_id="m", prompt="person")[0]
    second = block.run(image=Batch(images[1:]), model_id="m", prompt="car")[0]
    a, b = first["predictions"], second["predictions"]
    assert a is not model.answer and b is not model.answer and a is not b
    assert a.xyxy is model.answer.xyxy and b.xyxy is model.answer.xyxy
    assert a.image_metadata["class_names"] == {0: "person"}
    assert b.image_metadata["class_names"] == {0: "car"}
    assert a.image_metadata["prediction_type"] == "object-detection"
    assert a.image_metadata["backend"] == "kept"
    assert a.image_metadata["workflows_v2_rows"]["namespace"] == first["inference_id"]
    assert b.image_metadata["workflows_v2_rows"]["namespace"] == second["inference_id"]
    assert model.answer.image_metadata == {"backend": "kept"}
    assert model.calls[0]["image"] is images[0].tensor_image


def test_empty_native_detections_remain_valid():
    model = _Fake()
    model.answer = Detections(
        torch.empty((0, 4)), torch.empty(0, dtype=torch.int64), torch.empty(0)
    )
    rows = Moondream2Block(models={"m": model}).run(
        image=Batch(_images()[:1]), model_id="m"
    )
    assert len(rows[0]["predictions"]) == 0
    assert rows[0]["predictions"].xyxy is model.answer.xyxy


def test_per_image_none_prompt_is_empty_class_name():
    model = _Fake()
    Moondream2Block(models={"m": model}).run(
        image=Batch(_images()[:2]), model_id="m", prompt=Batch([None, "cat"])
    )
    assert [call["text"] for call in model.calls] == ["", "cat"]


def test_wrong_model_method_runs_nothing():
    class Wrong:
        def detect(self, images, prompt):
            raise AssertionError("Must not be invoked")

    with pytest.raises(
        ValueError, match=r"model 'b' \(backend Wrong\) does not accept"
    ):
        Moondream2Block(models={"b": Wrong()}).run(
            image=Batch(_images()[:2]), model_id="b"
        )


class _NoTokenLimit:
    def __init__(self):
        self.calls = []

    def detect(self, images, classes, input_color_format=None):
        self.calls.append(classes[0])
        return [_Fake().answer]


def test_backend_without_token_limit_names_model_and_fix_before_any_call():
    model = _NoTokenLimit()
    block = Moondream2Block(models={"b": model})
    with pytest.raises(ValueError) as caught:
        block.run(image=Batch(_images()[:2]), model_id="b", max_new_tokens=5)
    assert str(caught.value) == (
        "roboflow_core/moondream2@v1: model 'b' (backend _NoTokenLimit) does not accept "
        "detect(images, classes, input_color_format, max_new_tokens). Remove "
        "max_new_tokens or set it to null for this model. Checked before any "
        "model runs."
    )
    assert model.calls == []
    block.run(image=Batch(_images()[:1]), model_id="b", prompt="cat")
    assert model.calls == ["cat"]


def test_model_without_detect_fails_before_any_call():
    with pytest.raises(ValueError, match=r"\(backend object\) has no detect\(\)"):
        Moondream2Block(models={"b": object()}).run(
            image=Batch(_images()[:2]), model_id="b"
        )


@pytest.mark.parametrize(
    "arguments,error",
    [
        ({"prompt": ["ok", 9]}, "prompt must be text or None, got 9"),
    ],
)
def test_per_image_errors_name_the_logical_index_before_loading(arguments, error):
    loads = []
    block = Moondream2Block(models=ModelCache(lambda request: loads.append(request)))
    ((field, values),) = arguments.items()
    with pytest.raises(ValueError) as caught:
        block.run(
            image=_rows(_images()[:2], (2, 5)),
            **{"model_id": "m", field: _rows(values, (2, 5))},
        )
    assert str(caught.value).startswith("roboflow_core/moondream2@v1: image at index [5]: ")
    assert error in str(caught.value)
    assert loads == []


def test_grayscale_rejected_before_loading():
    loads = []
    grayscale = ImageData.from_tensor(torch.zeros((1, 6, 8), dtype=torch.uint8))
    with pytest.raises(ValueError, match="3-channel"):
        Moondream2Block(models=ModelCache(lambda request: loads.append(request))).run(
            image=Batch([grayscale]), model_id="m"
        )
    assert loads == []
    with pytest.raises(ValueError) as caught:
        Moondream2Block(models=ModelCache(lambda request: loads.append(request))).run(
            image=_rows([_images()[0], grayscale], (0, 4)), model_id="m"
        )
    assert str(caught.value) == (
        "roboflow_core/moondream2@v1: image at index [4]: a local vision model needs a "
        "3-channel RGB image, got 1 channel(s)"
    )
    assert loads == []


def test_empty_compiled_workflow_emits_no_rows_and_loads_nothing():
    loads = []
    session = _plan(model_id="m").create_session(
        {"models": ModelCache(lambda request: loads.append(request))}
    )
    assert session.run({"image": []}).rows() == []
    assert loads == []


def _pinned_moondream_functions():
    """Execute only the pinned detect/postprocessor, without model imports."""
    import hashlib

    package = Path(inference_models.__file__).parent
    source = package / "models/moondream2/moondream2_hf.py"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == (
        "ef456eba40bd8de8ce32e142dc19142f507d614549b81273da3b6542572e5dc5"
    )
    tree = ast.parse(source.read_text())
    postprocess = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "post_process_detections"
    )
    backend = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MoonDream2HF"
    )
    detect = next(
        node
        for node in backend.body
        if isinstance(node, ast.FunctionDef) and node.name == "detect"
    )
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[future, postprocess, detect], type_ignores=[])
    )
    namespace = {
        "torch": torch,
        "Detections": Detections,
        "INFERENCE_MODELS_MOONDREAM2_DEFAULT_MAX_NEW_TOKENS": 700,
    }
    exec(compile(module, str(source), "exec"), namespace)

    return namespace


class _ActualMoondreamPython:
    """Use actual detect/postprocessing; replace weights and image encoding."""

    def __init__(self, objects):
        functions = _pinned_moondream_functions()
        self.detect = MethodType(functions["detect"], self)
        self._device = torch.device("cpu")
        self._lock = Lock()
        self._model = SimpleNamespace(detect=self._raw_detect)
        self.objects = objects
        self.calls = []
        self.encoded_images = []

    def _raw_detect(self, *, image, object, settings):
        self.calls.append((image, object, settings))

        return {"objects": self.objects}

    def encode_images(self, *, images, input_color_format):
        assert input_color_format == "rgb"
        self.encoded_images.extend(images)
        encoded = [
            SimpleNamespace(
                moondream_encoded_image=image,
                image_dimensions=SimpleNamespace(
                    height=image.shape[-2], width=image.shape[-1]
                ),
            )
            for image in images
        ]

        return encoded


class _CountDetections(Block):
    type = "test/count_moondream_detections@v1"
    outputs = {"count": Output(INTEGER_KIND, source="prediction")}

    class Params(BlockParams):
        prediction: Ref(OBJECT_DETECTION_PREDICTION_KIND) = Field(
            description="Native Moondream detections to count."
        )

    def run(self, *, prediction):
        return {"count": len(prediction)}


def _count_plan(mode):
    definition = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowBatchInput", "name": "image", "kind": ["image"]}],
        "steps": [
            {"type": _Source.type, "name": "source", "image": "$inputs.image"},
            {
                "type": Moondream2Block.type,
                "name": "moon",
                "image": "$steps.source.image",
                "model_id": "m",
                "prompt": "person",
            },
            {
                "type": _CountDetections.type,
                "name": "count",
                "prediction": "$steps.moon.predictions",
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "count",
                "selector": "$steps.count.count",
            },
            {
                "type": "JsonField",
                "name": "predictions",
                "selector": "$steps.moon.predictions",
            },
        ],
    }
    plan = compile_workflow(
        definition,
        catalogue=Catalogue([_Source, Moondream2Block, _CountDetections]),
        options=CompileOptions(block_execution=mode),
    )

    return plan


def test_actual_backend_zero_objects_return_valid_empty_detections_direct():
    model = _ActualMoondreamPython([])
    frame = _images()[0]
    raw = model.detect(
        [frame.tensor_image], classes=["person"], input_color_format="rgb"
    )
    assert raw[0].xyxy.shape == (0,)  # The actual backend representation at issue.
    rows = Moondream2Block(models={"m": model}).run(
        image=Batch([frame]), model_id="m", prompt="person"
    )
    prediction = rows[0]["predictions"]
    OBJECT_DETECTION_PREDICTION_KIND.check(prediction)
    assert len(prediction) == 0
    assert prediction.xyxy.shape == (0, 4)
    assert prediction.xyxy.dtype == raw[0].xyxy.dtype == torch.int32
    assert prediction.xyxy.device == raw[0].xyxy.device
    assert prediction.image_metadata["class_names"] == {0: "person"}
    assert model.encoded_images[-1] is frame.tensor_image
    assert model.calls[-1][2] == {"max_tokens": 700}


@pytest.mark.parametrize("mode", ["run", "phases"])
def test_actual_backend_zero_objects_reach_compiled_count_consumer(mode):
    model = _ActualMoondreamPython([])
    images = _images()
    session = _count_plan(mode).create_session({"models": {"m": model}})
    rows = session.run({"image": images}).rows()
    assert [row["count"] for row in rows] == [0, 0, 0]
    assert all(row["predictions"].xyxy.shape == (0, 4) for row in rows)
    assert all(row["predictions"].xyxy.dtype == torch.int32 for row in rows)
    assert all(
        encoded is image.tensor_image
        for encoded, image in zip(model.encoded_images, images)
    )


@pytest.mark.parametrize(
    "dtype", [torch.int32, torch.int64, torch.float16, torch.float32]
)
def test_empty_box_normalization_preserves_original_carrier_and_other_fields(dtype):
    model = _Fake()
    original = Detections(
        xyxy=torch.empty((0,), dtype=dtype),
        class_id=torch.empty((0,), dtype=torch.int64),
        confidence=torch.empty((0,), dtype=torch.float32),
        image_metadata={"backend": "preserved"},
        bboxes_metadata=[],
    )
    model.answer = original
    images = _images()[:2]
    rows = Moondream2Block(models={"m": model}).run(
        image=Batch(images), model_id="m", prompt=Batch(["person", "car"])
    )
    assert original.xyxy.shape == (0,)
    assert original.image_metadata == {"backend": "preserved"}
    assert rows[0]["predictions"] is not rows[1]["predictions"]
    for row, label in zip(rows, ["person", "car"]):
        prediction = row["predictions"]
        assert prediction is not original
        assert prediction.xyxy.shape == (0, 4)
        assert prediction.xyxy.dtype == original.xyxy.dtype
        assert prediction.xyxy.device == original.xyxy.device
        assert prediction.class_id is original.class_id
        assert prediction.confidence is original.confidence
        assert prediction.bboxes_metadata is original.bboxes_metadata
        assert prediction.image_metadata["backend"] == "preserved"
        assert prediction.image_metadata["class_names"] == {0: label}


@pytest.mark.parametrize("shape", [(4,), (1, 3), (0, 3), (0, 0), (1, 0)])
def test_empty_box_fix_does_not_accept_other_malformed_box_shapes(shape):
    model = _Fake()
    count = shape[0]
    model.answer = Detections(
        xyxy=torch.empty(shape),
        class_id=torch.zeros((count,), dtype=torch.int64),
        confidence=torch.ones((count,)),
    )
    with pytest.raises(ValueError, match="shape"):
        Moondream2Block(models={"m": model}).run(
            image=Batch(_images()[:1]), model_id="m"
        )
    assert model.answer.xyxy.shape == shape


def test_empty_box_fix_does_not_accept_mismatched_nonempty_detection_fields():
    model = _Fake()
    model.answer = Detections(torch.empty((0,)), torch.tensor([0]), torch.tensor([0.9]))
    with pytest.raises(ValueError, match="shape"):
        Moondream2Block(models={"m": model}).run(
            image=Batch(_images()[:1]), model_id="m"
        )
    assert model.answer.xyxy.shape == (0,)


def test_actual_backend_nonempty_geometry_is_preserved():
    model = _ActualMoondreamPython(
        [{"x_min": 0.125, "y_min": 0.25, "x_max": 0.75, "y_max": 0.9}]
    )
    frame = ImageData.from_tensor(torch.zeros((3, 20, 40), dtype=torch.uint8))
    result = Moondream2Block(models={"m": model}).run(
        image=Batch([frame]), model_id="m", prompt="person"
    )[0]
    assert result["predictions"].xyxy.tolist() == [[5, 5, 30, 18]]
    assert result["predictions"].class_id.tolist() == [0]
    assert result["predictions"].confidence.tolist() == [1.0]
    assert model.encoded_images[0] is frame.tensor_image
