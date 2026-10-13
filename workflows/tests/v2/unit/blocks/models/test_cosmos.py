"""Offline local-model and real compiled workflow contracts; fakes are not inference."""

import ast
import inspect
import os
from pathlib import Path

import inference_models
import pytest
import torch
from pydantic import Field
from roboflow_workflows.v2.blocks.models.cache import (
    ModelCache,
    ModelRequest,
    ModelRequestConflict,
)
from roboflow_workflows.v2.blocks.models.cosmos import (
    DEFAULT_MODEL_ID,
    Cosmos3EdgeBlock,
)
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import STRING_KIND
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref
from roboflow_workflows.v2.engine.plan import CompileOptions


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

    def prompt(
        self,
        images,
        prompt=None,
        input_color_format=None,
        max_new_tokens=512,
        do_sample=False,
        skip_special_tokens=True,
        return_thinking=False,
    ):
        assert len(images) == 1
        self.calls.append(
            {
                "image": images[0],
                "text": prompt,
                "max_new_tokens": max_new_tokens,
                "input_color_format": input_color_format,
            }
        )
        return [f"answer:{prompt}"]


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
        prediction: Ref(STRING_KIND) = Field(description="Cosmos text answer.")

    def run(self, *, prediction):
        return {"value": prediction}


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
                "type": Cosmos3EdgeBlock.type,
                "name": "model",
                "image": "$steps.source.image",
                **params,
            },
            {
                "type": _Consume.type,
                "name": "consume",
                "prediction": "$steps.model.output",
            },
        ],
        "outputs": [
            {"type": "JsonField", "name": name, "selector": selector}
            for name, selector in (
                ("output", "$steps.model.output"),
                ("model_id", "$steps.model.model_id"),
                ("inference_id", "$steps.model.inference_id"),
                ("consumed", "$steps.consume.value"),
            )
        ],
    }
    return compile_workflow(
        definition,
        catalogue=Catalogue([_Source, Cosmos3EdgeBlock, _Consume]),
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
    tree = ast.parse((package / "models/cosmos3/cosmos3_reasoner_hf.py").read_text())
    cls = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "Cosmos3EdgeReasoner"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "prompt"
    )
    signature = _signature(method)
    assert [name for name, _ in signature] == list(
        inspect.signature(_Fake.prompt).parameters
    )
    assert (
        dict(signature)["max_new_tokens"]
        == "INFERENCE_MODELS_COSMOS3_DEFAULT_MAX_NEW_TOKENS"
    )
    # Real backend defaults come from configuration, not block-invented values.
    config = ast.parse((package / "configuration.py").read_text())
    assignment = next(
        node
        for node in ast.walk(config)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "INFERENCE_MODELS_COSMOS3_DEFAULT_MAX_NEW_TOKENS"
    )
    value = next(
        keyword.value
        for keyword in assignment.value.keywords
        if keyword.arg == "default"
    )
    assert ast.literal_eval(value) == 512


@pytest.mark.parametrize("params", [{}, {"prompt": None}, {"prompt": ""}])
def test_omitted_none_and_empty_prompts_preserve_v1(params):
    model = _Fake()
    rows = Cosmos3EdgeBlock(models={DEFAULT_MODEL_ID: model}).run(
        image=Batch(_images()[:1]), **params
    )
    assert model.calls[0]["text"] == "Describe what's in this image."
    assert rows[0]["model_id"] == DEFAULT_MODEL_ID


@pytest.mark.parametrize("tokens", [None, 9])
def test_generation_limit_omits_none_and_preserves_explicit_value(tokens):
    model = _Fake()
    method = model.prompt
    sent = []

    def record(images, **keywords):
        sent.append(keywords)
        return method(images, **keywords)

    model.prompt = record
    Cosmos3EdgeBlock(models={"m": model}).run(
        image=Batch(_images()[:1]), model_id="m", max_new_tokens=tokens
    )
    assert ("max_new_tokens" in sent[0]) is (tokens is not None)
    assert model.calls[0]["max_new_tokens"] == (512 if tokens is None else tokens)


def test_empty_batch_does_no_model_work():
    loads = []
    block = Cosmos3EdgeBlock(models=ModelCache(lambda request: loads.append(request)))
    assert block.run(image=Batch([]), model_id="m") == []
    assert loads == []


@pytest.mark.parametrize(
    "invalid",
    [
        {"prompt": 7},
        {"prompt": Batch(["ok", 9])},
        {"prompt": Batch(["too-short"])},
        {"model_id": Batch(["ok", ""])},
        {"model_package_id": Batch([None, 4])},
        {"max_new_tokens": 0},
        {"max_new_tokens": True},
        {"max_new_tokens": 1.2},
    ],
)
def test_bad_resolved_parameters_fail_before_loading(invalid):
    loads = []
    block = Cosmos3EdgeBlock(models=ModelCache(lambda request: loads.append(request)))
    arguments = {"image": Batch(_images()[:2]), "model_id": "m", **invalid}
    with pytest.raises((ValueError, TypeError)):
        block.run(**arguments)
    assert loads == []


def _rows(values, rows):
    return Batch(values, indices=[(row,) for row in rows])


@pytest.mark.parametrize(
    "field", ["prompt", "system_prompt", "model_id", "model_package_id"]
)
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

    values = {
        "prompt": ["Is it wet?", "Is it dry?"],
        "system_prompt": ["Be brief.", None],
        "model_id": ["m", "m"],
        "model_package_id": [None, None],
    }
    with pytest.raises(ValueError, match=f"{field} rows .* do not match image rows"):
        Cosmos3EdgeBlock(models=ModelCache(load)).run(
            image=_rows(_images()[:2], image_rows),
            **{"model_id": "m", field: _rows(values[field], value_rows)},
        )
    assert loads == [] and model.calls == []


def test_aligned_sparse_per_image_values_follow_their_images_and_scalars_broadcast():
    a, b = _Fake(), _Fake()
    images = _images()
    rows = Cosmos3EdgeBlock(models={"a": a, "b": b}).run(
        image=_rows(images, (1, 3, 4)),
        model_id=_rows(["a", "b", "a"], (1, 3, 4)),
        prompt="Is it wet?",
        system_prompt=_rows(["Be brief.", None, "Explain."], (1, 3, 4)),
        max_new_tokens=5,
    )
    assert [row["model_id"] for row in rows] == ["a", "b", "a"]
    assert [call["text"] for call in a.calls] == [
        "Is it wet?<system_prompt>Be brief.",
        "Is it wet?<system_prompt>Explain.",
    ]
    assert [call["text"] for call in b.calls] == ["Is it wet?"]
    sent = [call["image"] for call in a.calls + b.calls]
    expected = [images[0], images[2], images[1]]
    assert all(tensor is image.tensor_image for tensor, image in zip(sent, expected))
    assert [call["max_new_tokens"] for call in a.calls + b.calls] == [5, 5, 5]


def test_all_models_resolve_before_first_call():
    model = _Fake()
    block = Cosmos3EdgeBlock(models={"a": model})
    with pytest.raises(LookupError):
        block.run(image=Batch(_images()[:2]), model_id=Batch(["a", "missing"]))
    assert model.calls == []


def test_exact_package_request_is_required():
    model = _Fake()
    with pytest.raises(ModelRequestConflict):
        Cosmos3EdgeBlock(models={"m": model}).run(
            image=Batch(_images()[:1]), model_id="m", model_package_id="p"
        )
    assert model.calls == []


@pytest.mark.parametrize("bad", [None, "text", [], ["a", "b"], [{}], [7]])
def test_bad_model_output_type_or_cardinality_fails(bad):
    model = _Fake()

    def wrong(images, **keywords):
        return bad

    model.prompt = wrong
    with pytest.raises((ValueError, TypeError)):
        Cosmos3EdgeBlock(models={"m": model}).run(
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
            {"type": "WorkflowBatchInput", "name": "model", "kind": ["string"]},
            {"type": "WorkflowBatchInput", "name": "package", "kind": ["string"]},
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
        "model": ["a", "b", "a"],
        "package": ["p", None, "p"],
        "question": ["first", "second", "third"],
        "tokens": 12,
    }
    rows = session.run(arguments).rows()
    again = session.run(arguments).rows()
    assert loaded == [ModelRequest("a", "p"), ModelRequest("b")]
    assert [row["model_id"] for row in rows] == ["a", "b", "a"]
    assert [row["consumed"] for row in rows] == [
        "answer:first",
        "answer:second",
        "answer:third",
    ]
    assert len({row["inference_id"] for row in rows + again}) == 6
    calls = models[ModelRequest("a", "p")].calls
    assert [call["text"] for call in calls] == ["first", "third"] * 2
    assert calls[0]["image"] is images[0].tensor_image
    assert calls[1]["image"] is images[2].tensor_image
    assert all(
        call["input_color_format"] == "rgb" and call["max_new_tokens"] == 12
        for call in calls
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
    assert model.calls[0]["text"] == "Describe what's in this image."


@pytest.mark.skipif(
    not os.environ.get("COSMOS_LOCAL_PACKAGE"),
    reason="Opt-in existing local package only; no downloads",
)
def test_existing_real_package_offline():
    package = Path(os.environ["COSMOS_LOCAL_PACKAGE"]).resolve()
    assert package.is_dir(), "Supply an existing local package directory"
    assert (
        inference_models.OFFLINE_MODE
    ), "Real check requires OFFLINE_MODE=true at process startup"
    from inference_models import AutoModel

    model = AutoModel.from_pretrained(str(package))
    rows = Cosmos3EdgeBlock(models={"cached": model}).run(
        image=Batch(_images()[:1]), model_id="cached", prompt="Is the walkway clear?"
    )
    assert len(rows) == 1


@pytest.mark.parametrize("system", [None, ""])
def test_no_system_instruction_preserves_backend_fine_tune_default(system):
    model = _Fake()
    Cosmos3EdgeBlock(models={"m": model}).run(
        image=Batch(_images()[:1]), model_id="m", system_prompt=system
    )
    assert model.calls[0]["text"] == "Describe what's in this image."
    assert "<system_prompt>" not in model.calls[0]["text"]


def test_system_prompt_is_appended_per_image_with_v1_sentinel():
    model = _Fake()
    Cosmos3EdgeBlock(models={"m": model}).run(
        image=Batch(_images()),
        model_id="m",
        prompt=Batch(["question", None, ""]),
        system_prompt=Batch(["inspect", None, "brief"]),
    )
    assert [call["text"] for call in model.calls] == [
        "question<system_prompt>inspect",
        "Describe what's in this image.",
        "Describe what's in this image.<system_prompt>brief",
    ]


@pytest.mark.parametrize(
    "invalid",
    [
        {"system_prompt": 4},
        {"system_prompt": Batch(["ok", 4])},
        {"system_prompt": Batch(["short"])},
        {"prompt": "question<system_prompt>hidden"},
        {"system_prompt": "inspect<system_prompt>hidden"},
    ],
)
def test_bad_system_or_reserved_marker_fails_before_loading(invalid):
    loaded = []
    with pytest.raises(ValueError):
        Cosmos3EdgeBlock(models=ModelCache(lambda request: loaded.append(request))).run(
            image=Batch(_images()[:2]), model_id="m", **invalid
        )
    assert loaded == []


def test_wrong_later_model_method_runs_nothing():
    model = _Fake()

    class Wrong:
        def prompt(self, images, question):
            raise AssertionError("Must not be invoked")

    with pytest.raises(TypeError):
        Cosmos3EdgeBlock(models={"a": model, "b": Wrong()}).run(
            image=Batch(_images()[:2]), model_id=Batch(["a", "b"])
        )
    assert model.calls == []


def test_compiled_dynamic_system_prompt_composition():
    model = _Fake()
    plan = _plan(
        inputs=({"type": "WorkflowBatchInput", "name": "system", "kind": ["string"]},),
        model_id="m",
        prompt="question",
        system_prompt="$inputs.system",
    )
    rows = (
        plan.create_session({"models": {"m": model}})
        .run({"image": _images()[:2], "system": [None, "brief"]})
        .rows()
    )
    assert [row["output"] for row in rows] == [
        "answer:question",
        "answer:question<system_prompt>brief",
    ]


def test_source_backend_parser_preserves_trained_default():
    from types import SimpleNamespace

    package = Path(inference_models.__file__).parent
    tree = ast.parse((package / "models/cosmos3/cosmos3_reasoner_hf.py").read_text())
    cls = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "Cosmos3EdgeReasoner"
    )
    parser = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "_parse_prompt"
    )
    # Execute only this pinned pure-text method, without importing model libraries.
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            parser,
        ],
        type_ignores=[],
    )
    namespace = {
        "DEFAULT_PROMPT": "Describe what's in this image.",
        "SYSTEM_PROMPT_SENTINEL": "<system_prompt>",
    }
    exec(
        compile(ast.fix_missing_locations(module), "pinned_cosmos_parser", "exec"),
        namespace,
    )
    model = _Fake()
    block = Cosmos3EdgeBlock(models={"m": model})
    block.run(image=Batch(_images()[:1]), model_id="m")
    parse = namespace["_parse_prompt"]
    assert parse(
        SimpleNamespace(default_system_prompt="trained instruction"),
        model.calls[0]["text"],
    ) == ("Describe what's in this image.", "trained instruction")
    block.run(
        image=Batch(_images()[:1]),
        model_id="m",
        prompt="question",
        system_prompt="override",
    )
    assert parse(
        SimpleNamespace(default_system_prompt="trained instruction"),
        model.calls[-1]["text"],
    ) == ("question", "override")


def test_grayscale_rejected_before_loading():
    loads = []
    grayscale = ImageData.from_tensor(torch.zeros((1, 6, 8), dtype=torch.uint8))
    with pytest.raises(ValueError, match="3-channel"):
        Cosmos3EdgeBlock(models=ModelCache(lambda request: loads.append(request))).run(
            image=Batch([grayscale]), model_id="m"
        )
    assert loads == []


def test_empty_compiled_workflow_emits_no_rows_and_loads_nothing():
    loads = []
    session = _plan(model_id="m").create_session(
        {"models": ModelCache(lambda request: loads.append(request))}
    )
    assert session.run({"image": []}).rows() == []
    assert loads == []
