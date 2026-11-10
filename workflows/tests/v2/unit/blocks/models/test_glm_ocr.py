"""The GLM-OCR block on fake models, directly and in compiled V2 workflows.

``FakeGlmOcr`` copies the parameter names and defaults of
``GlmOcrHF.prompt``; a test compares them with the backend source. It is NOT
model inference: it records every call and answers each image with text that
echoes what reached it::

    answer = "<prompt received>|<H>x<W>|<mean pixel>|<max_new_tokens>"

The V1 prompt strings are compared with the V1 block source, so a V1 change
fails a test. The last test runs a real local package only when
``GLM_OCR_PACKAGE`` names one; otherwise it skips, which is not evidence that
real inference works.
"""

import ast
import importlib.util
import inspect
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pytest
import torch
from inference_models.configuration import (
    INFERENCE_MODELS_GLM_OCR_DEFAULT_DO_SAMPLE,
    INFERENCE_MODELS_GLM_OCR_DEFAULT_MAX_NEW_TOKENS,
)
from roboflow_workflows.v2.blocks.formatters.json_parser.v1 import JsonParserBlock
from roboflow_workflows.v2.blocks.models.cache import (
    ModelCache,
    ModelRequest,
    auto_model_loader,
)
from roboflow_workflows.v2.blocks.models.glm_ocr.v1 import BLOCKS, FIXED_TASK_PROMPTS, STRUCTURED_ANSWERING_PROMPT_TEMPLATE, TASK_TYPES, GlmOcrBlock
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.errors import (
    ParamsValidationError,
    WorkflowExecutionError,
)
from roboflow_workflows.v2.engine.introspection import discover_workload
from roboflow_workflows.v2.engine.plan import CompileOptions
from tests.v2.support.compilation.fixture_blocks import Gate

BACKEND_SOURCE = "models/glm_ocr/glm_ocr_hf.py"
V1_SOURCE = "core_steps/models/foundation/glm_ocr/v1.py"
MODES = ("run", "phases")
OUTPUTS = ("parsed_output", "model_id", "inference_id")
STRUCTURE = {"serial": "The serial number", "date": "The production date"}


class FakeGlmOcr:
    """Records ``prompt`` calls; ``GlmOcrHF.prompt`` parameters."""

    def __init__(self, request: Optional[ModelRequest] = None):
        self.request = request
        self.calls: List[Dict[str, Any]] = []

    def prompt(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        prompt: str = None,
        input_color_format: str = None,
        max_new_tokens: Optional[int] = INFERENCE_MODELS_GLM_OCR_DEFAULT_MAX_NEW_TOKENS,
        do_sample: bool = INFERENCE_MODELS_GLM_OCR_DEFAULT_DO_SAMPLE,
        skip_special_tokens: bool = True,
        **kwargs,
    ) -> List[str]:
        self.calls.append(
            {
                "images": images,
                "prompt": prompt,
                "input_color_format": input_color_format,
                "max_new_tokens": max_new_tokens,
                "kwargs": kwargs,
            }
        )
        # The backend puts `images` into one image slot: one answer.
        _, height, width = images.shape
        mean = round(images.float().mean().item())
        answers = [f"{prompt}|{height}x{width}|{mean}|{max_new_tokens}"]

        return answers


class FakeLoader:
    """``ModelCache`` loader building one ``model_type`` per request."""

    def __init__(self, model_type: type = FakeGlmOcr):
        self.model_type = model_type
        self.loaded: List[ModelRequest] = []
        self.models: List[Any] = []

    def __call__(self, request: ModelRequest) -> Any:
        self.loaded.append(request)
        model = self.model_type(request)
        self.models.append(model)

        return model


class JsonGlmOcr(FakeGlmOcr):
    """Answers like a structured-answering model: JSON in Markdown markers."""

    def prompt(self, images: torch.Tensor, **kwargs: Any) -> List[str]:
        super().prompt(images, **kwargs)
        document = {"serial": f"S-{round(images.float().mean().item())}"}
        answers = ["```json\n" + json.dumps(document, indent=4) + "\n```"]

        return answers


def _image(value: int, image_id: str, *, width: int = 8) -> ImageData:
    pixels = torch.full((3, 6, width), value, dtype=torch.uint8)
    image = ImageData.from_tensor(pixels, image_id=image_id)

    return image


def _images() -> List[ImageData]:
    images = [_image(20, "x", width=8), _image(100, "y", width=4), _image(200, "z")]

    return images


def _structured_prompt(structure: Dict[str, str]) -> str:
    prompt = STRUCTURED_ANSWERING_PROMPT_TEMPLATE.format(
        output_structure=json.dumps(structure, indent=4)
    )

    return prompt


def _plan(
    *,
    mode: str = "run",
    inputs: Tuple[Dict[str, Any], ...] = (),
    steps: Tuple[Dict[str, Any], ...] = (),
    outputs: Tuple[Tuple[str, str], ...] = (),
    **params: Any,
):
    definition = {
        "version": "2.0",
        "inputs": [
            {"type": "WorkflowBatchInput", "name": "image", "kind": ["image"]},
            *inputs,
        ],
        "steps": [
            *steps,
            {
                "type": GlmOcrBlock.type,
                "name": "ocr",
                "image": "$inputs.image",
                **params,
            },
        ],
        "outputs": [
            *(
                {"type": "JsonField", "name": name, "selector": f"$steps.ocr.{name}"}
                for name in OUTPUTS
            ),
            *(
                {"type": "JsonField", "name": name, "selector": selector}
                for name, selector in outputs
            ),
        ],
    }
    plan = compile_workflow(
        definition,
        catalogue=Catalogue([*BLOCKS, Gate, JsonParserBlock]),
        options=CompileOptions(block_execution=mode),
    )

    return plan


def _session(loader: FakeLoader, **params: Any) -> Any:
    session = _plan(**params).create_session({"models": ModelCache(loader)})

    return session


def _parameters(function_node: ast.FunctionDef) -> List[Tuple[str, Optional[str]]]:
    arguments = function_node.args
    positional = arguments.args
    defaults = [None] * (len(positional) - len(arguments.defaults)) + list(
        arguments.defaults
    )
    parameters = [
        (argument.arg, None if default is None else ast.unparse(default))
        for argument, default in zip(positional, defaults)
    ]
    if arguments.kwarg is not None:
        parameters.append((f"**{arguments.kwarg.arg}", None))

    return parameters


def _source_tree(package: str, relative: str) -> ast.Module:
    root = Path(importlib.util.find_spec(package).origin).parent
    tree = ast.parse((root / relative).read_text())

    return tree


def _module_constants(tree: ast.Module) -> Dict[str, Any]:
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                constants[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                continue

    return constants


# --- the fake and the prompts copy their sources --------------------------------


def test_the_fake_has_the_backend_prompt_parameter_names_and_defaults() -> None:
    tree = _source_tree("inference_models", BACKEND_SOURCE)
    (backend_prompt,) = [
        item
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "GlmOcrHF"
        for item in node.body
        if isinstance(item, ast.FunctionDef) and item.name == "prompt"
    ]
    fake_source = inspect.cleandoc("\n" + inspect.getsource(FakeGlmOcr.prompt))
    fake_prompt = ast.parse(fake_source).body[0]

    assert _parameters(fake_prompt) == _parameters(backend_prompt)


def test_prompts_task_types_and_defaults_are_the_v1_strings() -> None:
    tree = _source_tree("roboflow_workflows", V1_SOURCE)
    v1 = _module_constants(tree)
    (task_literal,) = [
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign) and node.targets[0].id == "TaskType"
    ]
    field_defaults = {
        node.target.id: next(
            ast.literal_eval(keyword.value)
            for keyword in node.value.keywords
            if keyword.arg == "default"
        )
        for node in ast.walk(tree)
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id in ("task_type", "model_version")
    }
    params = GlmOcrBlock.Params.model_fields

    assert STRUCTURED_ANSWERING_PROMPT_TEMPLATE == (
        v1["STRUCTURED_ANSWERING_PROMPT_TEMPLATE"]
    )
    assert dict(FIXED_TASK_PROMPTS) == {
        task: prompt for task, prompt in v1["TASK_TYPE_TO_PROMPT"].items() if prompt
    }
    assert TASK_TYPES == ast.literal_eval(task_literal.slice)
    assert params["task_type"].default == field_defaults["task_type"]
    assert params["model_id"].default == field_defaults["model_version"]


def test_the_fixed_prompts_are_the_backend_prompts() -> None:
    backend = _module_constants(_source_tree("inference_models", BACKEND_SOURCE))

    assert FIXED_TASK_PROMPTS["text-recognition"] == backend["TEXT_RECOGNITION_PROMPT"]
    assert (
        FIXED_TASK_PROMPTS["table-recognition"] == backend["TABLE_RECOGNITION_PROMPT"]
    )
    assert (
        FIXED_TASK_PROMPTS["formula-recognition"]
        == backend["FORMULA_RECOGNITION_PROMPT"]
    )


# --- direct calls ------------------------------------------------------------


@pytest.mark.parametrize(
    "params, expected_prompt",
    [
        ({}, "Text Recognition:"),
        ({"task_type": "text-recognition"}, "Text Recognition:"),
        ({"task_type": "table-recognition"}, "Table Recognition:"),
        ({"task_type": "formula-recognition"}, "Formula Recognition:"),
        (
            {"task_type": "structured-answering", "output_structure": STRUCTURE},
            _structured_prompt(STRUCTURE),
        ),
        ({"task_type": "custom", "prompt": "Read the label."}, "Read the label."),
        # V1 ignores prompt and output_structure where the task does not use them.
        (
            {
                "task_type": "table-recognition",
                "prompt": "x",
                "output_structure": STRUCTURE,
            },
            "Table Recognition:",
        ),
    ],
)
def test_each_task_type_sends_the_v1_prompt(
    params: Dict[str, Any], expected_prompt: str
) -> None:
    loader = FakeLoader()
    block = GlmOcrBlock(models=ModelCache(loader))

    (row,) = block.run(image=Batch([_image(9, "a")]), model_id="glm-ocr", **params)

    (call,) = loader.models[0].calls
    assert call["prompt"] == expected_prompt
    assert row["parsed_output"] == f"{expected_prompt}|6x8|9|{call['max_new_tokens']}"


def test_each_image_reaches_the_model_by_identity_as_rgb_with_backend_defaults() -> (
    None
):
    loader = FakeLoader()
    block = GlmOcrBlock(models=ModelCache(loader))
    images = _images()

    rows = block.run(image=Batch(images), model_id="glm-ocr")

    calls = loader.models[0].calls
    assert [call["images"] for call in calls] == [
        image.tensor_image for image in images
    ]
    assert all(
        call["images"] is image.tensor_image for call, image in zip(calls, images)
    )
    assert {call["input_color_format"] for call in calls} == {"rgb"}
    # Unset: the backend keeps its own default; nothing else is passed.
    assert {call["max_new_tokens"] for call in calls} == {
        INFERENCE_MODELS_GLM_OCR_DEFAULT_MAX_NEW_TOKENS
    }
    assert [call["kwargs"] for call in calls] == [{}, {}, {}]
    assert [row["parsed_output"].split("|")[1] for row in rows] == [
        "6x8",
        "6x4",
        "6x8",
    ]


def test_a_supplied_mapping_serves_its_model_and_names_a_missing_one() -> None:
    model = FakeGlmOcr()
    block = GlmOcrBlock(models={"glm-ocr": model})

    block.run(image=Batch([_image(1, "a")]), model_id="glm-ocr", max_new_tokens=7)
    with pytest.raises(LookupError, match="other"):
        block.run(image=Batch([_image(1, "a")]), model_id="other")

    (call,) = model.calls
    assert call["max_new_tokens"] == 7


@pytest.mark.parametrize(
    "params, message",
    [
        ({"task_type": "ocr"}, "Unsupported GLM-OCR task_type 'ocr'. Expected one of"),
        ({"task_type": None}, "Unsupported GLM-OCR task_type None"),
        ({"task_type": "custom"}, "`prompt` is required when task_type is 'custom'"),
        (
            {"task_type": "custom", "prompt": ""},
            "`prompt` is required when task_type is 'custom'",
        ),
        (
            {"task_type": "structured-answering"},
            "`output_structure` is required when task_type is 'structured-answering'",
        ),
        (
            {"task_type": "structured-answering", "output_structure": {}},
            "`output_structure` is required",
        ),
        (
            {"task_type": "structured-answering", "output_structure": {"a": 1}},
            "must map field names to text descriptions",
        ),
        (
            {"task_type": "custom", "prompt": Batch(["ok", None])},
            "`prompt` is required",
        ),
        (
            {"task_type": Batch(["custom", "custom"])},
            "task_type is one value for every image",
        ),
        (
            {"model_package_id": Batch([None, None])},
            "model_package_id is one value for every image",
        ),
        ({"prompt": Batch(["a", "b", "c"])}, "prompt has 3 values for 2 images"),
    ],
)
def test_malformed_task_inputs_fail_before_any_model_loads(
    params: Dict[str, Any], message: str
) -> None:
    loader = FakeLoader()
    block = GlmOcrBlock(models=ModelCache(loader))

    with pytest.raises(ValueError, match=re.escape(message)):
        block.run(image=Batch([_image(1, "a"), _image(2, "b")]), model_id="m", **params)

    assert loader.loaded == []


def test_a_grayscale_image_fails_before_any_model_loads() -> None:
    loader = FakeLoader()
    block = GlmOcrBlock(models=ModelCache(loader))
    gray = ImageData.from_tensor(torch.zeros((1, 4, 4), dtype=torch.uint8))

    with pytest.raises(ValueError, match="needs a 3-channel RGB image"):
        block.run(image=Batch([gray]), model_id="m")

    assert loader.loaded == []


class _TwoAnswers(FakeGlmOcr):
    def prompt(self, images: torch.Tensor, **kwargs: Any) -> List[str]:
        answers = super().prompt(images, **kwargs) * 2

        return answers


class _NotText(FakeGlmOcr):
    def prompt(self, images: torch.Tensor, **kwargs: Any) -> Any:
        super().prompt(images, **kwargs)

        return [{"text": "a"}]


@pytest.mark.parametrize("model_type", [_TwoAnswers, _NotText])
def test_an_answer_that_is_not_one_text_fails_the_step(model_type: type) -> None:
    block = GlmOcrBlock(models={"m": model_type()})

    with pytest.raises(ValueError, match="must answer one image with a list of one"):
        block.run(image=Batch([_image(1, "a")]), model_id="m")


def test_the_block_module_imports_neither_the_backend_nor_transformers() -> None:
    # The V2 image type already imports parts of inference_models; the GLM-OCR
    # backend and transformers load only with a model.
    code = (
        "import sys, roboflow_workflows.v2.blocks.models.glm_ocr.v1\n"
        "print(sorted(m for m in ('inference_models.models.glm_ocr.glm_ocr_hf', "
        "'transformers') if m in sys.modules))"
    )

    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )

    assert completed.stdout.strip() == "[]"


# --- compiled workflows --------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_selected_models_load_once_and_answers_keep_row_order(mode: str) -> None:
    loader = FakeLoader()
    session = _session(
        loader,
        mode=mode,
        inputs=({"type": "WorkflowParameter", "name": "model_id"},),
        model_id="$inputs.model_id",
    )
    assert loader.loaded == []  # compiling and constructing load nothing

    rows = session.run({"image": _images(), "model_id": "a"}).rows()
    other = session.run({"image": _images(), "model_id": "b"}).rows()
    again = session.run({"image": _images(), "model_id": "a"}).rows()

    assert loader.loaded == [ModelRequest("a"), ModelRequest("b")]
    assert [row["model_id"] for row in rows + other + again] == ["a"] * 3 + [
        "b"
    ] * 3 + ["a"] * 3
    assert [row["parsed_output"].split("|")[2] for row in rows] == ["20", "100", "200"]
    assert len({row["inference_id"] for row in rows + other + again}) == 9


@pytest.mark.parametrize("name", ["model_id", "task"])
def test_a_per_image_model_or_task_selector_fails_compilation(name: str) -> None:
    field = {"model_id": "model_id", "task": "task_type"}[name]
    with pytest.raises(Exception, match=f"{field} is passed once per call"):
        _session(
            FakeLoader(),
            inputs=({"type": "WorkflowBatchInput", "name": name},),
            **{field: f"$inputs.{name}"},
        )


@pytest.mark.parametrize("mode", MODES)
def test_the_default_step_reads_text_with_the_default_model(mode: str) -> None:
    loader = FakeLoader()
    session = _session(loader, mode=mode)

    (row,) = session.run({"image": _images()[:1]}).rows()

    assert loader.loaded == [ModelRequest("glm-ocr")]
    assert row["parsed_output"].startswith("Text Recognition:|6x8|20|")
    assert row["model_id"] == "glm-ocr"


def test_task_type_prompt_and_max_new_tokens_come_from_inputs_per_run() -> None:
    loader = FakeLoader()
    session = _session(
        loader,
        inputs=(
            {"type": "WorkflowParameter", "name": "task", "kind": ["string"]},
            {"type": "WorkflowParameter", "name": "prompt", "kind": ["string"]},
            {"type": "WorkflowParameter", "name": "tokens", "kind": ["integer"]},
        ),
        task_type="$inputs.task",
        prompt="$inputs.prompt",
        max_new_tokens="$inputs.tokens",
        output_structure=STRUCTURE,
    )
    image = _images()[:1]

    for task in TASK_TYPES:
        session.run({"image": image, "task": task, "prompt": "Mine.", "tokens": 5})

    prompts = [call["prompt"] for call in loader.models[0].calls]
    assert prompts == [
        "Text Recognition:",
        "Table Recognition:",
        "Formula Recognition:",
        _structured_prompt(STRUCTURE),
        "Mine.",
    ]
    assert {call["max_new_tokens"] for call in loader.models[0].calls} == {5}
    assert len(loader.loaded) == 1


def test_the_custom_prompt_comes_from_an_input_per_image() -> None:
    loader = FakeLoader()
    session = _session(
        loader,
        inputs=(
            {"type": "WorkflowParameter", "name": "task", "kind": ["string"]},
            {"type": "WorkflowBatchInput", "name": "prompt", "kind": ["string"]},
        ),
        task_type="$inputs.task",
        prompt="$inputs.prompt",
    )

    rows = session.run(
        {
            "image": _images(),
            "task": "custom",
            "prompt": ["Read the date.", "Read the time.", "Read the serial."],
        }
    ).rows()

    assert [row["parsed_output"].split("|")[0] for row in rows] == [
        "Read the date.",
        "Read the time.",
        "Read the serial.",
    ]


def test_output_structure_can_come_from_an_input() -> None:
    loader = FakeLoader()
    session = _session(
        loader,
        inputs=(
            {"type": "WorkflowParameter", "name": "fields", "kind": ["dictionary"]},
        ),
        task_type="structured-answering",
        output_structure="$inputs.fields",
    )

    session.run({"image": _images()[:1], "fields": {"total": "The total amount"}})

    (call,) = loader.models[0].calls
    assert call["prompt"] == _structured_prompt({"total": "The total amount"})


@pytest.mark.parametrize(
    "inputs, message",
    [
        ({"task": "ocr", "prompt": "x"}, "Unsupported GLM-OCR task_type 'ocr'"),
        ({"task": "custom", "prompt": ""}, "`prompt` is required"),
        ({"task": "structured-answering", "prompt": "x"}, "`output_structure` is"),
    ],
)
def test_a_selected_task_without_its_field_fails_the_step_before_loading(
    inputs: Dict[str, Any], message: str
) -> None:
    loader = FakeLoader()
    session = _session(
        loader,
        inputs=(
            {"type": "WorkflowParameter", "name": "task", "kind": ["string"]},
            {"type": "WorkflowParameter", "name": "prompt", "kind": ["string"]},
        ),
        task_type="$inputs.task",
        prompt="$inputs.prompt",
    )

    with pytest.raises(WorkflowExecutionError, match=message):
        session.run({"image": _images()[:1], **inputs})

    assert loader.loaded == []


@pytest.mark.parametrize("mode", MODES)
def test_invalid_selected_task_precedes_grayscale_check_before_loading(
    mode: str,
) -> None:
    loader = FakeLoader()
    session = _session(
        loader,
        mode=mode,
        inputs=({"type": "WorkflowParameter", "name": "task", "kind": ["string"]},),
        task_type="$inputs.task",
    )
    gray = ImageData.from_tensor(torch.zeros((1, 4, 4), dtype=torch.uint8))
    with pytest.raises(
        WorkflowExecutionError, match="Unsupported GLM-OCR task_type 'ocr'"
    ) as error:
        session.run({"image": [gray], "task": "ocr"})
    assert "needs a 3-channel RGB image" not in str(error.value)
    assert loader.loaded == []
    assert loader.models == []


@pytest.mark.parametrize(
    "inputs",
    [
        {"task": 3},
        {"task": ["custom"]},
    ],
)
def test_a_selected_task_type_of_the_wrong_kind_fails_before_loading(
    inputs: Dict[str, Any],
) -> None:
    loader = FakeLoader()
    session = _session(
        loader,
        inputs=({"type": "WorkflowParameter", "name": "task"},),
        task_type="$inputs.task",
    )

    with pytest.raises(
        WorkflowExecutionError,
        match=re.escape("parameter task_type is not a valid ['string']"),
    ):
        session.run({"image": _images()[:1], **inputs})

    assert loader.loaded == []


@pytest.mark.parametrize("mode", MODES)
def test_a_gated_image_is_skipped_and_the_others_keep_their_rows(mode: str) -> None:
    loader = FakeLoader()
    session = _session(
        loader,
        mode=mode,
        inputs=({"type": "WorkflowBatchInput", "name": "keep"},),
        steps=(
            {
                "type": Gate.type,
                "name": "keep",
                "value": "$inputs.keep",
                "next_steps": ["$steps.ocr"],
            },
        ),
    )

    rows = session.run({"image": _images(), "keep": [True, False, True]}).rows()

    assert len(loader.models[0].calls) == 2
    answers = [row["parsed_output"] for row in rows]
    assert answers[1] is None
    assert [answer.split("|")[2] for answer in (answers[0], answers[2])] == [
        "20",
        "200",
    ]


@pytest.mark.parametrize(
    "params",
    [
        {"task_type": "custom"},
        {"task_type": "custom", "prompt": ""},
        {"task_type": "structured-answering"},
        {"task_type": "structured-answering", "output_structure": {}},
        {"task_type": "structured-answering", "output_structure": {"a": 1}},
        {"task_type": "ocr"},
        {"max_new_tokens": 0},
        {"model_id": ""},
        {"prompt": 3},
        {"output_structure": "not a mapping"},
    ],
)
def test_invalid_literal_params_fail_at_compile_time(params: Dict[str, Any]) -> None:
    with pytest.raises(ParamsValidationError):
        _plan(**params)


def test_a_literal_custom_task_may_take_its_prompt_from_an_input() -> None:
    loader = FakeLoader()
    session = _session(
        loader,
        inputs=({"type": "WorkflowParameter", "name": "prompt", "kind": ["string"]},),
        task_type="custom",
        prompt="$inputs.prompt",
    )

    session.run({"image": _images()[:1], "prompt": "Read the plate."})
    with pytest.raises(WorkflowExecutionError, match="`prompt` is required"):
        session.run({"image": _images()[:1], "prompt": ""})

    assert [call["prompt"] for call in loader.models[0].calls] == ["Read the plate."]


def test_the_model_is_declared_as_a_dependent_vlm_resource() -> None:
    params = GlmOcrBlock.Params(
        image="$inputs.image", model_id="glm-ocr", model_package_id="pkg-1"
    )
    (declared,) = GlmOcrBlock.discover_dependent_resources(params)
    assert declared.details == {
        "resource": "models",
        "task_type": "vlm",
        "model_package_id": "pkg-1",
    }
    plan = _plan(model_id="glm-ocr", model_package_id="pkg-1")

    (usage,) = discover_workload(plan).resources.items

    assert (usage.resource_type, usage.identifier) == ("model", "glm-ocr")
    assert usage.used_by_steps == ["$steps.ocr"]


@pytest.mark.parametrize("mode", MODES)
def test_structured_answers_feed_a_json_parser_step(mode: str) -> None:
    plan = _plan(
        mode=mode,
        task_type="structured-answering",
        output_structure={"serial": "The serial number"},
        steps=(
            {
                "type": JsonParserBlock.type,
                "name": "parse",
                "raw_json": "$steps.ocr.parsed_output",
                "expected_fields": ["serial"],
            },
        ),
        outputs=(("serial", "$steps.parse.serial"),),
    )
    model = JsonGlmOcr()
    session = plan.create_session({"models": {"glm-ocr": model}})

    rows = session.run({"image": _images()}).rows()

    assert [row["serial"] for row in rows] == ["S-20", "S-100", "S-200"]
    assert {call["prompt"] for call in model.calls} == {
        _structured_prompt({"serial": "The serial number"})
    }


# --- real local package (opt-in) ----------------------------------------------


def test_a_real_local_package_returns_text_for_every_image() -> None:
    package = os.environ.get("GLM_OCR_PACKAGE")
    if not package or not (Path(package) / "model_config.json").is_file():
        pytest.skip(
            "UNVERIFIED: GLM_OCR_PACKAGE does not name a local inference_models "
            "package directory; real GLM-OCR inference was not run"
        )

    loader = auto_model_loader(
        device=torch.device(os.environ.get("GLM_OCR_DEVICE", "cpu"))
    )
    session = _plan(model_id=package, max_new_tokens=32).create_session(
        {"models": ModelCache(loader)}
    )
    pixels = torch.full((3, 64, 192), 255, dtype=torch.uint8)
    pixels[:, 24:40, 16:176] = 0  # a black bar on white; content is not checked

    rows = session.run(
        {"image": [ImageData.from_tensor(pixels), ImageData.from_tensor(pixels)]}
    ).rows()

    assert [type(row["parsed_output"]) for row in rows] == [str, str]


# --- one-per-call options and aligned prompts (r2: C1, C2) ---------------------


@pytest.mark.parametrize(
    "field, value",
    [
        ("max_new_tokens", Batch([64, 64])),
        ("output_structure", Batch([{"a": "b"}, {"a": "b"}])),
    ],
)
def test_a_batch_of_equal_option_values_fails_before_any_load(
    field: str, value: Batch
) -> None:
    loader = FakeLoader()
    block = GlmOcrBlock(models=ModelCache(loader))

    with pytest.raises(ValueError, match=f"{field} is one value for every image"):
        block.run(
            image=Batch([_image(1, "a"), _image(2, "b")]),
            model_id="m",
            **{field: value},
        )

    assert loader.loaded == []


@pytest.mark.parametrize(
    "prompt_indices",
    [[(0, 2), (1, 3)], [(1, 2), (0, 1)]],
    ids=["other-images", "reordered"],
)
def test_custom_prompts_must_carry_the_image_indices_before_any_load(
    prompt_indices: List[Tuple[int, ...]]
) -> None:
    loader = FakeLoader()
    block = GlmOcrBlock(models=ModelCache(loader))
    images = Batch([_image(1, "a"), _image(2, "b")], indices=[(0, 1), (1, 2)])
    prompts = Batch(["Read A.", "Read B."], indices=prompt_indices)

    with pytest.raises(ValueError, match="prompt rows .* do not match image rows"):
        block.run(image=images, model_id="m", task_type="custom", prompt=prompts)

    assert loader.loaded == []


def test_aligned_sparse_custom_prompts_follow_their_images() -> None:
    loader = FakeLoader()
    block = GlmOcrBlock(models=ModelCache(loader))
    indices = [(0, 1), (2, 5)]
    images = Batch([_image(1, "a"), _image(2, "b")], indices=indices)

    rows = block.run(
        image=images,
        model_id="m",
        task_type="custom",
        prompt=Batch(["Read A.", "Read B."], indices=indices),
    )

    assert [row["parsed_output"].split("|")[0] for row in rows] == [
        "Read A.",
        "Read B.",
    ]

