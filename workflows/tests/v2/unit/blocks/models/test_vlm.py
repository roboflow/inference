"""The four local VLM blocks called directly on fake prompt models.

The fakes (``tests/v2/support/models/fake_prompt_models.py``) echo what
reached them, so these tests check wiring: prompts, options, images, model
resolution and output shapes. They are not model inference. Compiled
workflows are in ``integration/blocks/models/test_vlm_workflows.py``.
"""

import ast
import importlib.util
import inspect
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
import torch
from roboflow_workflows.v2.blocks.models.cache import ModelCache, ModelRequest
from roboflow_workflows.v2.blocks.models._catalogue_vlm import BLOCKS
from roboflow_workflows.v2.blocks.models._shared.vlm import DEFAULT_PROMPT
from roboflow_workflows.v2.blocks.models.qwen3vl.v1 import Qwen3VLBlock
from roboflow_workflows.v2.blocks.models.qwen25vl.v1 import Qwen25VLBlock
from roboflow_workflows.v2.blocks.models.qwen3_5vl.v2 import Qwen35VLBlock
from roboflow_workflows.v2.blocks.models.smolvlm2.v1 import SmolVLM2Block
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.image import ImageData

from tests.v2.support.models.fake_prompt_models import (
    FakePromptLoader,
    FakeQwen3VL,
    FakeQwen25VL,
    FakeQwen35,
    FakeSmolVLM2,
)

QWEN25_SYSTEM = "You are a Qwen2.5-VL model that can answer questions about any image."
QWEN3_SYSTEM = "You are a Qwen3-VL model that can answer questions about any image."
QWEN35_SYSTEM = "You are a helpful assistant."
# Backend prompt() default of max_new_tokens: (value, environment variable that
# replaces it, or None when the backend writes the literal value).
BACKEND_MAX_NEW_TOKENS = {
    "qwen25vl": (512, "INFERENCE_MODELS_QWEN25_VL_DEFAULT_MAX_NEW_TOKENS"),
    "qwen3vl": (512, None),
    "qwen3_5": (512, None),
    "smolvlm2": (400, "INFERENCE_MODELS_SMOL_VLM_DEFAULT_MAX_NEW_TOKENS"),
}

# block, fake model, backend source (relative to inference_models), class name
FAMILIES = {
    "qwen25vl": (
        Qwen25VLBlock,
        FakeQwen25VL,
        "models/qwen25vl/qwen25vl_hf.py",
        "Qwen25VLHF",
    ),
    "qwen3vl": (Qwen3VLBlock, FakeQwen3VL, "models/qwen3vl/qwen3vl_hf.py", "Qwen3VLHF"),
    "qwen3_5": (Qwen35VLBlock, FakeQwen35, "models/qwen3_5/qwen3_5_hf.py", "Qwen35HF"),
    "smolvlm2": (
        SmolVLM2Block,
        FakeSmolVLM2,
        "models/smolvlm/smolvlm_hf.py",
        "SmolVLMHF",
    ),
}


def _image(value: int, image_id: str, *, height: int = 6, width: int = 8) -> ImageData:
    pixels = torch.full((3, height, width), value, dtype=torch.uint8)
    image = ImageData.from_tensor(pixels, image_id=image_id)

    return image


def _block(name: str) -> Tuple[Any, FakePromptLoader]:
    block_type, model_type, _, _ = FAMILIES[name]
    loader = FakePromptLoader(model_type)
    block = block_type(models=ModelCache(loader))

    return block, loader


def _prompt_parameters(function_node: ast.FunctionDef) -> List[Tuple[str, str]]:
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


def _backend_source(relative: str) -> ast.Module:
    package = Path(importlib.util.find_spec("inference_models").origin).parent
    tree = ast.parse((package / relative).read_text())

    return tree


def _source_prompt(relative: str, class_name: str) -> ast.FunctionDef:
    tree = _backend_source(relative)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "prompt":
                    return item

    raise AssertionError(f"{class_name}.prompt not found in {relative}")


def _environment_default(variable: str) -> int:
    for node in ast.walk(_backend_source("configuration.py")):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == variable
        ):
            (default,) = [kw.value for kw in node.value.keywords if kw.arg == "default"]
            return ast.literal_eval(default)

    raise AssertionError(f"{variable} not found in configuration.py")


def _record_keywords(model: Any) -> List[Dict[str, Any]]:
    """Make ``model.prompt`` also record the keywords the caller passed."""
    passed: List[Dict[str, Any]] = []
    fake_prompt = model.prompt

    def prompt(images: List[torch.Tensor], **keywords: Any) -> List[Any]:
        passed.append(keywords)
        answers = fake_prompt(images, **keywords)

        return answers

    model.prompt = prompt

    return passed


# --- fakes copy the backend contract -------------------------------------------


@pytest.mark.parametrize("name", FAMILIES)
def test_fake_prompt_has_the_backend_parameter_names_and_defaults(name: str) -> None:
    _, model_type, relative, class_name = FAMILIES[name]
    fake_source = inspect.getsource(model_type.prompt)
    fake_node = ast.parse(inspect.cleandoc("\n" + fake_source)).body[0]

    backend = _prompt_parameters(_source_prompt(relative, class_name))

    assert _prompt_parameters(fake_node) == backend


@pytest.mark.parametrize(
    "name, same_as_backend",
    [("qwen25vl", True), ("qwen3vl", False), ("qwen3_5", True)],
)
def test_only_qwen3vl_overrides_the_backend_system_prompt_default(
    name: str, same_as_backend: bool
) -> None:
    # The block always sends its default, so the backend's own never applies.
    # test_defaults_match_v1_and_each_image_gets_one_rgb_call pins the V1 values.
    block_type, _, relative, _ = FAMILIES[name]

    (backend_default,) = [
        ast.literal_eval(node.value)
        for node in ast.walk(_backend_source(relative))
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Attribute)
        and node.targets[0].attr == "default_system_prompt"
    ]

    assert (block_type.default_system_prompt == backend_default) is same_as_backend


@pytest.mark.parametrize("name", FAMILIES)
def test_max_new_tokens_descriptions_state_the_backend_prompt_default(
    name: str,
) -> None:
    # Reads the backend source; the process environment is not consulted.
    block_type, _, relative, class_name = FAMILIES[name]
    value, variable = BACKEND_MAX_NEW_TOKENS[name]
    prompt_parameters = dict(_prompt_parameters(_source_prompt(relative, class_name)))
    description = block_type.Params.model_fields["max_new_tokens"].description

    if variable is None:
        assert prompt_parameters["max_new_tokens"] == str(value)
        assert f"{value}, fixed" in description
    else:
        assert prompt_parameters["max_new_tokens"] == variable
        assert _environment_default(variable) == value
        assert f"{value}, or the {variable} environment variable" in description


# --- prompts, defaults and options ---------------------------------------------


@pytest.mark.parametrize(
    "name, expected_prompt, extra",
    [
        ("qwen25vl", f"{DEFAULT_PROMPT}<system_prompt>{QWEN25_SYSTEM}", {}),
        ("qwen3vl", f"{DEFAULT_PROMPT}<system_prompt>{QWEN3_SYSTEM}", {}),
        (
            "qwen3_5",
            f"{DEFAULT_PROMPT}<system_prompt>{QWEN35_SYSTEM}",
            {"enable_thinking": False},
        ),
        ("smolvlm2", DEFAULT_PROMPT, {}),
    ],
)
def test_defaults_match_v1_and_each_image_gets_one_rgb_call(
    name: str, expected_prompt: str, extra: Dict[str, Any]
) -> None:
    block, loader = _block(name)
    images = [_image(10, "a"), _image(200, "b", height=4)]

    results = block.run(image=Batch(images), model_id="m")

    (model,) = loader.models
    assert len(model.calls) == 2
    for call, image in zip(model.calls, images):
        (tensor,) = call["images"]
        assert tensor is image.tensor_image  # no copy, colour flip or move
        assert call["prompt"] == expected_prompt
        assert call["input_color_format"] == "rgb"
        assert call["kwargs"] == {}
        for keyword, value in extra.items():
            assert call[keyword] == value
    fake = type(model).__name__
    tokens = inspect.signature(type(model).prompt).parameters["max_new_tokens"].default
    assert [result["parsed_output"] for result in results] == [
        f"{fake}|{expected_prompt}|6x8|10|{tokens}",
        f"{fake}|{expected_prompt}|4x8|200|{tokens}",
    ]
    assert [result["model_id"] for result in results] == ["m", "m"]
    assert len({result["inference_id"] for result in results}) == 2


@pytest.mark.parametrize("name", FAMILIES)
def test_max_new_tokens_none_leaves_the_keyword_to_the_backend_default(
    name: str,
) -> None:
    block_type, model_type, _, _ = FAMILIES[name]
    model = model_type()
    passed = _record_keywords(model)
    signature = inspect.signature(model_type.prompt)

    block_type(models={"m": model}).run(
        image=Batch([_image(1, "a")]), model_id="m", max_new_tokens=None
    )

    (keywords,) = passed
    (call,) = model.calls
    assert "max_new_tokens" not in keywords
    assert call["max_new_tokens"] == signature.parameters["max_new_tokens"].default


@pytest.mark.parametrize("name", ["qwen25vl", "qwen3vl", "qwen3_5"])
def test_qwen_prompt_and_system_prompt_are_joined_by_the_marker(name: str) -> None:
    block, loader = _block(name)

    block.run(
        image=Batch([_image(1, "a")]),
        model_id="m",
        prompt="Count cars.",
        system_prompt="Answer with a number.",
        max_new_tokens=7,
    )

    (call,) = loader.models[0].calls
    assert call["prompt"] == "Count cars.<system_prompt>Answer with a number."
    assert call["max_new_tokens"] == 7


@pytest.mark.parametrize("name", ["qwen25vl", "qwen3vl", "qwen3_5"])
def test_empty_prompts_fall_back_to_the_defaults_like_v1(name: str) -> None:
    block, loader = _block(name)
    default_system = block.default_system_prompt

    block.run(image=Batch([_image(1, "a")]), model_id="m", prompt="", system_prompt="")

    (call,) = loader.models[0].calls
    assert call["prompt"] == f"{DEFAULT_PROMPT}<system_prompt>{default_system}"


def test_smolvlm_passes_a_prompt_with_the_marker_unchanged() -> None:
    block, loader = _block("smolvlm2")

    block.run(image=Batch([_image(1, "a")]), model_id="m", prompt="a <system_prompt> b")

    (call,) = loader.models[0].calls
    assert call["prompt"] == "a <system_prompt> b"
    assert call["images_to_single_prompt"] is True  # one image per call


def test_smolvlm_rejects_a_system_prompt_before_any_model_loads() -> None:
    block, loader = _block("smolvlm2")

    with pytest.raises(ValueError, match="has no system prompt"):
        block.run(image=Batch([_image(1, "a")]), model_id="m", system_prompt="x")

    assert loader.loaded == []
    assert "system_prompt" not in SmolVLM2Block.Params.model_fields


@pytest.mark.parametrize("field", ["prompt", "system_prompt"])
def test_a_qwen_marker_in_the_text_fails_before_any_model_loads(field: str) -> None:
    block, loader = _block("qwen3vl")

    with pytest.raises(ValueError, match=f"{field} contains '<system_prompt>'"):
        block.run(
            image=Batch([_image(1, "a"), _image(2, "b")]),
            model_id="m",
            **{field: "one<system_prompt>two"},
        )

    assert loader.loaded == []


def test_a_marker_in_a_later_image_prompt_fails_before_any_model_loads() -> None:
    block, loader = _block("qwen25vl")

    with pytest.raises(ValueError, match="prompt contains '<system_prompt>'"):
        block.run(
            image=Batch([_image(1, "a"), _image(2, "b")]),
            model_id="m",
            prompt=Batch(["fine", "one<system_prompt>two"]),
        )

    assert loader.loaded == []


# --- per-image values and model resolution -------------------------------------


def test_per_image_prompts_share_the_calls_one_model() -> None:
    block, loader = _block("smolvlm2")
    images = [_image(1, "a"), _image(2, "b"), _image(3, "c")]

    results = block.run(
        image=Batch(images), model_id="x", prompt=Batch(["first", None, "third"])
    )
    block.run(image=Batch(images[:1]), model_id="y", prompt="later")

    assert loader.loaded == [ModelRequest("x"), ModelRequest("y")]
    first, second = loader.models
    assert [call["prompt"] for call in first.calls] == [
        "first",
        DEFAULT_PROMPT,
        "third",
    ]
    assert [call["prompt"] for call in second.calls] == ["later"]
    assert [result["model_id"] for result in results] == ["x", "x", "x"]
    assert [result["parsed_output"].split("|")[3] for result in results] == [
        "1",
        "2",
        "3",
    ]


def test_one_system_prompt_joins_every_per_image_prompt() -> None:
    block, loader = _block("qwen3vl")

    block.run(
        image=Batch([_image(1, "a"), _image(2, "b")]),
        model_id="m",
        prompt=Batch(["Count cars.", "Count people."]),
        system_prompt="Answer with a number.",
    )

    (model,) = loader.models
    assert [call["prompt"] for call in model.calls] == [
        "Count cars.<system_prompt>Answer with a number.",
        "Count people.<system_prompt>Answer with a number.",
    ]


def test_a_per_image_system_prompt_fails_before_any_model_loads() -> None:
    block, loader = _block("qwen25vl")

    with pytest.raises(ValueError, match="system_prompt is one value for every image"):
        block.run(
            image=Batch([_image(1, "a"), _image(2, "b")]),
            model_id="m",
            system_prompt=Batch(["one", "two"]),
        )

    assert loader.loaded == []


def test_a_supplied_mapping_serves_models_and_names_missing_ones() -> None:
    model = FakeQwen25VL()
    block = Qwen25VLBlock(models={"qwen25-vl-7b": model})

    block.run(image=Batch([_image(1, "a")]), model_id="qwen25-vl-7b")
    with pytest.raises(LookupError, match="not in the 'models' resource"):
        block.run(image=Batch([_image(1, "a")]), model_id="other")

    assert len(model.calls) == 1


def test_per_image_model_ids_fail_before_any_model_call() -> None:
    model = FakeQwen3VL()
    block = Qwen3VLBlock(models={"m": model})

    with pytest.raises(ValueError, match="model_id is one value for every image"):
        block.run(
            image=Batch([_image(1, "a"), _image(2, "b")]),
            model_id=Batch(["m", "missing"]),
        )

    assert model.calls == []


def test_an_empty_batch_loads_no_model() -> None:
    block, loader = _block("qwen3_5")

    assert block.run(image=Batch([]), model_id="m") == []
    assert loader.loaded == []


def test_a_prompt_batch_of_the_wrong_length_fails() -> None:
    block, loader = _block("qwen3_5")

    with pytest.raises(ValueError, match="prompt has 1 values for 2 images"):
        block.run(
            image=Batch([_image(1, "a"), _image(2, "b")]),
            model_id="m",
            prompt=Batch(["only one"]),
        )

    assert loader.loaded == []


def test_a_grayscale_image_fails_before_any_model_loads() -> None:
    block, loader = _block("qwen25vl")
    gray = ImageData.from_tensor(torch.zeros((1, 4, 4), dtype=torch.uint8))

    with pytest.raises(ValueError, match="needs a 3-channel RGB image"):
        block.run(image=Batch([gray]), model_id="m")

    assert loader.loaded == []


class _ThinkingQwen35(FakeQwen35):
    def prompt(self, images: List[torch.Tensor], **kwargs: Any) -> List[Any]:
        answers = super().prompt(images, **{**kwargs, "enable_thinking": True})

        return answers


class _TwoAnswers(FakeQwen3VL):
    def prompt(self, images: List[torch.Tensor], **kwargs: Any) -> List[str]:
        answers = super().prompt(images, **kwargs) * 2

        return answers


@pytest.mark.parametrize("model", [_ThinkingQwen35(), _TwoAnswers()])
def test_an_answer_that_is_not_one_text_fails_the_step(model: Any) -> None:
    block = Qwen35VLBlock(models={"m": model})

    with pytest.raises(ValueError, match="must answer one image with a list of one"):
        block.run(image=Batch([_image(1, "a")]), model_id="m")


# --- declaration ---------------------------------------------------------------


@pytest.mark.parametrize(
    "block_type, default_model_id",
    [
        (Qwen25VLBlock, "qwen25-vl-7b"),
        (Qwen3VLBlock, "qwen3vl-2b-instruct"),
        (Qwen35VLBlock, "qwen3_5-0.8b"),
        (SmolVLM2Block, "smolvlm2/smolvlm-2.2b-instruct"),
    ],
)
def test_model_ids_default_to_v1_and_are_reported_as_vlm_resources(
    block_type: Any, default_model_id: str
) -> None:
    params = block_type.Params(image="$inputs.image")

    (resource,) = block_type.discover_dependent_resources(params)

    assert params.model_id == default_model_id
    assert resource.identifier == default_model_id
    assert resource.details == {
        "resource": "models",
        "task_type": "vlm",
        "model_package_id": None,
    }


def test_blocks_have_distinct_types_and_the_same_outputs() -> None:
    assert [block.type for block in BLOCKS] == [
        "roboflow_core/qwen25vl@v1",
        "roboflow_core/qwen3vl@v1",
        "roboflow_core/qwen3_5vl@v2",
        "roboflow_core/smolvlm2@v1",
    ]
    for block in BLOCKS:
        assert list(block.outputs) == ["parsed_output", "model_id", "inference_id"]


# --- one-per-call options and aligned prompts (r2: C1, C2) ---------------------


@pytest.mark.parametrize("name", ["qwen25vl", "smolvlm2"])
def test_a_batch_of_equal_generation_limits_fails_before_any_load(name: str) -> None:
    block, loader = _block(name)

    with pytest.raises(ValueError, match="max_new_tokens is one value for every image"):
        block.run(
            image=Batch([_image(1, "a"), _image(2, "b")]),
            model_id="m",
            max_new_tokens=Batch([64, 64]),
        )

    assert loader.loaded == []


@pytest.mark.parametrize(
    "prompt_indices",
    [[(0, 2), (1, 3)], [(1, 2), (0, 1)]],
    ids=["other-images", "reordered"],
)
def test_prompts_must_carry_the_image_indices_before_any_load(
    prompt_indices: List[Tuple[int, ...]]
) -> None:
    block, loader = _block("qwen3vl")
    images = Batch([_image(1, "a"), _image(2, "b")], indices=[(0, 1), (1, 2)])
    prompts = Batch(["Count cars.", "Count people."], indices=prompt_indices)

    with pytest.raises(ValueError, match="prompt rows .* do not match image rows"):
        block.run(image=images, model_id="m", prompt=prompts)

    assert loader.loaded == []


def test_aligned_sparse_prompts_follow_their_images() -> None:
    block, loader = _block("smolvlm2")
    indices = [(0, 1), (2, 5)]

    block.run(
        image=Batch([_image(1, "a"), _image(2, "b")], indices=indices),
        model_id="m",
        prompt=Batch(["Count cars.", "Count people."], indices=indices),
    )

    (model,) = loader.models
    assert [call["prompt"] for call in model.calls] == ["Count cars.", "Count people."]

