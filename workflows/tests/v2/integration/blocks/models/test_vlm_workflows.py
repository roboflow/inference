"""The four local VLM blocks in compiled V2 workflows with a ``ModelCache``.

Fake prompt models (``tests/v2/support/models/fake_prompt_models.py``) stand
in for loaded weights; nothing is downloaded and this is not real inference.
The tests check what a workflow builder relies on: models load at run time
once per request (one model per call), prompts can come from inputs per run
or per image while system prompts and limits are per run, gated
images are skipped, literal mistakes fail at compile time, and the text
answer feeds a downstream text block.
"""

import json
from typing import Any, Dict, List, Tuple

import pytest
import torch
from roboflow_workflows.v2.blocks.formatters.json_parser.v1 import JsonParserBlock
from roboflow_workflows.v2.blocks.models.cache import ModelCache, ModelRequest
from roboflow_workflows.v2.blocks.models._catalogue_vlm import BLOCKS
from roboflow_workflows.v2.blocks.models._shared.vlm import DEFAULT_PROMPT
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.errors import LineageError, ParamsValidationError
from roboflow_workflows.v2.engine.introspection import discover_workload
from roboflow_workflows.v2.engine.plan import CompileOptions

from tests.v2.support.compilation.fixture_blocks import Gate
from tests.v2.support.models.fake_prompt_models import (
    FakePromptLoader,
    FakeQwen3VL,
    FakeQwen25VL,
    FakeQwen35,
    FakeSmolVLM2,
)

MODES = ("run", "phases")
FAMILIES = {
    "roboflow_core/qwen25vl@v1": FakeQwen25VL,
    "roboflow_core/qwen3vl@v1": FakeQwen3VL,
    "roboflow_core/qwen3_5vl@v2": FakeQwen35,
    "roboflow_core/smolvlm2@v1": FakeSmolVLM2,
}
OUTPUTS = ("parsed_output", "model_id", "inference_id")


def _catalogue() -> Catalogue:
    catalogue = Catalogue([*BLOCKS, Gate, JsonParserBlock])

    return catalogue


def _plan(
    block_type: str,
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
            {"type": block_type, "name": "vlm", "image": "$inputs.image", **params},
        ],
        "outputs": [
            *(
                {"type": "JsonField", "name": name, "selector": f"$steps.vlm.{name}"}
                for name in OUTPUTS
            ),
            *(
                {"type": "JsonField", "name": name, "selector": selector}
                for name, selector in outputs
            ),
        ],
    }
    plan = compile_workflow(
        definition, catalogue=_catalogue(), options=CompileOptions(block_execution=mode)
    )

    return plan


def _images() -> List[ImageData]:
    images = [
        ImageData.from_tensor(
            torch.full((3, 6, width), value, dtype=torch.uint8), image_id=image_id
        )
        for value, width, image_id in ((20, 8, "x"), (100, 4, "y"), (200, 10, "z"))
    ]

    return images


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("block_type", FAMILIES)
def test_selected_models_load_once_and_answers_keep_row_order(
    block_type: str, mode: str
) -> None:
    loader = FakePromptLoader(FAMILIES[block_type])
    plan = _plan(
        block_type,
        mode=mode,
        inputs=({"type": "WorkflowParameter", "name": "model_id"},),
        model_id="$inputs.model_id",
    )

    session = plan.create_session({"models": ModelCache(loader)})
    assert loader.loaded == []  # compiling and constructing load nothing
    rows = session.run({"image": _images(), "model_id": "a"}).rows()
    other = session.run({"image": _images(), "model_id": "b"}).rows()
    again = session.run({"image": _images(), "model_id": "a"}).rows()

    assert loader.loaded == [ModelRequest("a"), ModelRequest("b")]
    assert [row["model_id"] for row in rows + other + again] == ["a"] * 3 + [
        "b"
    ] * 3 + ["a"] * 3
    sizes = [row["parsed_output"].split("|")[2] for row in rows]
    assert sizes == ["6x8", "6x4", "6x10"]
    assert len({row["inference_id"] for row in rows + other + again}) == 9


@pytest.mark.parametrize(
    "block_type, field",
    [
        ("roboflow_core/qwen25vl@v1", "model_id"),
        ("roboflow_core/smolvlm2@v1", "model_package_id"),
        ("roboflow_core/qwen3vl@v1", "system_prompt"),
        ("roboflow_core/qwen3_5vl@v2", "max_new_tokens"),
    ],
)
def test_a_per_image_configuration_selector_fails_compilation(
    block_type: str, field: str
) -> None:
    with pytest.raises(LineageError, match=f"vlm {field} is passed once per call"):
        _plan(
            block_type,
            inputs=({"type": "WorkflowBatchInput", "name": "value"},),
            **{"model_id": "m", field: "$inputs.value"},
        )


@pytest.mark.parametrize("mode", MODES)
def test_the_default_model_id_needs_no_step_parameter(mode: str) -> None:
    loader = FakePromptLoader(FakeQwen35)
    session = _plan("roboflow_core/qwen3_5vl@v2", mode=mode).create_session(
        {"models": ModelCache(loader)}
    )

    rows = session.run({"image": _images()[:1]}).rows()

    assert loader.loaded == [ModelRequest("qwen3_5-0.8b")]
    (call,) = loader.models[0].calls
    assert call["prompt"] == (
        f"{DEFAULT_PROMPT}<system_prompt>You are a helpful assistant."
    )
    assert call["enable_thinking"] is False
    assert rows[0]["model_id"] == "qwen3_5-0.8b"


def test_prompts_come_from_inputs_per_run_and_per_image() -> None:
    loader = FakePromptLoader(FakeQwen25VL)
    plan = _plan(
        "roboflow_core/qwen25vl@v1",
        inputs=(
            {"type": "WorkflowBatchInput", "name": "question", "kind": ["string"]},
            {"type": "WorkflowParameter", "name": "system", "kind": ["string"]},
            {"type": "WorkflowParameter", "name": "tokens", "kind": ["integer"]},
        ),
        model_id="m",
        prompt="$inputs.question",
        system_prompt="$inputs.system",
        max_new_tokens="$inputs.tokens",
    )
    session = plan.create_session({"models": ModelCache(loader)})

    rows = session.run(
        {
            "image": _images(),
            "question": ["Count cars.", "Name the colour.", None],
            "system": "Be brief.",
            "tokens": 12,
        }
    ).rows()

    calls = loader.models[0].calls
    assert [call["prompt"] for call in calls] == [
        "Count cars.<system_prompt>Be brief.",
        "Name the colour.<system_prompt>Be brief.",
        f"{DEFAULT_PROMPT}<system_prompt>Be brief.",
    ]
    assert [call["max_new_tokens"] for call in calls] == [12, 12, 12]
    assert rows[0]["parsed_output"].endswith("|20|12")


@pytest.mark.parametrize("mode", MODES)
def test_a_gated_image_is_skipped_and_the_others_keep_their_rows(mode: str) -> None:
    loader = FakePromptLoader(FakeSmolVLM2)
    plan = _plan(
        "roboflow_core/smolvlm2@v1",
        mode=mode,
        inputs=({"type": "WorkflowBatchInput", "name": "keep"},),
        steps=(
            {
                "type": Gate.type,
                "name": "keep",
                "value": "$inputs.keep",
                "next_steps": ["$steps.vlm"],
            },
        ),
        model_id="m",
    )
    session = plan.create_session({"models": ModelCache(loader)})

    rows = session.run({"image": _images(), "keep": [True, False, True]}).rows()

    assert len(loader.models[0].calls) == 2
    answers = [row["parsed_output"] for row in rows]
    assert answers[1] is None
    assert [answer.split("|")[3] for answer in (answers[0], answers[2])] == [
        "20",
        "200",
    ]


@pytest.mark.parametrize(
    "block_type, params",
    [
        ("roboflow_core/qwen3vl@v1", {"max_new_tokens": 0}),
        ("roboflow_core/qwen3vl@v1", {"model_id": ""}),
        ("roboflow_core/qwen3vl@v1", {"prompt": 3}),
        ("roboflow_core/smolvlm2@v1", {"system_prompt": "Be brief."}),
        ("roboflow_core/qwen3_5vl@v2", {"enable_thinking": True}),
    ],
)
def test_invalid_or_unported_literal_params_fail_at_compile_time(
    block_type: str, params: Dict[str, Any]
) -> None:
    with pytest.raises(ParamsValidationError):
        _plan(block_type, **params)


def test_the_model_is_declared_as_a_dependent_resource() -> None:
    plan = _plan("roboflow_core/qwen3vl@v1", model_id="qwen3vl-2b-instruct")

    (usage,) = discover_workload(plan).resources.items

    assert (usage.resource_type, usage.identifier) == ("model", "qwen3vl-2b-instruct")
    assert usage.used_by_steps == ["$steps.vlm"]


class _JsonQwen3VL(FakeQwen3VL):
    """Answers with JSON naming the image's mean pixel, like a JSON prompt."""

    def prompt(self, images: List[torch.Tensor], **kwargs: Any) -> List[str]:
        super().prompt(images, **kwargs)
        answers = [
            "```json\n"
            + json.dumps({"brightness": round(image.float().mean().item())})
            + "\n```"
            for image in images
        ]

        return answers


def test_the_text_answer_feeds_a_json_parser_step() -> None:
    plan = _plan(
        "roboflow_core/qwen3vl@v1",
        model_id="m",
        prompt="Return JSON with the brightness.",
        steps=(
            {
                "type": JsonParserBlock.type,
                "name": "parse",
                "raw_json": "$steps.vlm.parsed_output",
                "expected_fields": ["brightness"],
            },
        ),
        outputs=(("brightness", "$steps.parse.brightness"),),
    )
    session = plan.create_session({"models": {"m": _JsonQwen3VL()}})

    rows = session.run({"image": _images()}).rows()

    assert [row["brightness"] for row in rows] == [20, 100, 200]
