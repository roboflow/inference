"""``roboflow_core/qwen_vlm@v4`` on both backends, against in-process fakes only.

Native calls reach fake ``prompt`` models through a ``ModelCache``; OpenRouter
calls reach ``FakeChatServer`` (an ``httpx.MockTransport``) or a recording
``FakePlatformClient``. Fake answers are fixed strings, not model output, so
these tests check prompts, requests, ordering, validation and decoding.

Source-grounded checks compare against the latest V1 module
``core_steps/models/foundation/qwen_vlm/v4.py`` directly: model tables, native
prompts, OpenRouter messages and reasoning configs must be equal.
"""

import ast
import base64
import importlib.util
import json
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pytest
import torch
from roboflow_workflows.core_steps.common.openrouter import get_native_quantizations
from roboflow_workflows.core_steps.models.foundation.qwen_vlm import v4 as v1_qwen
from roboflow_workflows.v2.blocks.formatters.json_parser.v1 import JsonParserBlock
from roboflow_workflows.v2.blocks.geometry.detection_offset.v1 import DetectionOffsetBlock
from roboflow_workflows.v2.blocks.language._shared import openrouter_vendors as openrouter_vendors
from roboflow_workflows.v2.blocks.language.chat import ChatRequestError
from roboflow_workflows.v2.blocks.language.openrouter.v3 import MODEL_NATIVE_QUANTIZATIONS, OPENROUTER_TASKS
from roboflow_workflows.v2.blocks.models.qwen_vlm import v4 as qwen_vlm
from roboflow_workflows.v2.blocks.models.cache import ModelCache, ModelRequest
from roboflow_workflows.v2.blocks.models.qwen_vlm.v4 import QwenVLMBlock
from roboflow_workflows.v2.blocks.models.qwen3_5vl.v2 import Qwen35VLBlock
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.language import LANGUAGE_MODEL_OUTPUT_KIND
from roboflow_workflows.v2.data.predictions import (
    CLASSIFICATION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
)
from roboflow_workflows.v2.data.secrets import SECRET_KIND
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.declaration import spec_of

from inference_models.configuration import (
    INFERENCE_MODELS_QWEN25_VL_DEFAULT_MAX_NEW_TOKENS,
)
from inference_models.models.base.classification import ClassificationPrediction
from tests.v2.support.language.fake_chat import (
    FakeChatServer,
    FakePlatformClient,
    chat_completion_body,
    error_body,
)
from tests.v2.support.models.fake_prompt_models import (
    FakePromptLoader,
    FakePromptModel,
    FakeQwen25VL,
)

OPENROUTER_KEY = "sk-or-test-QWEN-7f3a"
ROBOFLOW_KEY = "rf-test-ROBOFLOW-qwen-11d2"
FAKE_OPENROUTER_URL = "http://openrouter.test/api/v1"
DETECTION_ANSWER = '[{"box_2d": [0, 0, 500, 500], "label": "car"}]'
CATALOGUE = Catalogue(
    [QwenVLMBlock, DetectionOffsetBlock, JsonParserBlock],
    kinds=[SECRET_KIND, LANGUAGE_MODEL_OUTPUT_KIND],
)
IMAGE_INPUT = {"type": "WorkflowBatchInput", "name": "image", "kind": ["image"]}


class FakeQwen35Scripted(FakePromptModel):
    """``Qwen35HF.prompt`` parameters; answers with ``reply(call index)``."""

    def __init__(self, request: Optional[ModelRequest] = None, *, reply: Any = None):
        super().__init__(request)
        self.reply = reply

    def prompt(
        self,
        images: List[torch.Tensor],
        prompt: str = None,
        input_color_format: str = None,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        skip_special_tokens: bool = True,
        enable_thinking: bool = False,
        **kwargs,
    ) -> List[Any]:
        echoes = self._answer(
            images,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            input_color_format=input_color_format,
            enable_thinking=enable_thinking,
            kwargs=kwargs,
        )
        if self.reply is None:
            return echoes

        answers = self.reply(len(self.calls) - 1)

        return answers


class FakeQwen38(FakePromptModel):
    """``Qwen38HF.prompt`` parameters: ``None`` defaults resolved by the backend."""

    def prompt(
        self,
        images: List[torch.Tensor],
        prompt: str = None,
        input_color_format: str = None,
        max_new_tokens: Optional[int] = None,
        do_sample: Optional[bool] = None,
        skip_special_tokens: bool = True,
        enable_thinking: bool = False,
        **kwargs,
    ) -> List[str]:
        answers = self._answer(
            images,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            input_color_format=input_color_format,
            enable_thinking=enable_thinking,
            kwargs=kwargs,
        )

        return answers


def _image(value: int = 0, *, width: int = 64, height: int = 48) -> ImageData:
    image = ImageData.from_tensor(
        torch.full((3, height, width), value, dtype=torch.uint8)
    )

    return image


def _textured_image(width: int = 40, height: int = 30) -> ImageData:
    pixels = torch.from_numpy(
        np.random.default_rng(7).integers(0, 255, (3, height, width), dtype=np.uint8)
    )
    image = ImageData.from_tensor(pixels)

    return image


def _bgr(image: ImageData) -> np.ndarray:
    pixels = image.tensor_image.permute(1, 2, 0).numpy()
    bgr = cv2.cvtColor(np.ascontiguousarray(pixels), cv2.COLOR_RGB2BGR)

    return bgr


def _answering(content: Any, **body: Any) -> FakeChatServer:
    server = FakeChatServer(
        lambda path, request: (200, chat_completion_body(content, **body))
    )

    return server


def _openrouter_block(server: FakeChatServer, **resources: Any) -> QwenVLMBlock:
    block = QwenVLMBlock(
        openrouter_base_url=FAKE_OPENROUTER_URL,
        openrouter_http_client=server.http_client(),
        **resources,
    )

    return block


def _native_block(model_type: type = FakeQwen35Scripted, **model_options: Any):
    loader = FakePromptLoader(partial(model_type, **model_options))
    block = QwenVLMBlock(models=ModelCache(loader))

    return block, loader


def _run(block: QwenVLMBlock, images: List[ImageData], resolved=None, **raw: Any):
    defaults = {"images": "$inputs.images", "api_key": OPENROUTER_KEY}
    params = spec_of(QwenVLMBlock).validate_params({**defaults, **raw})
    arguments = {name: getattr(params, name) for name in type(params).model_fields}
    arguments.update(images=Batch(images), **(resolved or {}))
    results = block.run(**arguments)

    return results


def _uploaded_size(body: Dict[str, Any]) -> tuple:
    url = body["messages"][0]["content"][0]["image_url"]["url"]
    data = base64.b64decode(url.split(",", 1)[1])
    decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    height, width = decoded.shape[:2]

    return width, height


# --- source grounding against V1 qwen_vlm@v4 ---------------------------------


def test_model_tables_and_defaults_match_v1() -> None:
    v1_variants = v1_qwen.MODEL_VARIANTS
    native = {
        label: variant["model_id"]
        for label, variant in v1_variants.items()
        if variant["backend"] == "native"
    }
    openrouter = {
        label: (variant["model_id"], variant.get("reasoning_required", False))
        for label, variant in v1_variants.items()
        if variant["backend"] == "openrouter"
    }

    assert list(qwen_vlm.NATIVE_MODELS.items()) == list(native.items())
    assert [
        (label, (model.slug, model.reasoning_required))
        for label, model in qwen_vlm.OPENROUTER_MODELS.items()
    ] == list(openrouter.items())
    assert qwen_vlm.THINKING_MODEL_VERSIONS == tuple(
        v1_qwen.NATIVE_THINKING_MODEL_VERSIONS
    )
    assert qwen_vlm.FINE_TUNED_LABEL == v1_qwen.FINE_TUNED_NATIVE_LABEL
    assert qwen_vlm.DEFAULT_NATIVE_MODEL_VERSION == v1_qwen.DEFAULT_NATIVE_MODEL_VERSION
    assert (
        qwen_vlm.DEFAULT_OPENROUTER_MODEL_VERSION
        == v1_qwen.DEFAULT_OPENROUTER_MODEL_VERSION
    )
    assert qwen_vlm.DEFAULT_MAX_TOKENS == v1_qwen.QWEN_OPENROUTER_DEFAULT_MAX_TOKENS
    assert qwen_vlm.QWEN_BOX_FORMAT == v1_qwen.DETECTION_BOX_FORMAT
    assert qwen_vlm.REASONING_LEVELS == tuple(v1_qwen.REASONING_EFFORT_OPTIONS)
    assert qwen_vlm.REASONING_EFFORT_METADATA == v1_qwen.REASONING_EFFORT_METADATA
    assert OPENROUTER_TASKS == tuple(v1_qwen.SUPPORTED_TASK_TYPES_LIST)
    assert (
        QwenVLMBlock.metadata["supported_model_variants"]
        == v1_qwen.NATIVE_SUPPORTED_VARIANTS
    )
    assert openrouter_vendors.ZAI_REQUIRED_REASONING_FALLBACK == (
        v1_qwen.REASONING_REQUIRED_FALLBACK_EFFORT
    )
    assert openrouter_vendors.OPENROUTER_MAX_BASE64_BYTES == (
        v1_qwen.OPENROUTER_MAX_BASE64_BYTES
    )


@pytest.mark.parametrize("task_type", OPENROUTER_TASKS)
@pytest.mark.parametrize("prompt", ["What is here?", ""])
def test_native_prompts_equal_v1(task_type: str, prompt: str) -> None:
    fields = {"output_structure": {"count": "cars"}, "classes": ["car", "bus"]}

    built = qwen_vlm.native_prompt(task_type, prompt=prompt, **fields)

    assert built == v1_qwen._build_native_prompt(
        task_type, prompt, fields["output_structure"], fields["classes"]
    )


@pytest.mark.parametrize("task_type", OPENROUTER_TASKS)
def test_openrouter_messages_equal_v1(task_type: str) -> None:
    server = _answering("ok")
    image = _textured_image()
    fields = {"prompt": "What?", "classes": ["car"], "output_structure": {"n": "x"}}

    _run(
        _openrouter_block(server),
        [image],
        backend="openrouter",
        task_type=task_type,
        **fields,
    )

    expected = v1_qwen.build_qwen_openrouter_prompts(
        images=[_bgr(image)], task_type=task_type, **fields
    )
    assert server.requests[0].body["messages"] == expected[0]


@pytest.mark.parametrize(
    "label", ["Qwen 3.7 Plus", "Qwen 3.8 Max", "Qwen 3 VL 8B Thinking"]
)
@pytest.mark.parametrize("effort", ["none", "low", "medium", "high"])
def test_reasoning_config_equals_v1(label: str, effort: str) -> None:
    server = _answering("ok")

    _run(
        _openrouter_block(server),
        [_image()],
        backend="openrouter",
        openrouter_model_version=label,
        prompt="Hi",
        reasoning_effort=effort,
    )

    required = v1_qwen.MODEL_VARIANTS[label].get("reasoning_required", False)
    assert server.requests[0].body["reasoning"] == v1_qwen.build_reasoning_config(
        effort, reasoning_required=required
    )


def test_fake_qwen38_copies_the_backend_prompt_signature() -> None:
    package = Path(importlib.util.find_spec("inference_models").origin).parent
    source = (package / "models/qwen3_8/qwen3_8_hf.py").read_text()
    backend = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef) and node.name == "prompt"
    )
    fake = next(
        node
        for node in ast.walk(ast.parse(Path(__file__).read_text()))
        if isinstance(node, ast.ClassDef) and node.name == "FakeQwen38"
    ).body[1]

    def parameters(function: ast.FunctionDef) -> List[tuple]:
        arguments = function.args.args
        defaults = [None] * (len(arguments) - len(function.args.defaults))
        defaults += [ast.unparse(default) for default in function.args.defaults]
        names = [argument.arg for argument in arguments]

        return list(zip(names, defaults))

    assert parameters(fake)[2:] == parameters(backend)[2:]


# --- OpenRouter backend ------------------------------------------------------


def test_openrouter_sends_one_image_first_message_with_v1_defaults() -> None:
    server = _answering("A cat.", reasoning="Looked at it.")

    results = _run(
        _openrouter_block(server), [_image()], backend="openrouter", prompt="What?"
    )

    request = server.requests[0]
    assert request.headers["authorization"] == f"Bearer {OPENROUTER_KEY}"
    assert request.body["model"] == "qwen/qwen3.7-plus"
    assert request.body["max_tokens"] == 2048
    assert "temperature" not in request.body
    assert request.body["reasoning"] == {"enabled": False}
    assert request.body["provider"] == {"data_collection": "deny"}
    assert [part["type"] for part in request.body["messages"][0]["content"]] == [
        "image_url",
        "text",
    ]
    assert results == [
        {
            "output": "A cat.",
            "classes": None,
            "thinking": "Looked at it.",
            "input_tokens": 13,
            "output_tokens": 5,
            "predictions": None,
            "error_status": False,
            "inference_id": results[0]["inference_id"],
        }
    ]


def test_openrouter_sends_set_temperature_none_tokens_and_quantization() -> None:
    server = _answering("ok")

    _run(
        _openrouter_block(server),
        [_image()],
        backend="openrouter",
        openrouter_model_version="Qwen 3.5 27B",
        prompt="Hi",
        temperature=0.4,
        max_tokens=None,
        privacy_level="zdr",
    )

    body = server.requests[0].body
    assert (body["temperature"], body["max_tokens"]) == (0.4, 2048)
    assert body["provider"] == {
        "data_collection": "deny",
        "zdr": True,
        "quantizations": list(get_native_quantizations(model="qwen/qwen3.5-27b")),
    }
    assert {
        model.slug: MODEL_NATIVE_QUANTIZATIONS.get(model.slug)
        for model in qwen_vlm.OPENROUTER_MODELS.values()
    } == {
        model.slug: get_native_quantizations(model=model.slug)
        for model in qwen_vlm.OPENROUTER_MODELS.values()
    }


def test_managed_key_posts_through_the_platform_proxy() -> None:
    platform = FakePlatformClient(lambda endpoint, payload: chat_completion_body("ok"))
    block = QwenVLMBlock(platform_client=platform, api_key=ROBOFLOW_KEY)

    _run(
        block,
        [_image()],
        backend="openrouter",
        prompt="Hi",
        api_key="rf_key:account",
    )

    post = platform.posts[0]
    assert (post["endpoint"], post["api_key"]) == ("apiproxy/openrouter", ROBOFLOW_KEY)
    assert post["payload"]["openrouter_api_key"] == "rf_key:account"
    assert post["payload"]["reasoning"] == {"enabled": False}
    assert "temperature" not in post["payload"]


def test_oversized_upload_shrinks_and_boxes_stay_in_original_pixels(
    monkeypatch,
) -> None:
    monkeypatch.setattr(openrouter_vendors, "OPENROUTER_MAX_BASE64_BYTES", 2_000)
    server = _answering(DETECTION_ANSWER)
    image = ImageData.from_tensor(
        torch.from_numpy(
            np.random.default_rng(3).integers(0, 255, (3, 100, 200), dtype=np.uint8)
        )
    )

    results = _run(
        _openrouter_block(server),
        [image],
        backend="openrouter",
        task_type="object-detection",
        classes=["car"],
    )

    width, height = _uploaded_size(server.requests[0].body)
    assert width < 200 and height < 100
    assert results[0]["predictions"].xyxy.tolist() == [[0.0, 0.0, 100.0, 50.0]]


def test_openrouter_answers_keep_image_order_under_concurrency() -> None:
    def reply(path: str, body: Dict[str, Any]) -> tuple:
        width, _ = _uploaded_size(body)
        return 200, chat_completion_body(f"width {width}")

    server = FakeChatServer(reply)
    images = [_image(width=width, height=8) for width in (16, 24, 32, 40, 48)]

    results = _run(
        _openrouter_block(server),
        images,
        backend="openrouter",
        prompt="Hi",
        max_concurrent_requests=3,
    )

    assert [result["output"] for result in results] == [
        "width 16",
        "width 24",
        "width 32",
        "width 40",
        "width 48",
    ]
    assert len({result["inference_id"] for result in results}) == 5


def test_a_failed_request_raises_without_the_key_and_decodes_nothing() -> None:
    server = FakeChatServer(
        lambda path, body: (401, error_body(f"bad key {OPENROUTER_KEY}"))
    )

    with pytest.raises(ChatRequestError) as caught:
        _run(_openrouter_block(server), [_image()], backend="openrouter", prompt="Hi")

    assert OPENROUTER_KEY not in str(caught.value)
    assert caught.value.__cause__ is None and caught.value.__context__ is None


@pytest.mark.parametrize(
    "resolved, message",
    [
        ({"openrouter_model_version": "Qwen 9"}, "Unknown OpenRouter Qwen variant"),
        ({"temperature": 3.0}, "temperature"),
        ({"temperature": True}, "temperature"),
        ({"api_key": None}, "resolved to None"),
        ({"prompt": 7}, "prompt must be text"),
        ({"classes": ["car", "car"]}, "repeat"),
        ({"classes": None}, "`classes` parameter required"),
    ],
)
def test_openrouter_checks_resolved_values_before_any_request(
    resolved: Dict[str, Any], message: str
) -> None:
    server = _answering(DETECTION_ANSWER)

    with pytest.raises(ValueError, match=message):
        _run(
            _openrouter_block(server),
            [_image()],
            resolved=resolved,
            backend="openrouter",
            task_type="object-detection",
            classes=["car"],
        )

    assert server.requests == []


def test_openrouter_empty_batch_sends_nothing() -> None:
    server = _answering("ok")

    results = _run(_openrouter_block(server), [], backend="openrouter", prompt="Hi")

    assert results == []
    assert server.requests == []


# --- native backend ----------------------------------------------------------


def test_native_runs_the_label_model_with_the_v1_prompt_and_options() -> None:
    block, loader = _native_block()

    results = _run(
        block, [_image(10), _image(200)], prompt="What?", model_package_id="p1"
    )

    assert loader.loaded == [ModelRequest("qwen3_5-2b", "p1")]
    calls = loader.models[0].calls
    expected_prompt = v1_qwen._build_native_prompt("unconstrained", "What?", None, None)
    assert [call["prompt"] for call in calls] == [expected_prompt] * 2
    assert [call["images"][0].float().mean().item() for call in calls] == [10, 200]
    assert {(call["max_new_tokens"], call["enable_thinking"]) for call in calls} == {
        (2048, False)
    }
    assert {call["input_color_format"] for call in calls} == {"rgb"}
    assert [result["output"].split("|")[3] for result in results] == ["10", "200"]
    assert {result["input_tokens"] for result in results} == {None}
    assert {result["output_tokens"] for result in results} == {None}
    assert {result["thinking"] for result in results} == {""}


@pytest.mark.parametrize(
    "model_version, model_id, thinking",
    [
        ("Qwen 3.5 VL 2B", "qwen3_5-2b", True),
        ("Qwen 3.8 VL 27B", "qwen3_8-27b", True),
        ("Qwen 3.5 VL 0.8B", "qwen3_5-0.8b", False),
        ("Fine-tuned model", "ws/3", True),
    ],
)
def test_enable_thinking_reaches_only_thinking_variants(
    model_version: str, model_id: str, thinking: bool
) -> None:
    def reply(index: int) -> List[Any]:
        return [{"thinking": "Two wheels.", "answer": "A bike."}]

    block, loader = _native_block(reply=reply)

    results = _run(
        block,
        [_image()],
        prompt="What?",
        model_version=model_version,
        fine_tuned_model_id="ws/3",
        enable_thinking=True,
    )

    assert loader.loaded == [ModelRequest(model_id)]
    assert loader.models[0].calls[0]["enable_thinking"] is thinking
    assert (results[0]["output"], results[0]["thinking"]) == ("A bike.", "Two wheels.")


def test_thinking_without_an_answer_keeps_the_trace_and_an_empty_output() -> None:
    block, _ = _native_block(
        reply=lambda index: [{"thinking": "Cut off", "answer": ""}]
    )

    results = _run(block, [_image()], prompt="What?", enable_thinking=True)

    assert (results[0]["output"], results[0]["thinking"]) == ("", "Cut off")


@pytest.mark.parametrize(
    "answers",
    [[], ["a", "b"], [None], [{"answer": 3}], [{"thinking": 1, "answer": "a"}], "a"],
)
def test_native_answer_of_another_shape_fails(answers: Any) -> None:
    block, _ = _native_block(reply=lambda index: answers)

    with pytest.raises(ValueError, match="must answer one image"):
        _run(block, [_image()], prompt="What?")


def test_qwen25_swallows_enable_thinking_and_none_tokens_keep_backend_default() -> None:
    block, loader = _native_block(FakeQwen25VL)

    _run(
        block,
        [_image()],
        prompt="What?",
        model_version="Qwen 2.5 VL 7B",
        max_tokens=None,
    )

    call = loader.models[0].calls[0]
    assert call["kwargs"] == {"enable_thinking": False}
    assert call["max_new_tokens"] == INFERENCE_MODELS_QWEN25_VL_DEFAULT_MAX_NEW_TOKENS


def test_qwen38_none_tokens_reach_the_backend_none_default() -> None:
    block, loader = _native_block(FakeQwen38)

    _run(
        block, [_image()], prompt="Hi", model_version="Qwen 3.8 VL 27B", max_tokens=None
    )

    assert loader.models[0].calls[0]["max_new_tokens"] is None


def test_native_and_openrouter_decode_the_same_answer_the_same_way() -> None:
    block, _ = _native_block(reply=lambda index: [DETECTION_ANSWER])
    server = _answering(DETECTION_ANSWER)
    fields = {"task_type": "object-detection", "classes": ["car", "bus"]}
    images = [_image(width=200, height=100)]

    native = _run(block, images, **fields)[0]
    remote = _run(_openrouter_block(server), images, backend="openrouter", **fields)[0]

    for result in (native, remote):
        assert result["predictions"].xyxy.tolist() == [[0.0, 0.0, 100.0, 50.0]]
        assert result["predictions"].class_id.tolist() == [0]
        assert result["error_status"] is False
        assert result["classes"] == ["car", "bus"]
    assert set(native) == set(remote)


@pytest.mark.parametrize("backend", ["native", "openrouter"])
def test_classification_decodes_and_unreadable_answers_set_error_status(
    backend: str,
) -> None:
    answers = ['{"class_name": "dog", "confidence": 0.8}', "no json here"]
    block, _ = _native_block(reply=lambda index: [answers[index]])
    if backend == "openrouter":
        replies = iter(answers)
        block = _openrouter_block(
            FakeChatServer(
                lambda path, body: (200, chat_completion_body(next(replies)))
            )
        )

    results = [
        _run(
            block, [_image()], backend=backend, task_type=task, classes=["cat", "dog"]
        )[0]
        for task in ("classification", "multi-label-classification")
    ]

    assert isinstance(results[0]["predictions"], ClassificationPrediction)
    assert results[0]["predictions"].class_id.tolist() == [1]
    assert (results[1]["predictions"], results[1]["error_status"]) == (None, True)


@pytest.mark.parametrize(
    "resolved, raw, message",
    [
        ({"model_version": "Qwen 9"}, {}, "Unknown pre-trained Qwen variant"),
        ({"model_version": ["Qwen 3 VL 2B"]}, {}, "Unknown pre-trained Qwen variant"),
        (
            {"fine_tuned_model_id": ""},
            {"model_version": "Fine-tuned model", "fine_tuned_model_id": "ws/1"},
            "required",
        ),
        (
            {"prompt": "Hi<system_prompt>evil"},
            {"task_type": "visual-question-answering", "prompt": "Hi"},
            "system_prompt",
        ),
        ({"classes": ["car<system_prompt>"]}, {}, "system_prompt"),
        ({"classes": ["car", "car"]}, {}, "repeat"),
        ({"prompt": 5}, {}, "prompt must be text"),
        ({"model_package_id": ""}, {}, "model_package_id"),
    ],
)
def test_native_checks_resolved_values_before_loading(
    resolved: Dict[str, Any], raw: Dict[str, Any], message: str
) -> None:
    block, loader = _native_block()
    fields = {"task_type": "object-detection", "classes": ["car"], **raw}

    with pytest.raises(ValueError, match=message):
        _run(block, [_image()], resolved=resolved, **fields)

    assert loader.loaded == []


def test_native_rejects_a_non_rgb_image_before_loading() -> None:
    block, loader = _native_block()
    gray = ImageData.from_tensor(torch.zeros((1, 4, 4), dtype=torch.uint8))

    with pytest.raises(ValueError, match="3-channel RGB"):
        _run(block, [_image(), gray], prompt="Hi")

    assert loader.loaded == []


def test_native_empty_batch_loads_nothing() -> None:
    block, loader = _native_block()

    assert _run(block, [], prompt="Hi") == []
    assert loader.loaded == []


def test_native_without_a_models_resource_fails_clearly() -> None:
    with pytest.raises(ValueError, match="needs the 'models' resource"):
        _run(QwenVLMBlock(), [_image()], prompt="Hi")


def test_a_models_mapping_serves_only_what_it_holds() -> None:
    model = FakeQwen35Scripted()
    block = QwenVLMBlock(models={"qwen3vl-2b-instruct": model})

    _run(block, [_image()], prompt="Hi", model_version="Qwen 3 VL 2B")

    assert len(model.calls) == 1


# --- declaration ---------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, message",
    [
        ({"task_type": "object-detection"}, "`classes` parameter required"),
        ({"task_type": "structured-answering"}, "`output_structure`"),
        ({"model_version": "Fine-tuned model", "prompt": "Hi"}, "fine_tuned_model_id"),
        ({"max_tokens": 1, "prompt": "Hi"}, "greater than 1"),
        ({"temperature": 2.5, "prompt": "Hi"}, "less than or equal to 2"),
        ({"prompt": 3}, "prompt"),
        ({"enable_thinking": "yes", "prompt": "Hi"}, "enable_thinking"),
        ({"reasoning_effort": "xhigh", "prompt": "Hi"}, "reasoning_effort"),
        ({"model_version": "Qwen 3.7 Plus", "prompt": "Hi"}, "model_version"),
        ({"max_concurrent_requests": 0, "prompt": "Hi"}, "greater than or equal to 1"),
    ],
)
def test_params_reject_invalid_literals(raw: Dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        spec_of(QwenVLMBlock).validate_params({"images": "$inputs.images", **raw})


def test_a_stale_fine_tuned_label_is_ignored_by_the_openrouter_backend() -> None:
    params = spec_of(QwenVLMBlock).validate_params(
        {
            "images": "$inputs.images",
            "backend": "openrouter",
            "model_version": "Fine-tuned model",
            "prompt": "Hi",
        }
    )

    assert params.model_version == "Fine-tuned model"


def test_describe_outputs_narrows_predictions_by_task() -> None:
    def outputs(task_type: str) -> Dict[str, Any]:
        params = spec_of(QwenVLMBlock).validate_params(
            {"images": "$inputs.images", "task_type": task_type, "classes": ["a"]}
        )
        return QwenVLMBlock.describe_outputs(params)

    assert list(outputs("ocr")) == [
        "output",
        "classes",
        "thinking",
        "input_tokens",
        "output_tokens",
        "predictions",
        "error_status",
        "inference_id",
    ]
    assert outputs("object-detection")["predictions"].kinds == (
        OBJECT_DETECTION_PREDICTION_KIND,
    )
    assert outputs("classification")["predictions"].kinds == (
        CLASSIFICATION_PREDICTION_KIND,
    )


@pytest.mark.parametrize(
    "raw, identifier",
    [
        ({}, "qwen3_5-2b"),
        ({"model_version": "$inputs.qwen"}, "$inputs.qwen"),
        ({"model_version": "Fine-tuned model", "fine_tuned_model_id": "ws/2"}, "ws/2"),
    ],
)
def test_native_dependency_matches_the_local_vlm_blocks(
    raw: Dict[str, Any], identifier: str
) -> None:
    spec = spec_of(QwenVLMBlock)
    params = spec.validate_params(
        {"images": "$inputs.images", "prompt": "Hi", "model_package_id": "p", **raw}
    )
    reference = spec_of(Qwen35VLBlock).validate_params(
        {"image": "$inputs.images", "model_id": identifier, "model_package_id": "p"}
    )

    resources = QwenVLMBlock.discover_dependent_resources(params)

    assert resources == Qwen35VLBlock.discover_dependent_resources(reference)
    assert QwenVLMBlock.discover_work_operations(params) == ["model_inference"]


@pytest.mark.parametrize(
    "label, identifier",
    [("Qwen 3.8 Max", "qwen/qwen3.8-max"), ("$inputs.qwen", "$inputs.qwen")],
)
def test_openrouter_dependency_names_the_slug_or_selector(
    label: str, identifier: str
) -> None:
    params = spec_of(QwenVLMBlock).validate_params(
        {
            "images": "$inputs.images",
            "backend": "openrouter",
            "openrouter_model_version": label,
            "prompt": "Hi",
        }
    )

    resources = QwenVLMBlock.discover_dependent_resources(params)

    assert [(r.resource_type, r.identifier, r.details) for r in resources] == [
        (
            "third_party_model",
            identifier,
            {"provider": "openrouter", "resource": "openrouter_http_client"},
        )
    ]
    assert QwenVLMBlock.discover_work_operations(params) == [
        "model_inference",
        "external_request",
        "image_encoding",
    ]


# --- compiled workflows --------------------------------------------------------


def _compile(steps: List[Dict[str, Any]], outputs: Dict[str, str], inputs=None):
    definition = {
        "version": "2.0",
        "inputs": inputs or [IMAGE_INPUT],
        "steps": steps,
        "outputs": [
            {"type": "JsonField", "name": name, "selector": selector}
            for name, selector in outputs.items()
        ],
    }
    plan = compile_workflow(definition, catalogue=CATALOGUE)

    return plan


def test_compiled_native_detections_feed_a_geometry_step() -> None:
    plan = _compile(
        [
            {
                "type": "roboflow_core/qwen_vlm@v4",
                "name": "qwen",
                "images": "$inputs.image",
                "task_type": "object-detection",
                "classes": ["car"],
            },
            {
                "type": "roboflow_core/detection_offset@v1",
                "name": "offset",
                "predictions": "$steps.qwen.predictions",
                "offset_width": 10,
                "offset_height": 10,
            },
        ],
        {"boxes": "$steps.offset.predictions", "tokens": "$steps.qwen.input_tokens"},
    )
    model = FakeQwen35Scripted(reply=lambda index: [DETECTION_ANSWER])
    session = plan.create_session(resources={"models": {"qwen3_5-2b": model}})

    rows = session.run(
        {"image": [_image(width=200, height=100), _image(width=40, height=20)]}
    ).rows()

    assert [row["boxes"].xyxy.tolist() for row in rows] == [
        [[0.0, 0.0, 105.0, 55.0]],
        [[0.0, 0.0, 25.0, 15.0]],
    ]
    assert [row["tokens"] for row in rows] == [None, None]
    assert len(model.calls) == 2


def test_compiled_openrouter_structured_answer_feeds_the_json_parser() -> None:
    plan = _compile(
        [
            {
                "type": "roboflow_core/qwen_vlm@v4",
                "name": "qwen",
                "images": "$inputs.image",
                "backend": "openrouter",
                "openrouter_model_version": "$inputs.qwen_model",
                "task_type": "structured-answering",
                "output_structure": {"count": "number of cars"},
                "api_key": "$inputs.key",
            },
            {
                "type": "roboflow_core/json_parser@v1",
                "name": "parse",
                "raw_json": "$steps.qwen.output",
                "expected_fields": ["count"],
            },
        ],
        {"count": "$steps.parse.count", "tokens": "$steps.qwen.output_tokens"},
        inputs=[
            IMAGE_INPUT,
            {"type": "WorkflowParameter", "name": "key", "kind": ["secret"]},
            {"type": "WorkflowParameter", "name": "qwen_model", "kind": ["string"]},
        ],
    )
    server = _answering('```json\n{"count": 3}\n```')
    session = plan.create_session(
        resources={"openrouter_http_client": server.http_client()}
    )

    result = session.run(
        {"image": [_image()], "key": OPENROUTER_KEY, "qwen_model": "Qwen 3.8 Max"}
    )

    assert result.rows() == [{"count": 3, "tokens": 5}]
    assert server.requests[0].body["model"] == "qwen/qwen3.8-max"
    assert server.requests[0].body["reasoning"] == {"effort": "low"}
    assert OPENROUTER_KEY not in json.dumps(result.rows(serialize=True))


def test_compiled_runtime_label_is_checked_before_loading() -> None:
    plan = _compile(
        [
            {
                "type": "roboflow_core/qwen_vlm@v4",
                "name": "qwen",
                "images": "$inputs.image",
                "model_version": "$inputs.qwen_model",
                "prompt": "Hi",
            }
        ],
        {"answer": "$steps.qwen.output"},
        inputs=[
            IMAGE_INPUT,
            {"type": "WorkflowParameter", "name": "qwen_model", "kind": ["string"]},
        ],
    )
    block_models: List[ModelRequest] = []

    def load(request: ModelRequest) -> FakePromptModel:
        block_models.append(request)
        return FakeQwen35Scripted(request)

    session = plan.create_session(resources={"models": ModelCache(load)})

    with pytest.raises(Exception) as caught:
        session.run({"image": [_image()], "qwen_model": "Qwen 2"})

    assert "Unknown pre-trained Qwen variant 'Qwen 2'" in _chain_text(caught.value)
    assert block_models == []
    rows = session.run({"image": [_image()], "qwen_model": "Qwen 3 VL 2B"}).rows()
    assert block_models == [ModelRequest("qwen3vl-2b-instruct")]
    assert rows[0]["answer"].startswith("FakeQwen35Scripted|Hi<system_prompt>")


def test_a_per_image_prompt_is_rejected_when_compiling() -> None:
    with pytest.raises(Exception, match="prompt is passed once per call"):
        _compile(
            [
                {
                    "type": "roboflow_core/qwen_vlm@v4",
                    "name": "qwen",
                    "images": "$inputs.image",
                    "prompt": "$inputs.question",
                }
            ],
            {"answer": "$steps.qwen.output"},
            inputs=[
                IMAGE_INPUT,
                {"type": "WorkflowBatchInput", "name": "question", "kind": ["string"]},
            ],
        )


def _chain_text(error: Optional[BaseException]) -> str:
    texts = []
    while error is not None:
        texts.append(f"{type(error).__name__}: {error}")
        error = error.__cause__ or error.__context__

    return "\n".join(texts)
